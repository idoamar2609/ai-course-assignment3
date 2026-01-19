import ext_plant
from collections import deque

id = ["318780301"]


class Controller:
    """
    Improved controller for ext_plant:

    Key upgrades vs your ex33.py:
    1) Reliability tracked per (robot, action_type) with Laplace smoothing.
    2) POUR uses expected-value (accounts for failure wasting water+turns).
    3) Intent/commitment per robot (reduces dithering/oscillation).
    4) Simple global plant assignment (robots split targets instead of dogpiling).
    5) Hopeless-robot suppression (handles extreme probs like 0.0001 fast).
    6) Corridor-safe movement still: uses wall-only BFS for planning, but avoids stepping into occupied cell.
    """

    # -----------------------
    # Init
    # -----------------------
    def __init__(self, game: ext_plant.Game):
        self.original_game = game
        self.rows = game.rows
        self.cols = game.cols
        self.walls = set(game.walls)

        self.plant_max_reward = game.get_plants_max_reward()
        self.capacities = game.get_capacities()
        self.horizon = game.get_max_steps()

        # Online reliability (per action type)
        # key: (rid, act) where act in {"UP","DOWN","LEFT","RIGHT","LOAD","POUR"}
        self.attempts = {}
        self.successes = {}
        self.fail_streak = {rid: 0 for rid in self.capacities}

        # Intent / commitment
        # ("PLANT", pos) or ("TAP", pos) or None
        self.intent = {rid: None for rid in self.capacities}
        self.intent_age = {rid: 0 for rid in self.capacities}

        # Last transition tracking
        self._last_state = None
        self._last_action = None  # "UP(11)" etc.

        # Tuning knobs (safe defaults across your suite)
        self.INTENT_TTL = 10              # steps before reconsidering target
        self.INTENT_IMPROVE_RATIO = 1.35  # if new target is 35% better, switch early

        self.WATER_COST = 1.2             # value/cost of wasting 1 unit of water
        self.TURN_COST = 0.12             # small penalty per action (prefers progress)
        self.HOPELESS_FAIL_STREAK = 6     # quick shutdown
        self.HOPELESS_POUR_THRESH = 0.15  # if learned p(pour) too low, avoid

    # -----------------------
    # Parsing helpers
    # -----------------------
    @staticmethod
    def _parse_action(s: str):
        """Parse 'ACTION(rid)' -> (ACTION, rid). Return (None, None) for RESET/invalid."""
        if s is None:
            return None, None
        s = s.strip()
        if s == "RESET":
            return "RESET", None
        if "(" not in s or not s.endswith(")"):
            return None, None
        act = s.split("(", 1)[0].strip().upper()
        inside = s.split("(", 1)[1][:-1].strip()
        if not inside.isdigit():
            return None, None
        return act, int(inside)

    @staticmethod
    def _robots_dict(robots_t):
        """robots_t: tuple of (rid, (r,c), load) -> dict rid -> ((r,c), load)"""
        return {rid: (pos, load) for (rid, pos, load) in robots_t}

    @staticmethod
    def _plants_dict(plants_t):
        """plants_t: tuple of ((r,c), need) -> dict pos -> need"""
        return {pos: need for (pos, need) in plants_t}

    @staticmethod
    def _taps_dict(taps_t):
        """taps_t: tuple of ((r,c), water) -> dict pos -> water"""
        return {pos: water for (pos, water) in taps_t}

    # -----------------------
    # Reliability model
    # -----------------------
    def _p(self, rid, act):
        """Laplace-smoothed success probability for (rid, act)."""
        a = self.attempts.get((rid, act), 0)
        s = self.successes.get((rid, act), 0)
        return (s + 1.0) / (a + 2.0)

    def _record(self, rid, act, success: bool):
        """Update counters for (rid, act) and rid fail streak."""
        self.attempts[(rid, act)] = self.attempts.get((rid, act), 0) + 1
        if success:
            self.successes[(rid, act)] = self.successes.get((rid, act), 0) + 1
            self.fail_streak[rid] = 0
        else:
            self.fail_streak[rid] = self.fail_streak.get(rid, 0) + 1

    def _infer_last_action_success(self, prev_state, action_str, curr_state):
        """
        Infer whether chosen action succeeded by comparing prev_state -> curr_state.

        MOVE success: robot ended at intended next cell.
        LOAD success: load increased by 1 at same pos.
        POUR success: plant need at pos decreased by 1 AND robot load decreased by 1.
            (On POUR fail, load decreases by 1 but plant doesn't change in this env.)
        """
        act, rid = self._parse_action(action_str)
        if act is None or act == "RESET" or rid is None:
            return None, None, None  # ignore

        prev_robots, prev_plants, _prev_taps, _ = prev_state
        curr_robots, curr_plants, _curr_taps, _ = curr_state

        prevR = self._robots_dict(prev_robots)
        currR = self._robots_dict(curr_robots)
        if rid not in prevR or rid not in currR:
            return rid, act, None

        (pr, pc), pload = prevR[rid]
        (cr, cc), cload = currR[rid]

        if act in ("UP", "DOWN", "LEFT", "RIGHT"):
            if act == "UP":
                intended = (pr - 1, pc)
            elif act == "DOWN":
                intended = (pr + 1, pc)
            elif act == "LEFT":
                intended = (pr, pc - 1)
            else:
                intended = (pr, pc + 1)
            return rid, act, ((cr, cc) == intended)

        if act == "LOAD":
            return rid, act, (cload == pload + 1 and (cr, cc) == (pr, pc))

        if act == "POUR":
            prevP = self._plants_dict(prev_plants)
            currP = self._plants_dict(curr_plants)
            prev_need = prevP.get((pr, pc), None)
            curr_need = currP.get((pr, pc), None)

            plant_decreased = False
            if prev_need is not None:
                if curr_need is None and prev_need == 1:
                    plant_decreased = True
                elif curr_need is not None and curr_need == prev_need - 1:
                    plant_decreased = True

            success = (plant_decreased and cload == pload - 1 and (cr, cc) == (pr, pc))
            return rid, act, success

        return rid, act, None

    def _update_reliability_stats(self, current_state):
        """Call once per choose_next_action, before choosing new action."""
        if self._last_state is None or self._last_action is None:
            self._last_state = current_state
            self._last_action = None
            return

        rid, act, success = self._infer_last_action_success(self._last_state, self._last_action, current_state)
        if rid is not None and act is not None and success is not None:
            self._record(rid, act, success)

        self._last_state = current_state
        self._last_action = None

    def _robot_is_hopeless(self, rid):
        """
        Suppress robots that are clearly failing to avoid wasting horizon.
        Works great for the 0.0001 case.
        """
        if self.fail_streak.get(rid, 0) >= self.HOPELESS_FAIL_STREAK:
            return True

        a_pour = self.attempts.get((rid, "POUR"), 0)
        a_move = (
            self.attempts.get((rid, "UP"), 0) +
            self.attempts.get((rid, "DOWN"), 0) +
            self.attempts.get((rid, "LEFT"), 0) +
            self.attempts.get((rid, "RIGHT"), 0)
        )

        # Don't disable with no data
        if a_pour + a_move < 6:
            return False

        p_pour = self._p(rid, "POUR") if a_pour > 0 else 0.5
        p_move = 0.0
        if a_move > 0:
            s_move = (
                self.successes.get((rid, "UP"), 0) +
                self.successes.get((rid, "DOWN"), 0) +
                self.successes.get((rid, "LEFT"), 0) +
                self.successes.get((rid, "RIGHT"), 0)
            )
            p_move = (s_move + 1.0) / (a_move + 2.0)
        else:
            p_move = 0.5

        return (p_pour < self.HOPELESS_POUR_THRESH and p_move < 0.45)

    # -----------------------
    # BFS
    # -----------------------
    def bfs_distance(self, start, goal, occupied):
        """BFS distance avoiding walls and occupied cells."""
        if start == goal:
            return 0
        q = deque([(start, 0)])
        vis = {start}
        while q:
            (r, c), d = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                nxt = (nr, nc)
                if nxt in vis or nxt in self.walls or nxt in occupied:
                    continue
                if nxt == goal:
                    return d + 1
                vis.add(nxt)
                q.append((nxt, d + 1))
        return float("inf")

    def bfs_path_ignore_robots(self, start, goal):
        """Wall-only BFS path; returns list of directions."""
        if start == goal:
            return []
        q = deque([(start, [])])
        vis = {start}
        while q:
            (r, c), path = q.popleft()
            for direction, (dr, dc) in [("UP", (-1, 0)), ("DOWN", (1, 0)), ("LEFT", (0, -1)), ("RIGHT", (0, 1))]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                nxt = (nr, nc)
                if nxt in vis or nxt in self.walls:
                    continue
                npath = path + [direction]
                if nxt == goal:
                    return npath
                vis.add(nxt)
                q.append((nxt, npath))
        return []

    @staticmethod
    def _next_cell(pos, direction):
        r, c = pos
        if direction == "UP":
            return (r - 1, c)
        if direction == "DOWN":
            return (r + 1, c)
        if direction == "LEFT":
            return (r, c - 1)
        if direction == "RIGHT":
            return (r, c + 1)
        return pos

    # -----------------------
    # Scoring / assignment
    # -----------------------
    def _expected_pour_value(self, rid, plant_pos):
        """
        Expected value for POUR on plant_pos for robot rid.
        Uses learned p(POUR) and plant reward, penalizes wasted water+turn.
        """
        rew = self.plant_max_reward.get(plant_pos, 1)
        p = self._p(rid, "POUR")
        return p * rew - (1.0 - p) * self.WATER_COST - self.TURN_COST

    def _score_plant_for_robot(self, rid, robot_pos, plant_pos, occupied_excl_self):
        """
        Score choosing plant_pos as target from robot_pos given current occupancy.
        A ratio-ish metric: expected reward per travel step.
        """
        dist = self.bfs_distance(robot_pos, plant_pos, occupied_excl_self)
        if dist == float("inf"):
            return -1e18
        # higher reward + higher p(pour) => better
        rew = self.plant_max_reward.get(plant_pos, 1)
        p_pour = self._p(rid, "POUR")
        # (rew*p) per unit distance. add small epsilon
        return (rew * p_pour) / (dist + 1.0)

    def _assign_plants_globally(self, robots_t, remaining_plants, occupied_all):
        """
        Greedy matching: pick highest (robot, plant) score first,
        avoid assigning same plant to multiple robots.
        Returns dict rid -> plant_pos.
        """
        if not remaining_plants:
            return {}

        pairs = []
        for rid, (r, c), load in robots_t:
            if load <= 0:
                continue
            if self._robot_is_hopeless(rid):
                continue
            robot_pos = (r, c)
            occ = occupied_all - {robot_pos}
            for plant_pos in remaining_plants.keys():
                v = self._score_plant_for_robot(rid, robot_pos, plant_pos, occ)
                pairs.append((v, rid, plant_pos))

        pairs.sort(reverse=True, key=lambda x: x[0])
        assigned = {}
        taken = set()
        for v, rid, plant_pos in pairs:
            if rid in assigned:
                continue
            if plant_pos in taken:
                continue
            assigned[rid] = plant_pos
            taken.add(plant_pos)
        return assigned

    def _best_tap_for_robot(self, robot_pos, taps_pos, occupied_excl_self):
        """Pick closest tap by BFS distance (occupancy-aware)."""
        best = None
        bestd = float("inf")
        for t in taps_pos:
            d = self.bfs_distance(robot_pos, t, occupied_excl_self)
            if d < bestd:
                bestd = d
                best = t
        return best, bestd

    def _refresh_intent(self, rid, robot_pos, load, remaining_plants, tap_positions, occupied_excl_self, assigned_plant):
        """
        Maintain intent with TTL and improvement ratio.
        """
        self.intent_age[rid] += 1

        # If robot has water, intent should be plant
        if load > 0:
            candidate = assigned_plant
            if candidate is None and remaining_plants:
                # fallback: local best
                best = None
                bestv = -1e18
                for ppos in remaining_plants:
                    v = self._score_plant_for_robot(rid, robot_pos, ppos, occupied_excl_self)
                    if v > bestv:
                        bestv, best = v, ppos
                candidate = best

            # Validate current
            cur = self.intent[rid]
            cur_pos = cur[1] if cur and cur[0] == "PLANT" else None
            cur_ok = (cur_pos is not None and cur_pos in remaining_plants)

            if (not cur_ok) or (self.intent_age[rid] > self.INTENT_TTL):
                if candidate is not None:
                    self.intent[rid] = ("PLANT", candidate)
                    self.intent_age[rid] = 0
                else:
                    self.intent[rid] = None
                    self.intent_age[rid] = 0
                return

            # consider switching if much better candidate exists
            if candidate is not None and cur_pos is not None and candidate != cur_pos:
                cur_v = self._score_plant_for_robot(rid, robot_pos, cur_pos, occupied_excl_self)
                cand_v = self._score_plant_for_robot(rid, robot_pos, candidate, occupied_excl_self)
                if cand_v > cur_v * self.INTENT_IMPROVE_RATIO:
                    self.intent[rid] = ("PLANT", candidate)
                    self.intent_age[rid] = 0
            return

        # If robot empty, intent should be tap
        candidate_tap = None
        if tap_positions:
            candidate_tap, _ = self._best_tap_for_robot(robot_pos, tap_positions, occupied_excl_self)

        cur = self.intent[rid]
        cur_pos = cur[1] if cur and cur[0] == "TAP" else None
        cur_ok = (cur_pos is not None and cur_pos in tap_positions)

        if (not cur_ok) or (self.intent_age[rid] > self.INTENT_TTL):
            if candidate_tap is not None:
                self.intent[rid] = ("TAP", candidate_tap)
                self.intent_age[rid] = 0
            else:
                self.intent[rid] = None
                self.intent_age[rid] = 0
            return

        if candidate_tap is not None and cur_pos is not None and candidate_tap != cur_pos:
            # switch if much closer
            cur_d = self.bfs_distance(robot_pos, cur_pos, occupied_excl_self)
            cand_d = self.bfs_distance(robot_pos, candidate_tap, occupied_excl_self)
            if cand_d + 2 < cur_d:  # decent improvement
                self.intent[rid] = ("TAP", candidate_tap)
                self.intent_age[rid] = 0

    # -----------------------
    # Main decision
    # -----------------------
    def choose_next_action(self, state):
        self._update_reliability_stats(state)

        robots_t, plants_t, taps_t, total_need = state

        if total_need == 0:
            self._last_action = "RESET"
            return "RESET"

        # occupied positions
        occupied_all = {(rr, cc) for (rid, (rr, cc), l) in robots_t}

        # plants/taps sets
        plant_needs = {pos: need for (pos, need) in plants_t}
        remaining_plants = {pos: need for pos, need in plant_needs.items() if need > 0}
        tap_positions = {pos for (pos, water) in taps_t}

        # short-horizon behavior (sip-load big tanks late-game)
        t_now = self.original_game.get_current_steps()
        steps_left = max(0, self.horizon - t_now)
        short_horizon = steps_left <= 35

        # global plant assignment for robots with load
        assigned = self._assign_plants_globally(robots_t, remaining_plants, occupied_all)

        best_action = None
        best_score = -1e18

        for rid, (r, c), load in robots_t:
            robot_pos = (r, c)
            cap = self.capacities[rid]
            occupied_excl_self = occupied_all - {robot_pos}

            # If hopeless and someone else exists, usually skip
            if self._robot_is_hopeless(rid) and len(robots_t) > 1:
                continue

            # Refresh intent (commitment)
            self._refresh_intent(
                rid,
                robot_pos,
                load,
                remaining_plants,
                tap_positions,
                occupied_excl_self,
                assigned.get(rid, None),
            )

            # Decide candidate actions for this robot
            act = None
            score = -1e18

            # 1) If on plant and have water: consider POUR by expected value
            if robot_pos in remaining_plants and load > 0:
                ev = self._expected_pour_value(rid, robot_pos)
                # Still allow POUR even if ev slightly negative when close to goal,
                # but prefer better robots due to p(POUR).
                score = 1000 + ev * 120.0
                act = f"POUR({rid})"

            # 2) If on tap: LOAD until target load, else go to plant intent
            elif robot_pos in tap_positions:
                max_need = max(remaining_plants.values()) if remaining_plants else 0
                if short_horizon and cap >= 8:
                    target_load = min(max_need, 3)
                else:
                    target_load = min(max_need, cap)

                if load < cap and load < target_load:
                    p_load = self._p(rid, "LOAD")
                    # expected benefit: having more water reduces future tap trips
                    score = 900 + (p_load * 20.0) + (target_load - load)
                    act = f"LOAD({rid})"
                else:
                    # move toward plant intent
                    tgt = self.intent[rid][1] if self.intent[rid] and self.intent[rid][0] == "PLANT" else None
                    if tgt is not None:
                        path = self.bfs_path_ignore_robots(robot_pos, tgt)
                        if path:
                            step = path[0]
                            nxt = self._next_cell(robot_pos, step)
                            if nxt not in occupied_excl_self and nxt not in self.walls:
                                # small bias toward higher-value target
                                v = self._score_plant_for_robot(rid, robot_pos, tgt, occupied_excl_self)
                                score = 850 + v * 500.0
                                act = f"{step}({rid})"

            # 3) If empty: go toward tap intent
            elif load == 0 and tap_positions:
                tgt = self.intent[rid][1] if self.intent[rid] and self.intent[rid][0] == "TAP" else None
                if tgt is not None:
                    path = self.bfs_path_ignore_robots(robot_pos, tgt)
                    if path:
                        step = path[0]
                        nxt = self._next_cell(robot_pos, step)
                        if nxt not in occupied_excl_self and nxt not in self.walls:
                            d = self.bfs_distance(robot_pos, tgt, occupied_excl_self)
                            p_move = self._p(rid, step) if step in ("UP", "DOWN", "LEFT", "RIGHT") else 0.5
                            score = 800 - d + (p_move - 0.5) * 30.0
                            act = f"{step}({rid})"

            # 4) Have water and not on plant: move toward plant intent
            elif load > 0 and remaining_plants:
                tgt = self.intent[rid][1] if self.intent[rid] and self.intent[rid][0] == "PLANT" else None
                if tgt is not None:
                    path = self.bfs_path_ignore_robots(robot_pos, tgt)
                    if path:
                        step = path[0]
                        nxt = self._next_cell(robot_pos, step)
                        if nxt not in occupied_excl_self and nxt not in self.walls:
                            v = self._score_plant_for_robot(rid, robot_pos, tgt, occupied_excl_self)
                            d = self.bfs_distance(robot_pos, tgt, occupied_excl_self)
                            score = 720 + v * 700.0 - 0.3 * d
                            act = f"{step}({rid})"

            # If candidate action exists, adjust by mild robot reliability
            if act is not None:
                # modest penalty for fail streak, not huge (avoid over-suppression)
                score -= 80.0 * min(self.fail_streak.get(rid, 0), 5)

                # mild bonus for capacity (bigger tanks more valuable)
                score += cap * 0.3

                if score > best_score:
                    best_score = score
                    best_action = act

        if best_action is not None:
            self._last_action = best_action
            return best_action

        # Fallback: any legal move to avoid RESET loops
        for rid, (r, c), _load in robots_t:
            robot_pos = (r, c)
            occupied_excl_self = occupied_all - {robot_pos}
            for direction, (dr, dc) in [("UP", (-1, 0)), ("DOWN", (1, 0)), ("LEFT", (0, -1)), ("RIGHT", (0, 1))]:
                nr, nc = r + dr, c + dc
                np2 = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if np2 not in self.walls and np2 not in occupied_excl_self:
                        act = f"{direction}({rid})"
                        self._last_action = act
                        return act

        self._last_action = "RESET"
        return "RESET"
