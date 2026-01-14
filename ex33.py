import ext_plant
from collections import deque
import numpy as np

id = ["318780301"]


class Controller:
    """
    Heuristic controller with corridor-safe movement and horizon-aware loading,
    PLUS online estimation of each robot's success probability.

    If a robot (e.g., 11) seems to fail repeatedly, we automatically down-prioritize it
    and prefer other robots (e.g., 10).
    """

    def __init__(self, game: ext_plant.Game):
        self.original_game = game
        self.rows = game.rows
        self.cols = game.cols
        self.walls = set(game.walls)
        self.plants = set(game.plants)
        self.taps = set(game.taps)
        self.plant_max_reward = game.get_plants_max_reward()
        self.capacities = game.get_capacities()
        self.horizon = game.get_max_steps()

        # --- Online reliability estimation ---
        # attempts/successes per robot, inferred from state transitions
        self.attempts = {rid: 0 for rid in self.capacities}
        self.successes = {rid: 0 for rid in self.capacities}
        self.fail_streak = {rid: 0 for rid in self.capacities}

        self._last_state = None
        self._last_action = None  # string like "UP(11)" or "RESET"

    # -----------------------
    # Helpers: parsing & state access
    # -----------------------
    @staticmethod
    def _parse_action(s: str):
        """Parse 'ACTION(rid)' -> (ACTION, rid). Return (None, None) for RESET/invalid."""
        if s is None:
            return None, None
        s = s.strip()
        if s == "RESET":
            return "RESET", None
        try:
            action, rid = ext_plant.Game.parse_robot_action  # type: ignore[attr-defined]
        except Exception:
            action = rid = None

        # We don't have direct access to Game.parse_robot_action as static; do minimal parse:
        # formats are like "UP(11)" "LOAD(10)"
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
    # Online success inference
    # -----------------------
    def _infer_last_action_success(self, prev_state, action_str, curr_state):
        """
        Infer whether the *chosen* action succeeded, by comparing prev_state -> curr_state.

        Notes:
        - MOVE success: robot ended at the intended next cell.
        - LOAD success: robot load increased by 1 (at same position).
        - POUR success: plant need at that position decreased by 1 AND robot load decreased by 1.
          (On POUR fail, robot load decreases by 1 but plant doesn't change.)
        - RESET: ignore for stats.
        """
        act, rid = self._parse_action(action_str)
        if act is None or act == "RESET" or rid is None:
            return None, None  # ignore

        prev_robots, prev_plants, prev_taps, _ = prev_state
        curr_robots, curr_plants, curr_taps, _ = curr_state

        prevR = self._robots_dict(prev_robots)
        currR = self._robots_dict(curr_robots)

        if rid not in prevR or rid not in currR:
            return rid, None

        (pr, pc), pload = prevR[rid]
        (cr, cc), cload = currR[rid]

        # Movement intended next cell
        if act in ("UP", "DOWN", "LEFT", "RIGHT"):
            if act == "UP":
                intended = (pr - 1, pc)
            elif act == "DOWN":
                intended = (pr + 1, pc)
            elif act == "LEFT":
                intended = (pr, pc - 1)
            else:
                intended = (pr, pc + 1)

            success = ((cr, cc) == intended)
            return rid, success

        if act == "LOAD":
            # success increases load by 1; failure keeps load
            success = (cload == pload + 1 and (cr, cc) == (pr, pc))
            return rid, success

        if act == "POUR":
            prevP = self._plants_dict(prev_plants)
            currP = self._plants_dict(curr_plants)

            # expected: load down by 1 always on POUR (even on fail in this env)
            # success: plant need decreases by 1 (or plant removed when hits 0)
            prev_need = prevP.get((pr, pc), None)
            curr_need = currP.get((pr, pc), None)

            plant_decreased = False
            if prev_need is not None:
                if curr_need is None and prev_need == 1:
                    plant_decreased = True
                elif curr_need is not None and curr_need == prev_need - 1:
                    plant_decreased = True

            success = (plant_decreased and cload == pload - 1 and (cr, cc) == (pr, pc))
            return rid, success

        return rid, None

    def _update_reliability_stats(self, current_state):
        """Call once per choose_next_action, before choosing new action."""
        if self._last_state is None or self._last_action is None:
            self._last_state = current_state
            self._last_action = None
            return

        rid, success = self._infer_last_action_success(self._last_state, self._last_action, current_state)
        if rid is not None and success is not None and rid in self.attempts:
            self.attempts[rid] += 1
            if success:
                self.successes[rid] += 1
                self.fail_streak[rid] = 0
            else:
                self.fail_streak[rid] += 1

        self._last_state = current_state
        self._last_action = None  # will be set when we return a new action

    def _estimated_p_success(self, rid):
        """
        Laplace-smoothed success estimate:
        p = (succ + 1) / (att + 2)
        """
        a = self.attempts.get(rid, 0)
        s = self.successes.get(rid, 0)
        return (s + 1.0) / (a + 2.0)

    def _reliability_adjustment(self, rid):
        """
        Convert estimated success + recent failure streak into a priority adjustment.

        - More successes => bonus
        - Repeated failures => strong penalty (so robot 10 will be preferred if 11 is failing)
        """
        p = self._estimated_p_success(rid)  # in [0,1]
        streak = self.fail_streak.get(rid, 0)

        # Bonus for reliable robots
        bonus = 250.0 * (p - 0.5)  # range roughly [-125, +125]

        # Penalty for repeated failures
        # After ~3-4 failures in a row, the robot gets heavily deprioritized.
        penalty = 180.0 * min(streak, 6)  # up to ~1080 penalty

        return bonus - penalty

    # -----------------------
    # BFS (robots as obstacles)
    # -----------------------
    def bfs_distance(self, start, goal, occupied):
        """BFS distance avoiding walls and occupied cells."""
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    nxt = (nr, nc)
                    if nxt in visited or nxt in self.walls or nxt in occupied:
                        continue
                    if nxt == goal:
                        return dist + 1
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))

        return float("inf")

    def bfs_path(self, start, goal, occupied):
        """BFS path avoiding walls and occupied cells."""
        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (r, c), path = queue.popleft()
            for direction, (dr, dc) in [("UP", (-1, 0)), ("DOWN", (1, 0)),
                                        ("LEFT", (0, -1)), ("RIGHT", (0, 1))]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    nxt = (nr, nc)
                    if nxt in visited or nxt in self.walls or nxt in occupied:
                        continue
                    new_path = path + [direction]
                    if nxt == goal:
                        return new_path
                    visited.add(nxt)
                    queue.append((nxt, new_path))

        return []

    # -----------------------
    # BFS (ignore robots, only walls)
    # -----------------------
    def bfs_path_ignore_robots(self, start, goal):
        """BFS path ignoring robots (only walls). Prevents false 'no path' in corridors."""
        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (r, c), path = queue.popleft()
            for direction, (dr, dc) in [("UP", (-1, 0)), ("DOWN", (1, 0)),
                                        ("LEFT", (0, -1)), ("RIGHT", (0, 1))]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    nxt = (nr, nc)
                    if nxt in visited or nxt in self.walls:
                        continue
                    new_path = path + [direction]
                    if nxt == goal:
                        return new_path
                    visited.add(nxt)
                    queue.append((nxt, new_path))

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

    def _pick_best_plant(self, robot_pos, remaining_plants, occupied_excl_self):
        """Choose plant by reward / distance tradeoff."""
        best_score = -1
        best_plant = None
        for plant_pos, need in remaining_plants.items():
            dist = self.bfs_distance(robot_pos, plant_pos, occupied_excl_self)
            if dist == float("inf"):
                continue
            rew = self.plant_max_reward.get(plant_pos, 1)
            score = (rew * 100.0) / (dist + 1.0)
            if score > best_score:
                best_score = score
                best_plant = plant_pos
        return best_plant

    def choose_next_action(self, state):
        # Update success stats based on what happened since last chosen action
        self._update_reliability_stats(state)

        robots_t, plants_t, taps_t, total_need = state

        # done
        if total_need == 0:
            self._last_action = "RESET"
            return "RESET"

        # occupied positions
        occupied_all = {(rr, cc) for (rid, (rr, cc), l) in robots_t}

        # plants / taps
        plant_needs = {pos: need for pos, need in plants_t}
        remaining_plants = {pos: need for pos, need in plant_needs.items() if need > 0}
        tap_positions = {pos for pos, water in taps_t}

        best_action = None
        best_priority = -1e18
        best_rid = None

        # horizon-aware: short horizons should "sip-load"
        t_now = self.original_game.get_current_steps()
        steps_left = max(0, self.horizon - t_now)
        short_horizon = steps_left <= 35

        for rid, (r, c), load in robots_t:
            robot_pos = (r, c)
            cap = self.capacities[rid]
            occupied_excl_self = occupied_all - {robot_pos}

            # compute max need among remaining plants
            max_need = max(remaining_plants.values()) if remaining_plants else 0

            # choose target load
            if short_horizon and cap >= 8:
                target_load = min(max_need, 3)  # sip-load
            else:
                target_load = min(max_need, cap)

            priority = -1
            action = None

            # Priority 1: POUR if at a plant and have water
            if robot_pos in remaining_plants and load > 0:
                plant_reward = self.plant_max_reward.get(robot_pos, 1)
                priority = 1000 + plant_reward
                action = f"POUR({rid})"

            # Priority 2: At tap - load only until target_load; then FORCE LEAVE to plants
            elif robot_pos in tap_positions:
                if load < cap and load < target_load:
                    priority = 900 + (target_load - load)
                    action = f"LOAD({rid})"
                else:
                    if remaining_plants:
                        best_plant = self._pick_best_plant(robot_pos, remaining_plants, occupied_excl_self)
                        if best_plant:
                            path = self.bfs_path_ignore_robots(robot_pos, best_plant)
                            if path:
                                step = path[0]
                                nxt = self._next_cell(robot_pos, step)
                                if nxt not in occupied_excl_self:
                                    priority = 950
                                    action = f"{step}({rid})"

            # Priority 3: Empty - go to nearest tap
            elif load == 0 and tap_positions:
                closest_tap = None
                best_d = float("inf")
                for tap in tap_positions:
                    d = self.bfs_distance(robot_pos, tap, occupied_excl_self)
                    if d < best_d:
                        best_d = d
                        closest_tap = tap

                if closest_tap is not None and best_d < float("inf"):
                    path = self.bfs_path_ignore_robots(robot_pos, closest_tap)
                    if path:
                        step = path[0]
                        nxt = self._next_cell(robot_pos, step)
                        if nxt not in occupied_excl_self:
                            priority = 800 - best_d
                            action = f"{step}({rid})"

            # Priority 4: Have water - go toward best plant (reward/distance)
            elif load > 0 and remaining_plants:
                best_plant = self._pick_best_plant(robot_pos, remaining_plants, occupied_excl_self)
                if best_plant:
                    path = self.bfs_path_ignore_robots(robot_pos, best_plant)
                    if path:
                        step = path[0]
                        nxt = self._next_cell(robot_pos, step)
                        if nxt not in occupied_excl_self:
                            dist = self.bfs_distance(robot_pos, best_plant, occupied_excl_self)
                            rew = self.plant_max_reward.get(best_plant, 1)
                            priority = 700 + (rew * 10) - (dist if dist != float("inf") else 50)
                            action = f"{step}({rid})"

            # --- Reliability / success-probability adjustment ---
            # If robot 11 keeps failing, it will get a big penalty and robot 10 will take over.
            if action is not None:
                priority += self._reliability_adjustment(rid)
                # small bias for bigger tanks (kept from your version)
                priority += cap * 0.5

            if action is not None and priority > best_priority:
                best_priority = priority
                best_action = action
                best_rid = rid

        if best_action:
            self._last_action = best_action
            return best_action

        # No decided action: do any legal move to avoid RESET loops
        for rid, (r, c), load in robots_t:
            robot_pos = (r, c)
            occupied_excl_self = occupied_all - {robot_pos}
            for direction, (dr, dc) in [("UP", (-1, 0)), ("DOWN", (1, 0)),
                                        ("LEFT", (0, -1)), ("RIGHT", (0, 1))]:
                nr, nc = r + dr, c + dc
                np2 = (nr, nc)
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if np2 not in self.walls and np2 not in occupied_excl_self:
                        act = f"{direction}({rid})"
                        self._last_action = act
                        return act

        self._last_action = "RESET"
        return "RESET"
