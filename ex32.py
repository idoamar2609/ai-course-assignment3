import ext_plant
from collections import deque
import numpy as np

id = ["318780301"]


class Controller:
    """Heuristic controller with corridor-safe movement and horizon-aware loading."""

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
            # big reward and close distance wins
            score = (rew * 100.0) / (dist + 1.0)
            if score > best_score:
                best_score = score
                best_plant = plant_pos
        return best_plant

    def choose_next_action(self, state):
        robots_t, plants_t, taps_t, total_need = state

        # done
        if total_need == 0:
            return "RESET"

        # occupied positions
        occupied_all = {(rr, cc) for (rid, (rr, cc), l) in robots_t}

        # plants / taps
        plant_needs = {pos: need for pos, need in plants_t}
        remaining_plants = {pos: need for pos, need in plant_needs.items() if need > 0}
        tap_positions = {pos for pos, water in taps_t}

        best_action = None
        best_priority = -1
        best_rid = None

        # horizon-aware: short horizons should "sip-load"
        t_now = self.original_game.get_current_steps()
        steps_left = max(0, self.horizon - t_now)
        short_horizon = steps_left <= 35  # works well for your hard cases

        for rid, (r, c), load in robots_t:
            robot_pos = (r, c)
            cap = self.capacities[rid]
            occupied_excl_self = occupied_all - {robot_pos}

            # compute max need among remaining plants
            max_need = max(remaining_plants.values()) if remaining_plants else 0

            # choose a "target load" that avoids loading forever
            if short_horizon and cap >= 8:
                target_load = min(max_need, 3)  # sip-load
            else:
                # normal: your original rule idea (enough to satisfy max-need plant)
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
                    priority = 900 + (target_load - load)  # finish sip quickly
                    action = f"LOAD({rid})"
                else:
                    # Already enough: leave tap towards best plant (priority higher than LOAD)
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
                # choose nearest tap by BFS distance
                closest_tap = None
                best_d = float("inf")
                for tap in tap_positions:
                    d = self.bfs_distance(robot_pos, tap, occupied_excl_self)
                    if d < best_d:
                        best_d = d
                        closest_tap = tap

                if closest_tap is not None and best_d < float("inf"):
                    # corridor-safe movement: ignore robots in planning, only avoid collision on next cell
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

            # Priority 5: Partial water - go to plant if close, else (maybe) tap
            elif remaining_plants and load > 0:
                # (rare fallback)
                closest_plant = None
                best_d = float("inf")
                for plant in remaining_plants:
                    d = self.bfs_distance(robot_pos, plant, occupied_excl_self)
                    if d < best_d:
                        best_d = d
                        closest_plant = plant
                if closest_plant is not None and best_d < float("inf"):
                    path = self.bfs_path_ignore_robots(robot_pos, closest_plant)
                    if path:
                        step = path[0]
                        nxt = self._next_cell(robot_pos, step)
                        if nxt not in occupied_excl_self:
                            priority = 650 - best_d
                            action = f"{step}({rid})"

            # bias toward bigger tanks so they don't get starved
            if action is not None:
                priority += cap * 0.5

            if priority > best_priority and action is not None:
                best_priority = priority
                best_action = action
                best_rid = rid

        if best_action:
            print(best_action)
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
                        print(f"{direction}({rid})")
                        return f"{direction}({rid})"
        print("RESET")
        return "RESET"
