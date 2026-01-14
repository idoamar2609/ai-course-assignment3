import ext_plant
from collections import deque
import numpy as np

id = ["318780301"]

class Controller:
    """This class is a controller for the ext_plant game."""

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.original_game = game
        self.rows = game.rows
        self.cols = game.cols
        self.walls = game.walls
        self.plants = game.plants
        self.taps = game.taps
        self.plant_max_reward = game.get_plants_max_reward()

    def bfs_distance(self, start, goal, occupied):
        """BFS to find shortest distance from start to goal, avoiding obstacles and occupied cells."""
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (r, c), dist = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    next_pos = (nr, nc)
                    if next_pos not in visited and next_pos not in self.walls and next_pos not in occupied:
                        if next_pos == goal:
                            return dist + 1
                        visited.add(next_pos)
                        queue.append((next_pos, dist + 1))

        return float('inf')

    def bfs_path(self, start, goal, occupied):
        """BFS to find shortest path from start to goal."""
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
                    next_pos = (nr, nc)
                    if next_pos not in visited and next_pos not in self.walls and next_pos not in occupied:
                        new_path = path + [direction]
                        if next_pos == goal:
                            return new_path
                        visited.add(next_pos)
                        queue.append((next_pos, new_path))

        return []

    def choose_next_action(self, state):
        """Choose the next action given a state."""
        robots_t, plants_t, taps_t, total_need = state

        # If all plants are watered
        if total_need == 0:
            return "RESET"

        # Build occupied positions
        occupied_all = {(rr, cc) for (rid, (rr, cc), l) in robots_t}

        # Get plant and tap positions
        plant_needs = {pos: need for pos, need in plants_t}
        remaining_plants = {pos: need for pos, need in plant_needs.items() if need > 0}
        tap_positions = {pos for pos, water in taps_t}

        best_action = None
        best_priority = -1
        best_rid = None

        # Evaluate all robots and choose the best action
        for rid, (r, c), load in robots_t:
            robot_pos = (r, c)
            cap = self.original_game._capacities[rid]
            occupied_excl_self = occupied_all - {robot_pos}

            # Priority 1: If at a plant with water, POUR (highest priority)
            if robot_pos in remaining_plants and load > 0:
                plant_reward = self.plant_max_reward.get(robot_pos, 1)
                priority = 1000 + plant_reward
                if priority > best_priority:
                    best_priority = priority
                    best_action = f"POUR({rid})"
                    best_rid = rid

            # Priority 2: If at a tap and not full, LOAD
            # elif robot_pos in tap_positions and load < cap:
            #     priority = 900
            #     if priority > best_priority:
            #         best_priority = priority
            #         best_action = f"LOAD({rid})"
            #         best_rid = rid
            elif robot_pos in tap_positions and load < cap:
                max_need = max(remaining_plants.values()) if remaining_plants else 0

                # horizon-aware target: don't waste turns overloading
                # with H=30, loading beyond 3-4 is usually bad
                target_load = min(max_need, 3)

                if load < target_load:
                    priority = 900
                    if priority > best_priority:
                        best_priority = priority
                        best_action = f"LOAD({rid})"
                        best_rid = rid
                else:
                    # already enough to leave tap; do NOT load more
                    pass



            # Priority 3: If no water, move to nearest tap
            elif load == 0 and tap_positions:
                distances_to_taps = {tap: self.bfs_distance(robot_pos, tap, occupied_excl_self)
                                    for tap in tap_positions}
                closest_tap = min(distances_to_taps, key=distances_to_taps.get)

                if distances_to_taps[closest_tap] < float('inf'):
                    path = self.bfs_path(robot_pos, closest_tap, occupied_excl_self)
                    if path:
                        priority = 800 - distances_to_taps[closest_tap]
                        if priority > best_priority:
                            best_priority = priority
                            best_action = f"{path[0]}({rid})"
                            best_rid = rid

            # Priority 4: If have water, move toward highest reward plant
            elif load > 0 and remaining_plants:
                best_plant_score = -1
                best_plant = None

                for plant_pos, need in remaining_plants.items():
                    dist = self.bfs_distance(robot_pos, plant_pos, occupied_excl_self)
                    if dist < float('inf'):
                        max_reward = self.plant_max_reward.get(plant_pos, 1)
                        # Strong preference for high reward plants
                        score = max_reward * 100 / (dist + 1)
                        if score > best_plant_score:
                            best_plant_score = score
                            best_plant = plant_pos

                if best_plant:
                    path = self.bfs_path(robot_pos, best_plant, occupied_excl_self)
                    if path:
                        dist = self.bfs_distance(robot_pos, best_plant, occupied_excl_self)
                        priority = 700 + (self.plant_max_reward.get(best_plant, 1) * 10) - dist
                        if priority > best_priority:
                            best_priority = priority
                            best_action = f"{path[0]}({rid})"
                            best_rid = rid

            # Priority 5: If have capacity but not full, go to tap
            elif load < cap and tap_positions:
                distances_to_taps = {tap: self.bfs_distance(robot_pos, tap, occupied_excl_self)
                                    for tap in tap_positions}
                closest_tap = min(distances_to_taps, key=distances_to_taps.get)

                if distances_to_taps[closest_tap] < float('inf'):
                    path = self.bfs_path(robot_pos, closest_tap, occupied_excl_self)
                    if path:
                        priority = 600 - distances_to_taps[closest_tap]
                        if priority > best_priority:
                            best_priority = priority
                            best_action = f"{path[0]}({rid})"
                            best_rid = rid

            # Priority 6: Reach any remaining plant
            elif remaining_plants:
                distances = {plant: self.bfs_distance(robot_pos, plant, occupied_excl_self)
                            for plant in remaining_plants}
                closest_plant = min(distances, key=distances.get)

                if distances[closest_plant] < float('inf'):
                    path = self.bfs_path(robot_pos, closest_plant, occupied_excl_self)
                    if path:
                        priority = 500 - distances[closest_plant]
                        if priority > best_priority:
                            best_priority = priority
                            best_action = f"{path[0]}({rid})"
                            best_rid = rid

        if best_action:
            print("BEST ACTION:", best_action, "FOR ROBOT:", best_rid)
            return best_action
        print("RESET")
        return "RESET"