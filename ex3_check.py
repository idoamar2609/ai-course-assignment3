import ext_plant
import ex33
import time
import numpy as np
import sys

# ANSI color codes for nicer terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
SEP = "\n" + ("=" * 60)


def draw_board(problem):
    """Draw a simple emoji board for the given problem dict."""
    rows, cols = problem.get("Size", (0, 0))
    walls = set(problem.get("Walls", []))
    taps = dict(problem.get("Taps", {}))
    plants = dict(problem.get("Plants", {}))
    robots = dict(problem.get("Robots", {}))

    # Build grid rows (r from 0..rows-1)
    grid_lines = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            pos = (r, c)
            if pos in walls:
                cell = "🧱 "
            elif pos in taps:
                # show tap emoji
                cell = "🚰 "
            elif pos in plants:
                cell = "🌱 "
            else:
                cell = "⬜ "

            # overlay robot if present (show last digit of id)
            robot_here = None
            for rid, (rr, cc, _carried, _cap) in robots.items():
                if (rr, cc) == pos:
                    robot_here = rid
                    break
            if robot_here is not None:
                cell = f"🤖{str(robot_here)[-1]}"

            row_cells.append(cell)
        grid_lines.append(" ".join(row_cells))

    print("\nBoard layout:")
    for line in grid_lines:
        print(line)
    # also print legends for taps/plants/robots (counts)
    if taps:
        taps_str = ", ".join([f"{pos}:{amt}" for pos, amt in taps.items()])
        print(f"Taps: {taps_str}")
    if plants:
        plants_str = ", ".join([f"{pos}:{need}" for pos, need in plants.items()])
        print(f"Plants: {plants_str}")
    if robots:
        robots_str = ", ".join(
            [f"{rid}:{(r,c)}" for rid, (r, c, _car, _cap) in robots.items()]
        )
        print(f"Robots: {robots_str}")
    # Note: do not print hidden problem details such as robot success probabilities
    # or plant reward distributions — these are unknown to the agent and should not
    # be revealed by the checker.


def is_action_legal(game: ext_plant.Game, chosen_action: str):
    """Return (True, '') if action legal in current state, else (False, reason).

    This mirrors the legality checks performed in `ext_plant.Game.submit_next_action` without changing the game state.
    """
    if chosen_action == "RESET":
        return True, ""

    # Parse action (may raise ValueError for bad format)
    try:
        action_name, robot_id = game.parse_robot_action(chosen_action)
    except Exception as e:
        return False, f"Bad action format or unknown action: {e}"

    # Get current state
    robots_t, plants_t, taps_t, total_water_need = game.get_current_state()

    # Find robot
    robot_entry = None
    for rid, (pos, load) in [(r[0], (r[1], r[2])) for r in robots_t]:
        pass
    robot_idx = None
    r = c = load = None
    for idx, (rid, (rr, cc), l) in enumerate(robots_t):
        if rid == robot_id:
            robot_idx = idx
            r, c, load = rr, cc, l
            break
    if robot_idx is None:
        return False, f"Robot {robot_id} not found in state"

    robot_pos = (r, c)
    plant_positions = {pos for (pos, need) in plants_t}
    tap_positions = {pos for (pos, water) in taps_t}

    # occupied positions by other robots
    occupied_positions = {
        (rr, cc) for (rid, (rr, cc), l) in robots_t if rid != robot_id
    }

    base_applicable = game.applicable_actions.get(robot_pos, [])

    dynamic_applicable = []
    for a in base_applicable:
        if a in ("UP", "DOWN", "LEFT", "RIGHT"):
            if a == "UP":
                target = (r - 1, c)
            elif a == "DOWN":
                target = (r + 1, c)
            elif a == "LEFT":
                target = (r, c - 1)
            else:
                target = (r, c + 1)
            if target in occupied_positions:
                continue
        dynamic_applicable.append(a)

    # Additional checks for POUR/LOAD
    if action_name in ("UP", "DOWN", "LEFT", "RIGHT"):
        if action_name not in dynamic_applicable:
            return (
                False,
                f"Move {action_name} not applicable from {robot_pos} (blocked or wall)",
            )
        return True, ""
    if action_name == "POUR":
        if action_name not in dynamic_applicable:
            return False, "POUR not applicable at this position"
        if load <= 0 or robot_pos not in plant_positions:
            return False, "POUR precondition failed: not standing on plant or no load"
        return True, ""
    if action_name == "LOAD":
        if action_name not in dynamic_applicable:
            return False, "LOAD not applicable at this position"
        cap = game.get_capacities().get(robot_id, None)
        if cap is None:
            return False, "Could not get robot capacity"
        if load >= cap or robot_pos not in tap_positions:
            return False, "LOAD precondition failed: at tap or capacity reached"
        return True, ""

    return False, "Unknown action or not applicable"


def solve(game: ext_plant.Game, run_idx: int, controller_module):
    policy = controller_module.Controller(game)
    for i in range(game.get_max_steps()):
        action = policy.choose_next_action(game.get_current_state())
        legal, reason = is_action_legal(game, action)
        if not legal:
            raise ValueError(
                f"Illegal action chosen by controller on run {run_idx}: {action!r}. Reason: {reason}"
            )
        game.submit_next_action(chosen_action=action)
        if game.get_done():
            break

    r = game.get_current_reward()
    state = game.get_current_state()
    state_str = str(state)

    finished_txt = "SUCCESS" if state[-1] else "FAILED"
    color = GREEN if state[-1] else RED

    print(
        f"Run {run_idx:2d}: {BOLD}{YELLOW}Reward: {r:3d}{RESET} | Steps: {game.get_max_steps():2d} | {color}{finished_txt}{RESET} \nState: {state_str}"
    )
    return r


def solve2(game: ext_plant.Game):
    policy = ex3_random.Controller(game)
    for i in range(game.get_max_steps()):
        game.submit_next_action(
            chosen_action=policy.choose_next_action(game.get_current_state())
        )
        if game.get_done():
            break
    print(
        "Game result:",
        game.get_current_state(),
        "\n\tFinished in",
        game.get_max_steps(),
        "Steps.\n\tReward result->",
        game.get_current_reward(),
    )
    print(
        "Game finished ",
        "" if game.get_current_state()[-1] else "un",
        "successfully.",
        sep="",
    )
    game.show_history()
    return game.get_current_reward()


problem_pdf = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.9,
    },
    "goal_reward": 10,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 60,
}


problem_pdf2 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.8,
    },
    "goal_reward": 12,
    "plants_reward": {
        (0, 2): [1, 3, 5, 7],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 60,
}

problem_pdf3 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.6,
    },
    "goal_reward": 30,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 60,
}

problem_new1_version1 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 60,
}
problem_new1_version2 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new1_version3 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 2,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 60,
}

problem_new2_version1 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 60,
}

problem_new2_version2 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new2_version3 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 20,
    "plants_reward": {
        (0, 0): [5, 7, 9],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 60,
}
problem_new2_version4 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 40,
}


problem_new3_version1 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 60,
}

problem_new3_version2 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.8,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 100,
}


problem_new3_version3 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 5,  # upper-right corrido
        (9, 0): 5,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.0001,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 210,  # "horizon": 70,
}
# reset ?
problem_new4_version1 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 140,  # "horizon": 70,
}

# reset ?
problem_new4_version2 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.85,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 80,
}


def append_lines(path: str, lines: list[str]):
    # writes in the SAME place you run the program (current directory)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    debug_mode = False
    n_runs = 30

    # Keep the same problem list as before
    problems = [
        ("problem_pdf", problem_pdf),
        ("problem_pdf2", problem_pdf2),
        ("problem_pdf3", problem_pdf3),
        ("problem_new1_version1", problem_new1_version1),
        ("problem_new1_version2", problem_new1_version2),
        ("problem_new1_version3", problem_new1_version3),
        ("problem_new2_version1", problem_new2_version1),
        ("problem_new2_version2", problem_new2_version2),
        ("problem_new2_version3", problem_new2_version3),
        ("problem_new2_version4", problem_new2_version4),
        ("problem_new3_version1", problem_new3_version1),
        ("problem_new3_version2", problem_new3_version2),
        ("problem_new3_version3", problem_new3_version3),
        ("problem_new4_version1", problem_new4_version1),
        ("problem_new4_version2", problem_new4_version2),
    ]

    # Official baselines (🟩) provided by user
    official_baseline_map = {
        "problem_pdf": (44.0, 0.0),
        "problem_pdf3": (37.0, 0.0),
        "problem_new1_version1": (83.0, 0.0),
        "problem_new1_version3": (57.0, 0.0),
        "problem_new2_version1": (85.0, 0.0),
        "problem_new2_version3": (88.0, 0.0),
        "problem_new3_version1": (18.0, 0.0),
        "problem_new3_version2": (20.0, 0.0),
        "problem_new4_version1": (35.0, 0.0),
        "problem_new4_version2": (12.0, 0.0),
    }

    # Estimated baselines (⬜) — kept for comparison but marked as estimated
    estimated_baseline_map = {
        "problem_pdf2": (28.33333, 0.0),
        "problem_new1_version2": (81.87778, 0.0),
        "problem_new2_version2": (123.715, 0.0),
        "problem_new2_version4": (41.65, 0.0),
        "problem_new3_version3": (8.922222, 0.0),
    }

    def get_baseline_info(pname: str):
        """Return (baseline_avg, baseline_time, is_estimated).
        Official baselines (in official_baseline_map) are not estimated.
        Estimated baselines come from estimated_baseline_map.
        If no baseline is available, returns (None, None, False).
        """
        if pname in official_baseline_map:
            a, t = official_baseline_map[pname]
            return a, t, False
        if pname in estimated_baseline_map:
            a, t = estimated_baseline_map[pname]
            return a, t, True
        return None, None, False

    summaries = []

    # parse CLI args
    args = sys.argv[1:]
    argstr = " ".join(args).lower()
    baseline_only = False
    if "baseline" in argstr:
        baseline_only = True

    # strict mode: abort on first timeout or illegal action when enabled
    strict_mode = False
    if "strict" in argstr:
        strict_mode = True
        print(
            f"{BOLD}{YELLOW}Strict mode enabled: will abort on first timeout or illegal action.{RESET}"
        )

    # choose controller module: default `ex3`, or `ex3_random` when 'random' passed
    if "random" in args:
        controller_module = ex3_random
    else:
        controller_module = ex33

    # allow explicit problem selection using a bracketed list in the CLI, e.g.:
    #   python ex3_check.py [problem_pdf,problem_new1_version3]
    selected_problem_names = None
    # note: preserve original names for missing-name reporting
    all_problem_names = [p for (p, _prob) in problems]
    for a in args:
        if a.startswith("[") and a.endswith("]"):
            inner = a[1:-1]
            names = [x.strip() for x in inner.split(",") if x.strip()]
            if names:
                selected_problem_names = names
                print(f"{BOLD}{YELLOW}Selected problems provided: {names}{RESET}")
            break

    # if a specific selection was provided, use that (takes precedence over baseline flag)
    if selected_problem_names is not None:
        problems = [(p, prob) for (p, prob) in problems if p in selected_problem_names]
        missing = [
            name for name in selected_problem_names if name not in all_problem_names
        ]
        if missing:
            print(f"{YELLOW}Warning: requested problems not found: {missing}{RESET}")
        print(
            f"{BOLD}{YELLOW}Running selected problems only; {len(problems)} problems will be executed.{RESET}"
        )
    elif baseline_only:
        problems = [(p, prob) for (p, prob) in problems if p in official_baseline_map]
        print(
            f"{BOLD}{YELLOW}Running baseline-only; {len(problems)} problems will be executed.{RESET}"
        )

    for idx, (pname, problem) in enumerate(problems, start=1):
        marker = "🟩" if pname in official_baseline_map else "⬜"
        print()
        print(f"{marker} *** Problem: {pname} ({idx}) ***")
        draw_board(problem)
        print(
            f"{marker} \n{SEP}\n{BOLD}{MAGENTA}--- Running {pname} (problem index slice item {idx}) ---{RESET}"
        )

        total_reward = 0.0
        horizon = problem.get("horizon", 0)
        time_limit = 20 + 0.5 * horizon

        run_times = []
        for seed in range(n_runs):
            problem["seed"] = seed
            game = ext_plant.create_pressure_plate_game((problem, debug_mode))

            try:
                run_start = time.time()
                run_reward = solve(game, seed, controller_module)
                run_duration = time.time() - run_start
            except ValueError as e:
                print(
                    f"{RED}ERROR: Illegal action detected during run {seed}: {e}{RESET}"
                )
                print(f"{RED}Aborting tests due to illegal action.{RESET}")
                sys.exit(1)
            except Exception as e:
                print(f"{RED}ERROR during run {seed}: {e}{RESET}")
                sys.exit(1)

            run_times.append(run_duration)
            total_reward += run_reward

            limit_ok = run_duration <= time_limit
            status_color = GREEN if limit_ok else RED
            print(
                f"Run {seed:2d} time: {run_duration:.2f}s | Limit: {time_limit:.2f}s | {status_color}{'OK' if limit_ok else 'TIMEOUT'}{RESET}"
            )

            # Strict timeout handling: abort on first timeout when strict_mode enabled
            if (not limit_ok) and strict_mode:
                print(
                    f"{RED}Strict mode: aborting on first timeout (run {seed}).{RESET}"
                )
                sys.exit(1)

        duration = sum(run_times)
        avg_time_per_run = duration / n_runs if n_runs else 0.0

        num_timeouts = sum(1 for t in run_times if t > time_limit)
        time_status = (
            f"{GREEN}PASS{RESET}"
            if num_timeouts == 0
            else f"{RED}TIMEOUT ({num_timeouts}/{n_runs} runs exceeded {time_limit:.1f}s){RESET}"
        )

        avg_reward = total_reward / n_runs if n_runs else 0.0
        baseline_avg, baseline_time, is_estimated = get_baseline_info(pname)
        summaries.append(
            (
                pname,
                avg_reward,
                duration,
                baseline_avg,
                baseline_time,
                is_estimated,
                time_status,
            )
        )

        if baseline_avg is not None:
            pct = (avg_reward / baseline_avg * 100) if baseline_avg > 0 else 0
            comp = (
                f"{GREEN}BETTER{RESET}"
                if avg_reward > baseline_avg
                else f"{RED}WORSE{RESET}"
            )
            label = "estimated baseline" if is_estimated else "baseline"
            print(
                f"\nAverage reward over {n_runs} runs: {avg_reward:.6f} ({pct:.1f}% of {label} {baseline_avg:.6f}) | {comp} than {label}"
            )
            print(
                f"{label.capitalize()} time: {baseline_time}s | Current time: {duration:.2f}s | Status: {time_status}"
            )
        else:
            print(f"\nAverage reward over {n_runs} runs: {avg_reward}")
            print(f"Time taken: {duration:.2f}s | Status: {time_status}")

    # Final summary across all problems run
    print(f"\n{SEP}\n{BOLD}{CYAN}=== Summary (per problem) ==={RESET}")
    for (
        pname,
        avg,
        dur,
        baseline_avg,
        baseline_time,
        is_estimated,
        t_status,
    ) in summaries:
        avg_col = f"{GREEN}{avg:.2f}{RESET}" if avg >= 0 else f"{RED}{avg:.2f}{RESET}"
        has_baseline = baseline_avg is not None
        if has_baseline:
            pct = (avg / baseline_avg * 100) if baseline_avg > 0 else 0
            comp = (
                f"{GREEN}BETTER{RESET}" if avg > baseline_avg else f"{RED}WORSE{RESET}"
            )
            baseline_label = "estimated baseline" if is_estimated else "baseline"
            baseline_str = f"{baseline_label} {YELLOW}{baseline_avg:.2f}{RESET} (time {YELLOW}{baseline_time:.2f}s{RESET})"
        else:
            pct = 0
            comp = ""
            baseline_str = "no baseline"
        marker = "🟩" if has_baseline and not is_estimated else "⬜"
        print(
            f"{marker} {BOLD}{pname}{RESET}: average = {avg_col} ({pct:.1f}% of {baseline_str}) | time = {YELLOW}{dur:.2f}s{RESET} | {comp} {t_status}"
        )

    total_time = sum(d for (_, _, d, _, _, _, _) in summaries)
    print(f"{BOLD}Total time for all problems: {RESET}{YELLOW}{total_time:.2f}s{RESET}")
    if summaries:
        overall_avg = sum(avg for (_, avg, _, _, _, _, _) in summaries) / len(summaries)
        print(
            f"{BOLD}Overall average across {len(summaries)} problems: {RESET}{YELLOW}{overall_avg:.2f}{RESET}"
        )


if __name__ == "__main__":
    main()
