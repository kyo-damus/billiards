"""Compare the learned WPA break with a small fixed shot grid on CPU."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import NineBallGoal, reference_angle_deg
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_physics_guard import InvalidPhysicalTransition, simulate_nine_ball_shot
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_session import create_nine_ball_rack
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER


def check_breaks(checkpoint, start_seed, racks):
    if racks < 1 or start_seed < 0:
        raise ValueError("Use positive --racks and nonnegative --start-seed")
    agent = NineBallMicroSACAgent(checkpoint, device="cpu", learned_break=True)
    decoder = WPA_NINE_BALL_BREAK_DECODER
    agent.break_decoder = decoder
    goal = NineBallGoal("BREAK", 1)
    context = NineBallContext()
    rules = NineBallRules(wpa_break_rules=True)
    first = create_nine_ball_rack(breaker=0, seed=start_seed)
    simulator = MultiBallFastFizSimulator(first.config)
    simulator.require_head_string_crossings()
    # Include the current decoder's minimum, midpoint and maximum speed,
    # center/offset aim, and three vertical tip offsets at 3 degrees elevation.
    grid = tuple(product((-1.0, 0.0, 1.0), (-0.5, 0.0, 0.5), (-0.5, 0.0, 0.5)))
    actor_counts, grid_counts = Counter(), Counter()
    grid_stats = [Counter() for _ in grid]
    grid_legal_racks = 0

    def trial(state, shot):
        try:
            _, result = simulate_nine_ball_shot(
                simulator, rules, state, context, target_ball=1, shot=shot,
            )
        except InvalidPhysicalTransition:
            return "physics_rejected"
        if result.foul:
            return "foul"
        if result.illegal_break:
            return "three_ball_short"
        return "legal_with_pot" if result.newly_pocketed else "legal_without_pot"

    for index in range(racks):
        state = create_nine_ball_rack(breaker=index % 2, seed=start_seed + index)
        actor_counts[trial(state, agent.plan_goal(state, goal, context).shot)] += 1
        angle = reference_angle_deg(state, goal)
        any_legal = False
        for knobs, counts in zip(grid, grid_stats, strict=True):
            speed, aim, tip_y = knobs
            action = (speed, aim, -0.6, 0.0, tip_y)
            outcome = trial(state, decoder.decode(action, angle))
            counts[outcome] += 1
            grid_counts[outcome] += 1
            any_legal |= outcome in ("legal_with_pot", "legal_without_pot")
        grid_legal_racks += any_legal

    best = max(range(len(grid)), key=lambda i: (
        grid_stats[i]["legal_with_pot"] + grid_stats[i]["legal_without_pot"],
        grid_stats[i]["legal_with_pot"],
    ))
    speed, aim, tip_y = grid[best]
    return {
        "start_seed": start_seed,
        "racks": racks,
        "actor_outcomes": dict(actor_counts),
        "grid_outcomes": dict(grid_counts),
        "grid_shots_per_rack": len(grid),
        "racks_with_any_legal_grid_shot": grid_legal_racks,
        "best_fixed_grid_action": [speed, aim, -0.6, 0.0, tip_y],
        "best_fixed_grid_outcomes": dict(grid_stats[best]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-checkpoint", type=Path, required=True)
    parser.add_argument("--racks", type=int, default=24)
    parser.add_argument("--start-seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check_breaks(args.micro_checkpoint, args.start_seed, args.racks), indent=2))


if __name__ == "__main__":
    main()
