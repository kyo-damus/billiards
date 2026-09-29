import argparse

import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)
from src.macro.position_feasibility import (
    PositionGoalFeasibilityChecker,
)
from src.macro.reachability_aware_position_generator import (
    ReachabilityAwarePositionCandidateGenerator,
)


def build_game_state(
    env,
):
    snapshot = env.sim.snapshot()

    return GameState(
        ball_positions=(
            snapshot.positions.copy()
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=(
            snapshot.pocket_indices.copy()
        ),
    )


def sample_state(
    env,
    seed,
):
    env.set_macro_action(
        MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=0,
        )
    )

    env.reset(
        seed=seed
    )

    return build_game_state(
        env
    )


def summarize(
    records,
):
    if not records:
        return {
            "count": 0,
            "pot_rate": 0.0,
            "reachable_rate": 0.0,
            "mean_best_distance": float("nan"),
        }

    pot_count = sum(
        int(
            item.feasibility.pot_found
        )
        for item in records
    )

    reachable_count = sum(
        int(
            item.feasibility.reachable
        )
        for item in records
    )

    distances = [
        float(
            item.feasibility
            .best_cue_distance
        )
        for item in records
        if (
            item.feasibility.pot_found
            and np.isfinite(
                item.feasibility
                .best_cue_distance
            )
        )
    ]

    return {
        "count": len(records),
        "pot_rate": (
            pot_count / len(records)
        ),
        "reachable_rate": (
            reachable_count
            / len(records)
        ),
        "mean_best_distance": (
            float(
                np.mean(distances)
            )
            if distances
            else float("nan")
        ),
    }


def main(
    args,
):
    state_env = (
        MicroBilliardEnv()
    )

    base_generator = (
        PositionAttackCandidateGenerator(
            target_radius=(
                args.radius
            )
        )
    )

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=args.grid_size
        )
    )

    reachability_generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                base_generator
            ),
            feasibility_checker=checker,
            top_k=args.top_k,
        )
    )

    baseline_records = []
    selected_records = []

    checked_states = 0
    fallback_states = 0
    states_with_reachable_before = 0
    states_with_reachable_after = 0

    attempts = 0

    while (
        checked_states < args.cases
        and attempts < args.cases * 20
    ):
        attempts += 1

        state = sample_state(
            state_env,
            seed=(
                args.seed
                + attempts * 100
            ),
        )

        ranked = (
            reachability_generator.rank(
                state
            )
        )

        if not ranked:
            continue

        selected = (
            reachability_generator
            .select_ranked(
                ranked
            )
        )

        if not any(
            item.feasibility.pot_found
            for item in ranked
        ):
            fallback_states += 1

        if any(
            item.feasibility.reachable
            for item in ranked
        ):
            states_with_reachable_before += 1

        if any(
            item.feasibility.reachable
            for item in selected
        ):
            states_with_reachable_after += 1

        baseline_records.extend(
            ranked
        )

        selected_records.extend(
            selected
        )

        checked_states += 1

        print(
            f"state={checked_states:3d} "
            f"baseline={len(ranked):3d} "
            f"selected={len(selected):2d} "
            f"pot_before="
            f"{sum(int(x.feasibility.pot_found) for x in ranked):2d} "
            f"reachable_before="
            f"{sum(int(x.feasibility.reachable) for x in ranked):2d} "
            f"reachable_after="
            f"{sum(int(x.feasibility.reachable) for x in selected):2d}"
        )

    if checked_states == 0:
        raise RuntimeError(
            "No POSITION candidates were generated."
        )

    before = summarize(
        baseline_records
    )

    after = summarize(
        selected_records
    )

    print()
    print(
        "=== Position Candidate Comparison ==="
    )
    print(
        f"states: {checked_states}"
    )
    print(
        f"grid size: "
        f"{args.grid_size} x "
        f"{args.grid_size}"
    )
    print(
        f"target radius: "
        f"{args.radius:.3f} m"
    )
    print(
        f"top-k: {args.top_k}"
    )
    print()

    print(
        "[Existing generator]"
    )
    print(
        f"candidates: "
        f"{before['count']}"
    )
    print(
        f"pot-found rate: "
        f"{before['pot_rate']:.3f}"
    )
    print(
        f"reachable rate: "
        f"{before['reachable_rate']:.3f}"
    )
    print(
        f"mean best cue distance: "
        f"{before['mean_best_distance']:.3f} m"
    )

    print()
    print(
        "[Reachability-aware]"
    )
    print(
        f"candidates: "
        f"{after['count']}"
    )
    print(
        f"pot-found rate: "
        f"{after['pot_rate']:.3f}"
    )
    print(
        f"reachable rate: "
        f"{after['reachable_rate']:.3f}"
    )
    print(
        f"mean best cue distance: "
        f"{after['mean_best_distance']:.3f} m"
    )

    print()
    print(
        "states with >=1 reachable candidate "
        f"(before): "
        f"{states_with_reachable_before / checked_states:.3f}"
    )
    print(
        "states with >=1 reachable candidate "
        f"(after): "
        f"{states_with_reachable_after / checked_states:.3f}"
    )
    print(
        "fallback state rate: "
        f"{fallback_states / checked_states:.3f}"
    )
    print(
        "candidate reduction: "
        f"{1.0 - after['count'] / before['count']:.3f}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
