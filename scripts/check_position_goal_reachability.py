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
    # reset時に必要な仮Action
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


def main(args):

    rng = np.random.default_rng(
        args.seed
    )

    state_env = MicroBilliardEnv()

    generator = (
        PositionAttackCandidateGenerator(
            target_radius=args.radius
        )
    )

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=args.grid_size
        )
    )

    checked = 0
    pot_reachable = 0
    position_reachable = 0

    best_distances = []

    attempts = 0

    while (
        checked < args.cases
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

        candidates = (
            generator.generate(
                state
            )
        )

        if len(candidates) == 0:
            continue

        candidate_index = int(
            rng.integers(
                0,
                len(candidates),
            )
        )

        candidate = (
            candidates[
                candidate_index
            ]
        )

        result = checker.check(
            state,
            candidate,
        )

        checked += 1

        pot_reachable += int(
            result.pot_found
        )

        position_reachable += int(
            result.reachable
        )

        if np.isfinite(
            result.best_cue_distance
        ):
            best_distances.append(
                result.best_cue_distance
            )

        print(
            f"case={checked:3d} "
            f"pot={int(result.pot_found)} "
            f"position={int(result.reachable)} "
            f"best_distance="
            f"{result.best_cue_distance:.3f}"
        )

    if checked == 0:
        raise RuntimeError(
            "No position candidates checked."
        )

    print()
    print(
        "=== Position Goal Reachability ==="
    )

    print(
        f"cases: {checked}"
    )

    print(
        "pot reachable: "
        f"{pot_reachable / checked:.3f}"
    )

    print(
        "position reachable: "
        f"{position_reachable / checked:.3f}"
    )

    if len(best_distances) > 0:

        print(
            "mean best cue distance: "
            f"{np.mean(best_distances):.3f} m"
        )

        print(
            "median best cue distance: "
            f"{np.median(best_distances):.3f} m"
        )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
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
    