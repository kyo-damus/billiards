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


def evaluate_set(
    checker,
    state,
    candidates,
    holdout_actions,
):
    results = []

    for candidate in candidates:
        result = checker.check_actions(
            state,
            candidate,
            holdout_actions,
        )

        results.append(
            result
        )

    return results


def summarize_results(
    results,
):
    if not results:
        return {
            "count": 0,
            "pot_rate": 0.0,
            "reachable_rate": 0.0,
            "mean_best_distance": float("nan"),
        }

    pot_results = [
        result
        for result in results
        if (
            result.pot_found
            and np.isfinite(
                result.best_cue_distance
            )
        )
    ]

    return {
        "count": len(results),
        "pot_rate": float(
            np.mean(
                [
                    result.pot_found
                    for result in results
                ]
            )
        ),
        "reachable_rate": float(
            np.mean(
                [
                    result.reachable
                    for result in results
                ]
            )
        ),
        "mean_best_distance": (
            float(
                np.mean(
                    [
                        result.best_cue_distance
                        for result in pot_results
                    ]
                )
            )
            if pot_results
            else float("nan")
        ),
    }


def best_distance_in_set(
    results,
):
    distances = [
        result.best_cue_distance
        for result in results
        if (
            result.pot_found
            and np.isfinite(
                result.best_cue_distance
            )
        )
    ]

    if not distances:
        return float("nan")

    return float(
        min(distances)
    )


def main(
    args,
):
    state_env = (
        MicroBilliardEnv()
    )

    base_generator = (
        PositionAttackCandidateGenerator(
            target_radius=args.radius
        )
    )

    coarse_checker = (
        PositionGoalFeasibilityChecker(
            grid_size=(
                args.coarse_grid_size
            )
        )
    )

    holdout_checker = (
        PositionGoalFeasibilityChecker(
            grid_size=2
        )
    )

    generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                base_generator
            ),
            feasibility_checker=(
                coarse_checker
            ),
            top_k=args.top_k,
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    # coarse gridとは独立した連続一様サンプル。
    # 同じhold-out action集合を全候補へ使う。
    holdout_actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            args.holdout_actions,
            2,
        ),
    ).astype(
        np.float32
    )

    selected_all = []
    random_all = []

    selected_state_has_pot = []
    selected_state_has_reachable = []
    selected_state_best_distance = []

    random_state_has_pot = []
    random_state_has_reachable = []
    random_state_best_distance = []

    checked_states = 0
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

        ranked = generator.rank(
            state
        )

        if not ranked:
            continue

        selected_ranked = (
            generator.select_ranked(
                ranked
            )
        )

        if not selected_ranked:
            continue

        selected_candidates = [
            item.candidate
            for item in selected_ranked
        ]

        n_select = len(
            selected_candidates
        )

        selected_results = (
            evaluate_set(
                holdout_checker,
                state,
                selected_candidates,
                holdout_actions,
            )
        )

        selected_all.extend(
            selected_results
        )

        selected_state_has_pot.append(
            any(
                result.pot_found
                for result
                in selected_results
            )
        )

        selected_state_has_reachable.append(
            any(
                result.reachable
                for result
                in selected_results
            )
        )

        selected_state_best_distance.append(
            best_distance_in_set(
                selected_results
            )
        )

        all_candidates = [
            item.candidate
            for item in ranked
        ]

        state_random_pot = []
        state_random_reachable = []
        state_random_distance = []

        for _ in range(
            args.random_repeats
        ):
            indices = rng.choice(
                len(all_candidates),
                size=n_select,
                replace=False,
            )

            random_candidates = [
                all_candidates[
                    int(index)
                ]
                for index in indices
            ]

            random_results = (
                evaluate_set(
                    holdout_checker,
                    state,
                    random_candidates,
                    holdout_actions,
                )
            )

            random_all.extend(
                random_results
            )

            state_random_pot.append(
                any(
                    result.pot_found
                    for result
                    in random_results
                )
            )

            state_random_reachable.append(
                any(
                    result.reachable
                    for result
                    in random_results
                )
            )

            state_random_distance.append(
                best_distance_in_set(
                    random_results
                )
            )

        random_state_has_pot.extend(
            state_random_pot
        )
        random_state_has_reachable.extend(
            state_random_reachable
        )
        random_state_best_distance.extend(
            state_random_distance
        )

        checked_states += 1

        print(
            f"state={checked_states:3d} "
            f"all={len(all_candidates):3d} "
            f"selected={n_select:2d} "
            f"holdout_pot="
            f"{sum(int(x.pot_found) for x in selected_results):2d} "
            f"holdout_reachable="
            f"{sum(int(x.reachable) for x in selected_results):2d}"
        )

    if checked_states == 0:
        raise RuntimeError(
            "No POSITION candidates evaluated."
        )

    selected_summary = (
        summarize_results(
            selected_all
        )
    )

    random_summary = (
        summarize_results(
            random_all
        )
    )

    selected_finite_distances = [
        value
        for value
        in selected_state_best_distance
        if np.isfinite(
            value
        )
    ]

    random_finite_distances = [
        value
        for value
        in random_state_best_distance
        if np.isfinite(
            value
        )
    ]

    print()
    print(
        "=== Hold-out Position Candidate Evaluation ==="
    )
    print(
        f"states: {checked_states}"
    )
    print(
        "coarse grid: "
        f"{args.coarse_grid_size} x "
        f"{args.coarse_grid_size}"
    )
    print(
        f"hold-out actions per candidate: "
        f"{args.holdout_actions}"
    )
    print(
        f"top-k: {args.top_k}"
    )
    print(
        f"random repeats: "
        f"{args.random_repeats}"
    )
    print(
        f"target radius: "
        f"{args.radius:.3f} m"
    )

    print()
    print(
        "[Random candidates]"
    )
    print(
        f"candidate trials: "
        f"{random_summary['count']}"
    )
    print(
        f"pot-found rate: "
        f"{random_summary['pot_rate']:.3f}"
    )
    print(
        f"reachable rate: "
        f"{random_summary['reachable_rate']:.3f}"
    )
    print(
        "mean best cue distance "
        "(pot-found candidates): "
        f"{random_summary['mean_best_distance']:.3f} m"
    )
    print(
        "set has pot rate: "
        f"{np.mean(random_state_has_pot):.3f}"
    )
    print(
        "set has reachable rate: "
        f"{np.mean(random_state_has_reachable):.3f}"
    )
    print(
        "mean best-of-set cue distance: "
        f"{np.mean(random_finite_distances):.3f} m"
        if random_finite_distances
        else (
            "mean best-of-set cue distance: nan"
        )
    )

    print()
    print(
        "[Reachability-aware candidates]"
    )
    print(
        f"candidate trials: "
        f"{selected_summary['count']}"
    )
    print(
        f"pot-found rate: "
        f"{selected_summary['pot_rate']:.3f}"
    )
    print(
        f"reachable rate: "
        f"{selected_summary['reachable_rate']:.3f}"
    )
    print(
        "mean best cue distance "
        "(pot-found candidates): "
        f"{selected_summary['mean_best_distance']:.3f} m"
    )
    print(
        "set has pot rate: "
        f"{np.mean(selected_state_has_pot):.3f}"
    )
    print(
        "set has reachable rate: "
        f"{np.mean(selected_state_has_reachable):.3f}"
    )
    print(
        "mean best-of-set cue distance: "
        f"{np.mean(selected_finite_distances):.3f} m"
        if selected_finite_distances
        else (
            "mean best-of-set cue distance: nan"
        )
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--coarse-grid-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--holdout-actions",
        type=int,
        default=100,
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
        "--random-repeats",
        type=int,
        default=3,
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
