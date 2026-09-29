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
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
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


def evaluate_candidates(
    checker,
    state,
    candidates,
    holdout_actions,
):
    return [
        checker.check_actions(
            state,
            candidate,
            holdout_actions,
        )
        for candidate in candidates
    ]


def summarize_candidate_results(
    results,
):
    if not results:
        return {
            "count": 0,
            "pot_count": 0,
            "reachable_count": 0,
            "distance_sum": 0.0,
            "distance_count": 0,
        }

    pot_count = sum(
        int(result.pot_found)
        for result in results
    )

    reachable_count = sum(
        int(result.reachable)
        for result in results
    )

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

    return {
        "count": len(results),
        "pot_count": pot_count,
        "reachable_count": (
            reachable_count
        ),
        "distance_sum": float(
            np.sum(distances)
        ) if distances else 0.0,
        "distance_count": len(
            distances
        ),
    }


class Aggregate:

    def __init__(self):
        self.candidate_count = 0
        self.pot_count = 0
        self.reachable_count = 0
        self.distance_sum = 0.0
        self.distance_count = 0

        self.state_count = 0
        self.nonempty_state_count = 0
        self.state_has_pot = 0
        self.state_has_reachable = 0

        self.best_set_distance_sum = 0.0
        self.best_set_distance_count = 0

    def add(
        self,
        results,
    ):
        summary = (
            summarize_candidate_results(
                results
            )
        )

        self.state_count += 1

        if results:
            self.nonempty_state_count += 1

        self.candidate_count += (
            summary["count"]
        )
        self.pot_count += (
            summary["pot_count"]
        )
        self.reachable_count += (
            summary["reachable_count"]
        )
        self.distance_sum += (
            summary["distance_sum"]
        )
        self.distance_count += (
            summary["distance_count"]
        )

        if any(
            result.pot_found
            for result in results
        ):
            self.state_has_pot += 1

        if any(
            result.reachable
            for result in results
        ):
            self.state_has_reachable += 1

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

        if distances:
            self.best_set_distance_sum += (
                float(
                    min(distances)
                )
            )
            self.best_set_distance_count += 1

    def print_summary(
        self,
        name,
    ):
        print()
        print(
            f"[{name}]"
        )
        print(
            "avg candidates/state: "
            f"{self.candidate_count / self.state_count:.2f}"
        )
        print(
            "nonempty state rate: "
            f"{self.nonempty_state_count / self.state_count:.3f}"
        )

        if self.candidate_count > 0:
            print(
                "pot-found rate: "
                f"{self.pot_count / self.candidate_count:.3f}"
            )
            print(
                "reachable rate: "
                f"{self.reachable_count / self.candidate_count:.3f}"
            )
        else:
            print(
                "pot-found rate: nan"
            )
            print(
                "reachable rate: nan"
            )

        if self.distance_count > 0:
            print(
                "mean best cue distance "
                "(pot-found candidates): "
                f"{self.distance_sum / self.distance_count:.3f} m"
            )
        else:
            print(
                "mean best cue distance "
                "(pot-found candidates): nan"
            )

        print(
            "set has pot rate: "
            f"{self.state_has_pot / self.state_count:.3f}"
        )
        print(
            "set has reachable rate: "
            f"{self.state_has_reachable / self.state_count:.3f}"
        )

        if (
            self.best_set_distance_count
            > 0
        ):
            print(
                "mean best-of-set cue distance: "
                f"{self.best_set_distance_sum / self.best_set_distance_count:.3f} m"
            )
        else:
            print(
                "mean best-of-set cue distance: nan"
            )


def main(
    args,
):
    rng = np.random.default_rng(
        args.seed
    )

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

    reachability_generator = (
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

    outcome_generator = (
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=(
                args.coarse_grid_size
            ),
            target_radius=(
                args.radius
            ),
            top_k=args.top_k,
            dedup_distance=(
                args.dedup_distance
            ),
        )
    )

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

    random_agg = Aggregate()
    reachability_agg = Aggregate()
    outcome_agg = Aggregate()

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

        base_candidates = (
            base_generator.generate(
                state
            )
        )

        if not base_candidates:
            continue

        ranked = (
            reachability_generator.rank(
                state
            )
        )

        reachability_candidates = [
            item.candidate
            for item in (
                reachability_generator
                .select_ranked(
                    ranked
                )
            )
        ]

        outcome_candidates = (
            outcome_generator.generate(
                state
            )
        )

        random_n = min(
            args.top_k,
            len(base_candidates),
        )

        for _ in range(
            args.random_repeats
        ):
            random_indices = (
                rng.choice(
                    len(base_candidates),
                    size=random_n,
                    replace=False,
                )
            )

            random_candidates = [
                base_candidates[
                    int(index)
                ]
                for index
                in random_indices
            ]

            random_agg.add(
                evaluate_candidates(
                    holdout_checker,
                    state,
                    random_candidates,
                    holdout_actions,
                )
            )

        reachability_results = (
            evaluate_candidates(
                holdout_checker,
                state,
                reachability_candidates,
                holdout_actions,
            )
        )

        outcome_results = (
            evaluate_candidates(
                holdout_checker,
                state,
                outcome_candidates,
                holdout_actions,
            )
        )

        reachability_agg.add(
            reachability_results
        )
        outcome_agg.add(
            outcome_results
        )

        checked_states += 1

        print(
            f"state={checked_states:3d} "
            f"base={len(base_candidates):3d} "
            f"reach={len(reachability_candidates):2d} "
            f"outcome={len(outcome_candidates):2d} "
            f"outcome_holdout_reachable="
            f"{sum(int(x.reachable) for x in outcome_results):2d}"
        )

    if checked_states == 0:
        raise RuntimeError(
            "No POSITION states evaluated."
        )

    print()
    print(
        "=== Position Generation Strategy Comparison ==="
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
        f"hold-out actions: "
        f"{args.holdout_actions}"
    )
    print(
        f"target radius: "
        f"{args.radius:.3f} m"
    )
    print(
        f"top-k: {args.top_k}"
    )
    print(
        f"dedup distance: "
        f"{args.dedup_distance:.3f} m"
    )
    print(
        "Random baseline state count includes "
        f"{args.random_repeats} repeats/state."
    )

    random_agg.print_summary(
        "Random existing candidates"
    )
    reachability_agg.print_summary(
        "Reachability-aware ranking"
    )
    outcome_agg.print_summary(
        "Outcome-guided generation"
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
        "--dedup-distance",
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
