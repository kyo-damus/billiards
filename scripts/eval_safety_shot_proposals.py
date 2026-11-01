"""Paired diagnostic of a baseline failed DIRECT and searched SAFETY shot.

The SAFETY winner is selected using deterministic FastFiz lookahead on
the same board; this is an in-model search result, not generalization.
"""

import argparse
import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import make_components
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.opponent_threat import OpponentDirectThreatEvaluator
from src.macro.outcome_guided_safety_shots import (
    OutcomeGuidedSafetyShotGenerator,
)
from src.micro.defense_shot import DefenseShotExecutor


def find_missed_direct_baseline(
    state,
    transition,
    direct_generator,
):
    """First deterministic non-potted, nonterminal turn-change DIRECT."""
    for candidate in direct_generator.generate(state):
        result = transition.step(state, candidate)
        if (
            not result.terminated
            and result.next_state.current_player != state.current_player
            and result.evaluation is not None
            and not result.evaluation.target_potted
            and not result.evaluation.scratch
        ):
            return result

    return None


def main(args):
    device = resolve_device(args.device)

    transition, _, direct_generator = make_components(
        args,
        device,
    )
    opponent_threat_evaluator = OpponentDirectThreatEvaluator(
        transition_model=transition,
        candidate_generator=direct_generator,
    )
    generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=DefenseShotExecutor(),
        opponent_threat_evaluator=opponent_threat_evaluator,
        powers=args.powers,
        angle_offsets_deg=args.angle_offsets,
        tip_offsets_mm=[
            (0.0, tip_y)
            for tip_y in args.tip_y_mm
        ],
        top_k=args.top_k,
    )
    env = MicroBilliardEnv()

    paired_direct = []
    paired_safety = []
    attempted = 0
    eligible = 0
    boards_with_safety = 0
    safety_zero_threat = 0

    print("=== Paired DIRECT miss vs searched SAFETY ===")
    print(f"Micro device: {device}")
    print(f"boards: {args.states}")
    print(
        "Shot grid: "
        f"powers={args.powers}, "
        f"offsets={args.angle_offsets}, "
        f"tip_y_mm={args.tip_y_mm}"
    )

    for board in range(args.states):
        state = random_state(
            env,
            args.start_seed + board,
        )

        safety = generator.generate(
            state
        )
        stats = generator.last_stats
        attempted += stats["attempted"]
        eligible += stats["eligible"]

        if safety:
            boards_with_safety += 1
            best = safety[0]
            safety_zero_threat += int(
                best.opponent_threat.successful_pot_count == 0
            )

        baseline = find_missed_direct_baseline(
            state,
            transition,
            direct_generator,
        )

        if baseline is not None and safety:
            threat = opponent_threat_evaluator.evaluate(
                baseline.next_state,
                defending_player=state.current_player,
            )
            paired_direct.append(
                threat.successful_pot_count
            )
            paired_safety.append(
                best.opponent_threat.successful_pot_count
            )

        if safety:
            print(
                f"board={board} tried={stats['attempted']} "
                f"eligible={stats['eligible']} "
                f"best_safety_threat="
                f"{best.opponent_threat.successful_pot_count} "
                f"geom={best.opponent_threat.geometric_candidate_count} "
                f"target={best.targeted_ball} "
                f"power={best.shot.power:.2f} "
                f"angle={best.shot.angle:.1f} "
                f"tip=({best.shot.tip_x_mm:.1f},"
                f"{best.shot.tip_y_mm:.1f})"
            )
        else:
            print(
                f"board={board} tried={stats['attempted']} "
                f"eligible=0 no safety shot"
            )

    print()
    print(f"total physical proposals: {attempted}")
    print(f"contact/turn-valid non-potted proposals: {eligible}")
    print(f"boards with safety proposals: {boards_with_safety}/{args.states}")
    print(
        "best safety eliminates opponent successful DIRECT "
        f"candidates: {safety_zero_threat}/{boards_with_safety}"
        if boards_with_safety
        else "no safety proposals available"
    )

    if paired_direct:
        direct_array = np.asarray(
            paired_direct,
            dtype=np.float64,
        )
        safety_array = np.asarray(
            paired_safety,
            dtype=np.float64,
        )
        print()
        print(f"paired boards: {len(paired_direct)}")
        print(
            "mean opponent successful DIRECT count after "
            f"failed attack: {direct_array.mean():.3f}"
        )
        print(
            "mean opponent successful DIRECT count after "
            f"searched safety: {safety_array.mean():.3f}"
        )
        print(
            "mean paired difference (safety - failed attack): "
            f"{(safety_array - direct_array).mean():+.3f}"
        )
        print(
            "safety better / tied / worse: "
            f"{np.count_nonzero(safety_array < direct_array)} / "
            f"{np.count_nonzero(safety_array == direct_array)} / "
            f"{np.count_nonzero(safety_array > direct_array)}"
        )

    print(
        "Caution: same-board counterfactual, with in-model "
        "lookahead optimization of Safety. This does not measure "
        "out-of-sample learned Defense performance."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument(
        "--states", type=int, default=5
    )
    parser.add_argument(
        "--start-seed", type=int, default=90000
    )
    parser.add_argument(
        "--powers", type=float, nargs="+",
        default=[0.65, 1.10],
    )
    parser.add_argument(
        "--angle-offsets", type=float, nargs="+",
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--tip-y-mm", type=float, nargs="+",
        default=[0.0, 5.0],
    )
    parser.add_argument(
        "--top-k", type=int, default=4
    )
    parser.add_argument(
        "--residual-scale", type=float, default=0.75
    )
    parser.add_argument(
        "--radius", type=float, default=0.10
    )
    parser.add_argument(
        "--outcome-grid-size", type=int, default=5
    )
    parser.add_argument(
        "--outcome-top-k", type=int, default=8
    )
    parser.add_argument(
        "--outcome-dedup-distance", type=float, default=0.10
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    if args.states < 1 or args.top_k < 1:
        parser.error("--states and --top-k must be >= 1")
    if not args.powers or any(
        not np.isfinite(value) or not 0.5 <= value <= 3.0
        for value in args.powers
    ):
        parser.error("--powers must be in [0.5, 3.0]")
    if not args.angle_offsets or any(
        not np.isfinite(value) or abs(value) > 35.0
        for value in args.angle_offsets
    ):
        parser.error("--angle-offsets must be in [-35, 35]")
    if not args.tip_y_mm or any(
        not np.isfinite(value) or abs(value) > 12.0
        for value in args.tip_y_mm
    ):
        parser.error("--tip-y-mm must be in [-12, 12]")
    return args


if __name__ == "__main__":
    main(parse_args())
