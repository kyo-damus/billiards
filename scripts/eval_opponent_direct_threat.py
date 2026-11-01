import argparse

import numpy as np

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import make_components
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.opponent_threat import OpponentDirectThreatEvaluator


def evaluate(args):
    device = resolve_device(args.device)
    transition, _, direct_generator = make_components(
        args,
        device,
    )
    threat_evaluator = OpponentDirectThreatEvaluator(
        transition_model=transition,
        candidate_generator=direct_generator,
    )
    state_env = MicroBilliardEnv()

    total_root_candidates = 0
    opponent_states = 0
    geometric_counts = []
    successful_counts = []
    best_rewards = []
    examples = []

    for offset in range(args.states):
        state = random_state(
            state_env,
            args.start_seed + offset,
        )

        # Find one nonterminal DIRECT attempt that passes
        # the turn to the opponent on this board.
        for candidate in direct_generator.generate(state):
            total_root_candidates += 1
            first = transition.step(state, candidate)

            if (
                first.terminated
                or first.next_state.current_player
                == state.current_player
            ):
                continue

            threat = threat_evaluator.evaluate(
                first.next_state,
                defending_player=state.current_player,
            )
            opponent_states += 1
            geometric_counts.append(
                threat.geometric_candidate_count
            )
            successful_counts.append(
                threat.successful_pot_count
            )
            best_rewards.append(
                threat.best_success_reward
            )

            if len(examples) < 3:
                examples.append(
                    (
                        offset,
                        candidate.action.target_ball,
                        candidate.action.target_pocket,
                        threat,
                    )
                )
            break

        if (
            args.log_interval > 0
            and (offset + 1) % args.log_interval == 0
        ):
            print(
                f"states checked: {offset + 1}/{args.states}, "
                f"opponent states: {opponent_states}"
            )

    print()
    print("=== Opponent DIRECT threat (FastFiz) ===")
    print(f"device: {device}")
    print(f"initial boards examined: {args.states}")
    print(f"first-shot candidates examined: {total_root_candidates}")
    print(f"nonterminal opponent-turn states: {opponent_states}")

    if not opponent_states:
        print(
            "No valid opponent-turn state found. "
            "Increase --states."
        )
        return

    print(
        "mean geometrically feasible DIRECT candidates: "
        f"{np.mean(geometric_counts):.3f}"
    )
    print(
        "mean actually successful DIRECT shots: "
        f"{np.mean(successful_counts):.3f}"
    )
    print(
        "opponent has >=1 successful shot rate: "
        f"{np.mean(np.asarray(successful_counts) > 0):.3f}"
    )
    print(
        "mean best successful local reward: "
        f"{np.mean(best_rewards):.3f}"
    )
    print(
        "Note: one deterministic Micro action per DIRECT candidate. "
        "This is an outcome diagnostic, not a success probability."
    )

    print()
    print("=== Example opponent-turn states ===")
    for offset, ball, pocket, threat in examples:
        best = threat.best_candidate
        best_label = (
            (
                best.action.target_ball,
                best.action.target_pocket,
            )
            if best is not None
            else None
        )
        print(
            f"board={offset} first=({ball},{pocket}) "
            f"feasible={threat.geometric_candidate_count} "
            f"successful={threat.successful_pot_count} "
            f"best_reward={threat.best_success_reward:.3f} "
            f"best_target={best_label}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--residual-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--states",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--outcome-grid-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--outcome-top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--outcome-dedup-distance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    if args.states < 1:
        parser.error("--states must be >= 1")

    return args


if __name__ == "__main__":
    evaluate(parse_args())
