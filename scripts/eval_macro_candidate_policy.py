import argparse

import numpy as np
import torch

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_macro_candidate_value import (
    make_components,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.types import (
    TacticalGoalType,
)
from src.macro.candidate_encoding import (
    encode_macro_candidate,
)
from src.macro.candidate_policy import (
    load_macro_candidate_policy_network,
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.encoding import (
    encode_game_state,
)


def predict_policy(
    network,
    state,
    candidates,
    device,
):
    state_tensor = torch.as_tensor(
        encode_game_state(
            state
        ),
        dtype=torch.float32,
        device=device,
    )

    candidate_tensor = torch.as_tensor(
        np.stack(
            [
                encode_macro_candidate(
                    candidate
                )
                for candidate
                in candidates
            ],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )

    network.eval()

    with torch.no_grad():
        logits = network(
            state_tensor,
            candidate_tensor,
        )

        probability = torch.softmax(
            logits,
            dim=0,
        )

    return (
        probability
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
        .reshape(-1)
    )


def calculate_state_metrics(
    probability,
    target_q,
):
    probability = np.asarray(
        probability,
        dtype=np.float32,
    )
    target_q = np.asarray(
        target_q,
        dtype=np.float32,
    )

    if (
        probability.ndim != 1
        or target_q.ndim != 1
        or len(probability)
        != len(target_q)
    ):
        raise ValueError(
            "probability and target_q "
            "must be matching 1-D arrays."
        )

    if len(target_q) == 0:
        raise ValueError(
            "At least one candidate is required."
        )

    best_q = float(
        np.max(
            target_q
        )
    )

    selected_index = int(
        np.argmax(
            probability
        )
    )

    selected_q = float(
        target_q[
            selected_index
        ]
    )

    top1_correct = float(
        np.isclose(
            selected_q,
            best_q,
            atol=1e-6,
        )
    )

    top_k = min(
        3,
        len(target_q),
    )

    top_indices = np.argsort(
        probability
    )[
        -top_k:
    ]

    top3_hit = float(
        np.any(
            np.isclose(
                target_q[
                    top_indices
                ],
                best_q,
                atol=1e-6,
            )
        )
    )

    optimal_mask = np.isclose(
        target_q,
        best_q,
        atol=1e-6,
    )

    optimal_prior_mass = float(
        np.sum(
            probability[
                optimal_mask
            ]
        )
    )

    regret = float(
        best_q
        - selected_q
    )

    return {
        "top1_correct": top1_correct,
        "top3_hit": top3_hit,
        "regret": regret,
        "optimal_prior_mass": (
            optimal_prior_mass
        ),
    }


def print_state_details(
    candidates,
    probability,
    targets,
):
    rows = []

    for (
        candidate,
        prior,
        target,
    ) in zip(
        candidates,
        probability,
        targets,
    ):
        goal = candidate.goal

        rows.append(
            (
                float(prior),
                float(target),
                goal.goal_type.value,
                (
                    candidate.action.target_ball,
                    candidate.action.target_pocket,
                ),
                (
                    (
                        goal.next_target_ball,
                        goal.next_target_pocket,
                    )
                    if (
                        goal.goal_type
                        == TacticalGoalType.POSITION_ATTACK
                    )
                    else None
                ),
            )
        )

    rows.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    print()
    print(
        "=== First state Candidate Policy details ==="
    )

    for (
        prior,
        target,
        goal_type,
        current_target,
        next_target,
    ) in rows:
        print(
            goal_type,
            "current=",
            current_target,
            "next=",
            next_target,
            f"prior={prior:.3f}",
            f"target_q={target:.3f}",
        )


def main(
    args,
):
    device = resolve_device(
        args.device
    )

    (
        policy_network,
        checkpoint,
    ) = (
        load_macro_candidate_policy_network(
            args.candidate_policy_checkpoint,
            device=device,
        )
    )

    (
        transition,
        root_generator,
        next_generator,
    ) = make_components(
        args,
        device,
    )

    sample_env = MicroBilliardEnv()

    state_metrics = []

    position_prior_masses = []
    direct_prior_masses = []

    first_details = None
    evaluated_states = 0
    total_candidates = 0

    for state_offset in range(
        args.states
    ):
        state = random_state(
            sample_env,
            args.start_seed
            + state_offset,
        )

        candidates = list(
            root_generator.generate(
                state
            )
        )

        if not candidates:
            continue

        probability = predict_policy(
            policy_network,
            state,
            candidates,
            device,
        )

        targets = []

        for candidate in candidates:
            result = (
                evaluate_two_step_candidate_target(
                    state=state,
                    candidate=candidate,
                    transition_model=(
                        transition
                    ),
                    next_candidate_generator=(
                        next_generator
                    ),
                    gamma=args.gamma,
                )
            )

            targets.append(
                result.target
            )

        targets = np.asarray(
            targets,
            dtype=np.float32,
        )

        metrics = calculate_state_metrics(
            probability,
            targets,
        )

        state_metrics.append(
            metrics
        )

        position_mask = np.asarray(
            [
                candidate.goal.goal_type
                == TacticalGoalType.POSITION_ATTACK
                for candidate in candidates
            ],
            dtype=bool,
        )

        direct_mask = np.asarray(
            [
                candidate.goal.goal_type
                == TacticalGoalType.DIRECT_ATTACK
                for candidate in candidates
            ],
            dtype=bool,
        )

        position_prior_masses.append(
            float(
                probability[
                    position_mask
                ].sum()
            )
        )

        direct_prior_masses.append(
            float(
                probability[
                    direct_mask
                ].sum()
            )
        )

        if (
            first_details is None
            and args.show_details
        ):
            first_details = (
                list(candidates),
                probability.copy(),
                targets.copy(),
            )

        evaluated_states += 1
        total_candidates += len(
            candidates
        )

    if evaluated_states == 0:
        raise RuntimeError(
            "No states with Macro candidates "
            "were evaluated."
        )

    top1 = np.mean(
        [
            metrics[
                "top1_correct"
            ]
            for metrics
            in state_metrics
        ]
    )

    top3 = np.mean(
        [
            metrics[
                "top3_hit"
            ]
            for metrics
            in state_metrics
        ]
    )

    mean_regret = np.mean(
        [
            metrics[
                "regret"
            ]
            for metrics
            in state_metrics
        ]
    )

    mean_optimal_prior_mass = np.mean(
        [
            metrics[
                "optimal_prior_mass"
            ]
            for metrics
            in state_metrics
        ]
    )

    print(
        "=== Macro Candidate Policy unseen evaluation ==="
    )
    print(
        f"device: {device}"
    )
    print(
        "checkpoint:",
        args.candidate_policy_checkpoint,
    )
    print(
        "checkpoint best val top-1:",
        checkpoint.get(
            "best_val_metrics",
            {},
        ).get(
            "top1_accuracy",
            "n/a",
        ),
    )
    print(
        f"states evaluated: {evaluated_states}"
    )
    print(
        f"candidates: {total_candidates}"
    )
    print(
        f"top-1 candidate accuracy: "
        f"{top1:.3f}"
    )
    print(
        f"top-3 hit rate: {top3:.3f}"
    )
    print(
        f"mean selected-Q regret: "
        f"{mean_regret:.4f}"
    )
    print(
        "mean prior mass on optimal candidates: "
        f"{mean_optimal_prior_mass:.3f}"
    )
    print(
        "mean DIRECT prior mass: "
        f"{np.mean(direct_prior_masses):.3f}"
    )
    print(
        "mean POSITION prior mass: "
        f"{np.mean(position_prior_masses):.3f}"
    )

    if first_details is not None:
        print_state_details(
            *first_details
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--candidate-policy-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--states",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=90000,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
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
        "--show-details",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    if args.states < 1:
        parser.error(
            "--states must be >= 1"
        )

    if not (
        0.0
        <= args.gamma
        <= 1.0
    ):
        parser.error(
            "--gamma must be between 0 and 1"
        )

    return args


if __name__ == "__main__":
    main(
        parse_args()
    )
