import argparse

import numpy as np
import torch

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.eval_position_residual import (
    load_base_agent,
    load_residual_network,
)
from scripts.train_macro_candidate_value import (
    calculate_metrics,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.game.simple_rules import (
    SimplifiedGameRules,
)
from src.goal.types import (
    TacticalGoalType,
)
from src.macro.candidate_encoding import (
    encode_macro_candidate,
)
from src.macro.candidate_generator import (
    ExpandedMacroCandidateGenerator,
    MacroCandidateGenerator,
)
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.encoding import (
    encode_game_state,
)
from src.macro.local_evaluation import (
    MacroLocalEvaluator,
)
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)
from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)
from src.micro.fixed_scale_position_residual_policy import (
    FixedScalePositionResidualPolicy,
)
from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
)


def predict_candidates(
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
        prediction = network(
            state_tensor,
            candidate_tensor,
        )

    return (
        prediction
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
        .reshape(-1)
    )


def make_components(
    args,
    device,
):
    base_agent = load_base_agent(
        args.base_checkpoint,
        device,
    )

    (
        residual_network,
        _,
    ) = load_residual_network(
        args.residual_checkpoint,
        device,
    )

    residual_policy = (
        PositionResidualMicroPolicy(
            base_agent=base_agent,
            residual_network=(
                residual_network
            ),
            device=device,
        )
    )

    micro_policy = (
        FixedScalePositionResidualPolicy(
            residual_policy=(
                residual_policy
            ),
            residual_scale=(
                args.residual_scale
            ),
            power_scale=1.0,
            angle_scale=0.0,
        )
    )

    transition = (
        PhysicsMacroTransitionModel(
            micro_agent=micro_policy,
            deterministic=True,
            game_rules=(
                SimplifiedGameRules()
            ),
            local_evaluator=(
                MacroLocalEvaluator()
            ),
        )
    )

    direct_generator = (
        MacroCandidateGenerator()
    )

    position_generator = (
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=(
                args.outcome_grid_size
            ),
            target_radius=(
                args.radius
            ),
            top_k=(
                args.outcome_top_k
            ),
            dedup_distance=(
                args.outcome_dedup_distance
            ),
        )
    )

    root_generator = (
        ExpandedMacroCandidateGenerator(
            direct_generator=(
                direct_generator
            ),
            position_generator=(
                position_generator
            ),
        )
    )

    return (
        transition,
        root_generator,
        direct_generator,
    )


def print_state_details(
    candidates,
    prediction,
    targets,
):
    rows = []

    for (
        candidate,
        pred,
        target,
    ) in zip(
        candidates,
        prediction,
        targets,
    ):
        goal = candidate.goal

        rows.append(
            (
                float(pred),
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
        "=== First state candidate Q details ==="
    )

    for (
        pred,
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
            f"pred_q={pred:.3f}",
            f"target_q={target:.3f}",
            f"error={pred - target:+.3f}",
        )


def main(
    args,
):
    device = resolve_device(
        args.device
    )

    (
        value_network,
        checkpoint,
    ) = (
        load_macro_candidate_value_network(
            args.candidate_value_checkpoint,
            device=device,
        )
    )

    if (
        bool(checkpoint.get("turn_aware", False))
        != bool(args.turn_aware)
    ):
        raise ValueError(
            "Candidate Q checkpoint turn_aware mode "
            "does not match --turn-aware."
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

    all_predictions = []
    all_targets = []
    all_state_ids = []
    all_goal_types = []

    first_details = None
    evaluated_states = 0

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

        prediction = predict_candidates(
            value_network,
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
                    turn_aware=args.turn_aware,
                )
            )

            targets.append(
                result.target
            )

        targets = np.asarray(
            targets,
            dtype=np.float32,
        )

        if (
            first_details is None
            and args.show_details
        ):
            first_details = (
                list(candidates),
                prediction.copy(),
                targets.copy(),
            )

        all_predictions.extend(
            prediction.tolist()
        )
        all_targets.extend(
            targets.tolist()
        )
        all_state_ids.extend(
            [
                evaluated_states
                for _ in candidates
            ]
        )
        all_goal_types.extend(
            [
                candidate.goal.goal_type
                for candidate in candidates
            ]
        )

        evaluated_states += 1

    if evaluated_states == 0:
        raise RuntimeError(
            "No states with Macro candidates "
            "were evaluated."
        )

    predictions = np.asarray(
        all_predictions,
        dtype=np.float32,
    )
    targets = np.asarray(
        all_targets,
        dtype=np.float32,
    )
    state_ids = np.asarray(
        all_state_ids,
        dtype=np.int64,
    )

    metrics = calculate_metrics(
        predictions,
        targets,
        state_ids,
    )

    print(
        "=== Macro candidate Q unseen evaluation ==="
    )
    print(
        f"device: {device}"
    )
    print(
        f"turn-aware targets: {args.turn_aware}"
    )
    print(
        "checkpoint:",
        args.candidate_value_checkpoint,
    )
    print(
        "checkpoint best val MSE:",
        checkpoint.get(
            "best_val_mse",
            "n/a",
        ),
    )
    print(
        f"states evaluated: {evaluated_states}"
    )
    print(
        f"samples: {len(targets)}"
    )
    print(
        f"MSE: {metrics['mse']:.6f}"
    )
    print(
        f"MAE: {metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {metrics['rmse']:.4f}"
    )
    print(
        "top-1 candidate accuracy: "
        f"{metrics['top1_accuracy']:.3f}"
    )

    goal_types = np.asarray(
        [
            goal_type.value
            for goal_type
            in all_goal_types
        ],
        dtype=object,
    )

    for goal_type in (
        TacticalGoalType.DIRECT_ATTACK,
        TacticalGoalType.POSITION_ATTACK,
    ):
        mask = (
            goal_types
            == goal_type.value
        )

        if not np.any(
            mask
        ):
            continue

        print(
            f"{goal_type.value}: "
            f"n={int(mask.sum())}, "
            f"target_mean="
            f"{targets[mask].mean():.3f}, "
            f"pred_mean="
            f"{predictions[mask].mean():.3f}"
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
        "--candidate-value-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--states",
        type=int,
        default=1,
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
        "--turn-aware",
        action="store_true",
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
