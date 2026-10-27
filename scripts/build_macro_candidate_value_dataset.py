import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.eval_position_residual import (
    load_base_agent,
    load_residual_network,
)
from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
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


GOAL_TYPE_TO_ID = {
    TacticalGoalType.DIRECT_ATTACK: 0,
    TacticalGoalType.POSITION_ATTACK: 1,
    TacticalGoalType.SAFETY: 2,
}


def resolve_device(
    requested,
):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def random_state(
    env,
    seed,
):
    placeholder = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=0,
    )

    env.set_macro_action(
        placeholder
    )

    env.reset(
        seed=seed
    )

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


def build_dataset(
    args,
):
    device = resolve_device(
        args.device
    )

    base_agent = load_base_agent(
        args.base_checkpoint,
        device,
    )

    (
        residual_network,
        residual_checkpoint,
    ) = load_residual_network(
        args.residual_checkpoint,
        device,
    )

    raw_residual_policy = (
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
                raw_residual_policy
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

    # 2手目はcheap DIRECT候補だけを見る。
    next_generator = (
        MacroCandidateGenerator()
    )

    sample_env = MicroBilliardEnv()

    encoded_states = []
    encoded_candidates = []
    q_targets = []

    immediate_rewards = []
    next_best_rewards = []
    next_candidate_counts = []

    state_ids = []
    goal_type_ids = []

    kept_states = 0

    for state_index in range(
        args.states
    ):
        state = random_state(
            sample_env,
            args.seed + state_index,
        )

        candidates = list(
            root_generator.generate(
                state
            )
        )

        if not candidates:
            continue

        state_encoding = (
            encode_game_state(
                state
            )
            .astype(
                np.float32
            )
        )

        current_state_id = (
            kept_states
        )
        kept_states += 1

        for candidate in candidates:
            target = (
                evaluate_two_step_candidate_target(
                    state=state,
                    candidate=candidate,
                    transition_model=transition,
                    next_candidate_generator=(
                        next_generator
                    ),
                    gamma=args.gamma,
                )
            )

            encoded_states.append(
                state_encoding.copy()
            )

            encoded_candidates.append(
                encode_macro_candidate(
                    candidate
                ).astype(
                    np.float32
                )
            )

            q_targets.append(
                target.target
            )
            immediate_rewards.append(
                target.immediate_reward
            )
            next_best_rewards.append(
                target.next_best_reward
            )
            next_candidate_counts.append(
                target.next_candidate_count
            )

            state_ids.append(
                current_state_id
            )

            goal_type_ids.append(
                GOAL_TYPE_TO_ID[
                    candidate.goal.goal_type
                ]
            )

        if (
            args.log_interval > 0
            and (
                state_index + 1
            )
            % args.log_interval
            == 0
        ):
            print(
                f"states processed: "
                f"{state_index + 1}/{args.states}, "
                f"samples: {len(q_targets)}"
            )

    if not q_targets:
        raise RuntimeError(
            "No MacroCandidate value samples "
            "were generated."
        )

    output_path = Path(
        args.output
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        state=np.stack(
            encoded_states,
            axis=0,
        ).astype(
            np.float32
        ),
        candidate=np.stack(
            encoded_candidates,
            axis=0,
        ).astype(
            np.float32
        ),
        q_target=np.asarray(
            q_targets,
            dtype=np.float32,
        ),
        immediate_reward=np.asarray(
            immediate_rewards,
            dtype=np.float32,
        ),
        next_best_reward=np.asarray(
            next_best_rewards,
            dtype=np.float32,
        ),
        next_candidate_count=np.asarray(
            next_candidate_counts,
            dtype=np.int32,
        ),
        state_id=np.asarray(
            state_ids,
            dtype=np.int32,
        ),
        goal_type=np.asarray(
            goal_type_ids,
            dtype=np.int32,
        ),
        gamma=np.asarray(
            args.gamma,
            dtype=np.float32,
        ),
        residual_scale=np.asarray(
            args.residual_scale,
            dtype=np.float32,
        ),
    )

    q_array = np.asarray(
        q_targets,
        dtype=np.float32,
    )

    print()
    print(
        "=== Macro candidate value dataset ==="
    )
    print(
        f"device: {device}"
    )
    print(
        f"requested states: {args.states}"
    )
    print(
        f"states with candidates: {kept_states}"
    )
    print(
        f"samples: {len(q_array)}"
    )
    print(
        "Q target mean / std: "
        f"{q_array.mean():.4f} / "
        f"{q_array.std():.4f}"
    )
    print(
        "Q target min / max: "
        f"{q_array.min():.4f} / "
        f"{q_array.max():.4f}"
    )
    print(
        "residual checkpoint mode:",
        residual_checkpoint.get(
            "residual_mode",
            "n/a",
        ),
    )
    print(
        f"output: {output_path}"
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
        "--output",
        type=str,
        default=(
            "macro_candidate_value_dataset.npz"
        ),
    )

    parser.add_argument(
        "--states",
        type=int,
        default=50,
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
        "--seed",
        type=int,
        default=91000,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
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
    build_dataset(
        parse_args()
    )
