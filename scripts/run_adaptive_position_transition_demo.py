import argparse

import numpy as np
import torch

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
from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)
from src.micro.adaptive_position_policy import (
    AdaptivePositionMicroPolicy,
)


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


def main(args):
    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(
        f"device: {device}"
    )

    micro_policy = (
        AdaptivePositionMicroPolicy
        .from_checkpoints(
            base_checkpoint=(
                args.base_checkpoint
            ),
            bc_checkpoint=(
                args.bc_checkpoint
            ),
            gate_checkpoint=(
                args.gate_checkpoint
            ),
            device=device,
        )
    )

    generator = (
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

    state_env = MicroBilliardEnv()

    candidate = None
    state = None

    for attempt in range(
        args.max_attempts
    ):
        state = random_state(
            state_env,
            args.seed + attempt,
        )

        candidates = generator.generate(
            state
        )

        if candidates:
            candidate = candidates[0]
            break

    if candidate is None:
        raise RuntimeError(
            "Could not generate an "
            "outcome-guided POSITION candidate."
        )

    transition = PhysicsMacroTransitionModel(
        micro_agent=micro_policy,
        deterministic=True,
    )

    result = transition.step(
        state,
        candidate,
    )

    selection = (
        micro_policy.last_selection
    )

    goal = candidate.goal

    cue_after = (
        result.next_state
        .ball_positions[0]
    )

    cue_target = np.asarray(
        goal.cue_target_position,
        dtype=np.float32,
    )

    cue_distance = float(
        np.linalg.norm(
            cue_after
            - cue_target
        )
    )

    print()
    print(
        "=== Adaptive POSITION transition smoke ==="
    )
    print(
        "current action:",
        (
            candidate.action.target_ball,
            candidate.action.target_pocket,
        ),
    )
    print(
        "next target:",
        (
            goal.next_target_ball,
            goal.next_target_pocket,
        ),
    )
    print(
        "selected lambda:",
        (
            selection.blend_lambda
            if selection is not None
            else None
        ),
    )
    print(
        "gate confidence:",
        (
            f"{selection.confidence:.3f}"
            if selection is not None
            else "n/a"
        ),
    )
    print(
        "cue target:",
        cue_target,
    )
    print(
        "cue after:",
        cue_after,
    )
    print(
        "cue distance:",
        f"{cue_distance:.3f} m",
    )
    print(
        "target radius:",
        f"{goal.cue_target_radius:.3f} m",
    )
    print(
        "transition reward:",
        result.reward,
    )
    print(
        "transition terminated:",
        result.terminated,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gate-checkpoint",
        type=str,
        required=True,
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
        default=90000,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
