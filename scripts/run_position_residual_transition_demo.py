import argparse
import numpy as np
import torch

from scripts.eval_position_residual import load_base_agent, load_residual_network
from src.common.types import GameState, MacroAction, Strategy
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.outcome_guided_position_generator import OutcomeGuidedPositionCandidateGenerator
from src.macro.physics_transition import PhysicsMacroTransitionModel
from src.micro.fixed_scale_position_residual_policy import FixedScalePositionResidualPolicy
from src.micro.position_residual_policy import PositionResidualMicroPolicy


def sample_state(env, seed):
    env.set_macro_action(
        MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=0,
        )
    )
    env.reset(seed=seed)
    snapshot = env.sim.snapshot()
    return GameState(
        ball_positions=snapshot.positions.copy(),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
        ball_pocket_indices=snapshot.pocket_indices.copy(),
    )


def main(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_agent = load_base_agent(args.base_checkpoint, device)
    residual_network, _ = load_residual_network(
        args.residual_checkpoint,
        device,
    )

    raw_policy = PositionResidualMicroPolicy(
        base_agent=base_agent,
        residual_network=residual_network,
        device=device,
    )
    micro_policy = FixedScalePositionResidualPolicy(
        residual_policy=raw_policy,
        residual_scale=args.residual_scale,
        power_scale=1.0,
        angle_scale=0.0,
    )

    generator = OutcomeGuidedPositionCandidateGenerator(
        grid_size=5,
        target_radius=0.10,
        top_k=8,
        dedup_distance=0.10,
    )
    env = MicroBilliardEnv()

    for attempt in range(100):
        state = sample_state(env, args.seed + attempt)
        candidates = generator.generate(state)
        if candidates:
            candidate = candidates[0]
            break
    else:
        raise RuntimeError("No POSITION candidate generated.")

    transition = PhysicsMacroTransitionModel(
        micro_agent=micro_policy,
        deterministic=True,
    )
    result = transition.step(state, candidate)
    selection = micro_policy.last_selection

    target = np.asarray(
        candidate.goal.cue_target_position,
        dtype=np.float32,
    )
    cue_after = result.next_state.ball_positions[0]
    distance = float(np.linalg.norm(cue_after - target))

    print("=== Position Residual transition ===")
    print("residual scale:", args.residual_scale)
    print("base action:", selection.base_action)
    print("raw residual:", selection.residual_action)
    print("cue distance:", f"{distance:.3f} m")
    print("target radius:", f"{candidate.goal.cue_target_radius:.3f} m")
    print("transition reward:", result.reward)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
