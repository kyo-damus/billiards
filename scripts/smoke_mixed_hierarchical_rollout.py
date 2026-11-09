"""Multi-turn end-to-end mixed Macro/Micro/FastFiz integration smoke.

Uses simplified two-object-ball game rules. Each turn performs a fresh
MCTS decision on the *executed* previous GameState. No training,
learning of high-elevation actions, or performance claims.
"""

import argparse

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.mixed_components import make_mixed_components
from src.controller.hierarchical_agent import HierarchicalAgent
from src.controller.mixed_hierarchical_rollout import (
    MixedHierarchicalRollout,
)
from src.controller.mixed_rollout_dataset import (
    save_mixed_rollout_npz,
)
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.candidate_policy import (
    load_macro_candidate_policy_network,
)
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)
from src.macro.mcts import MCTS
from src.macro.mixed_search import MixedRootGenerator, ZeroValue
from src.macro.value import load_macro_state_value_network


def build_agent(args, device):
    q, q_checkpoint = load_macro_candidate_value_network(
        args.candidate_value_checkpoint,
        device=device,
        expected_candidate_dim=31,
    )
    if (
        int(q_checkpoint.get("candidate_encoding_version", -1)) != 2
        or not bool(q_checkpoint.get("turn_aware", False))
    ):
        raise ValueError("Expected turn-aware mixed Candidate Q v2.")

    policy = None
    if args.candidate_policy_checkpoint:
        policy, policy_checkpoint = (
            load_macro_candidate_policy_network(
                args.candidate_policy_checkpoint,
                device=device,
                expected_candidate_dim=31,
            )
        )
        if (
            int(policy_checkpoint.get("candidate_encoding_version", -1)) != 2
            or not bool(policy_checkpoint.get("turn_aware", False))
        ):
            raise ValueError(
                "Expected turn-aware mixed Candidate Policy v2."
            )

    (
        transition,
        attack_generator,
        direct_generator,
        safety_shots,
        safety_adapter,
    ) = make_mixed_components(args, device)

    roots = MixedRootGenerator(
        attack_generator,
        safety_shots,
        safety_adapter,
    )

    state_value_checkpoint = getattr(
        args, "state_value_checkpoint", None
    )
    if state_value_checkpoint is not None:
        state_value_network, _ = load_macro_state_value_network(
            state_value_checkpoint, device=device,
        )
    else:
        state_value_network = ZeroValue().to(device)

    mcts = MCTS(
        policy_network=policy,
        value_network=state_value_network,
        action_generator=direct_generator,
        root_action_generator=roots,
        transition_model=transition,
        simulations=args.simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        device=device,
        candidate_mode=True,
        candidate_value_network=q,
        candidate_q_prior_visits=args.candidate_q_prior_visits,
        candidate_encoding_version=2,
        uniform_candidate_priors=policy is None,
        turn_aware=True,
        root_min_visits_before_puct=args.root_min_visits,
    )
    return HierarchicalAgent(
        mcts=mcts,
        transition_model=transition,
    )


def main(args):
    device = resolve_device(args.device)
    agent = build_agent(args, device)
    initial = random_state(
        MicroBilliardEnv(),
        args.seed,
    )
    rollout = MixedHierarchicalRollout(agent).run(
        initial,
        max_turns=args.max_turns,
    )

    print("=== Mixed hierarchical multi-turn smoke ===")
    print(f"device: {device}")
    print(f"initial seed: {args.seed}")
    print(f"max turns: {args.max_turns}")
    print(f"MCTS simulations per turn: {args.simulations}")
    print(
        "Macro Policy: "
        + (
            "learned 31-D"
            if args.candidate_policy_checkpoint
            else "uniform"
        )
    )
    print(
        "State V: "
        + (
            str(args.state_value_checkpoint)
            if args.state_value_checkpoint
            else "zero integration baseline"
        )
    )

    for turn in rollout.records:
        print(
            f"turn={turn.turn_index} "
            f"player={turn.player_before}->{turn.player_after} "
            f"strategy={turn.choice.action.strategy.value} "
            f"goal={turn.choice.goal.goal_type.value} "
            f"target_ball={turn.choice.action.target_ball} "
            f"target_pocket={turn.choice.action.target_pocket} "
            f"macro_local_reward={turn.reward:+.3f} "
            f"score_delta={turn.score_delta} "
            f"newly_pocketed={turn.newly_pocketed_balls} "
            f"terminated={turn.terminated}"
        )

    print(
        "final score: "
        f"{rollout.final_state.score.tolist()}"
    )
    print(
        "final pockets: "
        f"{rollout.final_state.ball_pocket_indices.tolist()}"
    )
    print(f"stop reason: {rollout.stop_reason}")
    print(f"game terminated: {rollout.terminated}")
    if args.save_rollout is not None:
        path = save_mixed_rollout_npz(
            rollout,
            args.save_rollout,
            seed=args.seed,
        )
        print(
            f"saved mixed rollout experience: {path} "
            f"({len(rollout.records)} executed turns)"
        )

    print(
        "Macro local rewards and SimplifiedGameRules score deltas "
        "are intentionally separate. Wrong-pocket shots can score "
        "a ball yet receive a negative Macro local reward."
    )
    print(
        "Uses simplified rules and selected model-based Safety "
        "candidates. This is a multi-turn execution smoke, "
        "not a game-strength benchmark."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint", required=True)
    parser.add_argument("--candidate-policy-checkpoint")
    parser.add_argument(
        "--state-value-checkpoint",
        help="Optional 12-D turn-aware state V trained on complete games.",
    )
    parser.add_argument(
        "--save-rollout",
        help=(
            "Optional .npz experience log of executed turns and "
            "MCTS root candidates; NOT supervised Q targets."
        ),
    )
    parser.add_argument("--seed", type=int, default=94000)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=15)
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument("--candidate-q-prior-visits", type=float, default=2.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--outcome-grid-size", type=int, default=5)
    parser.add_argument("--outcome-top-k", type=int, default=8)
    parser.add_argument("--outcome-dedup-distance", type=float, default=0.10)
    parser.add_argument("--safety-top-k", type=int, default=4)
    parser.add_argument(
        "--safety-powers", type=float, nargs="+", default=[0.65, 1.10]
    )
    parser.add_argument(
        "--safety-angles", type=float, nargs="+", default=[-12.0, 0.0, 12.0]
    )
    parser.add_argument(
        "--safety-tip-y-mm", type=float, nargs="+", default=[0.0, 5.0]
    )
    args = parser.parse_args()

    if args.max_turns < 1 or args.simulations < 1:
        parser.error("--max-turns and --simulations must be positive.")
    if args.root_min_visits < 0 or args.candidate_q_prior_visits < 0:
        parser.error("Root minimum visits and Q prior visits must be >= 0.")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be in [0, 1].")
    return args


if __name__ == "__main__":
    main(parse_args())
