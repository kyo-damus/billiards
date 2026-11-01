"""Smoke: Attack and Safety root candidates in turn-aware v2 MCTS.

The 31-D Candidate Q is used for unvisited branch estimates.
No 31-D Candidate Policy has been trained, so priors are uniform.
Only DIRECT candidates are generated deeper in the tree; this
matches the two-step teacher's next-action set at the first depth.

The value network is a neutral zero baseline. This is a wiring
experiment, not a win-rate or trained mixed-policy evaluation.
"""

import argparse

import torch
import torch.nn as nn

from scripts.build_macro_candidate_value_dataset import (
    random_state,
    resolve_device,
)
from scripts.build_mixed_candidate_value_dataset import (
    make_mixed_components,
)
from src.common.types import Strategy
from src.env.micro_billiard_env import MicroBilliardEnv
from src.macro.candidate_value import (
    load_macro_candidate_value_network,
)
from src.macro.mcts import MCTS


class ZeroValue(nn.Module):
    def forward(self, state):
        return torch.zeros(
            state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )


class MixedRootGenerator:
    """ATTACK plus outcome-guided executable SAFETY, root only."""

    def __init__(
        self,
        attack_generator,
        safety_shots,
        safety_adapter,
    ):
        self.attack_generator = attack_generator
        self.safety_shots = safety_shots
        self.safety_adapter = safety_adapter
        self.last_attack_count = 0
        self.last_safety_count = 0

    def generate(self, state):
        attacks = list(
            self.attack_generator.generate(state)
        )
        proposals = self.safety_shots.generate(state)
        safeties = self.safety_adapter.from_proposals(
            proposals
        )
        self.last_attack_count = len(attacks)
        self.last_safety_count = len(safeties)
        return attacks + safeties


def main(args):
    device = resolve_device(args.device)
    value, checkpoint = load_macro_candidate_value_network(
        args.candidate_value_checkpoint,
        device=device,
        expected_candidate_dim=31,
    )

    if (
        checkpoint.get("candidate_encoding_version") != 2
        or not bool(checkpoint.get("turn_aware", False))
    ):
        raise ValueError(
            "Expected a turn-aware 31D mixed Candidate Q v2 checkpoint."
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
    mcts = MCTS(
        policy_network=None,
        value_network=ZeroValue().to(device),
        action_generator=direct_generator,
        root_action_generator=roots,
        transition_model=transition,
        simulations=args.simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        device=device,
        candidate_mode=True,
        candidate_value_network=value,
        candidate_q_prior_visits=args.candidate_q_prior_visits,
        candidate_encoding_version=2,
        uniform_candidate_priors=True,
        turn_aware=True,
        root_min_visits_before_puct=args.root_min_visits,
    )

    env = MicroBilliardEnv()
    chosen = {Strategy.ATTACK: 0, Strategy.DEFENSE: 0}
    total_safety_visits = 0
    total_attack_visits = 0

    print("=== Turn-aware mixed Candidate Q MCTS v2 smoke ===")
    print(f"device: {device}; boards: {args.states}")
    print(
        f"simulations/board: {args.simulations}, "
        f"root_min_visits: {args.root_min_visits}, "
        f"Q pseudo visits: {args.candidate_q_prior_visits}"
    )

    for board in range(args.states):
        state = random_state(
            env, args.start_seed + board
        )
        root = mcts.search(state)

        attacks = [
            (candidate, child)
            for candidate, child in root.children.items()
            if candidate.action.strategy == Strategy.ATTACK
        ]
        safeties = [
            (candidate, child)
            for candidate, child in root.children.items()
            if candidate.action.strategy == Strategy.DEFENSE
        ]

        attack_visits = sum(
            child.visit_count for _, child in attacks
        )
        safety_visits = sum(
            child.visit_count for _, child in safeties
        )
        total_attack_visits += attack_visits
        total_safety_visits += safety_visits

        candidate, selected_child = max(
            root.children.items(),
            key=lambda item: item[1].visit_count,
        )
        chosen[candidate.action.strategy] += 1

        # Validate actual chosen MacroCandidate is executable.
        result = transition.step(state, candidate)

        print(
            f"board={board} "
            f"candidates Attack/Safety="
            f"{len(attacks)}/{len(safeties)} "
            f"visits Attack/Safety="
            f"{attack_visits}/{safety_visits} "
            f"chosen={candidate.goal.goal_type.value} "
            f"chosen_visits={selected_child.visit_count} "
            f"pred_q={selected_child.predicted_q:.3f} "
            f"executed_reward={result.reward:.3f} "
            f"next_player={result.next_state.current_player}"
        )
        if args.simulations < len(root.children):
            print(
                "  note: fewer simulations than root candidates; "
                "some candidates have not been executed."
            )

    print(
        "total root visits Attack/Safety: "
        f"{total_attack_visits}/{total_safety_visits}"
    )
    print(
        "chosen Attack/Safety: "
        f"{chosen[Strategy.ATTACK]}/{chosen[Strategy.DEFENSE]}"
    )
    print(
        "This is a mixed search integration smoke, not a "
        "fair win-rate comparison or trained Candidate Policy."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--candidate-value-checkpoint", required=True)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=94000)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--root-min-visits", type=int, default=1)
    parser.add_argument(
        "--candidate-q-prior-visits",
        type=float, default=2.0,
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--outcome-grid-size", type=int, default=5)
    parser.add_argument("--outcome-top-k", type=int, default=8)
    parser.add_argument("--outcome-dedup-distance", type=float, default=0.10)
    parser.add_argument("--safety-top-k", type=int, default=4)
    parser.add_argument(
        "--safety-powers", type=float, nargs="+",
        default=[0.65, 1.10],
    )
    parser.add_argument(
        "--safety-angles", type=float, nargs="+",
        default=[-12.0, 0.0, 12.0],
    )
    parser.add_argument(
        "--safety-tip-y-mm", type=float, nargs="+",
        default=[0.0, 5.0],
    )
    args = parser.parse_args()

    if args.states < 1 or args.simulations < 1:
        parser.error("--states and --simulations must be >= 1")
    if args.root_min_visits < 0:
        parser.error("--root-min-visits must be >= 0")
    if args.candidate_q_prior_visits < 0:
        parser.error("--candidate-q-prior-visits must be >= 0")

    return args


if __name__ == "__main__":
    main(parse_args())
