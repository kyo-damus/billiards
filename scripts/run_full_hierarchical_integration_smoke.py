import argparse

import numpy as np
import torch
import torch.nn as nn

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.controller.hierarchical_agent import (
    HierarchicalAgent,
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
from src.macro.candidate import (
    MacroCandidate,
)
from src.macro.candidate_generator import (
    ExpandedMacroCandidateGenerator,
    MacroCandidateGenerator,
)
from src.macro.mcts import MCTS
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)
from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)
from src.micro.adaptive_position_policy import (
    AdaptivePositionMicroPolicy,
)


class UniformCandidatePolicy(nn.Module):
    """
    Integration-only baseline.

    Macro Candidate Policy training is outside this smoke test;
    all root/tree candidates receive equal prior probability.
    """

    def forward(
        self,
        state,
        candidates,
    ):
        return torch.zeros(
            candidates.shape[0],
            dtype=torch.float32,
            device=candidates.device,
        )


class ZeroValue(nn.Module):
    """
    Integration-only value baseline.
    """

    def forward(
        self,
        state,
    ):
        return torch.zeros(
            1,
            dtype=torch.float32,
            device=state.device,
        )


class StaticCandidateGenerator:
    """
    Reuse the already generated expensive root candidates.
    """

    def __init__(
        self,
        candidates,
    ):
        self.candidates = list(
            candidates
        )

    def generate(
        self,
        state,
    ):
        return list(
            self.candidates
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


def main(
    args,
):
    if args.device == "auto":
        micro_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        micro_device = args.device

    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )

    print(
        f"micro device: {micro_device}"
    )
    print(
        "Macro policy/value: "
        "uniform/zero integration baseline"
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
            device=micro_device,
        )
    )

    transition = (
        PhysicsMacroTransitionModel(
            micro_agent=micro_policy,
            deterministic=True,
            game_rules=(
                SimplifiedGameRules()
            ),
        )
    )

    direct_generator = (
        MacroCandidateGenerator()
    )

    outcome_generator = (
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

    expensive_root_generator = (
        ExpandedMacroCandidateGenerator(
            direct_generator=(
                direct_generator
            ),
            position_generator=(
                outcome_generator
            ),
        )
    )

    # After a root POSITION shot, the current 2-object-ball
    # setup naturally leaves the remaining ball as the next
    # direct target. Deeper MCTS expansion therefore uses the
    # cheap DIRECT candidate generator only.
    tree_generator = (
        MacroCandidateGenerator()
    )

    state_env = MicroBilliardEnv()

    state = None
    root_candidates = None

    for attempt in range(
        args.max_attempts
    ):
        candidate_state = random_state(
            state_env,
            args.seed + attempt,
        )

        candidates = (
            expensive_root_generator
            .generate(
                candidate_state
            )
        )

        has_direct = any(
            candidate.goal.goal_type
            == TacticalGoalType.DIRECT_ATTACK
            for candidate in candidates
        )

        has_position = any(
            candidate.goal.goal_type
            == TacticalGoalType.POSITION_ATTACK
            for candidate in candidates
        )

        if (
            has_direct
            and has_position
        ):
            state = candidate_state
            root_candidates = candidates
            break

    if state is None:
        raise RuntimeError(
            "Could not find a state with both "
            "DIRECT and outcome-guided POSITION "
            "root candidates."
        )

    direct_count = sum(
        candidate.goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
        for candidate in root_candidates
    )

    position_count = sum(
        candidate.goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
        for candidate in root_candidates
    )

    print()
    print(
        "=== Root candidates ==="
    )
    print(
        f"total: {len(root_candidates)}"
    )
    print(
        f"DIRECT: {direct_count}"
    )
    print(
        f"POSITION: {position_count}"
    )

    root_generator = (
        StaticCandidateGenerator(
            root_candidates
        )
    )

    mcts = MCTS(
        policy_network=(
            UniformCandidatePolicy()
        ),
        value_network=(
            ZeroValue()
        ),
        action_generator=(
            tree_generator
        ),
        root_action_generator=(
            root_generator
        ),
        transition_model=(
            transition
        ),
        simulations=(
            args.simulations
        ),
        c_puct=(
            args.c_puct
        ),
        device="cpu",
        candidate_mode=True,
    )

    agent = HierarchicalAgent(
        mcts=mcts,
        transition_model=transition,
    )

    result = agent.step(
        state
    )

    decision = (
        result.macro_action
    )

    if not isinstance(
        decision,
        MacroCandidate,
    ):
        raise RuntimeError(
            "Candidate-mode MCTS did not return "
            "a MacroCandidate."
        )

    print()
    print(
        "=== Full hierarchical integration smoke ==="
    )
    print(
        "selected goal type:",
        decision.goal.goal_type.value,
    )
    print(
        "selected current target:",
        (
            decision.action.target_ball,
            decision.action.target_pocket,
        ),
    )

    if (
        decision.goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
    ):
        print(
            "selected next target:",
            (
                decision.goal.next_target_ball,
                decision.goal.next_target_pocket,
            ),
        )

        selection = (
            micro_policy.last_selection
        )

        print(
            "selected blend lambda:",
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

    else:
        print(
            "selected blend lambda: "
            "n/a (DIRECT uses base actor)"
        )

    print(
        "reward:",
        result.reward,
    )
    print(
        "terminated:",
        result.terminated,
    )
    print(
        "next positions:"
    )
    print(
        result.next_state.ball_positions
    )

    print()
    print(
        "Pipeline:"
    )
    print(
        "GameState -> Outcome-guided root candidates "
        "-> candidate-mode MCTS -> MacroCandidate "
        "-> Adaptive Micro -> FastFiz -> GameRules "
        "-> next GameState"
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
        "--simulations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
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
        default=30,
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
