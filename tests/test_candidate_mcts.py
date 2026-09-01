import numpy as np
import torch

from src.common.types import (
    GameState,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.candidate_generator import (
    ExpandedMacroCandidateGenerator,
)

from src.macro.candidate_policy import (
    MacroCandidatePolicyNetwork,
)

from src.macro.mcts import MCTS

from src.macro.transition import (
    MacroTransitionResult,
)

from src.macro.value import (
    MacroValueNetwork,
)


class DummyCandidateTransition:

    def __init__(self):
        self.last_choice = None

    def step(
        self,
        state,
        choice,
    ):
        self.last_choice = choice

        return MacroTransitionResult(
            next_state=GameState(
                ball_positions=(
                    state.ball_positions.copy()
                ),
                score=state.score.copy(),
                current_player=(
                    state.current_player
                ),
                ball_pocket_indices=(
                    state.ball_pocket_indices.copy()
                ),
            ),
            reward=0.0,
            terminated=False,
        )


def make_state():

    return GameState(
        ball_positions=np.array(
            [
                [0.80, 1.118],
                [0.40, 1.118],
                [0.80, 0.80],
            ],
            dtype=np.float32,
        ),

        score=np.zeros(
            2,
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_candidate_mcts_expands_candidates():

    torch.manual_seed(0)

    generator = (
        ExpandedMacroCandidateGenerator()
    )

    policy = (
        MacroCandidatePolicyNetwork()
    )

    value = MacroValueNetwork()

    transition = (
        DummyCandidateTransition()
    )

    mcts = MCTS(
        policy_network=policy,
        value_network=value,
        action_generator=generator,
        transition_model=transition,
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    assert len(root.children) > 0

    assert all(
        isinstance(
            choice,
            MacroCandidate,
        )
        for choice in root.children
    )


def test_candidate_mcts_contains_goal_types():

    torch.manual_seed(0)

    mcts = MCTS(
        policy_network=(
            MacroCandidatePolicyNetwork()
        ),
        value_network=(
            MacroValueNetwork()
        ),
        action_generator=(
            ExpandedMacroCandidateGenerator()
        ),
        transition_model=(
            DummyCandidateTransition()
        ),
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    goal_types = {
        candidate.goal.goal_type
        for candidate in root.children
    }

    assert (
        TacticalGoalType.DIRECT_ATTACK
        in goal_types
    )

    assert (
        TacticalGoalType.POSITION_ATTACK
        in goal_types
    )


def test_candidate_priors_sum_to_one():

    torch.manual_seed(0)

    mcts = MCTS(
        policy_network=(
            MacroCandidatePolicyNetwork()
        ),
        value_network=(
            MacroValueNetwork()
        ),
        action_generator=(
            ExpandedMacroCandidateGenerator()
        ),
        transition_model=(
            DummyCandidateTransition()
        ),
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    prior_sum = sum(
        child.prior
        for child in root.children.values()
    )

    assert np.isclose(
        prior_sum,
        1.0,
        atol=1e-5,
    )


def test_candidate_is_sent_to_transition():

    torch.manual_seed(0)

    transition = (
        DummyCandidateTransition()
    )

    mcts = MCTS(
        policy_network=(
            MacroCandidatePolicyNetwork()
        ),
        value_network=(
            MacroValueNetwork()
        ),
        action_generator=(
            ExpandedMacroCandidateGenerator()
        ),
        transition_model=transition,
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    mcts.search(
        make_state()
    )

    assert isinstance(
        transition.last_choice,
        MacroCandidate,
    )
    