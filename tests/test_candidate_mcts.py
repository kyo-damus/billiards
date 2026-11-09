import numpy as np
import torch
import torch.nn as nn

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

from src.macro.mcts import (
    MCTS,
    MCTSNode,
)

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


def test_candidate_mcts_root_contract():
    """A single search verifies root candidates, priors and diagnostics.

    These assertions previously ran four additional identical one-step
    MCTS searches, with the same seed and simulated transition.
    """
    torch.manual_seed(0)

    transition = DummyCandidateTransition()
    mcts = MCTS(
        policy_network=MacroCandidatePolicyNetwork(),
        value_network=MacroValueNetwork(),
        action_generator=ExpandedMacroCandidateGenerator(),
        transition_model=transition,
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(make_state())

    assert len(root.children) > 0
    assert all(
        isinstance(candidate, MacroCandidate)
        for candidate in root.children
    )
    goal_types = {
        candidate.goal.goal_type
        for candidate in root.children
    }
    assert TacticalGoalType.DIRECT_ATTACK in goal_types
    assert TacticalGoalType.POSITION_ATTACK in goal_types

    prior_sum = sum(child.prior for child in root.children.values())
    assert np.isclose(prior_sum, 1.0, atol=1e-5)

    assert isinstance(transition.last_choice, MacroCandidate)
    assert mcts.last_root is root


def test_force_evaluate_unvisited_prefers_unvisited_child():

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
        force_evaluate_unvisited=True,
    )

    candidates = (
        ExpandedMacroCandidateGenerator()
        .generate(
            make_state()
        )
    )

    assert len(candidates) >= 2

    visited = MCTSNode(
        prior=0.5,
        reward=10.0,
        state=make_state(),
        visit_count=1,
    )

    unvisited = MCTSNode(
        prior=0.5,
    )

    parent = MCTSNode(
        state=make_state(),
        visit_count=1,
        children={
            candidates[0]: visited,
            candidates[1]: unvisited,
        },
    )

    selected_action, selected_child = (
        mcts._select_child(
            parent
        )
    )

    assert selected_action == candidates[1]
    assert selected_child is unvisited


def test_min_visits_prefers_less_visited_child():

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

    candidates = (
        ExpandedMacroCandidateGenerator()
        .generate(
            make_state()
        )
    )

    assert len(candidates) >= 2

    more_visited = MCTSNode(
        prior=0.5,
        reward=10.0,
        state=make_state(),
        visit_count=2,
    )

    less_visited = MCTSNode(
        prior=0.5,
        reward=0.0,
        state=make_state(),
        visit_count=1,
    )

    parent = MCTSNode(
        state=make_state(),
        visit_count=3,
        children={
            candidates[0]: more_visited,
            candidates[1]: less_visited,
        },
    )

    selected_action, selected_child = (
        mcts._select_child(
            parent,
            min_visits=2,
        )
    )

    assert selected_action == candidates[1]
    assert selected_child is less_visited


class IncreasingCandidateValue(nn.Module):

    def forward(
        self,
        state,
        candidates,
    ):
        return torch.arange(
            candidates.shape[0],
            dtype=torch.float32,
            device=candidates.device,
        )


def test_candidate_value_predictions_are_attached_to_children():

    torch.manual_seed(0)

    mcts = MCTS(
        policy_network=(
            MacroCandidatePolicyNetwork()
        ),
        value_network=(
            MacroValueNetwork()
        ),
        candidate_value_network=(
            IncreasingCandidateValue()
        ),
        action_generator=(
            ExpandedMacroCandidateGenerator()
        ),
        transition_model=(
            DummyCandidateTransition()
        ),
        simulations=0,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    predicted = [
        child.predicted_q
        for child in root.children.values()
    ]

    assert predicted == [
        float(index)
        for index in range(
            len(predicted)
        )
    ]


def test_unvisited_candidate_uses_predicted_q():

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

    low = MCTSNode(
        prior=0.5,
        predicted_q=0.0,
    )

    high = MCTSNode(
        prior=0.5,
        predicted_q=2.0,
    )

    parent = MCTSNode(
        state=make_state(),
        visit_count=0,
    )

    assert (
        mcts._puct_score(
            parent,
            high,
        )
        >
        mcts._puct_score(
            parent,
            low,
        )
    )


def test_visited_candidate_uses_empirical_q():

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
        gamma=0.5,
    )

    child = MCTSNode(
        prior=0.0,
        reward=1.0,
        predicted_q=100.0,
        state=make_state(),
        visit_count=2,
        value_sum=4.0,
    )

    parent = MCTSNode(
        state=make_state(),
        visit_count=2,
    )

    # empirical Q = 1.0 + 0.5 * 2.0 = 2.0
    assert np.isclose(
        mcts._puct_score(
            parent,
            child,
        ),
        2.0,
    )



def test_candidate_q_prior_blends_with_empirical_q():

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
        gamma=0.5,
        candidate_q_prior_visits=2.0,
    )

    child = MCTSNode(
        prior=0.0,
        reward=1.0,
        predicted_q=4.0,
        state=make_state(),
        visit_count=2,
        value_sum=4.0,
    )

    # empirical Q = 1 + 0.5 * 2 = 2
    # prior 2 visits at Q=4, empirical 2 visits at Q=2
    # blended Q = 3
    assert np.isclose(
        mcts._candidate_q_value(
            child
        ),
        3.0,
    )


def test_candidate_q_prior_fades_with_more_visits():

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
        gamma=0.0,
        candidate_q_prior_visits=2.0,
    )

    child = MCTSNode(
        prior=0.0,
        reward=1.0,
        predicted_q=4.0,
        state=make_state(),
        visit_count=20,
        value_sum=0.0,
    )

    blended = (
        mcts._candidate_q_value(
            child
        )
    )

    assert np.isclose(
        blended,
        (
            20.0 * 1.0
            + 2.0 * 4.0
        )
        / 22.0,
    )
    assert blended < 1.3


def test_turn_aware_candidate_q_negates_opponent_continuation():
    mcts = MCTS(
        policy_network=MacroCandidatePolicyNetwork(),
        value_network=MacroValueNetwork(),
        action_generator=ExpandedMacroCandidateGenerator(),
        transition_model=DummyCandidateTransition(),
        simulations=0,
        device="cpu",
        candidate_mode=True,
        gamma=0.9,
        turn_aware=True,
    )

    parent = MCTSNode(
        state=make_state(),
    )
    opponent_state = make_state()
    opponent_state.current_player = 1

    child = MCTSNode(
        state=opponent_state,
        reward=0.0,
        visit_count=2,
        value_sum=4.0,
    )

    assert np.isclose(
        mcts._candidate_q_value(
            child,
            parent=parent,
        ),
        -1.8,
    )


def test_turn_aware_backup_negates_opponent_reward():
    mcts = MCTS(
        policy_network=MacroCandidatePolicyNetwork(),
        value_network=MacroValueNetwork(),
        action_generator=ExpandedMacroCandidateGenerator(),
        transition_model=DummyCandidateTransition(),
        simulations=0,
        device="cpu",
        candidate_mode=True,
        gamma=1.0,
        turn_aware=True,
    )

    root = MCTSNode(
        state=make_state(),
    )

    opponent_state = make_state()
    opponent_state.current_player = 1

    opponent = MCTSNode(
        state=opponent_state,
        reward=0.0,
    )
    opponent_next = MCTSNode(
        state=opponent_state,
        reward=1.0,
    )

    mcts._backup(
        [root, opponent, opponent_next],
        leaf_value=0.0,
    )

    assert np.isclose(
        root.value,
        -1.0,
    )
    assert np.isclose(
        opponent.value,
        1.0,
    )


def test_legacy_backup_keeps_previous_behavior():
    mcts = MCTS(
        policy_network=MacroCandidatePolicyNetwork(),
        value_network=MacroValueNetwork(),
        action_generator=ExpandedMacroCandidateGenerator(),
        transition_model=DummyCandidateTransition(),
        simulations=0,
        device="cpu",
        candidate_mode=True,
        gamma=1.0,
        turn_aware=False,
    )

    root = MCTSNode(
        state=make_state(),
    )

    opponent_state = make_state()
    opponent_state.current_player = 1

    opponent = MCTSNode(
        state=opponent_state,
        reward=0.0,
    )
    opponent_next = MCTSNode(
        state=opponent_state,
        reward=1.0,
    )

    mcts._backup(
        [root, opponent, opponent_next],
        leaf_value=0.0,
    )

    assert np.isclose(
        root.value,
        1.0,
    )
