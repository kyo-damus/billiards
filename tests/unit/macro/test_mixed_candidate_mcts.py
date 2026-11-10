"""Opt-in 31D mixed-strategy MCTS contract tests."""

import numpy as np
import pytest
import torch
from torch import nn

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate, PhysicalShotSpec
from src.macro.mcts import MCTS, MCTSNode
from src.macro.transition import MacroTransitionResult


def make_state(player=0):
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=player,
    )


def make_candidates():
    attack = MacroCandidate(
        action=MacroAction(Strategy.ATTACK, 0, 0),
        goal=TacticalGoal(
            TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
    )
    safety = MacroCandidate(
        action=MacroAction(Strategy.DEFENSE, 0, None),
        goal=TacticalGoal(
            TacticalGoalType.SAFETY,
            target_ball=0,
            cue_target_position=(0.55, 0.33),
            cue_target_radius=0.12,
        ),
        physical_shot=PhysicalShotSpec(
            angle=90.0,
            power=0.65,
            tip_y_mm=5.0,
        ),
    )
    return attack, safety


class FixedGenerator:
    def __init__(self, candidates):
        self.candidates = list(candidates)

    def generate(self, state):
        return list(self.candidates)


class ZeroValue(nn.Module):
    def forward(self, states):
        return torch.zeros(
            states.shape[0],
            dtype=states.dtype,
            device=states.device,
        )


class StrategyQ(nn.Module):
    candidate_dim = 31

    def forward(self, state, candidates):
        # Safety strategy is feature 1, Attack is feature 0.
        return candidates[:, 1] * 0.8 + candidates[:, 0] * 0.4


class DummyTransition:
    def __init__(self):
        self.visited = []

    def step(self, state, candidate):
        self.visited.append(candidate)
        return MacroTransitionResult(
            next_state=make_state(player=1),
            reward=(
                0.6
                if candidate.action.strategy == Strategy.ATTACK
                else 0.0
            ),
            terminated=True,
        )


def make_mcts(**kwargs):
    attack, safety = make_candidates()
    transition = DummyTransition()
    mcts = MCTS(
        policy_network=None,
        value_network=ZeroValue(),
        candidate_value_network=StrategyQ(),
        action_generator=FixedGenerator([]),
        root_action_generator=FixedGenerator([attack, safety]),
        transition_model=transition,
        simulations=4,
        candidate_mode=True,
        candidate_encoding_version=2,
        uniform_candidate_priors=True,
        turn_aware=True,
        root_min_visits_before_puct=1,
        candidate_q_prior_visits=2.0,
        device="cpu",
        **kwargs,
    )
    return mcts, transition, attack, safety


def test_mixed_mcts_attaches_31d_q_and_uniform_priors():
    mcts, transition, attack, safety = make_mcts()
    root = mcts.search(make_state())

    assert set(root.children) == {attack, safety}
    assert np.isclose(root.children[attack].prior, 0.5)
    assert np.isclose(root.children[safety].prior, 0.5)
    assert np.isclose(root.children[attack].predicted_q, 0.4)
    assert np.isclose(root.children[safety].predicted_q, 0.8)
    assert root.children[attack].visit_count >= 1
    assert root.children[safety].visit_count >= 1
    assert attack in transition.visited
    assert safety in transition.visited
    assert sum(x.visit_count for x in root.children.values()) == 4


def test_mixed_mcts_rejects_legacy_candidate_policy():
    with pytest.raises(ValueError, match="uniform_candidate_priors"):
        MCTS(
            policy_network=nn.Linear(1, 1),
            value_network=ZeroValue(),
            action_generator=FixedGenerator([]),
            transition_model=DummyTransition(),
            device="cpu",
            candidate_mode=True,
            turn_aware=True,
            candidate_encoding_version=2,
            uniform_candidate_priors=False,
        )


def test_mixed_mcts_requires_turn_aware():
    with pytest.raises(ValueError, match="turn_aware"):
        MCTS(
            policy_network=None,
            value_network=ZeroValue(),
            action_generator=FixedGenerator([]),
            transition_model=DummyTransition(),
            device="cpu",
            candidate_mode=True,
            turn_aware=False,
            candidate_encoding_version=2,
            uniform_candidate_priors=True,
        )


def test_mixed_mcts_rejects_25d_q():
    class WrongQ(StrategyQ):
        candidate_dim = 25

    with pytest.raises(ValueError, match="dimension mismatch"):
        MCTS(
            policy_network=None,
            value_network=ZeroValue(),
            action_generator=FixedGenerator([]),
            transition_model=DummyTransition(),
            device="cpu",
            candidate_mode=True,
            turn_aware=True,
            candidate_encoding_version=2,
            uniform_candidate_priors=True,
            candidate_value_network=WrongQ(),
        )



class StrategyPolicy(nn.Module):
    candidate_dim = 31

    def forward(self, state, candidates):
        return candidates[:, 1] * 3.0


def test_mixed_mcts_uses_learned_31d_candidate_priors():
    attack, safety = make_candidates()
    mcts = MCTS(
        policy_network=StrategyPolicy(),
        value_network=ZeroValue(),
        action_generator=FixedGenerator([]),
        root_action_generator=FixedGenerator([attack, safety]),
        transition_model=DummyTransition(),
        simulations=4,
        candidate_mode=True,
        candidate_encoding_version=2,
        uniform_candidate_priors=False,
        turn_aware=True,
        device="cpu",
    )
    root = mcts.search(make_state())
    assert root.children[safety].prior > root.children[attack].prior
    assert np.isclose(
        sum(child.prior for child in root.children.values()),
        1.0,
    )
    assert sum(child.visit_count for child in root.children.values()) == 4


def test_mixed_mcts_refuses_old_25d_candidate_policy():
    class OldPolicy(nn.Module):
        candidate_dim = 25

        def forward(self, state, candidates):
            return torch.zeros(candidates.shape[0])

    with pytest.raises(ValueError, match="31-D Candidate"):
        MCTS(
            policy_network=OldPolicy(),
            value_network=ZeroValue(),
            action_generator=FixedGenerator([]),
            transition_model=DummyTransition(),
            device="cpu",
            candidate_mode=True,
            turn_aware=True,
            candidate_encoding_version=2,
            uniform_candidate_priors=False,
        )


def test_root_visit_tie_breaker_default_keeps_first_candidate():
    mcts, _, attack, safety = make_mcts()
    root = MCTSNode(state=make_state(), children={
        attack: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=0.0, prior=0.2,
        ),
        safety: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=0.8, prior=0.8,
        ),
    })
    assert mcts.root_tie_breaker == "first"
    assert mcts._select_root_action(root) == attack


def test_empirical_q_tie_breaker_uses_observed_physical_rewards():
    mcts, _, attack, safety = make_mcts(root_tie_breaker="empirical_q")
    root = MCTSNode(state=make_state(), children={
        attack: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=0.0, predicted_q=4.0, prior=0.8,
        ),
        safety: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=0.6, predicted_q=-2.0, prior=0.2,
        ),
    })
    # Learned Candidate-Q predictions are not empirical child returns.
    assert mcts._select_root_action(root) == safety


def test_empirical_q_tie_breaker_negates_opponent_future_value():
    mcts, _, attack, safety = make_mcts(root_tie_breaker="empirical_q")
    root = MCTSNode(state=make_state(player=0), children={
        attack: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=1.0, value_sum=2.0, prior=0.6,
        ),
        safety: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=0.25, value_sum=0.0, prior=0.4,
        ),
    })
    # attack = +1 - gamma*2 < safety = +0.25
    assert mcts._select_root_action(root) == safety


def test_empirical_q_tie_breaker_preserves_visit_primary_and_prior_secondary():
    mcts, _, attack, safety = make_mcts(root_tie_breaker="empirical_q")
    root = MCTSNode(state=make_state(), children={
        attack: MCTSNode(
            state=make_state(player=1), visit_count=2,
            reward=-2.0, prior=0.1,
        ),
        safety: MCTSNode(
            state=make_state(player=1), visit_count=1,
            reward=1.0, prior=0.9,
        ),
    })
    assert mcts._select_root_action(root) == attack
    root.children[attack].visit_count = 1
    root.children[attack].reward = 1.0
    assert mcts._select_root_action(root) == safety
    root.children[safety].prior = 0.1
    assert mcts._select_root_action(root) == attack


def test_root_tie_breaker_rejects_invalid_name():
    with pytest.raises(ValueError, match="root_tie_breaker"):
        make_mcts(root_tie_breaker="unknown")


def test_select_action_resolves_real_search_visit_tie_without_fastfiz():
    """End-to-end search selects later Safety only in opt-in mode."""
    attack, safety = make_candidates()

    class FavorSafetyTransition:
        def step(self, state, candidate):
            return MacroTransitionResult(
                next_state=make_state(player=1),
                reward=0.8 if candidate == safety else 0.1,
                terminated=True,
            )

    def search_choice(mode):
        mcts = MCTS(
            policy_network=None,
            value_network=ZeroValue(),
            action_generator=FixedGenerator([]),
            root_action_generator=FixedGenerator([attack, safety]),
            transition_model=FavorSafetyTransition(),
            simulations=2,
            candidate_mode=True,
            candidate_encoding_version=2,
            uniform_candidate_priors=True,
            turn_aware=True,
            root_min_visits_before_puct=1,
            root_tie_breaker=mode,
            device="cpu",
        )
        choice = mcts.select_action(make_state())
        visits = [child.visit_count for child in mcts.last_root.children.values()]
        assert visits == [1, 1]
        return choice

    assert search_choice("first") == attack
    assert search_choice("empirical_q") == safety
