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
from src.macro.mcts import MCTS
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
