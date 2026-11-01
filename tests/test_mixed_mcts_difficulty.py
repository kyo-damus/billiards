"""Lightweight board-difficulty screen and mixed-root helper tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval_mixed_mcts_difficult_boards import (
    assess_attack_feasibility,
)
from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.candidate import MacroCandidate
from src.macro.mixed_search import (
    FixedCandidateGenerator,
    MixedRootGenerator,
)


def state():
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
    )


def attack(pocket):
    return MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=pocket,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=pocket,
        ),
    )


class FakeTransition:
    def __init__(self):
        self.calls = 0

    def step(self, current, candidate):
        self.calls += 1
        ball = candidate.action.target_pocket
        return SimpleNamespace(
            reward={0: 1.25, 1: 0.0, 2: -1.0}[ball],
            evaluation=SimpleNamespace(
                target_potted=(ball == 0),
                scratch=(ball == 2),
            ),
        )


def test_attack_difficulty_counts_executable_pots_not_geometry():
    transition = FakeTransition()
    choices = [attack(0), attack(1), attack(2)]
    result = assess_attack_feasibility(
        state(), choices, transition
    )

    assert result.candidate_count == 3
    assert result.successful_pot_count == 1
    assert np.isclose(result.best_immediate_reward, 1.25)
    assert transition.calls == 3


def test_attack_difficulty_rejects_defense_choices():
    transition = FakeTransition()
    candidate = attack(0)
    candidate = MacroCandidate(
        action=MacroAction(
            strategy=Strategy.DEFENSE,
            target_ball=0,
            target_pocket=None,
        ),
        goal=candidate.goal,
    )
    with pytest.raises(ValueError, match="ATTACK"):
        assess_attack_feasibility(
            state(), [candidate], transition
        )


def test_fixed_generator_snapshots_candidate_list():
    candidates = [attack(0)]
    generator = FixedCandidateGenerator(candidates)
    candidates.append(attack(1))
    assert len(generator.generate(state())) == 1


def test_mixed_root_generator_reuses_evaluated_shots():
    calls = {"attack": 0, "safety": 0, "adapter": 0}

    class AttackGen:
        def generate(self, current):
            calls["attack"] += 1
            return [attack(0)]

    class SafetyGen:
        def generate(self, current):
            calls["safety"] += 1
            return ["proposal"]

    class SafetyAdapter:
        def from_proposals(self, proposals):
            calls["adapter"] += 1
            assert proposals == ["proposal"]
            return [attack(1)]

    generator = MixedRootGenerator(
        AttackGen(),
        SafetyGen(),
        SafetyAdapter(),
    )
    choices = generator.generate(state())

    assert len(choices) == 2
    assert generator.last_attack_count == 1
    assert generator.last_safety_count == 1
    assert calls == {"attack": 1, "safety": 1, "adapter": 1}
