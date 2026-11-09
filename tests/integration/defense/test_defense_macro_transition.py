"""Tests for Attack/Defense transition dispatch and Defense reward contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.candidate import (
    MacroCandidate,
    PhysicalShotSpec,
)
from src.macro.defense_local_evaluation import (
    DefenseLocalEvaluation,
)
from src.macro.defense_transition import (
    DefenseMacroTransitionModel,
    MixedStrategyMacroTransitionModel,
)
from src.macro.transition import MacroTransitionResult


def state(player=0, cue=(0.5, 0.3)):
    return GameState(
        ball_positions=np.array(
            [
                cue,
                [0.5, 0.8],
                [0.9, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def safety_candidate():
    return MacroCandidate(
        action=MacroAction(
            strategy=Strategy.DEFENSE,
            target_ball=0,
            target_pocket=None,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.SAFETY,
            target_ball=0,
            cue_target_position=(0.55, 0.30),
            cue_target_radius=0.10,
        ),
        physical_shot=PhysicalShotSpec(
            angle=90.0,
            power=0.65,
            tip_y_mm=5.0,
        ),
    )


def direct_candidate():
    return MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=2,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=2,
        ),
    )


class FakeDefenseExecutor:
    def __init__(
        self,
        first_contact=0,
        scratched=False,
        pocketed=0,
        game_reward=0.0,
        turn_changed=True,
        terminated=False,
    ):
        self.calls = 0
        self.first_contact = first_contact
        self.scratched = scratched
        self.pocketed = pocketed
        self.game_reward = game_reward
        self.turn_changed = turn_changed
        self.terminated = terminated
        self.last_shot = None

    def execute(self, previous, target_ball, shot):
        self.calls += 1
        self.last_shot = shot

        next_state = state(
            player=1 if self.turn_changed else 0,
            cue=(0.55, 0.30),
        )

        return SimpleNamespace(
            next_state=next_state,
            game_reward=self.game_reward,
            terminated=self.terminated,
            turn_changed=self.turn_changed,
            scratched=self.scratched,
            first_contact_ball=self.first_contact,
            first_contact_verified=(
                self.first_contact == target_ball
            ),
            newly_pocketed_object_count=self.pocketed,
        )


class FakeAttackTransition:
    def __init__(self):
        self.calls = 0

    def step(self, previous, choice):
        self.calls += 1
        return MacroTransitionResult(
            next_state=state(player=1),
            reward=1.25,
            terminated=False,
        )


def test_defense_transition_executes_explicit_shot_and_changes_turn():
    executor = FakeDefenseExecutor()
    model = DefenseMacroTransitionModel(
        executor=executor
    )
    result = model.step(state(), safety_candidate())

    assert executor.calls == 1
    assert np.isclose(executor.last_shot.angle, 90.0)
    assert np.isclose(executor.last_shot.power, 0.65)
    assert np.isclose(executor.last_shot.tip_y_mm, 5.0)

    assert result.next_state.current_player == 1
    assert result.terminated is False
    assert np.isclose(result.reward, 0.0)
    assert isinstance(
        result.evaluation,
        DefenseLocalEvaluation,
    )
    assert result.evaluation.safety_eligible
    assert result.evaluation.goal_achieved
    assert result.evaluation.cue_region_reached
    # GameState stores float32 positions. Converting to Python
    # float for goal-distance calculations can leave sub-micron
    # numerical noise even when the positions conceptually match.
    assert np.isclose(
        result.evaluation.cue_distance,
        0.0,
        atol=1e-6,
    )


def test_defense_transition_penalizes_missed_first_contact():
    executor = FakeDefenseExecutor(first_contact=-1)
    model = DefenseMacroTransitionModel(
        executor=executor,
        wrong_contact_penalty=0.7,
    )
    result = model.step(state(), safety_candidate())

    assert np.isclose(result.reward, -0.7)
    assert not result.evaluation.safety_eligible
    assert not result.evaluation.goal_achieved


def test_defense_transition_does_not_double_penalize_scratch():
    executor = FakeDefenseExecutor(
        first_contact=-1,
        scratched=True,
        game_reward=-1.0,
        terminated=True,
    )
    result = DefenseMacroTransitionModel(
        executor=executor
    ).step(
        state(),
        safety_candidate(),
    )
    assert np.isclose(result.reward, -1.0)
    assert result.terminated
    assert not result.evaluation.safety_eligible


def test_pot_does_not_count_as_safety_success():
    executor = FakeDefenseExecutor(
        pocketed=1,
        game_reward=1.0,
        turn_changed=False,
    )
    result = DefenseMacroTransitionModel(
        executor=executor
    ).step(
        state(),
        safety_candidate(),
    )

    assert np.isclose(result.reward, 1.0)
    assert not result.evaluation.goal_achieved
    assert not result.evaluation.safety_eligible


def test_defense_transition_rejects_unexecutable_anchor():
    candidate = MacroCandidate(
        action=MacroAction(
            strategy=Strategy.DEFENSE,
            target_ball=None,
            target_pocket=None,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.SAFETY,
            cue_target_position=(0.55, 0.30),
            cue_target_radius=0.10,
        ),
    )
    executor = FakeDefenseExecutor()
    with pytest.raises(ValueError, match="physical_shot"):
        DefenseMacroTransitionModel(
            executor=executor
        ).step(state(), candidate)

    assert executor.calls == 0


def test_mixed_strategy_dispatches_without_touching_attack_model():
    attack = FakeAttackTransition()
    defense_executor = FakeDefenseExecutor()
    defense = DefenseMacroTransitionModel(
        executor=defense_executor
    )
    model = MixedStrategyMacroTransitionModel(
        attack_transition_model=attack,
        defense_transition_model=defense,
    )

    attack_result = model.step(state(), direct_candidate())
    assert attack.calls == 1
    assert defense_executor.calls == 0
    assert np.isclose(attack_result.reward, 1.25)

    defense_result = model.step(state(), safety_candidate())
    assert attack.calls == 1
    assert defense_executor.calls == 1
    assert isinstance(
        defense_result.evaluation,
        DefenseLocalEvaluation,
    )
    assert model.last_evaluation is defense_result.evaluation


def test_mixed_strategy_dispatches_legacy_macro_action_to_attack():
    attack = FakeAttackTransition()
    defense = DefenseMacroTransitionModel(
        executor=FakeDefenseExecutor()
    )
    mixed = MixedStrategyMacroTransitionModel(
        attack_transition_model=attack,
        defense_transition_model=defense,
    )

    mixed.step(
        state(),
        direct_candidate().action,
    )

    assert attack.calls == 1


def test_defense_physical_transition_integrated_fastfiz():
    # Physical smoke: don't assume shot hits the intended ball
    # or scores; check the transition contract only.
    from src.micro.defense_shot import DefenseShotExecutor

    original = state()
    before = original.ball_positions.copy()
    model = DefenseMacroTransitionModel(
        executor=DefenseShotExecutor(),
    )
    result = model.step(original, safety_candidate())

    assert result.next_state.ball_positions.shape == (3, 2)
    assert isinstance(result.reward, float)
    assert result.evaluation.cue_distance is not None
    assert result.evaluation.first_contact_ball in (-1, 0, 1)
    np.testing.assert_array_equal(original.ball_positions, before)
