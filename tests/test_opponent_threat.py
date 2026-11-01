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
from src.macro.candidate import MacroCandidate
from src.macro.local_evaluation import MacroLocalEvaluation
from src.macro.opponent_threat import OpponentDirectThreatEvaluator
from src.macro.transition import MacroTransitionResult


def make_state(player=1, cue_pocketed=False):
    return GameState(
        ball_positions=np.array(
            [
                [0.4, 1.0],
                [0.5, 1.4],
                [0.8, 1.8],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.array(
            [0 if cue_pocketed else -1, -1, -1],
            dtype=np.int32,
        ),
    )


def candidate(pocket):
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


class StaticGenerator:
    def __init__(self, candidates):
        self.candidates = candidates

    def generate(self, state):
        return list(self.candidates)


class DummyTransition:
    def __init__(self, outcomes):
        # pocket -> (pot, scratch, local reward)
        self.outcomes = outcomes
        self.calls = 0

    def step(self, state, action):
        self.calls += 1
        pot, scratch, reward = self.outcomes[
            action.action.target_pocket
        ]
        evaluation = MacroLocalEvaluation(
            target_potted=pot,
            wrong_pocket=False,
            scratch=scratch,
            position_goal_satisfied=False,
            cue_distance=None,
            position_quality=None,
            next_shot_feasible=False,
            score=reward,
        )
        return MacroTransitionResult(
            next_state=state,
            reward=reward,
            terminated=False,
            evaluation=evaluation,
        )


def test_opponent_direct_threat_counts_executable_success():
    choices = [candidate(0), candidate(1), candidate(2)]
    transition = DummyTransition(
        {
            0: (False, False, 0.0),
            1: (True, False, 1.25),
            2: (True, True, -1.0),
        }
    )
    evaluator = OpponentDirectThreatEvaluator(
        transition_model=transition,
        candidate_generator=StaticGenerator(choices),
    )

    threat = evaluator.evaluate(
        state=make_state(player=1),
        defending_player=0,
    )

    assert threat.geometric_candidate_count == 3
    assert threat.successful_pot_count == 1
    assert threat.has_successful_direct
    assert np.isclose(threat.best_success_reward, 1.25)
    assert threat.best_candidate == choices[1]
    assert transition.calls == 3


def test_opponent_direct_threat_zero_when_no_shots_work():
    transition = DummyTransition(
        {
            0: (False, False, 0.0),
            1: (False, True, -1.0),
        }
    )
    evaluator = OpponentDirectThreatEvaluator(
        transition_model=transition,
        candidate_generator=StaticGenerator(
            [candidate(0), candidate(1)]
        ),
    )

    threat = evaluator.evaluate(
        make_state(),
        defending_player=0,
    )

    assert threat.geometric_candidate_count == 2
    assert threat.successful_pot_count == 0
    assert not threat.has_successful_direct
    assert threat.best_candidate is None
    assert threat.best_success_reward == 0.0


def test_opponent_direct_threat_empty_candidates():
    transition = DummyTransition({})
    evaluator = OpponentDirectThreatEvaluator(
        transition_model=transition,
        candidate_generator=StaticGenerator([]),
    )

    threat = evaluator.evaluate(
        make_state(),
        defending_player=0,
    )

    assert threat.geometric_candidate_count == 0
    assert threat.successful_pot_count == 0
    assert transition.calls == 0


def test_opponent_direct_threat_checks_turn_and_scratch():
    evaluator = OpponentDirectThreatEvaluator(
        transition_model=DummyTransition({}),
        candidate_generator=StaticGenerator([]),
    )

    with pytest.raises(ValueError, match="still has the turn"):
        evaluator.evaluate(
            make_state(player=0),
            defending_player=0,
        )

    with pytest.raises(ValueError, match="terminal"):
        evaluator.evaluate(
            make_state(cue_pocketed=True),
            defending_player=0,
        )
