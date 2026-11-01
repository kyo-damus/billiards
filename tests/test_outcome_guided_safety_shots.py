from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.common.types import GameState
from src.macro.outcome_guided_safety_shots import (
    OutcomeGuidedSafetyShotGenerator,
    is_eligible_safety_shot,
)


def make_state(player=0, cue_pocketed=False, cue_x=0.50):
    return GameState(
        ball_positions=np.array(
            [
                [cue_x, 0.30],
                [0.50, 0.80],
                [0.90, 1.50],
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


class DummyExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, state, target_ball, shot):
        self.calls += 1

        next_state = make_state(
            player=1,
            cue_x=0.60 if shot.tip_y_mm > 0 else 0.50,
        )

        return SimpleNamespace(
            next_state=next_state,
            first_contact_verified=(target_ball == 0),
            first_contact_ball=0 if target_ball == 0 else -1,
            scratched=bool(shot.angle > 95.0),
            terminated=False,
            turn_changed=True,
            newly_pocketed_object_count=0,
        )


class DummyThreat:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, defending_player):
        self.calls += 1
        assert defending_player == 0
        assert state.current_player == 1

        successes = (
            0
            if state.ball_positions[0, 0] > 0.55
            else 2
        )
        return SimpleNamespace(
            successful_pot_count=successes,
            geometric_candidate_count=10,
        )


def test_eligible_requires_first_contact_and_turn_change():
    good = SimpleNamespace(
        first_contact_verified=True,
        first_contact_ball=0,
        scratched=False,
        terminated=False,
        turn_changed=True,
        newly_pocketed_object_count=0,
    )
    assert is_eligible_safety_shot(good, 0)
    assert not is_eligible_safety_shot(good, 1)

    for field, value in [
        ("scratched", True),
        ("terminated", True),
        ("turn_changed", False),
        ("newly_pocketed_object_count", 1),
        ("first_contact_verified", False),
    ]:
        altered = SimpleNamespace(**vars(good))
        setattr(altered, field, value)
        assert not is_eligible_safety_shot(altered, 0)


def test_shot_generator_ranks_lower_opponent_threat():
    executor = DummyExecutor()
    threat = DummyThreat()
    generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=executor,
        opponent_threat_evaluator=threat,
        powers=(0.65,),
        angle_offsets_deg=(0.0, 12.0),
        tip_offsets_mm=((0.0, 0.0), (0.0, 5.0)),
        top_k=2,
    )

    candidates = generator.generate(make_state())

    # 2 balls x 2 offsets x 2 tip positions.
    assert generator.last_stats["attempted"] == 8
    assert generator.last_stats["eligible"] == 2
    assert generator.last_stats["wrong_or_no_first_contact"] == 4
    assert generator.last_stats["scratch"] == 2
    assert threat.calls == 2
    assert len(candidates) == 2
    assert candidates[0].shot.tip_y_mm == 5.0
    assert candidates[0].opponent_threat.successful_pot_count == 0
    assert candidates[1].opponent_threat.successful_pot_count == 2


def test_pocketed_cue_has_no_shots():
    executor = DummyExecutor()
    threat = DummyThreat()
    generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=executor,
        opponent_threat_evaluator=threat,
    )

    assert generator.generate(
        make_state(cue_pocketed=True)
    ) == []
    assert executor.calls == 0
    assert threat.calls == 0


def test_rejects_invalid_grid():
    import pytest

    with pytest.raises(ValueError):
        OutcomeGuidedSafetyShotGenerator(
            defense_executor=DummyExecutor(),
            opponent_threat_evaluator=DummyThreat(),
            powers=(),
        )
