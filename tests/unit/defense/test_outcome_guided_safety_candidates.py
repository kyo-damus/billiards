from types import SimpleNamespace

import numpy as np
import pytest

from src.common.types import (
    GameState,
    MacroAction,
    ShotAction,
    Strategy,
)
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
    MACRO_CANDIDATE_V2_DIM,
    encode_macro_candidate,
    encode_macro_candidate_v2,
)
from src.macro.outcome_guided_safety_candidates import (
    OutcomeGuidedSafetyMacroCandidateGenerator,
)


def make_state():
    return GameState(
        ball_positions=np.asarray(
            [
                [0.5, 0.3],
                [0.5, 0.8],
                [0.9, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
    )


class FakeShotGenerator:
    def __init__(self, shots):
        self.shots = list(shots)

    def generate(self, state):
        return [
            SimpleNamespace(
                targeted_ball=0,
                shot=shot,
                result=SimpleNamespace(
                    next_state=state,
                ),
            )
            for shot in self.shots
        ]


def make_defense_shot_candidates():
    state = make_state()
    generator = OutcomeGuidedSafetyMacroCandidateGenerator(
        FakeShotGenerator(
            [
                ShotAction(
                    angle=90.0,
                    power=0.65,
                    tip_y_mm=0.0,
                ),
                ShotAction(
                    angle=90.0,
                    power=0.65,
                    tip_y_mm=5.0,
                ),
            ]
        )
    )
    return generator.generate(state)


def test_safety_adapter_generates_hashable_distinct_candidates():
    candidates = make_defense_shot_candidates()

    assert len(candidates) == 2
    assert len(set(candidates)) == 2
    assert all(
        candidate.action.strategy == Strategy.DEFENSE
        and candidate.action.target_ball == 0
        and candidate.action.target_pocket is None
        and candidate.goal.goal_type == TacticalGoalType.SAFETY
        and candidate.physical_shot is not None
        for candidate in candidates
    )


def test_v2_encoder_distinguishes_tip_offsets():
    a, b = make_defense_shot_candidates()

    encoded_a = encode_macro_candidate_v2(a)
    encoded_b = encode_macro_candidate_v2(b)

    assert MACRO_CANDIDATE_DIM == 25
    assert MACRO_CANDIDATE_V2_DIM == 31
    assert encoded_a.shape == (31,)
    assert encoded_b.shape == (31,)
    np.testing.assert_allclose(
        encoded_a[:25],
        encoded_b[:25],
    )
    assert not np.allclose(encoded_a, encoded_b)

    assert encoded_a[25] == 1.0
    assert np.isclose(encoded_b[30], 5.0 / 12.0)


def test_legacy_attack_encoder_cannot_silently_encode_defense():
    a, _ = make_defense_shot_candidates()

    with pytest.raises(ValueError, match="v2"):
        encode_macro_candidate(a)


def test_v2_keeps_attack_base_features_unchanged():
    candidate = MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=1,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=1,
        ),
    )

    old = encode_macro_candidate(candidate)
    new = encode_macro_candidate_v2(candidate)

    np.testing.assert_allclose(new[:25], old)
    np.testing.assert_allclose(
        new[25:],
        np.zeros(6, dtype=np.float32),
    )


def test_v2_rejects_unexecutable_safety_anchor():
    candidate = MacroCandidate(
        action=MacroAction(
            strategy=Strategy.DEFENSE,
            target_ball=None,
            target_pocket=None,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.SAFETY,
            cue_target_position=(0.4, 0.5),
            cue_target_radius=0.12,
        ),
    )

    with pytest.raises(ValueError, match="physical_shot"):
        encode_macro_candidate_v2(candidate)



def test_adapter_reuses_evaluated_proposals_without_second_generation():
    state = make_state()

    class CountingGenerator(FakeShotGenerator):
        def __init__(self, shots):
            super().__init__(shots)
            self.calls = 0

        def generate(self, state):
            self.calls += 1
            return super().generate(state)

    generator = CountingGenerator(
        [
            ShotAction(
                angle=45.0,
                power=0.65,
                tip_y_mm=5.0,
            )
        ]
    )

    adapter = OutcomeGuidedSafetyMacroCandidateGenerator(
        shot_generator=generator,
    )

    proposals = generator.generate(state)
    assert generator.calls == 1

    result = adapter.from_proposals(proposals)
    assert generator.calls == 1
    assert len(result) == 1
    assert np.isclose(
        result[0].physical_shot.tip_y_mm,
        5.0,
    )
