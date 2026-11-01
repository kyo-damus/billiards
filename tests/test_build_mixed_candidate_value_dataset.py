"""Fast, mocked test for mixed Q dataset schema and strategy coverage."""

from types import SimpleNamespace

import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    ShotAction,
    Strategy,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.candidate import MacroCandidate
from src.macro.outcome_guided_safety_candidates import (
    OutcomeGuidedSafetyMacroCandidateGenerator,
)
from src.macro.transition import MacroTransitionResult


def sample_state():
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
    )


class FakeGenerator:
    def __init__(self, choices):
        self.choices = choices

    def generate(self, state):
        return list(self.choices)


class FakeSafetyGenerator:
    def __init__(self):
        self.last_stats = {}

    def generate(self, state):
        self.last_stats = {
            "attempted": 1,
            "eligible": 1,
        }
        next_state = sample_state()
        next_state.current_player = 1
        return [
            SimpleNamespace(
                targeted_ball=0,
                shot=ShotAction(
                    angle=90.0,
                    power=0.65,
                    tip_y_mm=5.0,
                ),
                result=SimpleNamespace(
                    next_state=next_state,
                ),
            )
        ]


class FakeMixedTransition:
    def step(self, state, candidate):
        next_state = sample_state()
        next_state.current_player = 1
        return MacroTransitionResult(
            next_state=next_state,
            reward=(
                0.5
                if candidate.action.strategy == Strategy.ATTACK
                else 0.0
            ),
            terminated=True,
        )


def test_mixed_dataset_has_31_dim_and_both_strategies(
    tmp_path,
    monkeypatch,
):
    import scripts.build_mixed_candidate_value_dataset as script

    attack = MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
    )

    safety_shots = FakeSafetyGenerator()
    safety_adapter = OutcomeGuidedSafetyMacroCandidateGenerator(
        shot_generator=safety_shots,
    )

    monkeypatch.setattr(
        script,
        "make_mixed_components",
        lambda args, device: (
            FakeMixedTransition(),
            FakeGenerator([attack]),
            FakeGenerator([attack]),
            safety_shots,
            safety_adapter,
        ),
    )
    monkeypatch.setattr(
        script,
        "MicroBilliardEnv",
        lambda: object(),
    )
    monkeypatch.setattr(
        script,
        "random_state",
        lambda env, seed: sample_state(),
    )

    output = tmp_path / "mixed_v2.npz"
    args = SimpleNamespace(
        device="cpu",
        states=2,
        seed=0,
        gamma=0.99,
        residual_scale=0.75,
        output=str(output),
        log_interval=0,
    )

    script.build_dataset(args)

    with np.load(output) as data:
        assert data["state"].shape[0] == 4
        assert data["candidate"].shape == (4, 31)
        assert np.array_equal(
            data["strategy"],
            np.asarray([0, 1, 0, 1], dtype=np.int32),
        )
        assert np.array_equal(
            data["state_id"],
            np.asarray([0, 0, 1, 1], dtype=np.int32),
        )
        assert int(data["candidate_dim"]) == 31
        assert int(data["candidate_encoding_version"]) == 2
        assert bool(data["turn_aware"])
        assert np.isclose(float(data["residual_scale"]), 0.75)
        assert str(data["next_action_set"]) == "DIRECT_ONLY"

        # Terminal first step prevents second-step lookahead.
        np.testing.assert_allclose(
            data["q_target"],
            [0.5, 0.0, 0.5, 0.0],
        )
