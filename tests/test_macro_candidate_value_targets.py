from dataclasses import dataclass

import numpy as np

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
)
from src.macro.candidate_value_targets import (
    evaluate_two_step_candidate_target,
)
from src.macro.transition import (
    MacroTransitionResult,
)


def make_state(
    marker=0.0,
    player=0,
):
    return GameState(
        ball_positions=np.array(
            [
                [marker, 0.0],
                [0.4, 0.8],
                [0.9, 1.4],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=player,
    )


def make_candidate(
    pocket,
):
    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=pocket,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=pocket,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


class StaticGenerator:

    def __init__(
        self,
        candidates,
    ):
        self.candidates = list(
            candidates
        )

    def generate(
        self,
        state,
    ):
        return list(
            self.candidates
        )


class DummyTransition:

    def __init__(
        self,
        first_reward=1.0,
        next_rewards=None,
        terminated=False,
        switch_turn=False,
    ):
        self.switch_turn = bool(
            switch_turn
        )
        self.first_reward = float(
            first_reward
        )
        self.next_rewards = (
            dict(next_rewards)
            if next_rewards is not None
            else {}
        )
        self.terminated = bool(
            terminated
        )
        self.calls = 0

    def step(
        self,
        state,
        candidate,
    ):
        self.calls += 1

        # marker=0 is root state,
        # marker=1 is next state.
        is_root = bool(
            np.isclose(
                state.ball_positions[0, 0],
                0.0,
            )
        )

        if is_root:
            reward = self.first_reward
            next_state = make_state(
                marker=1.0,
                player=(
                    1 if self.switch_turn else 0
                ),
            )
            terminated = (
                self.terminated
            )

        else:
            reward = float(
                self.next_rewards.get(
                    candidate.action.target_pocket,
                    0.0,
                )
            )
            next_state = state
            terminated = False

        return MacroTransitionResult(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )


def test_two_step_target_uses_best_next_reward():

    next_candidates = [
        make_candidate(0),
        make_candidate(1),
        make_candidate(2),
    ]

    transition = DummyTransition(
        first_reward=1.0,
        next_rewards={
            0: 0.25,
            1: 1.25,
            2: -0.5,
        },
    )

    result = (
        evaluate_two_step_candidate_target(
            state=make_state(),
            candidate=make_candidate(3),
            transition_model=transition,
            next_candidate_generator=(
                StaticGenerator(
                    next_candidates
                )
            ),
            gamma=0.9,
        )
    )

    assert np.isclose(
        result.immediate_reward,
        1.0,
    )
    assert np.isclose(
        result.next_best_reward,
        1.25,
    )
    assert np.isclose(
        result.target,
        2.125,
    )
    assert (
        result.next_candidate_count
        == 3
    )


def test_terminated_target_has_no_lookahead():

    transition = DummyTransition(
        first_reward=-1.0,
        terminated=True,
    )

    result = (
        evaluate_two_step_candidate_target(
            state=make_state(),
            candidate=make_candidate(3),
            transition_model=transition,
            next_candidate_generator=(
                StaticGenerator(
                    [
                        make_candidate(0)
                    ]
                )
            ),
            gamma=0.99,
        )
    )

    assert result.terminated is True
    assert result.next_candidate_count == 0
    assert result.next_best_reward == 0.0
    assert result.target == -1.0
    assert transition.calls == 1


def test_no_next_candidate_falls_back_to_immediate_reward():

    transition = DummyTransition(
        first_reward=0.5,
    )

    result = (
        evaluate_two_step_candidate_target(
            state=make_state(),
            candidate=make_candidate(3),
            transition_model=transition,
            next_candidate_generator=(
                StaticGenerator([])
            ),
            gamma=0.99,
        )
    )

    assert result.terminated is False
    assert result.next_candidate_count == 0
    assert result.next_best_reward == 0.0
    assert result.target == 0.5


def test_two_step_target_penalizes_opponent_success_when_turn_changes():
    transition = DummyTransition(
        first_reward=0.0,
        next_rewards={
            0: 0.25,
            1: 1.25,
        },
        switch_turn=True,
    )

    result = evaluate_two_step_candidate_target(
        state=make_state(),
        candidate=make_candidate(3),
        transition_model=transition,
        next_candidate_generator=StaticGenerator(
            [
                make_candidate(0),
                make_candidate(1),
            ]
        ),
        gamma=0.9,
        turn_aware=True,
    )

    assert np.isclose(
        result.next_best_reward,
        1.25,
    )
    assert np.isclose(
        result.target,
        -1.125,
    )


def test_two_step_target_preserves_same_player_success():
    transition = DummyTransition(
        first_reward=1.0,
        next_rewards={
            0: 0.25,
            1: 1.25,
        },
        switch_turn=False,
    )

    result = evaluate_two_step_candidate_target(
        state=make_state(),
        candidate=make_candidate(3),
        transition_model=transition,
        next_candidate_generator=StaticGenerator(
            [
                make_candidate(0),
                make_candidate(1),
            ]
        ),
        gamma=0.9,
        turn_aware=True,
    )

    assert np.isclose(
        result.target,
        2.125,
    )


def test_attack_dataset_reuses_factory_and_keeps_residual_metadata(
    tmp_path, monkeypatch, capsys,
):
    """Keep the legacy 25-D dataset and residual checkpoint log intact."""
    from types import SimpleNamespace

    import scripts.build_macro_candidate_value_dataset as script
    from src.common.types import GameState, MacroAction, Strategy
    from src.goal.types import TacticalGoal, TacticalGoalType
    from src.macro.candidate import MacroCandidate

    candidate = MacroCandidate(
        action=MacroAction(Strategy.ATTACK, target_ball=0, target_pocket=0),
        goal=TacticalGoal(
            TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
    )

    class OneCandidate:
        def generate(self, state):
            return [candidate]

    calls = []

    def fake_components(args, device):
        calls.append(device)
        return (
            (object(), OneCandidate(), OneCandidate()),
            {"residual_mode": "power-only"},
        )

    monkeypatch.setattr(
        script, "build_attack_components_with_metadata", fake_components
    )
    monkeypatch.setattr(script, "MacroCandidateGenerator", OneCandidate)
    monkeypatch.setattr(script, "MicroBilliardEnv", lambda: object())
    monkeypatch.setattr(
        script, "random_state",
        lambda env, seed: GameState(
            ball_positions=np.array(
                [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
                dtype=np.float32,
            ),
            score=np.zeros(2, dtype=np.float32),
            current_player=0,
        ),
    )
    monkeypatch.setattr(
        script,
        "evaluate_two_step_candidate_target",
        lambda **kwargs: SimpleNamespace(
            target=0.75,
            immediate_reward=0.5,
            next_best_reward=0.25,
            next_candidate_count=1,
        ),
    )
    output = tmp_path / "attack_legacy.npz"
    args = SimpleNamespace(
        device="cpu",
        states=1,
        seed=11,
        gamma=0.99,
        turn_aware=False,
        residual_scale=0.75,
        log_interval=0,
        output=str(output),
    )

    script.build_dataset(args)

    assert calls == ["cpu"]
    with np.load(output, allow_pickle=False) as data:
        assert data["state"].shape == (1, 12)
        assert data["candidate"].shape == (1, 25)
        np.testing.assert_allclose(data["q_target"], [0.75])
        np.testing.assert_allclose(data["immediate_reward"], [0.5])
        assert data["state_id"].tolist() == [0]
        assert not bool(data["turn_aware"])
    assert "residual checkpoint mode: power-only" in capsys.readouterr().out
