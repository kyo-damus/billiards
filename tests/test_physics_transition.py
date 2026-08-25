import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)

from src.game.rules import (
    GameTransitionResult,
)


class DummyMicroAgent:
    """
    SACの代わりに固定行動を返す。
    transitionの接続テスト専用。
    """

    def select_action(
        self,
        observation,
        deterministic=False,
    ):
        assert observation.shape == (12,)

        return np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )


def make_game_state():
    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.50, 0.30],

                # object ball 0
                [0.50, 0.80],

                # object ball 1
                [0.90, 1.50],
            ],
            dtype=np.float32,
        ),

        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),

        current_player=0,
    )


def make_macro_action():
    return MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=2,
    )


def test_transition_returns_game_state():
    transition = PhysicsMacroTransitionModel(
        micro_agent=DummyMicroAgent(),
    )

    result = transition.step(
        make_game_state(),
        make_macro_action(),
    )

    assert isinstance(
        result.next_state,
        GameState,
    )

    assert (
        result.next_state.ball_positions.shape
        == (3, 2)
    )

    assert isinstance(
        result.reward,
        float,
    )


def test_transition_preserves_game_metadata():
    transition = PhysicsMacroTransitionModel(
        micro_agent=DummyMicroAgent(),
    )

    state = make_game_state()

    result = transition.step(
        state,
        make_macro_action(),
    )

    np.testing.assert_allclose(
        result.next_state.score,
        state.score,
    )

    assert (
        result.next_state.current_player
        == state.current_player
    )


def test_micro_end_is_not_game_end():
    transition = PhysicsMacroTransitionModel(
        micro_agent=DummyMicroAgent(),
    )

    result = transition.step(
        make_game_state(),
        make_macro_action(),
    )

    # Microは1ショットで終了するが、
    # ゲーム全体はまだ終了扱いしない。
    assert result.terminated is False

class DummyGameRules:

    def apply(
        self,
        previous_state,
        physical_state,
        shot_info,
    ):
        return GameTransitionResult(
            next_state=physical_state,
            reward=123.0,
            terminated=True,
            turn_changed=False,
        )


def test_transition_uses_game_rules():

    transition = PhysicsMacroTransitionModel(
        micro_agent=DummyMicroAgent(),
        game_rules=DummyGameRules(),
    )

    result = transition.step(
        make_game_state(),
        make_macro_action(),
    )

    assert result.reward == 123.0
    assert result.terminated is True
    