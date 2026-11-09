import numpy as np

from src.common.types import (
    GameState,
    Strategy,
)

from src.macro.action_generator import (
    MacroActionGenerator,
)


def make_simple_game_state():
    """
    object ball 0 -> W pocket が
    直接狙える簡単な配置。
    """

    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.80, 1.118],

                # object ball 0
                [0.40, 1.118],

                # object ball 1
                [0.80, 0.40],
            ],
            dtype=np.float32,
        ),

        score=np.array(
            [0, 0],
            dtype=np.float32,
        ),

        current_player=0,
    )


def test_generate_returns_actions():
    generator = MacroActionGenerator()

    game_state = make_simple_game_state()

    actions = generator.generate(
        game_state
    )

    assert len(actions) > 0


def test_expected_direct_shot_exists():
    generator = MacroActionGenerator()

    game_state = make_simple_game_state()

    actions = generator.generate(
        game_state
    )

    found = any(
        action.strategy == Strategy.ATTACK
        and action.target_ball == 0
        and action.target_pocket == 1
        for action in actions
    )

    assert found


def test_generated_actions_are_valid():
    generator = MacroActionGenerator()

    actions = generator.generate(
        make_simple_game_state()
    )

    for action in actions:

        assert action.strategy == Strategy.ATTACK

        assert action.target_ball in (
            0,
            1,
        )

        assert action.target_pocket in range(6)


def test_invalid_ball_positions_shape():
    generator = MacroActionGenerator()

    game_state = GameState(
        ball_positions=np.zeros(
            (2, 2),
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
    )

    try:
        generator.generate(game_state)

    except ValueError:
        return

    assert False, "ValueError was not raised"

def test_pocketed_target_is_not_generated():
    generator = MacroActionGenerator()

    state = make_simple_game_state()

    state.ball_pocket_indices = np.array(
        [
            -1,  # cue
             0,  # object ball 0 is pocketed
            -1,  # object ball 1
        ],
        dtype=np.int32,
    )

    actions = generator.generate(state)

    assert all(
        action.target_ball != 0
        for action in actions
    )

