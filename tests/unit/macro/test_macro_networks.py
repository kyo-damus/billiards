import numpy as np
import torch

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.macro.encoding import (
    MACRO_STATE_DIM,
    NUM_MACRO_ACTIONS,
    encode_game_state,
    macro_action_to_index,
    index_to_macro_action,
)

from src.macro.policy import (
    MacroPolicyNetwork,
)

from src.macro.value import (
    MacroValueNetwork,
)


def make_game_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.5, 0.5],
                [0.3, 1.0],
                [0.8, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        current_player=0,
    )


def test_encode_game_state():
    encoded = encode_game_state(
        make_game_state()
    )

    assert encoded.shape == (
        MACRO_STATE_DIM,
    )


def test_macro_action_index_roundtrip():
    for ball in (0, 1):
        for pocket in range(6):

            action = MacroAction(
                strategy=Strategy.ATTACK,
                target_ball=ball,
                target_pocket=pocket,
            )

            index = macro_action_to_index(
                action
            )

            restored = index_to_macro_action(
                index
            )

            assert restored == action


def test_policy_output_shape():
    policy = MacroPolicyNetwork()

    state = torch.zeros(
        4,
        MACRO_STATE_DIM,
    )

    logits = policy(state)

    assert logits.shape == (
        4,
        NUM_MACRO_ACTIONS,
    )


def test_value_output_shape():
    value = MacroValueNetwork()

    state = torch.zeros(
        4,
        MACRO_STATE_DIM,
    )

    output = value(state)

    assert output.shape == (
        4,
        1,
    )

def test_encoding_contains_pocket_information():
    state = make_game_state()

    state.ball_pocket_indices = np.array(
        [
            -1,
             0,
            -1,
        ],
        dtype=np.int32,
    )

    encoded = encode_game_state(state)

    assert encoded.shape == (
        MACRO_STATE_DIM,
    )

    # 位置6要素の次がpocket flags
    np.testing.assert_array_equal(
        encoded[6:9],
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
    )
    