import numpy as np
import torch

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)
from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)


NUM_TARGET_BALLS = 2
NUM_POCKETS = 6
NUM_MACRO_ACTIONS = NUM_TARGET_BALLS * NUM_POCKETS

MACRO_STATE_DIM = 12


def encode_game_state(
    game_state: GameState,
) -> np.ndarray:
    """
    GameState -> Macro network入力ベクトル

    0-5 : 3球の正規化座標
    6-7 : score
    8   : current_player
    """

    positions = np.asarray(
        game_state.ball_positions,
        dtype=np.float32,
    )

    if positions.shape != (3, 2):
        raise ValueError(
            "ball_positions must have shape (3, 2)"
        )

    normalized = positions.copy()

    normalized[:, 0] /= TABLE_WIDTH
    normalized[:, 1] /= TABLE_LENGTH

    score = np.asarray(
        game_state.score,
        dtype=np.float32,
    )

    if score.shape != (2,):
        raise ValueError(
            "score must have shape (2,)"
        )

    pocket_flags = (
        np.asarray(
            game_state.ball_pocket_indices,
            dtype=np.int32,
        )
        >= 0
    ).astype(np.float32)

    return np.concatenate(
        [
            # 6
            normalized.reshape(-1),

            # 3
            pocket_flags,

            # 2
            score,

            # 1
            np.array(
                [game_state.current_player],
                dtype=np.float32,
            ),
        ]
    )


def macro_action_to_index(
    action: MacroAction,
) -> int:
    if action.strategy != Strategy.ATTACK:
        raise NotImplementedError(
            "Only ATTACK actions are currently supported."
        )

    if action.target_ball not in (0, 1):
        raise ValueError("Invalid target_ball.")

    if action.target_pocket not in range(6):
        raise ValueError("Invalid target_pocket.")

    return (
        action.target_ball * NUM_POCKETS
        + action.target_pocket
    )


def index_to_macro_action(
    index: int,
) -> MacroAction:
    if index not in range(NUM_MACRO_ACTIONS):
        raise ValueError(
            f"Invalid macro action index: {index}"
        )

    target_ball = index // NUM_POCKETS
    target_pocket = index % NUM_POCKETS

    return MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=target_ball,
        target_pocket=target_pocket,
    )


def game_state_to_tensor(
    game_state: GameState,
    device=None,
) -> torch.Tensor:

    encoded = encode_game_state(
        game_state
    )

    return torch.as_tensor(
        encoded,
        dtype=torch.float32,
        device=device,
    )
