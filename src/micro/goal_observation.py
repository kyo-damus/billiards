import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)


PHYSICAL_OBSERVATION_DIM = 9


def build_physical_observation(
    positions,
    pocket_indices,
):
    """
    TacticalGoalに依存しない物理状態。

    6:
        cue / object0 / object1 のxy座標

    3:
        各球のpocketed flag
    """

    positions = np.asarray(
        positions,
        dtype=np.float32,
    )

    pocket_indices = np.asarray(
        pocket_indices,
        dtype=np.int32,
    )

    if positions.shape != (3, 2):
        raise ValueError(
            f"positions must have shape (3, 2), "
            f"got {positions.shape}"
        )

    if pocket_indices.shape != (3,):
        raise ValueError(
            "pocket_indices must have shape (3,)"
        )

    normalized_positions = (
        positions.copy()
    )

    normalized_positions[:, 0] /= (
        TABLE_WIDTH
    )

    normalized_positions[:, 1] /= (
        TABLE_LENGTH
    )

    pocket_flags = (
        pocket_indices >= 0
    ).astype(np.float32)

    return np.concatenate(
        [
            normalized_positions.reshape(-1),
            pocket_flags,
        ]
    ).astype(np.float32)
    