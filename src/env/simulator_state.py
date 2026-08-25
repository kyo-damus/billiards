from dataclasses import dataclass

import numpy as np


@dataclass
class SimulatorSnapshot:
    """
    FastFizの1時点の状態。

    positions:
        cue, obj0, obj1 の座標
        shape = (3, 2)

    pocket_indices:
        -1 = テーブル上
         0 = SW
         1 = W
         2 = NW
         3 = NE
         4 = E
         5 = SE

        shape = (3,)
    """

    positions: np.ndarray
    pocket_indices: np.ndarray

    def __post_init__(self):
        self.positions = np.asarray(
            self.positions,
            dtype=np.float32,
        )

        self.pocket_indices = np.asarray(
            self.pocket_indices,
            dtype=np.int32,
        )

        if self.positions.shape != (3, 2):
            raise ValueError(
                "positions must have shape (3, 2)"
            )

        if self.pocket_indices.shape != (3,):
            raise ValueError(
                "pocket_indices must have shape (3,)"
            )

    def copy(self):
        return SimulatorSnapshot(
            positions=self.positions.copy(),
            pocket_indices=self.pocket_indices.copy(),
        )
        