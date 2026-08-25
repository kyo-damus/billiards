from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Strategy(Enum):
    ATTACK = "attack"
    DEFENSE = "defense"


@dataclass
class GameState:
    """
    ゲーム全体の状態。

    ball_positions:
        shape = (3, 2)

        0 = cue
        1 = object ball 0
        2 = object ball 1

    ball_pocket_indices:
        shape = (3,)

        -1 = テーブル上
         0 = SW
         1 = W
         2 = NW
         3 = NE
         4 = E
         5 = SE
    """

    ball_positions: np.ndarray
    score: np.ndarray
    current_player: int

    # 既存コードとの後方互換性のため、
    # 指定しなければ全球テーブル上とする。
    ball_pocket_indices: np.ndarray = field(
        default_factory=lambda: np.full(
            3,
            -1,
            dtype=np.int32,
        )
    )

    def is_pocketed(
        self,
        ball_index: int,
    ) -> bool:
        """
        ball_index:
            0 = cue
            1 = object ball 0
            2 = object ball 1
        """
        return bool(
            self.ball_pocket_indices[ball_index] >= 0
        )


@dataclass(frozen=True)
class MacroAction:
    """Macro が決定し、Micro に渡す戦術レベルの行動."""

    strategy: Strategy
    target_ball: int | None
    target_pocket: int | None


@dataclass
class ShotAction:
    """Micro が決定し、物理シミュレータへ渡す実際のショット."""

    angle: float
    power: float
    spin_x: float = 0.0
    spin_y: float = 0.0