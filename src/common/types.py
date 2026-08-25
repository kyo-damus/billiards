from dataclasses import dataclass
from enum import Enum

import numpy as np


class Strategy(Enum):
    ATTACK = "attack"
    DEFENSE = "defense"


@dataclass
class GameState:
    """Macro / Micro が共通して参照する現在のゲーム状態."""

    ball_positions: np.ndarray
    score: np.ndarray
    current_player: int


@dataclass
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