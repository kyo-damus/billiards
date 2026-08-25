from dataclasses import dataclass
from typing import Protocol

from src.common.types import (
    GameState,
    MacroAction,
)


@dataclass
class MacroTransitionResult:
    """
    MacroActionを1回実行した結果。
    """

    next_state: GameState
    reward: float
    terminated: bool


class MacroTransitionModel(Protocol):
    """
    MCTSから見た状態遷移インターフェース。

    将来的には、
        MacroAction
            ↓
        Micro SAC
            ↓
        FastFiz
            ↓
        next GameState

    を実装する。
    """

    def step(
        self,
        state: GameState,
        action: MacroAction,
    ) -> MacroTransitionResult:
        ...