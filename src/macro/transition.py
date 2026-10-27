from dataclasses import dataclass
from typing import Protocol

from src.common.types import (
    GameState,
    MacroAction,
)

from src.macro.local_evaluation import (
    MacroLocalEvaluation,
)


@dataclass
class MacroTransitionResult:
    """
    MacroActionを1回実行した結果。
    """

    next_state: GameState
    reward: float
    terminated: bool

    # 局所評価を有効化した場合のみ設定する。
    # 既存Transitionとの後方互換性のためoptional。
    evaluation: MacroLocalEvaluation | None = None


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
