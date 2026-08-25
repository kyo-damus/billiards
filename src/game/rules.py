from dataclasses import dataclass
from typing import Protocol

from src.common.types import GameState


@dataclass
class GameTransitionResult:
    """
    1ショット後にゲームルールを適用した結果。
    """

    next_state: GameState

    reward: float

    terminated: bool

    turn_changed: bool


class GameRules(Protocol):
    """
    ビリヤードのゲームルールを表すインターフェース。

    FastFizが計算するのは「物理結果」。

    GameRulesはその結果に対して、

        ・得点
        ・手番変更
        ・ゲーム終了
        ・Macro用報酬

    を決定する。
    """

    def apply(
        self,
        previous_state: GameState,
        physical_state: GameState,
        shot_info: dict,
    ) -> GameTransitionResult:
        ...