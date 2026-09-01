from dataclasses import dataclass
from enum import Enum


class TacticalGoalType(Enum):
    DIRECT_ATTACK = "direct_attack"
    POSITION_ATTACK = "position_attack"
    SAFETY = "safety"


@dataclass(frozen=True)
class TacticalGoal:
    """
    MacroからMicroへ渡す戦術目標。

    現段階では DIRECT_ATTACK のみ実運用する。
    その他は将来拡張用。
    """

    goal_type: TacticalGoalType

    target_ball: int | None = None
    target_pocket: int | None = None

    cue_target_position: tuple[float, float] | None = None
    cue_target_radius: float | None = None

    # POSITION_ATTACK用
    next_target_ball: int | None = None
    next_target_pocket: int | None = None