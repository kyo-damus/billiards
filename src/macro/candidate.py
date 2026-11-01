from dataclasses import dataclass

from src.common.types import MacroAction
from src.goal.types import TacticalGoal


@dataclass(frozen=True)
class PhysicalShotSpec:
    """
    Immutable physical shot for an executable Defense proposal.

    angle is an absolute heading in degrees; power is cue speed
    in m/s; tip offsets are in millimeters.

    Frozen and scalar-only so MacroCandidate remains hashable for
    MCTS child dictionaries. Existing Attack candidates use None.
    """

    angle: float
    power: float
    tip_x_mm: float = 0.0
    tip_y_mm: float = 0.0


@dataclass(frozen=True)
class MacroCandidate:
    """
    Macro探索における1つの戦術候補。

    action:
        MicroBilliardEnvへ渡す
        物理的なショット対象

    goal:
        Micro Policyが達成すべき
        戦術的な条件
    """

    action: MacroAction
    goal: TacticalGoal

    # Explicit shot is only for executable Defense candidates.
    # Attack remains goal-conditioned and chooses its shot via Micro.
    physical_shot: PhysicalShotSpec | None = None