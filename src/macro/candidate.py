from dataclasses import dataclass

from src.common.types import MacroAction
from src.goal.types import TacticalGoal


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