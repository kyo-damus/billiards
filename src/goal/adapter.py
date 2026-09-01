from src.common.types import (
    MacroAction,
    Strategy,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


def macro_action_to_tactical_goal(
    action: MacroAction,
) -> TacticalGoal:
    """
    現在のMacroActionを、
    Microが解釈するTacticalGoalへ変換する。
    """

    if action.strategy == Strategy.ATTACK:
        return TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=action.target_ball,
            target_pocket=action.target_pocket,
        )

    raise NotImplementedError(
        f"Unsupported strategy: {action.strategy}"
    )