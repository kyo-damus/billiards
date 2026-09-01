import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


GOAL_TYPE_DIM = 3
TARGET_BALL_DIM = 2
TARGET_POCKET_DIM = 6

# type 3
# + ball 2
# + pocket 6
# + cue region flag 1
# + cue target xy 2
# + cue radius 1
TACTICAL_GOAL_DIM = 15


GOAL_TYPE_TO_INDEX = {
    TacticalGoalType.DIRECT_ATTACK: 0,
    TacticalGoalType.POSITION_ATTACK: 1,
    TacticalGoalType.SAFETY: 2,
}


def encode_tactical_goal(
    goal: TacticalGoal,
) -> np.ndarray:
    """
    TacticalGoal -> NN入力用固定長ベクトル
    """

    vector = np.zeros(
        TACTICAL_GOAL_DIM,
        dtype=np.float32,
    )

    # --------------------------------
    # Goal type: 0..2
    # --------------------------------

    type_index = GOAL_TYPE_TO_INDEX[
        goal.goal_type
    ]

    vector[type_index] = 1.0

    offset = GOAL_TYPE_DIM

    # --------------------------------
    # Target ball: 3..4
    # --------------------------------

    if goal.target_ball is not None:
        if goal.target_ball not in (0, 1):
            raise ValueError(
                f"Invalid target_ball: {goal.target_ball}"
            )

        vector[
            offset + goal.target_ball
        ] = 1.0

    offset += TARGET_BALL_DIM

    # --------------------------------
    # Target pocket: 5..10
    # --------------------------------

    if goal.target_pocket is not None:
        if goal.target_pocket not in range(6):
            raise ValueError(
                f"Invalid target_pocket: "
                f"{goal.target_pocket}"
            )

        vector[
            offset + goal.target_pocket
        ] = 1.0

    offset += TARGET_POCKET_DIM

    # --------------------------------
    # Cue target region
    # --------------------------------

    if goal.cue_target_position is not None:

        vector[offset] = 1.0

        x, y = goal.cue_target_position

        vector[offset + 1] = (
            x / TABLE_WIDTH
        )

        vector[offset + 2] = (
            y / TABLE_LENGTH
        )

        if goal.cue_target_radius is not None:
            vector[offset + 3] = (
                goal.cue_target_radius
                / TABLE_WIDTH
            )

    return vector