import math

import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)

from src.common.types import (
    Strategy,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)


STRATEGY_DIM = 2
GOAL_TYPE_DIM = 3
TARGET_BALL_DIM = 2
TARGET_POCKET_DIM = 6
CUE_REGION_DIM = 4
NEXT_BALL_DIM = 2
NEXT_POCKET_DIM = 6

MACRO_CANDIDATE_DIM = (
    STRATEGY_DIM
    + GOAL_TYPE_DIM
    + TARGET_BALL_DIM
    + TARGET_POCKET_DIM
    + CUE_REGION_DIM
    + NEXT_BALL_DIM
    + NEXT_POCKET_DIM
)
# 25


STRATEGY_TO_INDEX = {
    Strategy.ATTACK: 0,
    Strategy.DEFENSE: 1,
}


GOAL_TYPE_TO_INDEX = {
    TacticalGoalType.DIRECT_ATTACK: 0,
    TacticalGoalType.POSITION_ATTACK: 1,
    TacticalGoalType.SAFETY: 2,
}


def encode_macro_candidate(
    candidate: MacroCandidate,
) -> np.ndarray:
    """
    MacroCandidateをPolicy入力用の
    固定長ベクトルへ変換する。

    候補数は可変だが、
    各候補の表現次元は固定。
    """

    vector = np.zeros(
        MACRO_CANDIDATE_DIM,
        dtype=np.float32,
    )

    action = candidate.action
    goal = candidate.goal

    # --------------------------------
    # Strategy: 0..1
    # --------------------------------

    vector[
        STRATEGY_TO_INDEX[
            action.strategy
        ]
    ] = 1.0

    offset = STRATEGY_DIM

    # --------------------------------
    # Goal type: 2..4
    # --------------------------------

    vector[
        offset
        + GOAL_TYPE_TO_INDEX[
            goal.goal_type
        ]
    ] = 1.0

    offset += GOAL_TYPE_DIM

    # --------------------------------
    # Current target ball: 5..6
    # --------------------------------

    if goal.target_ball is not None:

        if goal.target_ball not in (0, 1):
            raise ValueError(
                "target_ball must be 0 or 1"
            )

        vector[
            offset + goal.target_ball
        ] = 1.0

    offset += TARGET_BALL_DIM

    # --------------------------------
    # Current target pocket: 7..12
    # --------------------------------

    if goal.target_pocket is not None:

        if goal.target_pocket not in range(6):
            raise ValueError(
                "target_pocket must be 0..5"
            )

        vector[
            offset + goal.target_pocket
        ] = 1.0

    offset += TARGET_POCKET_DIM

    # --------------------------------
    # Cue target region: 13..16
    # --------------------------------

    if goal.cue_target_position is not None:

        vector[offset] = 1.0

        x, y = (
            goal.cue_target_position
        )

        vector[offset + 1] = (
            float(x) / TABLE_WIDTH
        )

        vector[offset + 2] = (
            float(y) / TABLE_LENGTH
        )

        if goal.cue_target_radius is not None:

            table_diag = math.sqrt(
                TABLE_WIDTH ** 2
                + TABLE_LENGTH ** 2
            )

            vector[offset + 3] = (
                float(
                    goal.cue_target_radius
                )
                / table_diag
            )

    offset += CUE_REGION_DIM

    # --------------------------------
    # Next target ball: 17..18
    # --------------------------------

    if goal.next_target_ball is not None:

        if (
            goal.next_target_ball
            not in (0, 1)
        ):
            raise ValueError(
                "next_target_ball "
                "must be 0 or 1"
            )

        vector[
            offset
            + goal.next_target_ball
        ] = 1.0

    offset += NEXT_BALL_DIM

    # --------------------------------
    # Next pocket: 19..24
    # --------------------------------

    if (
        goal.next_target_pocket
        is not None
    ):

        if (
            goal.next_target_pocket
            not in range(6)
        ):
            raise ValueError(
                "next_target_pocket "
                "must be 0..5"
            )

        vector[
            offset
            + goal.next_target_pocket
        ] = 1.0

    return vector
    