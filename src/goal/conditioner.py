import math

import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
    POCKET_POSITIONS,
)

from src.env.shot_geometry import (
    ghost_ball_position,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)


MICRO_GOAL_DIM = 25


GOAL_TYPE_INDEX = {
    TacticalGoalType.DIRECT_ATTACK: 0,
    TacticalGoalType.POSITION_ATTACK: 1,
    TacticalGoalType.SAFETY: 2,
}


def decode_positions(
    physical_observation,
):
    obs = np.asarray(
        physical_observation,
        dtype=np.float32,
    )

    if obs.shape != (
        PHYSICAL_OBSERVATION_DIM,
    ):
        raise ValueError(
            "invalid physical observation shape"
        )

    positions = (
        obs[:6]
        .reshape(3, 2)
        .copy()
    )

    positions[:, 0] *= TABLE_WIDTH
    positions[:, 1] *= TABLE_LENGTH

    return positions


def encode_micro_goal(
    physical_observation,
    goal: TacticalGoal,
):
    """
    TacticalGoalを現在の盤面に対する
    幾何的なGoal表現へ変換する。
    """

    features = np.zeros(
        MICRO_GOAL_DIM,
        dtype=np.float32,
    )

    # --------------------------------
    # goal type: 3
    # --------------------------------

    goal_type_index = (
        GOAL_TYPE_INDEX[goal.goal_type]
    )

    features[
        goal_type_index
    ] = 1.0

    positions = decode_positions(
        physical_observation
    )

    cue = positions[0]

    # --------------------------------
    # Attack系Goal
    # --------------------------------

    if (
        goal.target_ball is not None
        and goal.target_pocket is not None
    ):

        if goal.target_ball not in (0, 1):
            raise ValueError(
                "target_ball must be 0 or 1"
            )

        if goal.target_pocket not in range(6):
            raise ValueError(
                "target_pocket must be 0..5"
            )

        target = positions[
            goal.target_ball + 1
        ]

        pocket = np.asarray(
            POCKET_POSITIONS[
                goal.target_pocket
            ],
            dtype=np.float32,
        )

        ghost = ghost_ball_position(
            target,
            pocket,
        )

        cue_to_target = (
            target - cue
        )

        target_to_pocket = (
            pocket - target
        )

        cue_to_ghost = (
            ghost - cue
        )

        # relative geometry
        features[3:5] = (
            cue_to_target
            / np.array(
                [
                    TABLE_WIDTH,
                    TABLE_LENGTH,
                ],
                dtype=np.float32,
            )
        )

        features[5:7] = (
            target_to_pocket
            / np.array(
                [
                    TABLE_WIDTH,
                    TABLE_LENGTH,
                ],
                dtype=np.float32,
            )
        )

        features[7:9] = (
            cue_to_ghost
            / np.array(
                [
                    TABLE_WIDTH,
                    TABLE_LENGTH,
                ],
                dtype=np.float32,
            )
        )

        ideal_angle = math.atan2(
            cue_to_ghost[1],
            cue_to_ghost[0],
        )

        features[9] = math.sin(
            ideal_angle
        )

        features[10] = math.cos(
            ideal_angle
        )

        table_diag = math.sqrt(
            TABLE_WIDTH ** 2
            + TABLE_LENGTH ** 2
        )

        features[11] = (
            np.linalg.norm(
                cue_to_target
            )
            / table_diag
        )

        features[12] = (
            np.linalg.norm(
                target_to_pocket
            )
            / table_diag
        )

    # --------------------------------
    # cue target region
    # --------------------------------

    if (
        goal.cue_target_position
        is not None
    ):
        features[13] = 1.0

        cue_target = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        relative = (
            cue_target - cue
        )

        features[14] = (
            relative[0]
            / TABLE_WIDTH
        )

        features[15] = (
            relative[1]
            / TABLE_LENGTH
        )

        if (
            goal.cue_target_radius
            is not None
        ):
            table_diag = math.sqrt(
                TABLE_WIDTH ** 2
                + TABLE_LENGTH ** 2
            )

            features[16] = (
                goal.cue_target_radius
                / table_diag
            )
    
    # --------------------------------
    # Explicit goal identity
    # --------------------------------

    # target ball one-hot
    features[
        17 + goal.target_ball
    ] = 1.0

    # target pocket one-hot
    features[
        19 + goal.target_pocket
    ] = 1.0

    return features