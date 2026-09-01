import math

import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)

from src.goal.conditioner import (
    decode_positions,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


POSITION_PROGRESS_SCALE = 3.0
POSITION_SUCCESS_BONUS = 5.0


def calculate_goal_conditioned_reward(
    base_reward: float,
    goal: TacticalGoal,
    before_observation,
    after_observation,
    shot_info,
):
    """
    MicroBilliardEnvが返すbase rewardへ、
    TacticalGoal固有の報酬を追加する。

    DIRECT_ATTACK:
        既存rewardをそのまま利用。

    POSITION_ATTACK:
        + cue ballが目標領域へ近づいた量
        + pot成功かつ目標領域内ならbonus

    Returns
    -------
    total_reward: float
    goal_info: dict
    """

    base_reward = float(
        base_reward
    )

    # ==================================
    # DIRECT ATTACK
    # ==================================

    if (
        goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
    ):
        return (
            base_reward,
            {
                "goal_achieved": bool(
                    shot_info.get(
                        "success",
                        False,
                    )
                ),
                "position_progress": 0.0,
                "position_bonus": 0.0,
                "cue_region_reached": False,
            },
        )

    # ==================================
    # POSITION ATTACK
    # ==================================

    if (
        goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
    ):

        if (
            goal.cue_target_position
            is None
        ):
            raise ValueError(
                "POSITION_ATTACK requires "
                "cue_target_position."
            )

        if (
            goal.cue_target_radius
            is None
        ):
            raise ValueError(
                "POSITION_ATTACK requires "
                "cue_target_radius."
            )

        before_positions = (
            decode_positions(
                before_observation
            )
        )

        after_positions = (
            decode_positions(
                after_observation
            )
        )

        cue_before = (
            before_positions[0]
        )

        cue_after = (
            after_positions[0]
        )

        cue_target = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        distance_before = float(
            np.linalg.norm(
                cue_before
                - cue_target
            )
        )

        distance_after = float(
            np.linalg.norm(
                cue_after
                - cue_target
            )
        )

        table_diag = math.sqrt(
            TABLE_WIDTH ** 2
            + TABLE_LENGTH ** 2
        )

        normalized_progress = (
            distance_before
            - distance_after
        ) / table_diag

        scratched = bool(
            shot_info.get(
                "scratched",
                False,
            )
        )

        direct_success = bool(
            shot_info.get(
                "success",
                False,
            )
        )

        cue_region_reached = bool(
            distance_after
            <= goal.cue_target_radius
        )

        # scratch時にposition shapingを
        # 利用して得をしないよう無効化
        if scratched:
            progress_reward = 0.0

        else:
            progress_reward = (
                POSITION_PROGRESS_SCALE
                * normalized_progress
            )

        position_success = bool(
            direct_success
            and cue_region_reached
            and not scratched
        )

        position_bonus = (
            POSITION_SUCCESS_BONUS
            if position_success
            else 0.0
        )

        total_reward = (
            base_reward
            + progress_reward
            + position_bonus
        )

        return (
            float(total_reward),
            {
                "goal_achieved": (
                    position_success
                ),
                "position_progress": float(
                    progress_reward
                ),
                "position_bonus": float(
                    position_bonus
                ),
                "cue_region_reached": (
                    cue_region_reached
                ),
                "cue_distance_before": (
                    distance_before
                ),
                "cue_distance_after": (
                    distance_after
                ),
            },
        )

    raise NotImplementedError(
        f"Reward is not implemented "
        f"for {goal.goal_type}"
    )
    