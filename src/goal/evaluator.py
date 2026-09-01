import numpy as np

from src.common.types import GameState

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


class TacticalGoalEvaluator:

    def is_achieved(
        self,
        goal: TacticalGoal,
        state: GameState,
    ) -> bool:

        if (
            goal.goal_type
            == TacticalGoalType.DIRECT_ATTACK
        ):
            return self._direct_attack_achieved(
                goal,
                state,
            )

        if (
            goal.goal_type
            == TacticalGoalType.POSITION_ATTACK
        ):
            return self._position_attack_achieved(
                goal,
                state,
            )

        raise NotImplementedError(
            f"Goal evaluation is not implemented "
            f"for {goal.goal_type}"
        )

    def _direct_attack_achieved(
        self,
        goal: TacticalGoal,
        state: GameState,
    ) -> bool:

        if goal.target_ball is None:
            return False

        if goal.target_pocket is None:
            return False

        # GameState:
        # 0 = cue
        # 1 = object ball 0
        # 2 = object ball 1
        state_index = goal.target_ball + 1

        return bool(
            state.ball_pocket_indices[
                state_index
            ]
            == goal.target_pocket
        )

    def _position_attack_achieved(
        self,
        goal: TacticalGoal,
        state: GameState,
    ) -> bool:
        """
        POSITION_ATTACK成功条件:

        1. 指定球が指定ポケットへ入る
        2. cue ballがscratchしていない
        3. cue ballが指定領域内に停止
        """

        # まずpot成功
        if not self._direct_attack_achieved(
            goal,
            state,
        ):
            return False

        if (
            goal.cue_target_position
            is None
        ):
            return False

        if (
            goal.cue_target_radius
            is None
        ):
            return False

        # cue ballがscratch
        if state.is_pocketed(0):
            return False

        cue_position = np.asarray(
            state.ball_positions[0],
            dtype=np.float32,
        )

        target_position = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(
                cue_position
                - target_position
            )
        )

        return bool(
            distance
            <= goal.cue_target_radius
        )