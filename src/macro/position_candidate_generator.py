import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
    POCKET_POSITIONS,
)

from src.common.types import (
    MacroAction,
)

from src.env.shot_geometry import (
    BALL_DIAMETER,
    BALL_RADIUS,
    ghost_ball_position,
    is_direct_shot_feasible,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.macro.action_generator import (
    MacroActionGenerator,
)

from src.macro.candidate import (
    MacroCandidate,
)


DEFAULT_SETUP_DISTANCE = 0.25
DEFAULT_TARGET_RADIUS = 0.10


class PositionAttackCandidateGenerator:
    """
    現在の球をポケットした後に、
    次球を直接狙いやすい位置へ
    手球を残すPOSITION_ATTACK候補を生成する。
    """

    def __init__(
        self,
        action_generator=None,
        setup_distance=DEFAULT_SETUP_DISTANCE,
        target_radius=DEFAULT_TARGET_RADIUS,
    ):
        self.action_generator = (
            action_generator
            if action_generator is not None
            else MacroActionGenerator()
        )

        self.setup_distance = float(
            setup_distance
        )

        self.target_radius = float(
            target_radius
        )

    def generate(
        self,
        state,
    ):
        """
        現在実行可能なDirect Attackを基点として、
        次球用のcue target regionを付加する。
        """

        current_actions = (
            self.action_generator.generate(
                state
            )
        )

        candidates = []

        for action in current_actions:

            if action.target_ball is None:
                continue

            if action.target_pocket is None:
                continue

            # 現在2 object ballsなので
            # もう一方を次球とする
            next_target_ball = (
                1 - action.target_ball
            )

            next_state_index = (
                next_target_ball + 1
            )

            # 次球が既に落ちているなら
            # Position Playは不要
            if state.is_pocketed(
                next_state_index
            ):
                continue

            next_target_pos = (
                state.ball_positions[
                    next_state_index
                ]
            )

            for next_pocket in range(6):

                cue_target = (
                    self._build_cue_target_position(
                        next_target_pos,
                        next_pocket,
                    )
                )

                if cue_target is None:
                    continue

                # その位置から次球を直接狙えるか
                if not is_direct_shot_feasible(
                    cue_pos=cue_target,
                    target_pos=next_target_pos,
                    other_pos=None,
                    pocket_pos=(
                        POCKET_POSITIONS[
                            next_pocket
                        ]
                    ),
                    table_width=TABLE_WIDTH,
                    table_length=TABLE_LENGTH,
                ):
                    continue

                goal = TacticalGoal(
                    goal_type=(
                        TacticalGoalType.POSITION_ATTACK
                    ),

                    # 今回入れる球
                    target_ball=(
                        action.target_ball
                    ),
                    target_pocket=(
                        action.target_pocket
                    ),

                    # 今回のショット後に
                    # 手球を残したい場所
                    cue_target_position=(
                        float(cue_target[0]),
                        float(cue_target[1]),
                    ),

                    cue_target_radius=(
                        self.target_radius
                    ),

                    # 次の狙い
                    next_target_ball=(
                        next_target_ball
                    ),
                    next_target_pocket=(
                        next_pocket
                    ),
                )

                candidates.append(
                    MacroCandidate(
                        action=action,
                        goal=goal,
                    )
                )

        return candidates

    def _build_cue_target_position(
        self,
        target_pos,
        pocket_index,
    ):
        """
        次球→ポケットの直線を利用し、
        ghost ballよりさらに後方に
        手球の停止目標を置く。

        pocket
          ↑
        target
          ↑
        ghost
          ↑
        cue target
        """

        target_pos = np.asarray(
            target_pos,
            dtype=np.float32,
        )

        pocket_pos = np.asarray(
            POCKET_POSITIONS[
                pocket_index
            ],
            dtype=np.float32,
        )

        direction = (
            pocket_pos - target_pos
        )

        distance = float(
            np.linalg.norm(direction)
        )

        if distance < 1e-8:
            return None

        direction /= distance

        ghost = ghost_ball_position(
            target_pos,
            pocket_pos,
        )

        cue_target = (
            ghost
            - direction
            * self.setup_distance
        )

        # --------------------------------
        # target region全体が
        # テーブル内に入るよう確認
        # --------------------------------

        margin = (
            BALL_RADIUS
            + self.target_radius
        )

        x = float(cue_target[0])
        y = float(cue_target[1])

        if not (
            margin
            <= x
            <= TABLE_WIDTH - margin
        ):
            return None

        if not (
            margin
            <= y
            <= TABLE_LENGTH - margin
        ):
            return None

        return cue_target.astype(
            np.float32
        )
        