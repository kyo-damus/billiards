import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)
from src.common.types import (
    MacroAction,
    Strategy,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.candidate import (
    MacroCandidate,
)


class SafetyCandidateGenerator:
    """
    Defense/SAFETY用の初期候補生成器。

    現段階ではMicroでDefenseを実行しないため、
    「手球をどの安全領域へ置きたいか」という
    TacticalGoal候補だけを生成する。

    候補位置はポケットを避けたレール中央寄りの
    固定anchorから作り、残っている的球から遠い順に
    top_k件を返す。

    これは戦術的最適解ではなく、
    Defense候補表現をMacro側へ導入するための
    v0 heuristic。
    """

    def __init__(
        self,
        target_radius=0.12,
        rail_margin=0.18,
        top_k=4,
    ):
        self.target_radius = float(
            target_radius
        )
        self.rail_margin = float(
            rail_margin
        )
        self.top_k = int(
            top_k
        )

        if self.target_radius <= 0.0:
            raise ValueError(
                "target_radius must be positive."
            )

        if self.rail_margin <= 0.0:
            raise ValueError(
                "rail_margin must be positive."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

    def _anchors(self):
        margin = self.rail_margin

        return [
            (
                margin,
                TABLE_LENGTH / 2.0,
            ),
            (
                TABLE_WIDTH - margin,
                TABLE_LENGTH / 2.0,
            ),
            (
                TABLE_WIDTH / 2.0,
                margin,
            ),
            (
                TABLE_WIDTH / 2.0,
                TABLE_LENGTH - margin,
            ),
        ]

    def generate(
        self,
        state,
    ):
        if state.is_pocketed(0):
            return []

        object_positions = []

        for state_index in (1, 2):
            if not state.is_pocketed(
                state_index
            ):
                object_positions.append(
                    np.asarray(
                        state.ball_positions[
                            state_index
                        ],
                        dtype=np.float32,
                    )
                )

        if not object_positions:
            return []

        ranked = []

        for anchor in self._anchors():
            anchor_array = np.asarray(
                anchor,
                dtype=np.float32,
            )

            min_object_distance = min(
                float(
                    np.linalg.norm(
                        anchor_array
                        - object_position
                    )
                )
                for object_position
                in object_positions
            )

            ranked.append(
                (
                    min_object_distance,
                    anchor,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        candidates = []

        for _, anchor in ranked[
            : self.top_k
        ]:
            action = MacroAction(
                strategy=Strategy.DEFENSE,
                target_ball=None,
                target_pocket=None,
            )

            goal = TacticalGoal(
                goal_type=(
                    TacticalGoalType.SAFETY
                ),
                cue_target_position=(
                    float(anchor[0]),
                    float(anchor[1]),
                ),
                cue_target_radius=(
                    self.target_radius
                ),
            )

            candidates.append(
                MacroCandidate(
                    action=action,
                    goal=goal,
                )
            )

        return candidates
