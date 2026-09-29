from dataclasses import dataclass

import numpy as np

from src.common.table import (
    POCKET_POSITIONS,
    TABLE_LENGTH,
    TABLE_WIDTH,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.env.shot_geometry import (
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


@dataclass(frozen=True)
class OutcomeGuidedPositionWitness:
    """
    生成したPOSITION候補と、
    その候補を作る根拠になったcoarse action。

    witness_action:
        normalized [power, angle_offset]

    next_shot_distance:
        shot後の手球位置から、
        次球を指定ポケットへ狙うghost-ball位置までの距離。
        小さいものを簡易的に優先する。
    """

    candidate: MacroCandidate
    witness_action: tuple[float, float]
    next_shot_distance: float


class OutcomeGuidedPositionCandidateGenerator:
    """
    「理想位置を先に決める」のではなく、
    現在ショットで実際に到達できた結果から
    POSITION_ATTACK Goalを逆生成する。

    手順:
        1. 現在可能なDirect Attackを生成
        2. coarseなMicro action gridをFastFizで試す
        3. 指定pot成功 + non-scratchの結果を残す
        4. shot後の手球位置から次球を直接狙えるか確認
        5. 実際に到達した手球位置をcue_target_positionにする

    したがって、生成されたGoalには少なくとも
    coarse grid上で1つのwitness actionが存在する。

    注意:
        これは「学習済みMicroがそのGoalを達成できる」
        ことを保証するものではない。
    """

    def __init__(
        self,
        action_generator=None,
        grid_size=5,
        target_radius=0.10,
        top_k=8,
        dedup_distance=0.10,
    ):
        self.action_generator = (
            action_generator
            if action_generator is not None
            else MacroActionGenerator()
        )

        if grid_size < 2:
            raise ValueError(
                "grid_size must be >= 2."
            )

        if target_radius <= 0.0:
            raise ValueError(
                "target_radius must be positive."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if dedup_distance <= 0.0:
            raise ValueError(
                "dedup_distance must be positive."
            )

        self.grid_size = int(
            grid_size
        )
        self.target_radius = float(
            target_radius
        )
        self.top_k = int(
            top_k
        )
        self.dedup_distance = float(
            dedup_distance
        )

        self.env = MicroBilliardEnv()

    def generate_with_witnesses(
        self,
        state,
    ) -> list[OutcomeGuidedPositionWitness]:

        macro_actions = (
            self.action_generator.generate(
                state
            )
        )

        values = np.linspace(
            -1.0,
            1.0,
            self.grid_size,
            dtype=np.float32,
        )

        # 同じnext shotに対して近いcue targetが
        # 大量に並ぶのを避けるため、空間binごとに
        # 最良候補だけ残す。
        best_by_key = {}

        for macro_action in macro_actions:

            if macro_action.target_ball is None:
                continue

            if macro_action.target_pocket is None:
                continue

            next_target_ball = (
                1 - macro_action.target_ball
            )
            next_state_index = (
                next_target_ball + 1
            )

            if state.is_pocketed(
                next_state_index
            ):
                continue

            for power_action in values:
                for angle_action in values:

                    micro_action = np.array(
                        [
                            power_action,
                            angle_action,
                        ],
                        dtype=np.float32,
                    )

                    self.env.reset_to_game_state(
                        state,
                        macro_action,
                    )

                    (
                        _,
                        _,
                        _,
                        _,
                        info,
                    ) = self.env.step(
                        micro_action
                    )

                    if not bool(
                        info.get(
                            "success",
                            False,
                        )
                    ):
                        continue

                    if bool(
                        info.get(
                            "scratched",
                            False,
                        )
                    ):
                        continue

                    snapshot = (
                        self.env.sim.snapshot()
                    )

                    # 意図せず次球まで落ちた場合は
                    # Position Playの次目標にできない。
                    if (
                        snapshot.pocket_indices[
                            next_state_index
                        ]
                        >= 0
                    ):
                        continue

                    cue_after = np.asarray(
                        snapshot.positions[0],
                        dtype=np.float32,
                    )

                    next_target_pos = np.asarray(
                        snapshot.positions[
                            next_state_index
                        ],
                        dtype=np.float32,
                    )

                    for (
                        next_pocket,
                        pocket_pos,
                    ) in POCKET_POSITIONS.items():

                        if not is_direct_shot_feasible(
                            cue_pos=cue_after,
                            target_pos=(
                                next_target_pos
                            ),
                            other_pos=None,
                            pocket_pos=pocket_pos,
                            table_width=(
                                TABLE_WIDTH
                            ),
                            table_length=(
                                TABLE_LENGTH
                            ),
                        ):
                            continue

                        ghost = (
                            ghost_ball_position(
                                next_target_pos,
                                pocket_pos,
                            )
                        )

                        next_shot_distance = float(
                            np.linalg.norm(
                                cue_after - ghost
                            )
                        )

                        goal = TacticalGoal(
                            goal_type=(
                                TacticalGoalType
                                .POSITION_ATTACK
                            ),
                            target_ball=(
                                macro_action
                                .target_ball
                            ),
                            target_pocket=(
                                macro_action
                                .target_pocket
                            ),
                            cue_target_position=(
                                float(
                                    cue_after[0]
                                ),
                                float(
                                    cue_after[1]
                                ),
                            ),
                            cue_target_radius=(
                                self.target_radius
                            ),
                            next_target_ball=(
                                next_target_ball
                            ),
                            next_target_pocket=(
                                next_pocket
                            ),
                        )

                        candidate = (
                            MacroCandidate(
                                action=(
                                    macro_action
                                ),
                                goal=goal,
                            )
                        )

                        witness = (
                            OutcomeGuidedPositionWitness(
                                candidate=(
                                    candidate
                                ),
                                witness_action=(
                                    float(
                                        micro_action[
                                            0
                                        ]
                                    ),
                                    float(
                                        micro_action[
                                            1
                                        ]
                                    ),
                                ),
                                next_shot_distance=(
                                    next_shot_distance
                                ),
                            )
                        )

                        key = (
                            macro_action.target_ball,
                            macro_action.target_pocket,
                            next_target_ball,
                            next_pocket,
                            int(
                                np.floor(
                                    cue_after[0]
                                    / self.dedup_distance
                                )
                            ),
                            int(
                                np.floor(
                                    cue_after[1]
                                    / self.dedup_distance
                                )
                            ),
                        )

                        previous = (
                            best_by_key.get(
                                key
                            )
                        )

                        if (
                            previous is None
                            or next_shot_distance
                            < previous.next_shot_distance
                        ):
                            best_by_key[
                                key
                            ] = witness

        witnesses = list(
            best_by_key.values()
        )

        witnesses.sort(
            key=lambda item: (
                item.next_shot_distance
            )
        )

        return witnesses[
            : self.top_k
        ]

    def generate(
        self,
        state,
    ):
        return [
            item.candidate
            for item
            in self.generate_with_witnesses(
                state
            )
        ]
