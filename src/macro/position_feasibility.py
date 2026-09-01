from dataclasses import dataclass

import numpy as np

from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)


@dataclass
class PositionFeasibilityResult:
    """
    有限個のaction gridを探索した結果。

    reachable=False は数学的な到達不能を
    証明するものではなく、
    指定grid内で到達例が見つからなかったことを表す。
    """

    reachable: bool
    pot_found: bool

    best_cue_distance: float

    tested_actions: int

    best_action: np.ndarray | None


class PositionGoalFeasibilityChecker:
    """
    FastFizを使ってPOSITION_ATTACKが
    現在のMicro action spaceで実現可能そうか調べる。

    action:
        [normalized_power,
         normalized_angle_offset]
    """

    def __init__(
        self,
        grid_size=9,
    ):
        if grid_size < 2:
            raise ValueError(
                "grid_size must be >= 2"
            )

        self.grid_size = int(
            grid_size
        )

        self.env = MicroBilliardEnv()

    def check(
        self,
        state,
        candidate: MacroCandidate,
    ) -> PositionFeasibilityResult:

        goal = candidate.goal

        if (
            goal.goal_type
            != TacticalGoalType.POSITION_ATTACK
        ):
            raise ValueError(
                "PositionGoalFeasibilityChecker "
                "requires POSITION_ATTACK."
            )

        if (
            goal.cue_target_position
            is None
        ):
            raise ValueError(
                "cue_target_position is required."
            )

        if (
            goal.cue_target_radius
            is None
        ):
            raise ValueError(
                "cue_target_radius is required."
            )

        target_position = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        # normalized SAC action
        values = np.linspace(
            -1.0,
            1.0,
            self.grid_size,
            dtype=np.float32,
        )

        best_distance = float("inf")
        best_action = None

        pot_found = False
        tested_actions = 0

        for power_action in values:

            for angle_action in values:

                action = np.array(
                    [
                        power_action,
                        angle_action,
                    ],
                    dtype=np.float32,
                )

                self.env.reset_to_game_state(
                    state,
                    candidate.action,
                )

                (
                    _,
                    _,
                    _,
                    _,
                    info,
                ) = self.env.step(
                    action
                )

                tested_actions += 1

                # 指定球→指定ポケットに
                # 成功したショットだけを見る
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

                pot_found = True

                snapshot = (
                    self.env.sim.snapshot()
                )

                cue_position = (
                    snapshot.positions[0]
                )

                distance = float(
                    np.linalg.norm(
                        cue_position
                        - target_position
                    )
                )

                if (
                    distance
                    < best_distance
                ):
                    best_distance = distance
                    best_action = action.copy()

                if (
                    distance
                    <= goal.cue_target_radius
                ):
                    return (
                        PositionFeasibilityResult(
                            reachable=True,
                            pot_found=True,
                            best_cue_distance=(
                                distance
                            ),
                            tested_actions=(
                                tested_actions
                            ),
                            best_action=(
                                action.copy()
                            ),
                        )
                    )

        return PositionFeasibilityResult(
            reachable=False,
            pot_found=pot_found,
            best_cue_distance=(
                best_distance
            ),
            tested_actions=(
                tested_actions
            ),
            best_action=best_action,
        )
        