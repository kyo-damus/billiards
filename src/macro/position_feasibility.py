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
    有限個のaction候補を探索した結果。

    reachable=False は数学的な到達不能を
    証明するものではなく、
    指定したaction集合内で到達例が
    見つからなかったことを表す。
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
        """
        power x angle の等間隔gridで評価する。
        """

        values = np.linspace(
            -1.0,
            1.0,
            self.grid_size,
            dtype=np.float32,
        )

        actions = np.array(
            [
                [
                    power_action,
                    angle_action,
                ]
                for power_action in values
                for angle_action in values
            ],
            dtype=np.float32,
        )

        return self.check_actions(
            state,
            candidate,
            actions,
        )

    def check_actions(
        self,
        state,
        candidate: MacroCandidate,
        actions,
    ) -> PositionFeasibilityResult:
        """
        任意に与えたnormalized action集合で評価する。

        coarse gridで候補を選び、
        別のhold-out action集合で再評価する用途にも使う。
        """

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

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        if (
            actions.ndim != 2
            or actions.shape[1] != 2
        ):
            raise ValueError(
                "actions must have shape (N, 2)."
            )

        if len(actions) == 0:
            raise ValueError(
                "actions must not be empty."
            )

        if np.any(
            actions < -1.0
        ) or np.any(
            actions > 1.0
        ):
            raise ValueError(
                "actions must be normalized "
                "to [-1, 1]."
            )

        target_position = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        best_distance = float("inf")
        best_action = None

        pot_found = False
        tested_actions = 0

        for action in actions:

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

            # 指定球→指定ポケットに成功し、
            # scratchしていないショットだけを見る。
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
                best_action = (
                    action.copy()
                )

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
