from dataclasses import dataclass

import numpy as np

from src.common.types import GameState
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.action_generator import MacroActionGenerator


@dataclass(frozen=True)
class MacroLocalEvaluationConfig:
    """
    Macro候補の1-step局所評価に使う暫定重み。

    学習済みValueではなく、FastFizで観測できる
    ショット結果をMCTSのedge rewardへ変換する。
    """

    pot_reward: float = 1.0
    wrong_pocket_penalty: float = 0.5
    scratch_penalty: float = 1.0
    position_weight: float = 0.5
    next_shot_weight: float = 0.25


@dataclass(frozen=True)
class MacroLocalEvaluation:
    """
    1つのMacroCandidateをFastFizで実行した結果の局所評価。
    """

    target_potted: bool
    wrong_pocket: bool
    scratch: bool

    position_goal_satisfied: bool
    cue_distance: float | None
    position_quality: float | None

    next_shot_feasible: bool

    score: float


class MacroLocalEvaluator:
    """
    FastFizの1-shot結果からMacro用の局所rewardを計算する。

    v0:
      - 指定球を指定ポケットへ入れたか
      - wrong pocket
      - scratch
      - POSITION_ATTACKの手球配置品質

    次ショット可能性は後続段階で追加する。
    """

    def __init__(
        self,
        config: MacroLocalEvaluationConfig | None = None,
        action_generator=None,
    ):
        self.config = (
            config
            if config is not None
            else MacroLocalEvaluationConfig()
        )
        self.action_generator = (
            action_generator
            if action_generator is not None
            else MacroActionGenerator()
        )

    def evaluate(
        self,
        goal: TacticalGoal,
        state: GameState,
        shot_info: dict,
    ) -> MacroLocalEvaluation:

        target_potted = bool(
            shot_info.get(
                "success",
                self._target_potted(
                    goal,
                    state,
                ),
            )
        )

        scratch = bool(
            shot_info.get(
                "scratched",
                state.is_pocketed(0),
            )
        )

        wrong_pocket = bool(
            shot_info.get(
                "wrong_pocket",
                self._wrong_pocket(
                    goal,
                    state,
                ),
            )
        )

        (
            position_goal_satisfied,
            cue_distance,
            position_quality,
        ) = self._position_metrics(
            goal=goal,
            state=state,
            target_potted=target_potted,
            scratch=scratch,
        )

        next_shot_feasible = False

        if (
            target_potted
            and not scratch
        ):
            next_shot_feasible = bool(
                self.action_generator.generate(
                    state
                )
            )

        score = 0.0

        if (
            target_potted
            and not scratch
        ):
            score += self.config.pot_reward

        if wrong_pocket:
            score -= (
                self.config
                .wrong_pocket_penalty
            )

        if scratch:
            score -= (
                self.config
                .scratch_penalty
            )

        # POSITIONの配置加点は、
        # 指定球の正常な入球かつscratchなしの場合のみ。
        if (
            goal.goal_type
            == TacticalGoalType.POSITION_ATTACK
            and target_potted
            and not scratch
            and position_goal_satisfied
            and position_quality is not None
        ):
            score += (
                self.config.position_weight
                * position_quality
            )

        if (
            target_potted
            and not scratch
            and next_shot_feasible
        ):
            score += (
                self.config.next_shot_weight
            )

        return MacroLocalEvaluation(
            target_potted=target_potted,
            wrong_pocket=wrong_pocket,
            scratch=scratch,
            position_goal_satisfied=(
                position_goal_satisfied
            ),
            cue_distance=cue_distance,
            position_quality=position_quality,
            next_shot_feasible=(
                next_shot_feasible
            ),
            score=float(score),
        )

    @staticmethod
    def _target_potted(
        goal: TacticalGoal,
        state: GameState,
    ) -> bool:
        if (
            goal.target_ball is None
            or goal.target_pocket is None
        ):
            return False

        state_index = goal.target_ball + 1

        return bool(
            state.ball_pocket_indices[
                state_index
            ]
            == goal.target_pocket
        )

    @staticmethod
    def _wrong_pocket(
        goal: TacticalGoal,
        state: GameState,
    ) -> bool:
        if (
            goal.target_ball is None
            or goal.target_pocket is None
        ):
            return False

        state_index = goal.target_ball + 1
        pocket_index = int(
            state.ball_pocket_indices[
                state_index
            ]
        )

        return bool(
            pocket_index >= 0
            and pocket_index
            != goal.target_pocket
        )

    @staticmethod
    def _position_metrics(
        goal: TacticalGoal,
        state: GameState,
        target_potted: bool,
        scratch: bool,
    ) -> tuple[
        bool,
        float | None,
        float | None,
    ]:
        if (
            goal.goal_type
            != TacticalGoalType.POSITION_ATTACK
        ):
            return False, None, None

        if goal.cue_target_position is None:
            raise ValueError(
                "POSITION_ATTACK requires "
                "cue_target_position."
            )

        if (
            goal.cue_target_radius is None
            or goal.cue_target_radius <= 0.0
        ):
            raise ValueError(
                "POSITION_ATTACK requires a positive "
                "cue_target_radius."
            )

        # scratch後のcue位置は配置品質として扱わない。
        if scratch:
            return False, None, 0.0

        cue_position = np.asarray(
            state.ball_positions[0],
            dtype=np.float32,
        )

        cue_target = np.asarray(
            goal.cue_target_position,
            dtype=np.float32,
        )

        cue_distance = float(
            np.linalg.norm(
                cue_position
                - cue_target
            )
        )

        radius = float(
            goal.cue_target_radius
        )

        position_quality = float(
            np.exp(
                -(
                    cue_distance
                    / radius
                ) ** 2
            )
        )

        position_goal_satisfied = bool(
            target_potted
            and cue_distance <= radius
        )

        return (
            position_goal_satisfied,
            cue_distance,
            position_quality,
        )
