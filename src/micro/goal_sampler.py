from dataclasses import dataclass

import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.macro.candidate_generator import (
    MacroCandidateGenerator,
)

from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)

from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)


@dataclass
class GoalTrainingSample:
    observation: np.ndarray
    goal: TacticalGoal
    macro_action: MacroAction


class MixedTacticalGoalSampler:
    """
    Goal typeを先に選び、
    そのGoal type内から候補を選択する。

    これにより、
    POSITION_ATTACK候補数が多いことによる
    sampling biasを防ぐ。

    position_mode:
        "ideal"
            従来の幾何的な理想位置から
            POSITION Goalを生成する。

        "outcome"
            coarseなMicro actionをFastFizで試し、
            実際に到達できた手球位置から
            POSITION Goalを生成する。
    """

    def __init__(
        self,
        position_probability: float = 0.5,
        position_mode: str = "ideal",
        outcome_grid_size: int = 5,
        outcome_top_k: int = 8,
        outcome_dedup_distance: float = 0.10,
    ):
        if not (
            0.0
            <= position_probability
            <= 1.0
        ):
            raise ValueError(
                "position_probability must "
                "be between 0 and 1."
            )

        if position_mode not in (
            "ideal",
            "outcome",
        ):
            raise ValueError(
                "position_mode must be "
                "'ideal' or 'outcome'."
            )

        self.position_probability = (
            float(position_probability)
        )

        self.position_mode = (
            position_mode
        )

        self.direct_generator = (
            MacroCandidateGenerator()
        )

        if (
            position_mode
            == "ideal"
        ):
            self.position_generator = (
                PositionAttackCandidateGenerator()
            )

        else:
            self.position_generator = (
                OutcomeGuidedPositionCandidateGenerator(
                    grid_size=(
                        outcome_grid_size
                    ),
                    top_k=(
                        outcome_top_k
                    ),
                    dedup_distance=(
                        outcome_dedup_distance
                    ),
                )
            )

    def sample_goal_type(
        self,
        rng,
    ) -> TacticalGoalType:

        if (
            rng.random()
            < self.position_probability
        ):
            return (
                TacticalGoalType.POSITION_ATTACK
            )

        return (
            TacticalGoalType.DIRECT_ATTACK
        )

    def sample(
        self,
        env,
        rng,
        seed,
        max_attempts=100,
    ) -> GoalTrainingSample:

        goal_type = (
            self.sample_goal_type(rng)
        )

        for attempt in range(
            max_attempts
        ):
            # reset()内部ではMacroActionを参照するため、
            # reset用の仮Actionを設定しておく。
            placeholder_action = MacroAction(
                strategy=Strategy.ATTACK,
                target_ball=0,
                target_pocket=0,
            )

            env.set_macro_action(
                placeholder_action
            )

            env.reset(
                seed=seed + attempt
            )

            state = (
                self._build_game_state(
                    env
                )
            )

            if (
                goal_type
                == TacticalGoalType.DIRECT_ATTACK
            ):
                candidates = (
                    self.direct_generator
                    .generate(state)
                )

            elif (
                goal_type
                == TacticalGoalType.POSITION_ATTACK
            ):
                candidates = (
                    self.position_generator
                    .generate(state)
                )

            else:
                raise RuntimeError(
                    f"Unsupported goal type: "
                    f"{goal_type}"
                )

            if len(candidates) == 0:
                continue

            index = int(
                rng.integers(
                    0,
                    len(candidates),
                )
            )

            candidate = candidates[index]

            # 実際に実行するMacroActionへ更新
            env.set_macro_action(
                candidate.action
            )

            observation = (
                env.get_physical_observation()
            )

            return GoalTrainingSample(
                observation=observation,
                goal=candidate.goal,
                macro_action=candidate.action,
            )

        raise RuntimeError(
            "Could not sample a feasible "
            f"{goal_type.value} goal."
        )

    def _build_game_state(
        self,
        env,
    ) -> GameState:

        snapshot = env.sim.snapshot()

        return GameState(
            ball_positions=(
                snapshot.positions.copy()
            ),

            score=np.zeros(
                2,
                dtype=np.float32,
            ),

            current_player=0,

            ball_pocket_indices=(
                snapshot.pocket_indices.copy()
            ),
        )
