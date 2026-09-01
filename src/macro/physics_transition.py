import numpy as np

from src.common.types import GameState
from src.env.micro_billiard_env import MicroBilliardEnv

from src.macro.transition import (
    MacroTransitionResult,
)

from src.goal.adapter import (
    macro_action_to_tactical_goal,
)

from src.macro.candidate import (
    MacroCandidate,
)

class PhysicsMacroTransitionModel:
    """
    MacroActionをMicro SAC + FastFizで
    実際に1ショット実行する状態遷移モデル。

    MCTS
      ↓ MacroAction
    Micro policy
      ↓ continuous action
    FastFiz
      ↓
    next GameState
    """

    def __init__(
        self,
        micro_agent,
        env=None,
        deterministic=True,
        game_rules=None,
    ):
        self.micro_agent = micro_agent
        self.game_rules = game_rules

        if env is None:
            env = MicroBilliardEnv()

        self.env = env

        # MCTS内部では同じ状態・行動に対して
        # なるべく決定論的な結果にする
        self.deterministic = deterministic

    def step(
        self,
        state,
        action,
    ) -> MacroTransitionResult:

        # --------------------------------
        # MacroCandidate / MacroAction
        # を共通形式へ変換
        # --------------------------------

        if isinstance(
            action,
            MacroCandidate,
        ):
            macro_action = action.action
            tactical_goal = action.goal

        else:
            macro_action = action

            tactical_goal = (
                macro_action_to_tactical_goal(
                    macro_action
                )
            )

        # --------------------------------
        # Physical state restore
        # --------------------------------

        self.env.reset_to_game_state(
            state,
            macro_action,
        )

        observation = (
            self.env.get_physical_observation()
        )

        # --------------------------------
        # Goal-conditioned Micro
        # --------------------------------

        micro_action = (
            self.micro_agent.select_action(
                observation,
                tactical_goal,
                deterministic=self.deterministic,
            )
        )

        # --------------------------------
        # FastFiz
        # --------------------------------

        (
            _,
            micro_reward,
            _,
            _,
            info,
        ) = self.env.step(
            micro_action
        )

        snapshot = self.env.sim.snapshot()

        physical_state = GameState(
            ball_positions=(
                snapshot.positions.copy()
            ),
            score=state.score.copy(),
            current_player=state.current_player,
            ball_pocket_indices=(
                snapshot.pocket_indices.copy()
            ),
        )

        # --------------------------------
        # Game Rules
        # --------------------------------

        if self.game_rules is not None:

            game_result = (
                self.game_rules.apply(
                    previous_state=state,
                    physical_state=physical_state,
                    shot_info=info,
                )
            )

            return MacroTransitionResult(
                next_state=(
                    game_result.next_state
                ),
                reward=float(
                    game_result.reward
                ),
                terminated=bool(
                    game_result.terminated
                ),
            )

        # backward-compatible path
        return MacroTransitionResult(
            next_state=physical_state,
            reward=float(micro_reward),
            terminated=False,
        )