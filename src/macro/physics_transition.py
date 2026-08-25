import numpy as np

from src.common.types import GameState
from src.env.micro_billiard_env import MicroBilliardEnv

from src.macro.transition import (
    MacroTransitionResult,
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
        # Macro state -> Micro observation
        # --------------------------------

        observation = (
            self.env.reset_to_game_state(
                game_state=state,
                macro_action=action,
            )
        )

        # --------------------------------
        # Micro SAC
        # --------------------------------

        micro_action = (
            self.micro_agent.select_action(
                observation,
                deterministic=self.deterministic,
            )
        )

        # --------------------------------
        # FastFizで1ショット
        # --------------------------------

        (
            _,
            reward,
            _micro_terminated,
            _truncated,
            info,
        ) = self.env.step(
            micro_action
        )

        # --------------------------------
        # Physical state -> GameState
        # --------------------------------

        snapshot = self.env.sim.snapshot()

        next_positions = (
            snapshot.positions.copy()
        )

        next_pocket_indices = (
            snapshot.pocket_indices.copy()
        )

        physical_state = GameState(
            ball_positions=next_positions,

            score=np.asarray(
                state.score,
                dtype=np.float32,
            ).copy(),

            current_player=state.current_player,

            ball_pocket_indices=(
                next_pocket_indices
            ),
        )

        # ------------------------------------------------
        # Game rules
        # ------------------------------------------------

        if self.game_rules is not None:

            game_result = self.game_rules.apply(
                previous_state=state,
                physical_state=physical_state,
                shot_info=info,
            )

            return MacroTransitionResult(
                next_state=game_result.next_state,
                reward=game_result.reward,
                terminated=game_result.terminated,
            )


        # ------------------------------------------------
        # 後方互換:
        # GameRules未指定時は従来のMicro rewardを使用
        # ------------------------------------------------

        return MacroTransitionResult(
            next_state=physical_state,
            reward=float(reward),
            terminated=False,
        )