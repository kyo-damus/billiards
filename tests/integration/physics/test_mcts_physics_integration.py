import numpy as np
import torch

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.game.simple_rules import (
    SimplifiedGameRules,
)

from src.macro.mcts import MCTS
from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)

from src.macro.policy import MacroPolicyNetwork
from src.macro.value import MacroValueNetwork


class FixedMacroActionGenerator:
    """
    Integration test用。

    テーブル上に残っている的球から
    決定論的に1つだけMacroActionを生成する。
    """

    def generate(self, state):

        pockets = state.ball_pocket_indices

        # object ball 0 が残っている
        if pockets[1] < 0:
            return [
                MacroAction(
                    strategy=Strategy.ATTACK,
                    target_ball=0,
                    target_pocket=1,
                )
            ]

        # object ball 0 が落ちていて、
        # object ball 1 が残っている
        if pockets[2] < 0:
            return [
                MacroAction(
                    strategy=Strategy.ATTACK,
                    target_ball=1,
                    target_pocket=5,
                )
            ]

        return []

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)

class LowPowerMicroAgent:
    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        assert observation.shape == (
            PHYSICAL_OBSERVATION_DIM,
        )

        return np.array(
            [-1.0, 0.0],
            dtype=np.float32,
        )


def make_integration_state():
    """
    Wポケット方向へほぼ一直線の簡単な盤面。

    power=0.5なので、
    1ショットで即ゲーム終了しにくい配置にする。
    """

    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.80, 1.118],

                # object ball 0
                [0.40, 1.118],

                # object ball 1
                [0.80, 0.40],
            ],
            dtype=np.float32,
        ),

        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_mcts_reaches_depth_two_with_real_physics():
    torch.manual_seed(0)

    policy = MacroPolicyNetwork()
    value = MacroValueNetwork()

    transition = PhysicsMacroTransitionModel(
        micro_agent=LowPowerMicroAgent(),
        deterministic=True,
        game_rules=SimplifiedGameRules(),
    )

    mcts = MCTS(
        policy_network=policy,
        value_network=value,
        action_generator=FixedMacroActionGenerator(),
        transition_model=transition,

        # 2手先を見るだけなので極小
        simulations=3,

        device="cpu",
    )

    root = mcts.search(
        make_integration_state()
    )

    # ----------------------------
    # depth 1
    # ----------------------------

    assert len(root.children) == 1

    first_child = next(
        iter(root.children.values())
    )

    # FastFiz transitionが実際に実行された
    assert first_child.state is not None

    assert (
        first_child.state.ball_positions.shape
        == (3, 2)
    )

    # ----------------------------
    # depth 2
    # ----------------------------

    assert first_child.expanded
    assert len(first_child.children) == 1

    second_action, second_child = next(
        iter(first_child.children.items())
    )

    # MCTSが次状態を見て、
    # テーブル上に存在する球を選んでいる
    target_state_index = (
        second_action.target_ball + 1
    )

    assert (
        first_child.state.ball_pocket_indices[
            target_state_index
        ]
        < 0
    )

    # 2回目のFastFiz transitionも実行された
    assert second_child.state is not None

    assert (
        second_child.state.ball_positions.shape
        == (3, 2)
    )

    assert second_child.state is not root.state
    