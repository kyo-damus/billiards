import numpy as np
import torch

from src.common.types import GameState

from src.controller.hierarchical_agent import (
    HierarchicalAgent,
    HierarchicalStepResult,
)

from src.game.simple_rules import SimplifiedGameRules

from src.macro.action_generator import MacroActionGenerator
from src.macro.mcts import MCTS
from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)
from src.macro.policy import MacroPolicyNetwork
from src.macro.value import MacroValueNetwork

from src.micro.sac import SACAgent


def make_initial_state():
    """
    直接ショット候補が存在する簡単な初期盤面。
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


def test_full_hierarchical_agent_smoke():
    """
    GameState
        ↓
    MacroActionGenerator
        ↓
    Policy / Value / MCTS
        ↓
    MacroAction
        ↓
    Micro SAC
        ↓
    FastFiz
        ↓
    GameRules
        ↓
    Next GameState

    が1回最後まで通ることを確認する。
    """

    np.random.seed(0)
    torch.manual_seed(0)

    # --------------------------------
    # Macro
    # --------------------------------

    action_generator = MacroActionGenerator()

    policy = MacroPolicyNetwork()
    value = MacroValueNetwork()

    # --------------------------------
    # Micro
    # --------------------------------

    # 学習性能を見るテストではないので、
    # 未学習SACを決定論的に使用する。
    micro_agent = SACAgent(
        state_dim=12,
        action_dim=2,
        device="cpu",
    )

    # --------------------------------
    # Physics + Game Rules
    # --------------------------------

    transition = PhysicsMacroTransitionModel(
        micro_agent=micro_agent,
        deterministic=True,
        game_rules=SimplifiedGameRules(),
    )

    # --------------------------------
    # MCTS
    # --------------------------------

    mcts = MCTS(
        policy_network=policy,
        value_network=value,
        action_generator=action_generator,
        transition_model=transition,

        # smoke testなので極小
        simulations=2,

        device="cpu",
    )

    # --------------------------------
    # Hierarchical Agent
    # --------------------------------

    agent = HierarchicalAgent(
        mcts=mcts,
        transition_model=transition,
    )

    initial_state = make_initial_state()

    # 実行前にMacro候補が存在することを確認
    feasible_actions = (
        action_generator.generate(
            initial_state
        )
    )

    assert len(feasible_actions) > 0

    # =================================
    # Full hierarchical step
    # =================================

    result = agent.step(
        initial_state
    )

    assert isinstance(
        result,
        HierarchicalStepResult,
    )

    # MCTSが選んだ行動が、
    # 初期盤面で生成された候補のいずれか
    assert (
        result.macro_action
        in feasible_actions
    )

    # FastFizを通った次状態
    assert isinstance(
        result.next_state,
        GameState,
    )

    assert (
        result.next_state.ball_positions.shape
        == (3, 2)
    )

    assert (
        result.next_state.ball_pocket_indices.shape
        == (3,)
    )

    assert (
        result.next_state.score.shape
        == (2,)
    )

    assert isinstance(
        result.reward,
        float,
    )

    assert isinstance(
        result.terminated,
        bool,
    )

    # 初期GameStateそのものを書き換えていない
    assert result.next_state is not initial_state
    