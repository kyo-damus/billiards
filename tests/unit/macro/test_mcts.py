import numpy as np
import torch
import torch.nn as nn

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.macro.encoding import (
    MACRO_STATE_DIM,
    NUM_MACRO_ACTIONS,
)

from src.macro.mcts import MCTS

from src.macro.transition import (
    MacroTransitionResult,
)


def make_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.5, 0.4],
                [0.5, 0.8],
                [0.9, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        current_player=0,
    )


class DummyActionGenerator:

    def generate(self, state):
        return [
            MacroAction(
                strategy=Strategy.ATTACK,
                target_ball=0,
                target_pocket=0,
            ),
            MacroAction(
                strategy=Strategy.ATTACK,
                target_ball=0,
                target_pocket=1,
            ),
        ]


class DummyPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        # device取得用
        self.dummy = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, state):
        batch_size = state.shape[0]

        return torch.zeros(
            batch_size,
            NUM_MACRO_ACTIONS,
            device=state.device,
        )


class DummyValue(nn.Module):

    def __init__(self):
        super().__init__()

        self.dummy = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, state):
        return torch.zeros(
            state.shape[0],
            1,
            device=state.device,
        )


class DummyTransition:

    def step(
        self,
        state,
        action,
    ):
        # pocket 0を選べば高報酬、
        # pocket 1なら低報酬。

        if action.target_pocket == 0:
            reward = 1.0
        else:
            reward = 0.0

        return MacroTransitionResult(
            next_state=state,
            reward=reward,
            terminated=True,
        )


def test_mcts_search_runs():

    mcts = MCTS(
        policy_network=DummyPolicy(),
        value_network=DummyValue(),
        action_generator=DummyActionGenerator(),
        transition_model=DummyTransition(),
        simulations=20,
        device="cpu",
    )

    root = mcts.search(
        make_state()
    )

    assert root.visit_count == 20
    assert len(root.children) == 2


def test_mcts_prefers_higher_reward_action():

    mcts = MCTS(
        policy_network=DummyPolicy(),
        value_network=DummyValue(),
        action_generator=DummyActionGenerator(),
        transition_model=DummyTransition(),
        simulations=30,
        device="cpu",
    )

    action = mcts.select_action(
        make_state()
    )

    assert action.target_ball == 0
    assert action.target_pocket == 0
    