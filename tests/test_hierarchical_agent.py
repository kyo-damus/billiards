import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.controller.hierarchical_agent import (
    HierarchicalAgent,
    HierarchicalStepResult,
)

from src.macro.transition import (
    MacroTransitionResult,
)


def make_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.5, 0.3],
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


class DummyMCTS:
    def select_action(self, state):
        return MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=2,
        )


class DummyTransition:
    def step(
        self,
        state,
        action,
    ):
        next_state = GameState(
            ball_positions=(
                state.ball_positions.copy()
            ),
            score=state.score.copy(),
            current_player=state.current_player,
        )

        return MacroTransitionResult(
            next_state=next_state,
            reward=1.0,
            terminated=False,
        )


def test_select_macro_action():
    agent = HierarchicalAgent(
        mcts=DummyMCTS(),
        transition_model=DummyTransition(),
    )

    action = agent.select_macro_action(
        make_state()
    )

    assert action.strategy == Strategy.ATTACK
    assert action.target_ball == 0
    assert action.target_pocket == 2


def test_execute_macro_action():
    agent = HierarchicalAgent(
        mcts=DummyMCTS(),
        transition_model=DummyTransition(),
    )

    state = make_state()

    action = agent.select_macro_action(state)

    result = agent.execute_macro_action(
        state,
        action,
    )

    assert isinstance(
        result,
        MacroTransitionResult,
    )

    assert result.reward == 1.0


def test_hierarchical_step():
    agent = HierarchicalAgent(
        mcts=DummyMCTS(),
        transition_model=DummyTransition(),
    )

    result = agent.step(
        make_state()
    )

    assert isinstance(
        result,
        HierarchicalStepResult,
    )

    assert result.macro_action.target_ball == 0
    assert result.macro_action.target_pocket == 2

    assert result.next_state.ball_positions.shape == (
        3,
        2,
    )

    assert result.reward == 1.0
    assert result.terminated is False
    