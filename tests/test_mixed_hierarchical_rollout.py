"""Fast unit tests for multi-turn state and stop-condition handling."""

import numpy as np
import pytest

from src.common.types import GameState, MacroAction, Strategy
from src.controller.hierarchical_agent import (
    HierarchicalAgent,
    HierarchicalStepResult,
)
from src.controller.mixed_hierarchical_rollout import (
    MixedHierarchicalRollout,
)
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.macro.mcts import NoFeasibleMacroActionError
from src.macro.transition import MacroTransitionResult


def make_state(
    *,
    player=0,
    score=(0.0, 0.0),
    pockets=(-1, -1, -1),
):
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.asarray(score, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.asarray(pockets, dtype=np.int32),
    )


def direct_candidate():
    return MacroCandidate(
        action=MacroAction(
            Strategy.ATTACK, target_ball=0, target_pocket=0
        ),
        goal=TacticalGoal(
            TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
    )


class CountingMCTS:
    def __init__(self):
        self.current_players = []

    def select_action(self, state):
        self.current_players.append(state.current_player)
        return direct_candidate()


class CountingTransition:
    def __init__(self):
        self.inputs = []

    def step(self, state, action):
        self.inputs.append(state)
        # First turn: pocket an object and keep the turn.
        # Second turn: miss and pass the turn to opponent.
        index = len(self.inputs)
        if index == 1:
            next_state = make_state(
                player=0, score=(1.0, 0.0), pockets=(-1, 0, -1)
            )
        else:
            next_state = make_state(
                player=1, score=(1.0, 0.0), pockets=(-1, 0, -1)
            )
        return MacroTransitionResult(
            next_state=next_state,
            reward=1.0 if index == 1 else 0.0,
            terminated=False,
        )


def test_rollout_replans_from_executed_next_state():
    planner = CountingMCTS()
    transition = CountingTransition()
    agent = HierarchicalAgent(
        mcts=planner, transition_model=transition
    )

    result = MixedHierarchicalRollout(agent).run(
        make_state(), max_turns=3
    )

    assert len(result.records) == 3
    assert planner.current_players == [0, 0, 1]
    assert transition.inputs[1] is not transition.inputs[0]
    assert transition.inputs[1].score.tolist() == [1.0, 0.0]
    assert transition.inputs[2].current_player == 1
    assert result.stop_reason == "max_turns"
    assert not result.terminated
    assert result.final_state.score.tolist() == [1.0, 0.0]


def test_rollout_handles_no_feasible_root_without_faking_terminal():
    class NoChoices:
        def step(self, state):
            raise NoFeasibleMacroActionError(
                "No feasible MacroActions from root state."
            )

    initial = make_state()
    result = MixedHierarchicalRollout(NoChoices()).run(
        initial, max_turns=2
    )
    assert not result.records
    assert result.final_state is initial
    assert result.stop_reason == "no_feasible_macro_candidates"
    assert not result.terminated


def test_rollout_stops_on_terminal_execution():
    class OneShot:
        def step(self, state):
            return HierarchicalStepResult(
                macro_action=direct_candidate(),
                next_state=make_state(
                    pockets=(-1, 0, 3), score=(2.0, 0.0)
                ),
                reward=2.0,
                terminated=True,
            )

    result = MixedHierarchicalRollout(OneShot()).run(
        make_state(), max_turns=3
    )
    assert result.terminated
    assert result.stop_reason == "game_terminated"
    assert len(result.records) == 1
    assert result.final_state.score.tolist() == [2.0, 0.0]


def test_rollout_rejects_terminal_initial_state_and_bad_horizon():
    class Uncalled:
        def step(self, state):
            raise AssertionError("Must not plan from terminal state.")

    rollout = MixedHierarchicalRollout(Uncalled())
    result = rollout.run(
        make_state(pockets=(-1, 0, 3)),
        max_turns=3,
    )
    assert result.terminated
    assert result.stop_reason == "initial_terminal_state"
    with pytest.raises(ValueError, match="max_turns"):
        rollout.run(make_state(), max_turns=0)
