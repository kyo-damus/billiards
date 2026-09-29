import numpy as np
import torch
import torch.nn as nn

from src.common.types import GameState
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.common.types import MacroAction, Strategy
from src.macro.mcts import MCTS
from src.macro.transition import MacroTransitionResult


class RecordingGenerator:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = 0

    def generate(self, state):
        self.calls += 1
        return [self.candidate]


class ZeroCandidatePolicy(nn.Module):
    def forward(self, state, candidates):
        return torch.zeros(
            candidates.shape[0],
            dtype=torch.float32,
            device=candidates.device,
        )


class ZeroValue(nn.Module):
    def forward(self, state):
        return torch.zeros(
            1,
            dtype=torch.float32,
            device=state.device,
        )


class DummyTransition:
    def step(self, state, action):
        return MacroTransitionResult(
            next_state=GameState(
                ball_positions=(
                    state.ball_positions.copy()
                ),
                score=state.score.copy(),
                current_player=state.current_player,
                ball_pocket_indices=(
                    state.ball_pocket_indices.copy()
                ),
            ),
            reward=0.0,
            terminated=False,
        )


def make_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.8, 1.1],
                [0.4, 1.1],
                [0.8, 0.8],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def make_candidate(pocket):
    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=pocket,
    )

    goal = TacticalGoal(
        goal_type=TacticalGoalType.DIRECT_ATTACK,
        target_ball=0,
        target_pocket=pocket,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


def test_mcts_uses_separate_root_generator():

    root_generator = RecordingGenerator(
        make_candidate(0)
    )

    tree_generator = RecordingGenerator(
        make_candidate(1)
    )

    mcts = MCTS(
        policy_network=ZeroCandidatePolicy(),
        value_network=ZeroValue(),
        action_generator=tree_generator,
        root_action_generator=root_generator,
        transition_model=DummyTransition(),
        simulations=1,
        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    assert root_generator.calls == 1
    assert tree_generator.calls == 1

    root_choices = list(
        root.children.keys()
    )

    assert (
        root_choices[0]
        .action.target_pocket
        == 0
    )
