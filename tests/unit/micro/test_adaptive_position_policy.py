import numpy as np
import torch
import torch.nn as nn

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.micro.adaptive_position_policy import (
    AdaptivePositionMicroPolicy,
)


class DummyAgent:

    def __init__(self, action):
        self.action = np.asarray(
            action,
            dtype=np.float32,
        )
        self.device = torch.device(
            "cpu"
        )

    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        return self.action.copy()

    def build_conditioned_state(
        self,
        observation,
        goal,
    ):
        return np.zeros(
            34,
            dtype=np.float32,
        )


class DummyGate(nn.Module):

    def __init__(self, logits):
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.as_tensor(
                logits,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        return (
            self.fixed_logits
            .unsqueeze(0)
            .expand(
                x.shape[0],
                -1,
            )
        )


def make_position_goal():
    return TacticalGoal(
        goal_type=(
            TacticalGoalType
            .POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
        cue_target_position=(
            0.5,
            1.0,
        ),
        cue_target_radius=0.10,
        next_target_ball=1,
        next_target_pocket=5,
    )


def test_position_policy_blends_selected_lambda():

    policy = AdaptivePositionMicroPolicy(
        base_agent=DummyAgent(
            [0.0, 0.0]
        ),
        bc_agent=DummyAgent(
            [1.0, -1.0]
        ),
        gate=DummyGate(
            [-2.0, 3.0]
        ),
        lambdas=[
            0.0,
            0.5,
        ],
        device="cpu",
    )

    action = policy.select_action(
        np.zeros(
            9,
            dtype=np.float32,
        ),
        make_position_goal(),
        deterministic=True,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.5, -0.5],
            dtype=np.float32,
        ),
    )

    assert (
        policy.last_selection
        is not None
    )

    assert (
        policy.last_selection
        .blend_lambda
        == 0.5
    )


def test_direct_goal_preserves_base_actor():

    policy = AdaptivePositionMicroPolicy(
        base_agent=DummyAgent(
            [0.25, -0.25]
        ),
        bc_agent=DummyAgent(
            [1.0, 1.0]
        ),
        gate=DummyGate(
            [-2.0, 3.0]
        ),
        lambdas=[
            0.0,
            1.0,
        ],
        device="cpu",
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType
            .DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
    )

    action = policy.select_action(
        np.zeros(
            9,
            dtype=np.float32,
        ),
        goal,
        deterministic=True,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.25, -0.25],
            dtype=np.float32,
        ),
    )

    assert (
        policy.last_selection
        is None
    )


def test_low_confidence_falls_back_to_base():

    policy = AdaptivePositionMicroPolicy(
        base_agent=DummyAgent(
            [0.2, -0.2]
        ),
        bc_agent=DummyAgent(
            [1.0, 1.0]
        ),
        gate=DummyGate(
            [0.0, 0.1]
        ),
        lambdas=[
            0.0,
            1.0,
        ],
        selection_threshold=0.9,
        device="cpu",
    )

    action = policy.select_action(
        np.zeros(
            9,
            dtype=np.float32,
        ),
        make_position_goal(),
        deterministic=True,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.2, -0.2],
            dtype=np.float32,
        ),
    )

    assert (
        policy.last_selection
        .blend_lambda
        == 0.0
    )
