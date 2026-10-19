import numpy as np
import torch
import torch.nn as nn

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
    PositionResidualNetwork,
    compose_residual_action,
)
from src.micro.fixed_scale_position_residual_policy import (
    FixedScalePositionResidualPolicy,
)


class DummyBaseAgent:

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


class ConstantResidual(nn.Module):

    def __init__(self, residual):
        super().__init__()

        self.register_buffer(
            "residual",
            torch.as_tensor(
                residual,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        conditioned_state,
        base_action,
    ):
        return (
            self.residual
            .unsqueeze(0)
            .expand(
                conditioned_state.shape[0],
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


def test_zero_initialized_network_preserves_base_action():

    network = PositionResidualNetwork(
        conditioned_state_dim=34,
        action_dim=2,
        hidden_dim=32,
        zero_init_output=True,
    )

    policy = PositionResidualMicroPolicy(
        base_agent=DummyBaseAgent(
            [0.25, -0.25]
        ),
        residual_network=network,
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
            [0.25, -0.25],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        policy.last_selection.residual_action,
        np.zeros(
            2,
            dtype=np.float32,
        ),
    )


def test_position_policy_adds_residual_and_clips():

    policy = PositionResidualMicroPolicy(
        base_agent=DummyBaseAgent(
            [0.8, -0.8]
        ),
        residual_network=ConstantResidual(
            [0.5, -0.5]
        ),
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
            [1.0, -1.0],
            dtype=np.float32,
        ),
    )


def test_direct_goal_preserves_base_action():

    policy = PositionResidualMicroPolicy(
        base_agent=DummyBaseAgent(
            [0.2, -0.4]
        ),
        residual_network=ConstantResidual(
            [0.5, 0.5]
        ),
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
            [0.2, -0.4],
            dtype=np.float32,
        ),
    )

    assert (
        policy.last_selection
        is None
    )


def test_compose_residual_action_requires_matching_shape():

    try:
        compose_residual_action(
            np.zeros(
                2,
                dtype=np.float32,
            ),
            np.zeros(
                3,
                dtype=np.float32,
            ),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError for mismatched shapes."
        )



def test_fixed_scale_wrapper_applies_power_only_residual():

    raw_policy = PositionResidualMicroPolicy(
        base_agent=DummyBaseAgent(
            [0.2, -0.3]
        ),
        residual_network=ConstantResidual(
            [0.8, 0.4]
        ),
        device="cpu",
    )

    policy = FixedScalePositionResidualPolicy(
        residual_policy=raw_policy,
        residual_scale=0.75,
        power_scale=1.0,
        angle_scale=0.0,
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
            [0.8, -0.3],
            dtype=np.float32,
        ),
    )
