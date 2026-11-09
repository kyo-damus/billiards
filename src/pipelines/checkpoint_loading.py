"""Checkpoint loading shared by Micro evaluation and Macro pipelines.

Keep the original checkpoint metadata and state-dict behavior unchanged.
"""

import torch

from src.micro.goal_conditioned_sac import GoalConditionedSACAgent
from src.micro.position_residual_policy import PositionResidualNetwork


def load_base_agent(
    checkpoint_path,
    device,
):
    agent = GoalConditionedSACAgent(
        action_dim=2,
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    agent.agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.agent.critic.load_state_dict(
        checkpoint["critic"]
    )

    if "critic_target" in checkpoint:
        agent.agent.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

    return agent


def load_residual_network(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = (
        "residual_network",
        "conditioned_state_dim",
        "action_dim",
        "hidden_dim",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Residual checkpoint is missing keys: "
            + ", ".join(missing)
        )

    network = PositionResidualNetwork(
        conditioned_state_dim=int(
            checkpoint[
                "conditioned_state_dim"
            ]
        ),
        action_dim=int(
            checkpoint[
                "action_dim"
            ]
        ),
        hidden_dim=int(
            checkpoint[
                "hidden_dim"
            ]
        ),
        zero_init_output=False,
    ).to(
        device
    )

    network.load_state_dict(
        checkpoint["residual_network"]
    )
    network.eval()

    return network, checkpoint
