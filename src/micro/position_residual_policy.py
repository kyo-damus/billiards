from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.goal.types import TacticalGoalType


@dataclass(frozen=True)
class PositionResidualSelection:
    """
    POSITION用の補正結果を保持する。

    base_action:
        既存のGoal-conditioned Micro actorの出力。

    residual_action:
        Position Residual Networkが出力した補正量。

    final_action:
        base_action + residual_action を [-1, 1] にclipした最終action。
    """

    base_action: np.ndarray
    residual_action: np.ndarray
    final_action: np.ndarray


class PositionResidualNetwork(nn.Module):
    """
    Base actionに対する補正量だけを出力する小規模MLP。

    input:
        conditioned_state = observation + encoded TacticalGoal
        base_action

    output:
        residual action [delta_power, delta_angle_offset]

    最終層を0初期化することで、学習前はResidual=0となり、
    Base Policyの挙動をそのまま保つ。
    """

    def __init__(
        self,
        conditioned_state_dim=34,
        action_dim=2,
        hidden_dim=128,
        zero_init_output=True,
    ):
        super().__init__()

        self.conditioned_state_dim = int(
            conditioned_state_dim
        )
        self.action_dim = int(
            action_dim
        )

        input_dim = (
            self.conditioned_state_dim
            + self.action_dim
        )

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )
        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        self.out = nn.Linear(
            hidden_dim,
            self.action_dim,
        )

        if zero_init_output:
            nn.init.zeros_(
                self.out.weight
            )
            nn.init.zeros_(
                self.out.bias
            )

    def forward(
        self,
        conditioned_state,
        base_action,
    ):
        x = torch.cat(
            [
                conditioned_state,
                base_action,
            ],
            dim=-1,
        )

        x = F.relu(
            self.fc1(x)
        )
        x = F.relu(
            self.fc2(x)
        )

        return self.out(x)


def compose_residual_action(
    base_action,
    residual_action,
):
    """
    Base actionとResidualを加算し、
    Micro actionの正規化範囲 [-1, 1] に収める。
    """

    base = np.asarray(
        base_action,
        dtype=np.float32,
    )

    residual = np.asarray(
        residual_action,
        dtype=np.float32,
    )

    if base.shape != residual.shape:
        raise ValueError(
            "base_action and residual_action "
            "must have the same shape."
        )

    return np.clip(
        base + residual,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


class PositionResidualMicroPolicy:
    """
    DIRECT_ATTACK:
        既存Base Policyをそのまま使用する。

    POSITION_ATTACK:
        Base Policyを固定したまま、
        learned residualだけを加えてactionを補正する。

        a_final = clip(a_base + delta_a, -1, 1)
    """

    def __init__(
        self,
        base_agent,
        residual_network,
        device=None,
    ):
        self.base_agent = base_agent

        if device is None:
            device = base_agent.device

        self.device = torch.device(
            device
        )

        self.residual_network = (
            residual_network.to(
                self.device
            )
        )
        self.residual_network.eval()

        self.last_selection = None

    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        base_action = (
            self.base_agent.select_action(
                observation,
                goal,
                deterministic=deterministic,
            )
        )

        base_action = np.asarray(
            base_action,
            dtype=np.float32,
        )

        if (
            goal.goal_type
            != TacticalGoalType.POSITION_ATTACK
        ):
            self.last_selection = None
            return base_action

        conditioned_state = (
            self.base_agent
            .build_conditioned_state(
                observation,
                goal,
            )
        )

        state_tensor = torch.as_tensor(
            conditioned_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        base_tensor = torch.as_tensor(
            base_action,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            residual = (
                self.residual_network(
                    state_tensor,
                    base_tensor,
                )
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        final_action = (
            compose_residual_action(
                base_action,
                residual,
            )
        )

        self.last_selection = (
            PositionResidualSelection(
                base_action=(
                    base_action.copy()
                ),
                residual_action=(
                    residual.copy()
                ),
                final_action=(
                    final_action.copy()
                ),
            )
        )

        return final_action
