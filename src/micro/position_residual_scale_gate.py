from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.goal.types import (
    TacticalGoalType,
)
from src.micro.position_residual_policy import (
    compose_residual_action,
)


POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM = 38


def build_position_residual_scale_features(
    conditioned_state,
    base_action,
    residual_action,
):
    """
    Gate input:
        conditioned state: 34
        base action:         2
        predicted residual:  2
    total:                  38
    """

    conditioned_state = np.asarray(
        conditioned_state,
        dtype=np.float32,
    )
    base_action = np.asarray(
        base_action,
        dtype=np.float32,
    )
    residual_action = np.asarray(
        residual_action,
        dtype=np.float32,
    )

    if conditioned_state.shape != (
        34,
    ):
        raise ValueError(
            "conditioned_state must have "
            "shape (34,)."
        )

    if base_action.shape != (
        2,
    ):
        raise ValueError(
            "base_action must have shape (2,)."
        )

    if residual_action.shape != (
        2,
    ):
        raise ValueError(
            "residual_action must have shape (2,)."
        )

    features = np.concatenate(
        [
            conditioned_state,
            base_action,
            residual_action,
        ]
    ).astype(
        np.float32
    )

    if features.shape != (
        POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM,
    ):
        raise RuntimeError(
            "Unexpected residual scale gate "
            "feature dimension."
        )

    return features


class PositionResidualScaleGate(nn.Module):

    def __init__(
        self,
        input_dim=(
            POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM
        ),
        hidden_dim=128,
        num_scales=5,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                num_scales,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


@dataclass(frozen=True)
class PositionResidualScaleSelection:
    residual_scale: float
    confidence: float
    base_action: np.ndarray
    residual_action: np.ndarray
    final_action: np.ndarray


class AdaptivePositionResidualPolicy:
    """
    DIRECT_ATTACK:
        Preserve the original Base Policy.

    POSITION_ATTACK:
        Predict a power residual first, then use a learned
        gate to choose how strongly that residual should be
        applied.

        a_final = clip(
            a_base + scale * residual,
            -1,
            1,
        )
    """

    def __init__(
        self,
        base_agent,
        residual_network,
        gate,
        scales,
        selection_threshold=0.0,
        device=None,
    ):
        self.base_agent = base_agent

        self.scales = [
            float(value)
            for value in scales
        ]

        if not self.scales:
            raise ValueError(
                "scales must not be empty."
            )

        if 0.0 not in self.scales:
            raise ValueError(
                "scales must include 0.0."
            )

        if not (
            0.0
            <= selection_threshold
            <= 1.0
        ):
            raise ValueError(
                "selection_threshold must be "
                "between 0 and 1."
            )

        self.selection_threshold = float(
            selection_threshold
        )

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
        self.gate = gate.to(
            self.device
        )

        self.residual_network.eval()
        self.gate.eval()

        self.last_selection = None

    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        base_action = np.asarray(
            self.base_agent.select_action(
                observation,
                goal,
                deterministic=deterministic,
            ),
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
            residual_action = (
                self.residual_network(
                    state_tensor,
                    base_tensor,
                )
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        features = (
            build_position_residual_scale_features(
                conditioned_state,
                base_action,
                residual_action,
            )
        )

        feature_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.gate(
                feature_tensor
            )
            probabilities = torch.sigmoid(
                logits
            )
            confidence, choice = torch.max(
                probabilities,
                dim=1,
            )

        choice_index = int(
            choice.item()
        )
        confidence_value = float(
            confidence.item()
        )

        if (
            confidence_value
            < self.selection_threshold
        ):
            choice_index = (
                self.scales.index(
                    0.0
                )
            )

        residual_scale = self.scales[
            choice_index
        ]

        scaled_residual = (
            residual_action
            * residual_scale
        ).astype(
            np.float32
        )

        final_action = (
            compose_residual_action(
                base_action,
                scaled_residual,
            )
        )

        self.last_selection = (
            PositionResidualScaleSelection(
                residual_scale=(
                    residual_scale
                ),
                confidence=(
                    confidence_value
                ),
                base_action=(
                    base_action.copy()
                ),
                residual_action=(
                    residual_action.copy()
                ),
                final_action=(
                    final_action.copy()
                ),
            )
        )

        return final_action
