from dataclasses import dataclass

import numpy as np
import torch

from src.goal.types import (
    TacticalGoalType,
)
from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)
from src.micro.position_blend_gate import (
    PositionBlendGate,
    build_position_blend_gate_features,
)


@dataclass(frozen=True)
class PositionBlendSelection:
    blend_lambda: float
    confidence: float
    base_action: np.ndarray
    bc_action: np.ndarray
    blended_action: np.ndarray


class AdaptivePositionMicroPolicy:
    """
    DIRECT_ATTACK:
        use the original goal-conditioned Micro actor.

    POSITION_ATTACK:
        evaluate the learned per-lambda success gate and
        blend the original actor with the witness-BC actor.

        a = (1 - lambda) * a_base + lambda * a_bc

    The gate is only used for POSITION_ATTACK so the DIRECT
    policy is preserved.
    """

    def __init__(
        self,
        base_agent,
        bc_agent,
        gate,
        lambdas,
        selection_threshold=0.0,
        device=None,
    ):
        self.base_agent = base_agent
        self.bc_agent = bc_agent
        self.gate = gate

        self.lambdas = [
            float(value)
            for value in lambdas
        ]

        if not self.lambdas:
            raise ValueError(
                "lambdas must not be empty."
            )

        if 0.0 not in self.lambdas:
            raise ValueError(
                "lambdas must include 0.0."
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

        self.gate = self.gate.to(
            self.device
        )
        self.gate.eval()

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

        if (
            goal.goal_type
            != TacticalGoalType.POSITION_ATTACK
        ):
            self.last_selection = None

            return np.asarray(
                base_action,
                dtype=np.float32,
            )

        bc_action = (
            self.bc_agent.select_action(
                observation,
                goal,
                deterministic=deterministic,
            )
        )

        conditioned_state = (
            self.base_agent
            .build_conditioned_state(
                observation,
                goal,
            )
        )

        features = (
            build_position_blend_gate_features(
                conditioned_state,
                base_action,
                bc_action,
            )
        )

        features_tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(
            0
        )

        with torch.no_grad():
            logits = self.gate(
                features_tensor
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
                self.lambdas.index(
                    0.0
                )
            )

        blend_lambda = (
            self.lambdas[
                choice_index
            ]
        )

        blended_action = (
            (1.0 - blend_lambda)
            * np.asarray(
                base_action,
                dtype=np.float32,
            )
            + blend_lambda
            * np.asarray(
                bc_action,
                dtype=np.float32,
            )
        )

        blended_action = np.clip(
            blended_action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        self.last_selection = (
            PositionBlendSelection(
                blend_lambda=(
                    blend_lambda
                ),
                confidence=(
                    confidence_value
                ),
                base_action=(
                    np.asarray(
                        base_action,
                        dtype=np.float32,
                    ).copy()
                ),
                bc_action=(
                    np.asarray(
                        bc_action,
                        dtype=np.float32,
                    ).copy()
                ),
                blended_action=(
                    blended_action.copy()
                ),
            )
        )

        return blended_action

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint,
        bc_checkpoint,
        gate_checkpoint,
        device="cpu",
    ):
        base_agent = GoalConditionedSACAgent(
            action_dim=2,
            device=device,
        )

        bc_agent = GoalConditionedSACAgent(
            action_dim=2,
            device=device,
        )

        base_data = torch.load(
            base_checkpoint,
            map_location=device,
        )

        bc_data = torch.load(
            bc_checkpoint,
            map_location=device,
        )

        base_agent.agent.actor.load_state_dict(
            base_data[
                "actor"
            ]
        )

        bc_agent.agent.actor.load_state_dict(
            bc_data[
                "actor"
            ]
        )

        base_agent.agent.actor.eval()
        bc_agent.agent.actor.eval()

        gate_data = torch.load(
            gate_checkpoint,
            map_location=device,
        )

        lambdas = [
            float(value)
            for value in gate_data[
                "lambdas"
            ]
        ]

        gate = PositionBlendGate(
            input_dim=int(
                gate_data[
                    "input_dim"
                ]
            ),
            hidden_dim=int(
                gate_data[
                    "hidden_dim"
                ]
            ),
            num_lambdas=len(
                lambdas
            ),
        ).to(
            device
        )

        gate.load_state_dict(
            gate_data[
                "gate"
            ]
        )

        threshold = float(
            gate_data.get(
                "selection_threshold",
                0.0,
            )
        )

        return cls(
            base_agent=base_agent,
            bc_agent=bc_agent,
            gate=gate,
            lambdas=lambdas,
            selection_threshold=threshold,
            device=device,
        )
