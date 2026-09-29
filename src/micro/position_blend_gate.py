import numpy as np
import torch
import torch.nn as nn


POSITION_BLEND_GATE_INPUT_DIM = 40


def build_position_blend_gate_features(
    conditioned_state,
    base_action,
    bc_action,
):
    """
    Gate input:
        conditioned state: 34
        base action:         2
        BC action:           2
        action delta:        2
    total:                  40
    """

    conditioned_state = np.asarray(
        conditioned_state,
        dtype=np.float32,
    )

    base_action = np.asarray(
        base_action,
        dtype=np.float32,
    )

    bc_action = np.asarray(
        bc_action,
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

    if bc_action.shape != (
        2,
    ):
        raise ValueError(
            "bc_action must have shape (2,)."
        )

    delta = (
        bc_action
        - base_action
    )

    features = np.concatenate(
        [
            conditioned_state,
            base_action,
            bc_action,
            delta,
        ]
    ).astype(
        np.float32
    )

    if features.shape != (
        POSITION_BLEND_GATE_INPUT_DIM,
    ):
        raise RuntimeError(
            "Unexpected gate feature dimension."
        )

    return features


def choose_oracle_lambda_index(
    goal_success,
    pot_success,
    scratched,
    cue_distances,
    zero_index=0,
):
    """
    Training target for the adaptive gate.

    Priority:
        1. POSITION Goal success
           -> choose smallest cue distance.
        2. Non-scratch pot
           -> choose smallest cue distance.
        3. Otherwise preserve the base actor
           -> lambda=0 index.

    This is an oracle label computed from simulator outcomes
    and is used only as supervised training data.
    """

    goal_success = np.asarray(
        goal_success,
        dtype=bool,
    )

    pot_success = np.asarray(
        pot_success,
        dtype=bool,
    )

    scratched = np.asarray(
        scratched,
        dtype=bool,
    )

    cue_distances = np.asarray(
        cue_distances,
        dtype=np.float32,
    )

    n = len(
        goal_success
    )

    if not (
        len(pot_success) == n
        and len(scratched) == n
        and len(cue_distances) == n
    ):
        raise ValueError(
            "Outcome arrays must have "
            "the same length."
        )

    if not (
        0 <= zero_index < n
    ):
        raise ValueError(
            "zero_index is out of range."
        )

    successful = np.flatnonzero(
        goal_success
    )

    if len(successful) > 0:
        finite = [
            int(index)
            for index in successful
            if np.isfinite(
                cue_distances[
                    index
                ]
            )
        ]

        if finite:
            return min(
                finite,
                key=lambda index: float(
                    cue_distances[
                        index
                    ]
                ),
            )

        return int(
            successful[0]
        )

    pot_candidates = np.flatnonzero(
        pot_success
        & (~scratched)
    )

    if len(pot_candidates) > 0:
        finite = [
            int(index)
            for index in pot_candidates
            if np.isfinite(
                cue_distances[
                    index
                ]
            )
        ]

        if finite:
            return min(
                finite,
                key=lambda index: float(
                    cue_distances[
                        index
                    ]
                ),
            )

        return int(
            pot_candidates[0]
        )

    return int(
        zero_index
    )


class PositionBlendGate(nn.Module):

    def __init__(
        self,
        input_dim=POSITION_BLEND_GATE_INPUT_DIM,
        hidden_dim=128,
        num_lambdas=8,
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
                num_lambdas,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )
