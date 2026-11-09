import numpy as np
import torch

from src.micro.position_residual_scale_gate import (
    POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM,
    PositionResidualScaleGate,
    build_position_residual_scale_features,
)


def test_scale_gate_feature_shape():

    conditioned_state = np.zeros(
        34,
        dtype=np.float32,
    )
    base_action = np.array(
        [0.2, -0.1],
        dtype=np.float32,
    )
    residual_action = np.array(
        [0.5, 0.0],
        dtype=np.float32,
    )

    features = (
        build_position_residual_scale_features(
            conditioned_state,
            base_action,
            residual_action,
        )
    )

    assert features.shape == (
        POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM,
    )

    np.testing.assert_allclose(
        features[-4:-2],
        base_action,
    )
    np.testing.assert_allclose(
        features[-2:],
        residual_action,
    )


def test_scale_gate_forward_shape():

    model = PositionResidualScaleGate(
        hidden_dim=32,
        num_scales=5,
    )

    x = torch.zeros(
        (
            7,
            POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM,
        ),
        dtype=torch.float32,
    )

    logits = model(
        x
    )

    assert logits.shape == (
        7,
        5,
    )
