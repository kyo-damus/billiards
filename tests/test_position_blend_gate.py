import numpy as np
import torch

from src.micro.position_blend_gate import (
    POSITION_BLEND_GATE_INPUT_DIM,
    PositionBlendGate,
    build_position_blend_gate_features,
    choose_oracle_lambda_index,
)


def test_gate_feature_dimension():

    features = (
        build_position_blend_gate_features(
            np.zeros(
                34,
                dtype=np.float32,
            ),
            np.array(
                [0.1, -0.2],
                dtype=np.float32,
            ),
            np.array(
                [0.3, 0.4],
                dtype=np.float32,
            ),
        )
    )

    assert features.shape == (
        POSITION_BLEND_GATE_INPUT_DIM,
    )


def test_oracle_prefers_success():

    index = choose_oracle_lambda_index(
        goal_success=[
            False,
            True,
            True,
        ],
        pot_success=[
            True,
            True,
            True,
        ],
        scratched=[
            False,
            False,
            False,
        ],
        cue_distances=[
            0.05,
            0.09,
            0.04,
        ],
        zero_index=0,
    )

    assert index == 2


def test_oracle_falls_back_to_best_pot_distance():

    index = choose_oracle_lambda_index(
        goal_success=[
            False,
            False,
            False,
        ],
        pot_success=[
            True,
            True,
            False,
        ],
        scratched=[
            False,
            False,
            False,
        ],
        cue_distances=[
            0.40,
            0.20,
            0.10,
        ],
        zero_index=0,
    )

    assert index == 1


def test_gate_forward_shape():

    model = PositionBlendGate(
        num_lambdas=8,
    )

    x = torch.zeros(
        5,
        POSITION_BLEND_GATE_INPUT_DIM,
    )

    y = model(
        x
    )

    assert y.shape == (
        5,
        8,
    )
