import numpy as np

from scripts.eval_position_residual import (
    build_scaled_residual_action,
)
from src.micro.position_residual_policy import (
    PositionResidualSelection,
)


def make_selection():
    return PositionResidualSelection(
        base_action=np.array(
            [0.2, -0.3],
            dtype=np.float32,
        ),
        residual_action=np.array(
            [0.6, 0.4],
            dtype=np.float32,
        ),
        final_action=np.array(
            [0.8, 0.1],
            dtype=np.float32,
        ),
    )


def test_scaled_residual_action_half_scale():

    action = build_scaled_residual_action(
        make_selection(),
        residual_scale=0.5,
        power_scale=1.0,
        angle_scale=1.0,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.5, -0.1],
            dtype=np.float32,
        ),
    )


def test_scaled_residual_action_power_only():

    action = build_scaled_residual_action(
        make_selection(),
        residual_scale=1.0,
        power_scale=1.0,
        angle_scale=0.0,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [0.8, -0.3],
            dtype=np.float32,
        ),
    )


def test_scaled_residual_action_clips():

    selection = PositionResidualSelection(
        base_action=np.array(
            [0.9, -0.9],
            dtype=np.float32,
        ),
        residual_action=np.array(
            [0.5, -0.5],
            dtype=np.float32,
        ),
        final_action=np.array(
            [1.0, -1.0],
            dtype=np.float32,
        ),
    )

    action = build_scaled_residual_action(
        selection,
        residual_scale=1.0,
        power_scale=1.0,
        angle_scale=1.0,
    )

    np.testing.assert_allclose(
        action,
        np.array(
            [1.0, -1.0],
            dtype=np.float32,
        ),
    )
