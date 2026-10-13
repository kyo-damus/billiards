import numpy as np

from scripts.build_position_residual_dataset import (
    build_residual_targets,
)


def test_build_residual_targets():

    base = np.array(
        [
            [0.2, -0.3],
            [-0.5, 0.4],
        ],
        dtype=np.float32,
    )

    witness = np.array(
        [
            [0.7, -0.1],
            [-0.2, -0.1],
        ],
        dtype=np.float32,
    )

    residual = build_residual_targets(
        base,
        witness,
    )

    expected = np.array(
        [
            [0.5, 0.2],
            [0.3, -0.5],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        residual,
        expected,
    )


def test_build_residual_targets_rejects_shape_mismatch():

    try:
        build_residual_targets(
            np.zeros(
                (2, 2),
                dtype=np.float32,
            ),
            np.zeros(
                (3, 2),
                dtype=np.float32,
            ),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError for mismatched shapes."
        )
