import numpy as np

from scripts.train_position_residual import (
    calculate_metrics,
    load_residual_dataset,
    make_split,
)


def test_make_split_is_deterministic():

    train_a, val_a = make_split(
        sample_count=20,
        validation_fraction=0.20,
        seed=123,
    )

    train_b, val_b = make_split(
        sample_count=20,
        validation_fraction=0.20,
        seed=123,
    )

    np.testing.assert_array_equal(
        train_a,
        train_b,
    )
    np.testing.assert_array_equal(
        val_a,
        val_b,
    )

    assert len(train_a) == 16
    assert len(val_a) == 4

    assert set(train_a).isdisjoint(
        set(val_a)
    )


def test_calculate_metrics():

    target = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    prediction = np.array(
        [
            [0.5, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )

    metrics = calculate_metrics(
        prediction,
        target,
    )

    assert np.isclose(
        metrics["mse"],
        0.125,
    )

    np.testing.assert_allclose(
        metrics["mae_per_dim"],
        np.array(
            [0.25, 0.25],
            dtype=np.float32,
        ),
    )

    assert np.isclose(
        metrics["mean_error_norm"],
        0.5,
    )


def test_load_residual_dataset_checks_identity(
    tmp_path,
):

    states = np.zeros(
        (4, 34),
        dtype=np.float32,
    )

    base = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
        ],
        dtype=np.float32,
    )

    witness = base + np.array(
        [
            [0.2, -0.1],
            [0.1, -0.2],
            [0.3, 0.0],
            [-0.1, 0.1],
        ],
        dtype=np.float32,
    )

    residual = (
        witness - base
    )

    path = (
        tmp_path
        / "dataset.npz"
    )

    np.savez_compressed(
        path,
        conditioned_state=states,
        base_action=base,
        witness_action=witness,
        residual_target=residual,
    )

    (
        loaded_states,
        loaded_base,
        loaded_witness,
        loaded_residual,
    ) = load_residual_dataset(
        path
    )

    np.testing.assert_allclose(
        loaded_states,
        states,
    )
    np.testing.assert_allclose(
        loaded_base,
        base,
    )
    np.testing.assert_allclose(
        loaded_witness,
        witness,
    )
    np.testing.assert_allclose(
        loaded_residual,
        residual,
    )
