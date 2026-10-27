import numpy as np

from scripts.train_macro_candidate_value import (
    calculate_metrics,
    make_group_split,
)


def test_group_split_keeps_state_groups_separate():

    state_ids = np.array(
        [
            0, 0, 0,
            1, 1,
            2, 2, 2,
            3, 3,
        ],
        dtype=np.int64,
    )

    train_indices, val_indices = (
        make_group_split(
            state_ids,
            validation_fraction=0.25,
            seed=0,
        )
    )

    train_state_ids = set(
        state_ids[
            train_indices
        ].tolist()
    )

    val_state_ids = set(
        state_ids[
            val_indices
        ].tolist()
    )

    assert train_state_ids
    assert val_state_ids
    assert (
        train_state_ids
        .isdisjoint(
            val_state_ids
        )
    )


def test_top1_metric_accepts_tied_best_targets():

    target = np.array(
        [
            1.25,
            1.25,
            0.0,
        ],
        dtype=np.float32,
    )

    # index 1を最大予測にするが、
    # index 0と教師値は同率最大。
    prediction = np.array(
        [
            0.8,
            1.0,
            -0.2,
        ],
        dtype=np.float32,
    )

    state_ids = np.array(
        [0, 0, 0],
        dtype=np.int64,
    )

    metrics = calculate_metrics(
        prediction,
        target,
        state_ids,
    )

    assert np.isclose(
        metrics[
            "top1_accuracy"
        ],
        1.0,
    )
