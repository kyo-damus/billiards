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



def test_load_dataset_accepts_explicit_mixed_candidate_v2(tmp_path):
    from scripts.train_macro_candidate_value import load_dataset
    from src.macro.encoding import MACRO_STATE_DIM

    output = tmp_path / "mixed.npz"
    np.savez_compressed(
        output,
        state=np.zeros((4, MACRO_STATE_DIM), dtype=np.float32),
        candidate=np.zeros((4, 31), dtype=np.float32),
        q_target=np.zeros(4, dtype=np.float32),
        state_id=np.array([0, 0, 1, 1], dtype=np.int32),
        candidate_encoding_version=np.asarray(2),
        turn_aware=np.asarray(True),
    )
    states, candidates, targets, state_ids = load_dataset(output)
    assert states.shape == (4, MACRO_STATE_DIM)
    assert candidates.shape == (4, 31)
    assert targets.shape == (4,)
    assert state_ids.shape == (4,)


def test_load_dataset_rejects_unmarked_31_dim_candidates(tmp_path):
    import pytest
    from scripts.train_macro_candidate_value import load_dataset
    from src.macro.encoding import MACRO_STATE_DIM

    output = tmp_path / "invalid_mixed.npz"
    np.savez_compressed(
        output,
        state=np.zeros((2, MACRO_STATE_DIM), dtype=np.float32),
        candidate=np.zeros((2, 31), dtype=np.float32),
        q_target=np.zeros(2, dtype=np.float32),
        state_id=np.array([0, 1], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="candidate_encoding_version"):
        load_dataset(output)


def test_load_dataset_keeps_attack_v0_compatible(tmp_path):
    from scripts.train_macro_candidate_value import load_dataset
    from src.macro.encoding import MACRO_STATE_DIM

    output = tmp_path / "attack_v0.npz"
    np.savez_compressed(
        output,
        state=np.zeros((2, MACRO_STATE_DIM), dtype=np.float32),
        candidate=np.zeros((2, 25), dtype=np.float32),
        q_target=np.zeros(2, dtype=np.float32),
        state_id=np.array([0, 1], dtype=np.int32),
    )
    _, candidates, _, _ = load_dataset(output)
    assert candidates.shape == (2, 25)
