"""Check moved Macro training utilities and backward-compatible exports."""

import numpy as np

from scripts import training_utils as legacy
from src.pipelines import training_utils as new


def test_training_utils_legacy_reexports_same_functions():
    assert legacy.resolve_device is new.resolve_device
    assert legacy.make_group_split is new.make_group_split
    assert legacy.make_groups is new.make_groups


def test_moved_split_still_keeps_state_groups_disjoint():
    state_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    train_a, val_a = new.make_group_split(state_ids, 0.25, 123)
    train_b, val_b = legacy.make_group_split(state_ids, 0.25, 123)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert set(state_ids[train_a]).isdisjoint(set(state_ids[val_a]))
    assert len(train_a) + len(val_a) == len(state_ids)
    groups = new.make_groups(train_a, state_ids)
    assert all(len(set(state_ids[group])) == 1 for group in groups)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups)), np.sort(train_a)
    )


def test_resolve_device_explicit_value_remains_unchanged():
    assert new.resolve_device("cpu") == "cpu"
    assert legacy.resolve_device("cpu") == "cpu"
