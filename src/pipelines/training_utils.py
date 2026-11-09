"""Shared utility functions for Macro data collection and training.

Implementations were moved unchanged from the dataset, Q, and Policy
scripts. Group splitting is deterministic and is used to avoid leakage
between state/episode groups. Public imports in the original scripts
remain available for backward compatibility.
"""

import numpy as np
import torch


def resolve_device(
    requested,
):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def make_group_split(
    state_ids,
    validation_fraction,
    seed,
):
    unique_ids = np.unique(
        state_ids
    )

    if len(unique_ids) < 2:
        raise ValueError(
            "At least two distinct states "
            "are required."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled = rng.permutation(
        unique_ids
    )

    val_state_count = max(
        1,
        int(
            round(
                len(unique_ids)
                * validation_fraction
            )
        ),
    )
    val_state_count = min(
        val_state_count,
        len(unique_ids) - 1,
    )

    val_ids = set(
        shuffled[
            :val_state_count
        ].tolist()
    )

    val_mask = np.asarray(
        [
            state_id in val_ids
            for state_id in state_ids
        ],
        dtype=bool,
    )

    train_indices = np.flatnonzero(
        ~val_mask
    )
    val_indices = np.flatnonzero(
        val_mask
    )

    return (
        train_indices,
        val_indices,
    )


def make_groups(
    indices,
    state_ids,
):
    groups = []

    selected_state_ids = np.unique(
        state_ids[
            indices
        ]
    )

    index_set = set(
        np.asarray(
            indices,
            dtype=np.int64,
        ).tolist()
    )

    for state_id in selected_state_ids:
        group = [
            int(index)
            for index in np.flatnonzero(
                state_ids == state_id
            )
            if int(index) in index_set
        ]

        if group:
            groups.append(
                np.asarray(
                    group,
                    dtype=np.int64,
                )
            )

    return groups
