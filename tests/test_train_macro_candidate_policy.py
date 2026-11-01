import numpy as np
import torch

from scripts.train_macro_candidate_policy import (
    make_groups,
    target_policy_distribution,
)


def test_target_policy_distribution_prefers_high_q():

    q_values = torch.tensor(
        [
            0.0,
            1.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    probability = (
        target_policy_distribution(
            q_values,
            temperature=0.5,
        )
    )

    assert np.isclose(
        float(
            probability.sum().item()
        ),
        1.0,
    )
    assert (
        probability[2]
        > probability[1]
        > probability[0]
    )


def test_make_groups_keeps_states_separate():

    state_ids = np.array(
        [
            0, 0,
            1, 1, 1,
            2, 2,
        ],
        dtype=np.int64,
    )

    indices = np.array(
        [
            0, 1,
            5, 6,
        ],
        dtype=np.int64,
    )

    groups = make_groups(
        indices,
        state_ids,
    )

    assert len(groups) == 2

    grouped_state_ids = [
        set(
            state_ids[
                group
            ].tolist()
        )
        for group in groups
    ]

    assert all(
        len(group_ids) == 1
        for group_ids
        in grouped_state_ids
    )
