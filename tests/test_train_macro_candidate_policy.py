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



def test_train_policy_on_tiny_mixed_31d_dataset(tmp_path):
    from types import SimpleNamespace
    from scripts.train_macro_candidate_policy import train
    from src.macro.encoding import MACRO_STATE_DIM
    from src.macro.candidate_policy import (
        load_macro_candidate_policy_network,
    )

    dataset = tmp_path / "tiny_mixed.npz"
    checkpoint = tmp_path / "tiny_mixed_policy.pt"
    # Four independent boards, each with two distinct candidates.
    state = np.zeros((8, MACRO_STATE_DIM), dtype=np.float32)
    candidate = np.zeros((8, 31), dtype=np.float32)
    candidate[::2, 0] = 1.0  # ATTACK
    candidate[1::2, 1] = 1.0  # DEFENSE
    np.savez_compressed(
        dataset,
        state=state,
        candidate=candidate,
        state_id=np.repeat(np.arange(4), 2),
        q_target=np.tile([1.0, -1.0], 4).astype(np.float32),
        candidate_encoding_version=np.array(2),
        turn_aware=np.array(True),
    )
    args = SimpleNamespace(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        device="cpu",
        seed=123,
        validation_fraction=0.5,
        hidden_dim=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        temperature=0.5,
        grad_clip=5.0,
        epochs=2,
        log_interval=1,
    )
    train(args)
    network, data = load_macro_candidate_policy_network(
        checkpoint, expected_candidate_dim=31
    )
    assert data["candidate_dim"] == 31
    assert data["candidate_encoding_version"] == 2
    assert data["turn_aware"] is True
    with torch.no_grad():
        scores = network(
            torch.zeros(MACRO_STATE_DIM), torch.zeros(2, 31)
        )
    assert scores.shape == (2,)



def test_policy_trainer_rejects_executed_only_mc_returns(tmp_path):
    from types import SimpleNamespace
    import pytest
    from scripts.train_macro_candidate_policy import train

    path = tmp_path / "selected_only.npz"
    np.savez_compressed(
        path,
        state=np.zeros((2, 12), dtype=np.float32),
        candidate=np.zeros((2, 31), dtype=np.float32),
        q_target=np.array([0.5, 1.0], dtype=np.float32),
        state_id=np.array([0, 1], dtype=np.int32),
        candidate_encoding_version=np.array(2),
        turn_aware=np.array(True),
        target_kind=np.array("selected_action_mc_return"),
    )
    with pytest.raises(ValueError, match="cannot train Candidate Policy"):
        train(SimpleNamespace(dataset=str(path), device="cpu", seed=1))
