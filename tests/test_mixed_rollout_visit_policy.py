"""MCTS root visits train Candidate Policy even from truncated rollouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.build_mixed_rollout_visit_dataset import (
    build_visit_dataset,
)
from scripts.train_macro_visit_policy import (
    load_visit_dataset, train,
)
from src.macro.candidate_policy import (
    load_macro_candidate_policy_network,
)


def write_rollout(path, visit_counts=(1, 3), completed=False):
    state = np.zeros((1, 12), dtype=np.float32)
    state[0, -1] = 0.0
    candidates = np.zeros((2, 31), dtype=np.float32)
    candidates[0, 0] = 1.0
    candidates[1, 1] = 1.0
    np.savez_compressed(
        path,
        format_version=np.array(1),
        candidate_encoding_version=np.array(2),
        data_kind=np.array("executed_mixed_rollout"),
        state=state,
        root_turn_index=np.array([0, 0], dtype=np.int32),
        root_candidate=candidates,
        root_visit_count=np.asarray(visit_counts, dtype=np.int32),
        root_selected=np.array([False, True]),
        stop_reason=np.array(
            "game_terminated" if completed else "max_turns"
        ),
        rollout_terminated=np.array(completed),
    )
    return path


def test_builder_accepts_truncated_games_and_keeps_episode_groups(tmp_path):
    first = write_rollout(tmp_path / "a.npz", (1, 3), completed=False)
    second = write_rollout(tmp_path / "b.npz", (3, 1), completed=False)
    output, roots, candidates, skipped = build_visit_dataset(
        [first, second], tmp_path / "visits.npz"
    )
    assert (roots, candidates, skipped) == (2, 4, 0)
    state, cand, target, group, episodes = load_visit_dataset(output)
    assert state.shape == (4, 12)
    assert cand.shape == (4, 31)
    np.testing.assert_allclose(target, [0.25, 0.75, 0.75, 0.25])
    assert group.tolist() == [0, 0, 1, 1]
    assert episodes.tolist() == [0, 0, 1, 1]
    with np.load(output, allow_pickle=False) as data:
        assert str(data["target_kind"]) == "mcts_root_visit_distribution"
        assert bool(data["turn_aware"])
        assert "q_target" not in data


def test_visit_trainer_checkpoint_uses_31d_policy_loader(tmp_path):
    a = write_rollout(tmp_path / "a.npz")
    b = write_rollout(tmp_path / "b.npz")
    output, *_ = build_visit_dataset([a, b], tmp_path / "visits.npz")
    checkpoint = tmp_path / "policy.pt"
    train(
        SimpleNamespace(
            dataset=str(output),
            checkpoint=str(checkpoint),
            epochs=2,
            hidden_dim=16,
            learning_rate=1e-3,
            validation_fraction=0.5,
            seed=1,
            device="cpu",
        )
    )
    network, data = load_macro_candidate_policy_network(
        checkpoint, expected_candidate_dim=31
    )
    assert data["turn_aware"] is True
    assert data["candidate_encoding_version"] == 2
    assert data["target_kind"] == "mcts_root_visit_distribution"
    with torch.no_grad():
        output = network(torch.zeros(12), torch.zeros(2, 31))
    assert output.shape == (2,)


def test_builder_skips_empty_visit_roots(tmp_path):
    file = write_rollout(tmp_path / "empty.npz", (0, 0))
    with pytest.raises(ValueError, match="No MCTS roots"):
        build_visit_dataset([file], tmp_path / "none.npz")
    assert not (tmp_path / "none.npz").exists()


def test_builder_rejects_invalid_selected_root(tmp_path):
    file = write_rollout(tmp_path / "invalid.npz")
    with np.load(file, allow_pickle=False) as data:
        content = {k: data[k].copy() for k in data.files}
    content["root_selected"] = np.array([False, False])
    np.savez_compressed(file, **content)
    with pytest.raises(ValueError, match="expected one selected root"):
        build_visit_dataset([file], tmp_path / "out.npz")


def test_visit_training_requires_two_episodes(tmp_path):
    file = write_rollout(tmp_path / "a.npz")
    output, *_ = build_visit_dataset([file], tmp_path / "visits.npz")
    with pytest.raises(ValueError, match="At least two distinct states"):
        train(
            SimpleNamespace(
                dataset=str(output), checkpoint=str(tmp_path / "p.pt"),
                epochs=1, hidden_dim=16, learning_rate=1e-3,
                validation_fraction=0.5, seed=1, device="cpu"
            )
        )
