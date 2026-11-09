"""State V checkpoint path from complete executed-game returns."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train_macro_state_value import (
    load_state_value_dataset,
    train,
)
from src.common.types import GameState
from src.macro.value import load_macro_state_value_network
from src.macro.mcts import MCTS
from src.macro.mixed_search import ZeroValue


def example_dataset(path, episode_ids=(0, 0, 1, 1)):
    states = np.zeros((4, 12), dtype=np.float32)
    states[:, 0] = np.asarray([0.2, 0.4, 0.6, 0.8])
    states[2:, -1] = 1.0
    np.savez_compressed(
        path,
        state=states,
        q_target=np.asarray([0.49, 1.0, -1.49, 1.0], dtype=np.float32),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        target_kind=np.array("selected_action_mc_return"),
        reward_kind=np.array("macro_local_reward"),
        turn_aware=np.array(True),
        candidate_encoding_version=np.array(2),
    )
    return path


def make_args(dataset, checkpoint):
    return SimpleNamespace(
        dataset=str(dataset),
        checkpoint=str(checkpoint),
        epochs=2,
        hidden_dim=16,
        learning_rate=0.001,
        batch_size=2,
        validation_fraction=0.5,
        seed=123,
        device="cpu",
    )


def test_state_value_training_checkpoint_loader_and_mcts_leaf(tmp_path):
    data = example_dataset(tmp_path / "return.npz")
    checkpoint = tmp_path / "state_v.pt"
    assert train(make_args(data, checkpoint)) == checkpoint
    network, meta = load_macro_state_value_network(
        checkpoint, device="cpu"
    )
    assert meta["turn_aware"] is True
    assert meta["target_kind"] == "selected_action_mc_return"

    state = GameState(
        ball_positions=np.asarray(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
    )
    class EmptyGenerator:
        def generate(self, state):
            return []

    mcts = MCTS(
        policy_network=None,
        value_network=network,
        action_generator=EmptyGenerator(),
        transition_model=object(),
        device="cpu",
        candidate_mode=True,
        turn_aware=True,
        candidate_encoding_version=2,
        uniform_candidate_priors=True,
    )
    value = mcts._evaluate(state)
    assert np.isfinite(value)
    assert network(torch.zeros(1, 12)).shape == (1, 1)
    assert ZeroValue()(torch.zeros(1, 12)).shape == (1,)


def test_state_value_requires_multiple_completed_episodes(tmp_path):
    path = example_dataset(
        tmp_path / "one_episode.npz", episode_ids=(0, 0, 0, 0)
    )
    with pytest.raises(ValueError, match="At least two completed"):
        load_state_value_dataset(path)


def test_state_value_rejects_wrong_target_kind(tmp_path):
    path = example_dataset(tmp_path / "wrong.npz")
    with np.load(path, allow_pickle=False) as data:
        contents = {key: data[key].copy() for key in data.files}
    contents["target_kind"] = np.array("two_step_oracle")
    np.savez_compressed(path, **contents)
    with pytest.raises(ValueError, match="completed turn-aware"):
        load_state_value_dataset(path)


def test_state_value_loader_rejects_legacy_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"hidden_dim": 16, "state_dim": 12}, path)
    with pytest.raises(ValueError, match="missing keys"):
        load_macro_state_value_network(path)
