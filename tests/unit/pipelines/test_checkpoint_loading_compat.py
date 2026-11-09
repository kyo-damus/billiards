"""Verify checkpoint loaders migrated without breaking legacy imports."""

import pytest
import torch

from scripts import attack_components
from scripts import eval_position_residual
from scripts import eval_position_residual_sweep
from src.pipelines import checkpoint_loading


def test_checkpoint_loaders_keep_legacy_import_identity():
    for legacy in (
        attack_components,
        eval_position_residual,
        eval_position_residual_sweep,
    ):
        assert legacy.load_base_agent is checkpoint_loading.load_base_agent
        assert (
            legacy.load_residual_network
            is checkpoint_loading.load_residual_network
        )


def test_residual_checkpoint_required_keys_remain_validated(monkeypatch):
    monkeypatch.setattr(
        checkpoint_loading.torch,
        "load",
        lambda path, map_location: {"residual_network": {}},
    )
    with pytest.raises(ValueError, match="Residual checkpoint is missing keys"):
        checkpoint_loading.load_residual_network("dummy.pt", "cpu")


def test_base_checkpoint_critic_target_is_optional(monkeypatch):
    loaded = []

    class SpyModule:
        def load_state_dict(self, value):
            loaded.append(value)

    class SpyAgent:
        def __init__(self, *, action_dim, device):
            assert action_dim == 2
            assert device == "cpu"
            class Container:
                actor = SpyModule()
                critic = SpyModule()
                critic_target = SpyModule()

            self.agent = Container()

    monkeypatch.setattr(checkpoint_loading, "GoalConditionedSACAgent", SpyAgent)
    monkeypatch.setattr(
        checkpoint_loading.torch,
        "load",
        lambda path, map_location: {"actor": {"w": 1}, "critic": {"w": 2}},
    )
    checkpoint_loading.load_base_agent("dummy.pt", "cpu")
    assert loaded == [{"w": 1}, {"w": 2}]


def test_base_checkpoint_loads_optional_critic_target(monkeypatch):
    loaded = []

    class SpyModule:
        def load_state_dict(self, value):
            loaded.append(value)

    class SpyAgent:
        def __init__(self, *, action_dim, device):
            class Container:
                actor = SpyModule()
                critic = SpyModule()
                critic_target = SpyModule()

            self.agent = Container()

    monkeypatch.setattr(checkpoint_loading, "GoalConditionedSACAgent", SpyAgent)
    monkeypatch.setattr(
        checkpoint_loading.torch,
        "load",
        lambda path, map_location: {
            "actor": {"w": 1},
            "critic": {"w": 2},
            "critic_target": {"w": 3},
        },
    )
    checkpoint_loading.load_base_agent("dummy.pt", "cpu")
    assert loaded == [{"w": 1}, {"w": 2}, {"w": 3}]
