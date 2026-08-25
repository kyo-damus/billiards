import numpy as np

from src.common.types import MacroAction, Strategy
from src.env.micro_billiard_env import MicroBilliardEnv

import pytest


def make_attack_action(target_ball=0):
    return MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=target_ball,
        target_pocket=0,
    )


def test_reset():
    env = MicroBilliardEnv()

    env.set_macro_action(
        make_attack_action(target_ball=0)
    )

    obs, info = env.reset(seed=42)

    assert obs.shape == (12,)
    assert obs.dtype == np.float32


def test_step():
    env = MicroBilliardEnv()

    env.set_macro_action(
        make_attack_action(target_ball=0)
    )

    env.reset(seed=42)

    action = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    assert obs.shape == (12,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False

    assert info["target_ball"] == 0
    assert info["target_pocket"] == 0
    assert terminated is True
    assert "success" in info
    assert "scratched" in info


def test_target_ball_can_change():
    env = MicroBilliardEnv()

    env.set_macro_action(
        make_attack_action(target_ball=1)
    )

    obs, _ = env.reset(seed=42)

    assert obs.shape == (12,)


def test_same_seed_same_initial_state():
    env1 = MicroBilliardEnv()
    env2 = MicroBilliardEnv()

    macro_action = make_attack_action()

    env1.set_macro_action(macro_action)
    env2.set_macro_action(macro_action)

    env1.reset(seed=123)
    env2.reset(seed=123)

    np.testing.assert_allclose(
        env1.get_state(),
        env2.get_state(),
    )

def test_cannot_step_twice_without_reset():
    env = MicroBilliardEnv()

    env.set_macro_action(
        make_attack_action(target_ball=0)
    )

    env.reset(seed=42)

    action = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    env.step(action)

    with pytest.raises(RuntimeError):
        env.step(action)