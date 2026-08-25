import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

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

def test_initial_state_is_inside_table():
    env = MicroBilliardEnv()

    env.set_macro_action(
        make_attack_action(target_ball=0)
    )

    env.reset(seed=42)

    state = env.get_state()

    positions = [
        state[0:2],
        state[2:4],
        state[4:6],
    ]

    for pos in positions:
        assert 0.0 < pos[0] < env.TABLE_WIDTH
        assert 0.0 < pos[1] < env.TABLE_LENGTH

def test_all_target_pockets_are_supported():
    for pocket_id in range(6):
        env = MicroBilliardEnv()

        macro_action = MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=pocket_id,
        )

        env.set_macro_action(macro_action)

        obs, _ = env.reset(seed=42)

        assert obs.shape == (12,)
        assert env.macro_action.target_pocket == pocket_id

def test_invalid_target_pocket_is_rejected():
    env = MicroBilliardEnv()

    macro_action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=6,
    )

    with pytest.raises(ValueError):
        env.set_macro_action(macro_action)

def test_reset_to_game_state_restores_pocket_state():
    env = MicroBilliardEnv()

    state = GameState(
        ball_positions=np.array(
            [
                [0.5, 0.3],
                [0.5, 0.8],
                [0.04, 0.04],
            ],
            dtype=np.float32,
        ),

        score=np.zeros(
            2,
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [
                -1,
                -1,
                 0,
            ],
            dtype=np.int32,
        ),
    )

    macro_action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=2,
    )

    env.reset_to_game_state(
        state,
        macro_action,
    )

    assert env.sim.get_pocket_index(1) == 0