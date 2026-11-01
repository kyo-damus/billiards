import numpy as np
import pytest

from src.common.types import (
    GameState,
    ShotAction,
)
from src.micro.defense_shot import (
    DefenseShotExecutor,
    plan_defense_shot,
)


def make_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.50, 0.30],
                [0.50, 0.80],
                [0.90, 1.50],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_defense_shot_plans_direct_center_line():
    shot = plan_defense_shot(
        make_state(),
        target_ball=0,
        power=1.0,
        angle_offset_deg=10.0,
    )

    assert np.isclose(
        shot.angle,
        100.0,
    )
    assert np.isclose(
        shot.power,
        1.0,
    )


def test_defense_shot_validates_physical_inputs():
    with pytest.raises(ValueError):
        plan_defense_shot(
            make_state(),
            0,
            power=-0.5,
        )

    with pytest.raises(ValueError):
        plan_defense_shot(
            make_state(),
            0,
            power=1.0,
            angle_offset_deg=40.0,
        )


def test_defense_shot_can_execute_without_attack_action():
    state = make_state()
    before = state.ball_positions.copy()

    shot = plan_defense_shot(
        state,
        target_ball=0,
        power=1.0,
    )
    executor = DefenseShotExecutor()

    result = executor.execute(
        state,
        target_ball=0,
        shot=shot,
    )

    assert result.next_state.ball_positions.shape == (3, 2)
    assert result.next_state.ball_pocket_indices.shape == (3,)
    assert result.physical_state.ball_positions.shape == (3, 2)
    assert isinstance(result.game_reward, float)
    assert isinstance(result.turn_changed, bool)
    assert result.first_contact_verified is False

    # The reusable root board must not be mutated.
    np.testing.assert_array_equal(
        state.ball_positions,
        before,
    )


def test_defense_shot_rejects_spin_explicitly():
    executor = DefenseShotExecutor()

    with pytest.raises(ValueError, match="spin"):
        executor.execute(
            make_state(),
            target_ball=0,
            shot=ShotAction(
                angle=90.0,
                power=1.0,
                spin_x=0.5,
            ),
        )


def test_defense_shot_is_deterministic_from_restored_state():
    state = make_state()
    executor = DefenseShotExecutor()
    shot = plan_defense_shot(
        state,
        target_ball=0,
        power=1.0,
    )

    first = executor.execute(
        state,
        target_ball=0,
        shot=shot,
    )
    second = executor.execute(
        state,
        target_ball=0,
        shot=shot,
    )

    np.testing.assert_allclose(
        first.next_state.ball_positions,
        second.next_state.ball_positions,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        first.next_state.ball_pocket_indices,
        second.next_state.ball_pocket_indices,
    )
