"""Generalized rack state: no FastFiz or GPU required."""

import numpy as np
import pytest

from src.common.types import GameState
from src.game.rack_state import RackState, TableGameConfig


def state(config=None, positions=None, pockets=None):
    config = config or TableGameConfig.nine_ball_inventory()
    n = len(config.ball_numbers)
    return RackState(
        config=config,
        ball_positions=np.zeros((n, 2), dtype=np.float32) if positions is None else positions,
        ball_pocket_indices=np.full(n, -1, dtype=np.int32) if pockets is None else pockets,
        current_player=0,
        score=np.zeros(2, dtype=np.float32),
    )


def test_nine_ball_inventory_has_cue_plus_nine_numbered_balls():
    config = TableGameConfig.nine_ball_inventory()
    assert config.ball_numbers == tuple(range(10))
    assert state(config).ball_count == 10


def test_training_inventory_has_three_balls():
    assert TableGameConfig.two_object_training().ball_numbers == (0, 1, 2)


def test_explicit_ball_numbers_preserve_identity_and_pocket_state():
    config = TableGameConfig("practice", (2, 5, 9))
    rack = state(config, pockets=np.array([-1, 0, -1, 5]))
    assert rack.array_index_for_ball(5) == 2
    assert rack.on_table_object_balls() == (5,)
    assert rack.is_pocketed(2)
    assert rack.is_pocketed(9)
    with pytest.raises(ValueError, match="Unknown ball number"):
        rack.is_pocketed(1)


def test_state_copies_mutable_arrays_and_is_read_only():
    config = TableGameConfig.two_object_training()
    positions = np.zeros((3, 2), dtype=np.float32)
    pockets = np.full(3, -1)
    rack = state(config, positions=positions, pockets=pockets)
    positions[0, 0] = 7.0
    pockets[1] = 2
    assert rack.ball_positions[0, 0] == 0.0
    assert rack.ball_pocket_indices[1] == -1
    with pytest.raises(ValueError):
        rack.ball_positions[0, 0] = 1.0


@pytest.mark.parametrize("numbers", [(), (0,), (1, 1), (-1, 2)])
def test_invalid_ball_inventory_is_rejected(numbers):
    with pytest.raises(ValueError):
        TableGameConfig("test", numbers)


def test_mismatched_ball_count_and_invalid_pockets_are_rejected():
    config = TableGameConfig.nine_ball_inventory()
    with pytest.raises(ValueError, match="ball_positions"):
        state(config, positions=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="Pocket indices"):
        state(config, pockets=np.array([-1] * 9 + [6]))


def test_legacy_adapter_does_not_change_old_state():
    old = GameState(
        ball_positions=np.zeros((3, 2), dtype=np.float32),
        ball_pocket_indices=np.array([-1, 1, -1], dtype=np.int32),
        score=np.array([1.0, 0.0], dtype=np.float32),
        current_player=1,
    )
    adapted = RackState.from_legacy(old)
    assert adapted.config.rule_id == "simplified_two_object"
    assert adapted.ball_count == 3
    assert adapted.current_player == 1
    assert adapted.is_pocketed(1)
    np.testing.assert_array_equal(adapted.score, old.score)
    with pytest.raises(ValueError, match="Legacy GameState"):
        RackState.from_legacy(
            old, config=TableGameConfig.nine_ball_inventory()
        )
