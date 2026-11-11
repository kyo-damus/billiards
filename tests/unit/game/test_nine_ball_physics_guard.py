"""9-ball physical layout checks independent of native C++ compilation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, validate_physical_step, validate_rack_layout,
)
from src.game.nine_ball_session import create_nine_ball_rack


def test_fresh_rack_and_float_precision_tolerance():
    rack = create_nine_ball_rack(breaker=0, seed=101)
    validate_rack_layout(rack)
    positions = np.array([
        [0.2, 0.3], [0.7, 0.3], [0.2, 0.6], [0.7, 0.6],
        [0.2, 0.9], [0.7, 0.9], [0.2, 1.2], [0.7, 1.2],
        [0.2, 1.5], [0.7, 1.5],
    ], dtype=np.float32)
    positions[2] = positions[5] + np.asarray(
        [0.05714998, 0.0], dtype=np.float32,
    )
    validate_rack_layout(replace(rack, ball_positions=positions))


def test_substantial_overlap_rejected_only_for_active_balls():
    rack = create_nine_ball_rack(breaker=0, seed=102)
    positions = rack.ball_positions.copy()
    positions[2] = positions[5]  # 57mm, NOT numerical precision
    with pytest.raises(InvalidPhysicalTransition, match="2,5"):
        validate_rack_layout(replace(rack, ball_positions=positions))
    pockets = rack.ball_pocket_indices.copy()
    pockets[5] = 1
    validate_rack_layout(replace(
        rack, ball_positions=positions, ball_pocket_indices=pockets,
    ))


def test_native_precondition_is_not_a_game_foul():
    with pytest.raises(InvalidPhysicalTransition, match="precondition=2"):
        validate_physical_step(SimpleNamespace(
            precondition_code=2, placement_code=0,
        ))
    with pytest.raises(InvalidPhysicalTransition, match="placement=4"):
        validate_physical_step(SimpleNamespace(
            precondition_code=0, placement_code=4,
        ))
    validate_physical_step(SimpleNamespace(
        precondition_code=0, placement_code=0,
    ))
