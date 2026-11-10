"""Actual FastFiz 9-ball rack + one shot + rule transition (CPU only).

Run against the isolated CMake build via BILLIARDS_FASTFIZ_BUILD_DIR.
"""

import numpy as np

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_session import (
    NineBallRackSession, create_nine_ball_rack,
)
from src.micro.shot_control import PhysicalShot5D


def test_real_nine_ball_diamond_can_execute_one_physical_rule_turn():
    rack = create_nine_ball_rack(breaker=0, seed=12345)
    simulator = MultiBallFastFizSimulator(rack.config)
    session = NineBallRackSession(simulator, rack)
    assert simulator.native.get_ball_placement_code() == 0
    before_positions, before_pockets = simulator.snapshot()
    np.testing.assert_array_equal(before_pockets, np.full(10, -1))
    np.testing.assert_allclose(before_positions, rack.ball_positions, atol=1e-6)

    # Straight break towards the rack. No claim that a legal break/pot will
    # occur: the test verifies physics->events->rule logic and state sync.
    result = session.step(
        target_ball=1,
        shot=PhysicalShot5D(
            speed_m_s=2.5, azimuth_deg=90.0,
            elevation_deg=10.0, tip_x_mm=0.0, tip_y_mm=0.0,
        ),
    )
    assert result.next_state.ball_count == 10
    assert result.next_state.turn_index == 1
    assert result.next_state.break_shot is False
    assert result.next_state.current_player in (0, 1)
    assert result.rack_winner in (None, 0, 1)
    if result.rack_winner is None:
        snapshot_positions, snapshot_pockets = simulator.snapshot()
        np.testing.assert_allclose(
            snapshot_positions, result.next_state.ball_positions, atol=1e-6
        )
        np.testing.assert_array_equal(
            snapshot_pockets, result.next_state.ball_pocket_indices
        )
