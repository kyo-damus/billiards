"""Actual FastFiz 9-ball rack + one shot + rule transition (CPU only).

Run against the isolated CMake build via BILLIARDS_FASTFIZ_BUILD_DIR.
"""

from dataclasses import replace

import numpy as np

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    generate_nine_ball_goals, reference_angle_deg,
)
from src.game.nine_ball_physics_guard import simulate_nine_ball_shot
from src.game.nine_ball_position import generate_outcome_position_goals
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_session import (
    NineBallRackSession, create_nine_ball_rack,
)
from src.micro.shot_control import PhysicalShot5D, Shot5DDecoder


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


def test_position_goal_has_a_native_witness_and_leaves_the_next_ball():
    rack = create_nine_ball_rack(breaker=0, seed=12345)
    positions = np.zeros_like(rack.ball_positions)
    positions[[0, 1, 9]] = [(0.52, 1.68), (0.28, 1.94), (0.75, 1.2)]
    pockets = np.zeros_like(rack.ball_pocket_indices)
    pockets[[0, 1, 9]] = -1
    rack = replace(
        rack, ball_positions=positions, ball_pocket_indices=pockets, break_shot=False,
    )
    simulator = MultiBallFastFizSimulator(rack.config)
    context, rules = NineBallContext(), NineBallRules()
    goals = generate_outcome_position_goals(rack, context, simulator, rules)
    assert goals == generate_outcome_position_goals(rack, context, simulator, rules)
    assert [goal for goal in goals if goal.kind != "POSITION_ATTACK"] == [
        goal for goal in generate_nine_ball_goals(rack) if goal.kind != "POSITION_ATTACK"
    ]
    position, = (
        goal for goal in goals if goal.kind == "POSITION_ATTACK" and goal.target_pocket == 2
    )
    assert position.target_ball == 1
    assert position.explicit_shot is None
    decoder = Shot5DDecoder()
    heading = reference_angle_deg(rack, position)
    witnesses = []
    for speed in (-0.6, 0.0, 0.6):
        for tip_y in (0.0, -0.4, 0.4):
            physical, outcome = simulate_nine_ball_shot(
                simulator, rules, rack, context, target_ball=1,
                shot=decoder.decode((speed, 0.0, -0.8, 0.0, tip_y), heading),
            )
            if (
                physical.pocket_indices[1] == 2 and not outcome.foul
                and not outcome.turn_changed and not outcome.next_state.is_pocketed(9)
            ):
                witnesses.append(outcome.next_state.ball_positions[0].tolist())
    assert list(position.cue_stop_xy) in witnesses
