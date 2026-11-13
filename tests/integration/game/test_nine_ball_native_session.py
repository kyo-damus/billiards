"""Actual FastFiz 9-ball rack + one shot + rule transition (CPU only).

Run against the isolated CMake build via BILLIARDS_FASTFIZ_BUILD_DIR.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_agents import ShotPlan
from src.game.nine_ball_encoding_v3 import (
    NineBallGoal, generate_nine_ball_goals, reference_angle_deg,
)
from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
from src.game.nine_ball_mcts_v3 import NineBallMCTS
from src.game.nine_ball_physics_guard import NativeShotRejected, simulate_nine_ball_shot
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


@pytest.mark.parametrize("obstruction", ("rail", "ball"))
def test_mcts_replays_the_cue_clearance_adjustment_in_the_live_rack(monkeypatch, obstruction):
    import src.game.nine_ball_mcts_v3 as mcts

    rack = create_nine_ball_rack(breaker=0, seed=12345)
    positions = np.zeros_like(rack.ball_positions)
    positions[[0, 1, 9]] = (
        [(0.5, 0.038), (0.5, 0.21), (0.85, 1.1)] if obstruction == "rail"
        else [(0.5, 0.6), (0.5, 0.8), (0.5, 0.52)]
    )
    pockets = np.zeros_like(rack.ball_pocket_indices)
    pockets[[0, 1, 9]] = -1
    rack = replace(rack, ball_positions=positions, ball_pocket_indices=pockets, break_shot=False)
    simulator = MultiBallFastFizSimulator(rack.config)
    rules, context = NineBallRules(), NineBallContext()
    goal = NineBallGoal("DIRECT_ATTACK", 1)
    shot = PhysicalShot5D(2.2, 90, 10, 0, 0)
    with pytest.raises(NativeShotRejected) as exc:
        simulate_nine_ball_shot(simulator, rules, rack, context, target_ball=1, shot=shot)
    assert (exc.value.precondition, exc.value.placement) == (128, 0)

    monkeypatch.setattr(mcts, "generate_nine_ball_goals", lambda *args, **kwargs: (goal,))
    calls = []

    def plan_goal(state, chosen, context):
        calls.append(chosen)
        return ShotPlan(1, shot, "DIRECT", macro_goal=chosen)

    agent = NineBallMCTSAgent.__new__(NineBallMCTSAgent)
    agent.device = "cpu"
    agent.plan_goal = plan_goal
    agent.searcher = NineBallMCTS(agent, simulator=simulator, rules=rules, simulations=2, max_depth=1)
    plan = agent.choose_shot(rack, context)
    assert calls == [goal]
    assert plan.macro_goal == goal
    assert plan.shot.elevation_deg > shot.elevation_deg
    assert plan.macro_search["root_visits"] == [2]
    assert plan.macro_search["chosen_cue_clearance_adjusted"]

    live = NineBallRackSession(MultiBallFastFizSimulator(rack.config), rack)
    outcome = live.step(target_ball=plan.target_ball, shot=plan.shot)
    assert not outcome.foul
    assert live.last_physical.precondition_code == 0
    np.testing.assert_array_equal(
        outcome.next_state.ball_positions, agent.searcher.last_root.edges[0].child.state.ball_positions,
    )
