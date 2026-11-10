"""Versioned 10-ball and variable-ball encoders: no GPU or native build."""

from dataclasses import replace

import numpy as np
import pytest

from src.game.nine_ball_encoding_v3 import (
    MACRO_STATE_V3_DIM, MACRO_GOAL_V3_DIM, MICRO_CONDITIONED_V3_DIM,
    NineBallGoal, encode_rack_v3, encode_goal_v3,
    conditioned_observation_v3, generate_nine_ball_goals, reference_angle_deg,
)
from src.game.nine_ball_session import create_nine_ball_rack
from src.game.nine_ball_rules import NineBallContext
from src.game.rack_state import RackState, TableGameConfig
from src.micro.shot_control import PhysicalShot5D


def test_v3_state_has_actual_ball_ids_and_rule_context():
    rack = create_nine_ball_rack(breaker=1, seed=3)
    enc = encode_rack_v3(rack, NineBallContext((1, 2), True))
    assert enc.shape == (MACRO_STATE_V3_DIM,)
    assert enc[4 * 9] == 1.0
    assert enc[4 * 10] == 0.0
    assert enc[65] == 1.0
    assert enc[69] == pytest.approx(1 / 3)
    assert enc[70] == pytest.approx(2 / 3)
    assert enc[71] == 1.0
    assert np.all(np.isfinite(enc))


def test_two_ball_inventory_pads_absent_balls_and_preserves_id_2():
    config = TableGameConfig("training", (2,))
    rack = RackState(
        config=config,
        ball_positions=np.asarray([[0.5, 0.4], [0.5, 1.2]], dtype=np.float32),
        ball_pocket_indices=np.asarray([-1, -1], dtype=np.int32),
        current_player=0,
        score=np.zeros(2),
    )
    vector = encode_rack_v3(rack)
    assert vector[4 * 2] == 1.0
    assert vector[4 * 1] == 0.0
    assert vector.shape == (72,)


def test_pocketed_ball_state_is_distinct_from_missing_ball():
    rack = create_nine_ball_rack(breaker=0, seed=3)
    pockets = rack.ball_pocket_indices.copy()
    pockets[8] = 2
    after = replace(rack, ball_pocket_indices=pockets)
    vec = encode_rack_v3(after)
    assert vec[4 * 8] == 1.0
    assert vec[4 * 8 + 1] == 0.0


def test_v3_goal_encodes_actual_target_and_full_explicit_shot():
    goal = NineBallGoal(
        "SAFETY", 9,
        explicit_shot=PhysicalShot5D(1.2, 90, 10, 2, -3),
    )
    vec = encode_goal_v3(goal)
    assert vec.shape == (MACRO_GOAL_V3_DIM,)
    assert vec[3] == 1.0
    assert vec[4 + 9] == 1.0
    assert vec[29] == 1.0
    assert vec[30] == pytest.approx(0.4)
    assert vec[31] == pytest.approx(1.0)
    assert vec[34] == pytest.approx(2 / 12)


def test_generates_break_and_nonbreak_without_strategy_quota():
    rack = create_nine_ball_rack(breaker=0, seed=4)
    candidates = generate_nine_ball_goals(rack)
    assert len(candidates) == 1
    assert candidates[0].kind == "BREAK"
    assert candidates[0].target_ball == 1
    mid = replace(rack, break_shot=False)
    options = generate_nine_ball_goals(mid)
    assert options
    assert options[-1].kind == "SAFETY"
    assert all(x.target_ball == 1 for x in options)
    assert all(x.kind in ("DIRECT_ATTACK", "POSITION_ATTACK", "SAFETY") for x in options)
    assert all(
        conditioned_observation_v3(mid, x).shape == (MICRO_CONDITIONED_V3_DIM,)
        for x in options
    )
    for x in options:
        assert np.isfinite(reference_angle_deg(mid, x))


def test_disallows_candidates_before_ball_in_hand_placement():
    rack = replace(create_nine_ball_rack(breaker=0, seed=2), ball_in_hand=True)
    with pytest.raises(ValueError, match="placement"):
        generate_nine_ball_goals(rack)


def test_invalid_versioned_goals_are_rejected():
    for value in [
        ("UNKNOWN", 1, None),
        ("DIRECT_ATTACK", 16, None),
        ("DIRECT_ATTACK", 1, 6),
    ]:
        with pytest.raises(ValueError):
            NineBallGoal(value[0], value[1], value[2])
    with pytest.raises(ValueError, match="stop"):
        NineBallGoal("POSITION_ATTACK", 2, 3, cue_stop_xy=(0.4, 0.5))
