"""GPU-free decoding/sampling tests for holdout Micro audit."""
from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.evaluation.audit_nine_ball_micro_generalization import (
    goal_from_vector, sample_from_trajectory, select_cases,
)
from scripts.collect_nine_ball_self_play import rack_state_json, rule_context_json
from src.game.nine_ball_encoding_v3 import NineBallGoal, encode_goal_v3
from src.game.nine_ball_rules import NineBallContext
from src.game.nine_ball_session import create_nine_ball_rack
from src.game.rack_state import TableGameConfig


def test_decode_roundtrip_direct_and_position():
    for goal in (
        NineBallGoal("DIRECT_ATTACK", 1, 2),
        NineBallGoal("POSITION_ATTACK", 1, 2, (0.5, 0.6), 0.1),
        NineBallGoal("SAFETY", 1),
        NineBallGoal("BREAK", 1),
    ):
        got = goal_from_vector(encode_goal_v3(goal))
        assert got is not None
        np.testing.assert_allclose(encode_goal_v3(got), encode_goal_v3(goal))


def test_skips_table_choices_and_fixed_heuristic_shots():
    from src.micro.shot_control import PhysicalShot5D

    assert goal_from_vector(encode_goal_v3(NineBallGoal("ACCEPT_TABLE", 1))) is None
    explicit = NineBallGoal(
        "SAFETY", 1, explicit_shot=PhysicalShot5D(0.8, 0.0, 10.0, 0.0, 0.0)
    )
    assert goal_from_vector(encode_goal_v3(explicit)) is None


def test_reads_saved_proposed_and_evaluation_trajectory_goals(tmp_path):
    rack = replace(create_nine_ball_rack(breaker=0, seed=7), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)
    row = {
        "state": rack_state_json(rack),
        "rule_context": rule_context_json(NineBallContext()),
        "chosen_action": {"goal_v3": encode_goal_v3(goal).tolist()},
    }
    case = sample_from_trajectory(row, TableGameConfig.nine_ball_inventory())
    assert case.selected_goal.kind == "DIRECT_ATTACK"
    assert case.selected_goal.target_pocket == 0
    train = tmp_path / "train"
    eval_folder = tmp_path / "eval"
    train.mkdir()
    eval_folder.mkdir()
    (train / "match_seed_100.json").write_text(json.dumps({"trajectory": [row]}))
    (eval_folder / "seed_200_break_final.json").write_text(json.dumps({"trajectory": [row]}))
    a, boards_a, meta_a = select_cases(train, max_matches=1, per_goal=5, seed=1)
    b, boards_b, meta_b = select_cases(eval_folder, max_matches=1, per_goal=5, seed=2)
    assert len(a["DIRECT_ATTACK"]) == 1
    assert len(b["DIRECT_ATTACK"]) == 1
    assert boards_a == boards_b
    assert meta_a["sampled_match_files"] == meta_b["sampled_match_files"] == 1


def test_rejects_nonmicro_pocket_data():
    v = encode_goal_v3(NineBallGoal("DIRECT_ATTACK", 1, 0)).copy()
    v[21] = 1  # Invalid: both pocket 0 and pocket 1 are active
    with pytest.raises(ValueError):
        goal_from_vector(v)
