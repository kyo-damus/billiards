"""CPU-only unit tests for Phase31 same-board one-shot diagnostics."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments.diagnose_phase31_shot_outcomes import (
    load_cases, shot_metrics, summarize,
)
from src.game.nine_ball_agents import ShotPlan
from src.game.nine_ball_encoding_v3 import NineBallGoal
from src.game.nine_ball_rules import NineBallContext
from src.game.nine_ball_session import create_nine_ball_rack
from src.micro.shot_control import PhysicalShot5D


def test_shot_metrics_reports_real_pot_and_position_goal_only_when_met():
    state = create_nine_ball_rack(breaker=0, seed=123)
    pos = state.ball_positions[0].copy()
    goal = NineBallGoal(
        "POSITION_ATTACK", 1, 0, (float(pos[0]), float(pos[1])), 0.2
    )
    plan = ShotPlan(1, PhysicalShot5D(1.2, 45, 10, 0, 0), "test", 0, goal)
    pockets = state.ball_pocket_indices.copy()
    pockets[1] = 0
    new_state = replace(state, ball_pocket_indices=pockets)
    outcome = SimpleNamespace(
        next_state=new_state, next_context=NineBallContext(),
        foul=False, foul_reasons=(), newly_pocketed=(1,),
        rack_winner=None, turn_changed=False,
    )
    physical = SimpleNamespace(first_contact_ball=1)
    metrics = shot_metrics(state, NineBallContext(), plan, physical, outcome)
    assert metrics["first_contact_ball"] == 1
    assert metrics["target_ball_pocketed"]
    assert metrics["exact_target_pocket_success"]
    assert metrics["full_position_goal_success"]
    assert metrics["cue_stop_distance_m"] == pytest.approx(0)
    assert metrics["next_lowest_ball"] == 2
    assert isinstance(metrics["next_direct_pocket_options"], list)


def test_shot_metrics_does_not_credit_fouled_pot_and_suppresses_placement_proxy():
    state = create_nine_ball_rack(breaker=0, seed=123)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)
    plan = ShotPlan(1, PhysicalShot5D(1.2, 45, 10, 0, 0), "test", 0, goal)
    pockets = state.ball_pocket_indices.copy()
    pockets[1] = 0
    new_state = replace(
        state, ball_pocket_indices=pockets, current_player=1, ball_in_hand=True
    )
    outcome = SimpleNamespace(
        next_state=new_state, next_context=NineBallContext(),
        foul=True, foul_reasons=("cue_scratch",), newly_pocketed=(1,),
        rack_winner=None, turn_changed=True,
    )
    metrics = shot_metrics(
        state, NineBallContext(), plan,
        SimpleNamespace(first_contact_ball=1), outcome,
    )
    assert not metrics["exact_target_pocket_success"]
    assert metrics["next_direct_pocket_count"] is None
    assert metrics["ball_in_hand_next"]
    assert metrics["full_position_goal_success"] is None


def test_summarize_paired_counts():
    a = {
        "foul": False, "target_ball_pocketed": True,
        "exact_target_pocket_success": True, "rack_winner": None,
        "next_player": 0, "full_position_goal_success": True,
    }
    b = {
        "foul": True, "target_ball_pocketed": False,
        "exact_target_pocket_success": False, "rack_winner": None,
        "next_player": 1, "full_position_goal_success": None,
    }
    info = summarize(
        [{"acting_player": 0, "base": a, "final": b}], [], 1
    )
    assert info["compared"] == 1
    assert info["base"]["shooter_keeps_turn"] == 1
    assert info["final"]["fouls"] == 1
    assert info["base"]["position_goals_fully_satisfied"] == 1


def test_load_cases_rejects_tampered_logged_goal(tmp_path):
    evaluation = tmp_path / "evaluation"
    diagnostic = tmp_path / "diagnostic"
    evaluation.mkdir()
    diagnostic.mkdir()
    source = evaluation / "seed_1690000_break_base.json"
    source.write_text(json.dumps({
        "seed": 1690000, "winner": "base",
        "player_models": ["base", "final"],
        "trajectory": [{
            "player": 1, "state": {"current_player": 1},
            "rule_context": {}, "match_context": {},
            "chosen_action": {"goal_v3": [1.0, 0.0]},
        }],
    }))
    (diagnostic / "summary.json").write_text(json.dumps({
        "diagnostic_kind": "phase31_identical_state_root_mcts_replay",
        "evaluation_dir": str(evaluation),
        "compared_states": 1,
    }))
    (diagnostic / "details.json").write_text(json.dumps({
        "cases": [{
            "source_file": str(source), "match_seed": 1690000,
            "row_index": 0, "logged_choice_reproduced": True,
            "comparison": {"chosen_different": True, "candidate_sets_equal": True},
            "final": {"chosen_goal_v3": [0.0, 1.0]},
        }],
    }))
    with pytest.raises(ValueError, match="differs"):
        load_cases(diagnostic, evaluation)
