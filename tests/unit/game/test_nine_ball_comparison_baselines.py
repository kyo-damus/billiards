"""Fast CPU checks for Phase30 adapted comparison baselines."""
from dataclasses import replace

import pytest

from scripts.eval_nine_ball_comparison_baselines_v1 import _check_seed_isolation
from src.game.nine_ball_comparison_baselines import PotOnlyGreedyAgent
from src.game.nine_ball_rules import NineBallContext
from src.game.nine_ball_session import create_nine_ball_rack


@pytest.fixture
def selector():
    # Only test pure goal selection. Real checkpoint loading is tested in
    # test_nine_ball_learned_agent.py and DGX integration smoke.
    return object.__new__(PotOnlyGreedyAgent)


def test_baseline_only_attacks_nonbreak_positions(selector):
    state = replace(create_nine_ball_rack(breaker=0, seed=12), break_shot=False)
    goal = selector._choose_goal(state, NineBallContext())
    assert goal.kind == "DIRECT_ATTACK"
    assert goal.target_ball == min(state.on_table_object_balls())


def test_baseline_falls_back_to_lowest_ball_without_clear_pocket(selector, monkeypatch):
    import src.game.agents.baselines.pot_only_greedy as module
    monkeypatch.setattr(module, "direct_pocket_candidates", lambda state, target: ())
    state = replace(create_nine_ball_rack(breaker=0, seed=12), break_shot=False)
    goal = selector._choose_goal(state, NineBallContext())
    assert (goal.kind, goal.target_ball, goal.target_pocket) == ("DIRECT_ATTACK", 1, None)


def test_baseline_handles_wpa_break_and_table_choice(selector):
    state = create_nine_ball_rack(breaker=0, seed=12)
    assert selector._choose_goal(state, NineBallContext()).kind == "BREAK"
    decided = selector._choose_goal(
        replace(state, break_shot=False), NineBallContext(pending_choice="illegal_break")
    )
    assert decided.kind == "ACCEPT_TABLE"


def test_seed_preflight_rejects_overlap_and_accepts_new_seed():
    training = {
        "settings": {"race_to": 9},
        "rounds": [{"start_seed": 1400000, "matches": 512, "race_to": 9}],
    }
    with pytest.raises(ValueError, match="overlap"):
        _check_seed_isolation(training, 1400000, 1)
    with pytest.raises(ValueError, match="overlap"):
        _check_seed_isolation(training, 1400018, 1)
    _check_seed_isolation(training, 1460000, 32)
