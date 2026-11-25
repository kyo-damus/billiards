"""CPU-only tests for the Phase31 two-ply diagnostic comparison logic."""
from types import SimpleNamespace

import pytest

from scripts.experiments.diagnose_phase31_two_ply_values import (
    _match_target, _rank_summary, _reproduce,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal, encode_goal_v3


def _candidate(goal, visits, q, prior):
    return SimpleNamespace(
        goal=goal, visits=visits, q=q, prior=prior, invalid_reason=None,
    )


def test_one_step_ranking_reverses_even_if_both_root_candidates_are_visited():
    a = {
        "first_successor_value": {"available": True, "root_value": 0.35},
        "root_q": 0.15,
    }
    b = {
        "first_successor_value": {"available": True, "root_value": 0.25},
        "root_q": 0.20,
    }
    r = _rank_summary(a, b)
    assert r["one_step_margin_C_choice_minus_Cprime_choice"] == pytest.approx(0.10)
    assert r["depth2_root_q_margin_C_choice_minus_Cprime_choice"] == pytest.approx(-0.05)
    assert r["rank_reversed"]


def test_nonvisited_candidate_is_not_given_a_fabricated_mcts_q():
    a = {
        "first_successor_value": {"available": False, "reason": "not_visited_in_root_search"},
        "root_q": None,
    }
    b = {
        "first_successor_value": {"available": True, "root_value": 0.0},
        "root_q": 0.0,
    }
    r = _rank_summary(a, b)
    assert r["rank_reversed"] is None
    assert r["depth2_root_q_margin_C_choice_minus_Cprime_choice"] is None


def test_reproduce_rejects_changed_root_visit_distribution():
    a = NineBallGoal("DIRECT_ATTACK", 3, 0)
    b = NineBallGoal("DIRECT_ATTACK", 3, 1)
    edges = [_candidate(a, 7, 0.3, 0.55), _candidate(b, 5, 0.2, 0.45)]
    root = SimpleNamespace(edges=edges)
    old = {
        "chosen_goal_v3": encode_goal_v3(a).tolist(),
        "candidates": [
            {"goal_v3": encode_goal_v3(a).tolist(), "visits": 7},
            {"goal_v3": encode_goal_v3(b).tolist(), "visits": 5},
        ],
    }
    _reproduce(root, old)
    assert _match_target(root, encode_goal_v3(b).tolist()) is edges[1]
    edges[0].visits = 6
    with pytest.raises(ValueError, match="root visits"):
        _reproduce(root, old)


def test_match_target_requires_unique_exact_goal():
    a = NineBallGoal("DIRECT_ATTACK", 3, 0)
    root = SimpleNamespace(edges=[
        _candidate(a, 1, 0.0, 1.0),
        _candidate(a, 0, 0.0, 1.0),
    ])
    with pytest.raises(ValueError, match="exactly one"):
        _match_target(root, encode_goal_v3(a).tolist())
