"""CPU-only tests for same-state MCTS choice diagnostics (no FastFiz needed)."""
import json

import pytest

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _eligible_rows,
    _sample_cases,
    compare_roots,
)


def _root(chosen, *, visits=(8, 2), qs=(0.7, 0.2), reversed_order=False):
    candidates = [
        {"goal_v3": [1.0, 0.0], "prior": 0.65, "visits": visits[0],
         "q": qs[0], "invalid_reason": None},
        {"goal_v3": [0.0, 1.0], "prior": 0.35, "visits": visits[1],
         "q": qs[1], "invalid_reason": None},
    ]
    if reversed_order:
        candidates = list(reversed(candidates))
    selected = [1.0, 0.0] if chosen == 0 else [0.0, 1.0]
    selected_index = next(i for i, c in enumerate(candidates) if c["goal_v3"] == selected)
    return {
        "chosen_index": selected_index,
        "chosen_goal_v3": selected,
        "candidates": candidates,
    }


def test_compare_roots_aligns_candidates_even_if_they_have_different_order():
    base = _root(0)
    final = _root(1, visits=(1, 9), qs=(0.2, 0.8), reversed_order=True)
    result = compare_roots(base, final)
    assert result["chosen_different"] is True
    assert result["candidate_sets_equal"] is True
    assert result["normalized_visit_l1"] == pytest.approx(0.7)
    assert result["maximum_prior_difference"] == 0.0
    assert result["maximum_root_q_difference"] == pytest.approx(0.6)
    assert result["base_selected_visits"] == 8
    assert result["final_selected_visits"] == 9


def test_compare_roots_flags_candidate_mismatch():
    base = _root(0)
    final = _root(1)
    final["candidates"][1]["goal_v3"] = [0.0, 2.0]
    result = compare_roots(base, final)
    assert result["candidate_sets_equal"] is False
    assert result["normalized_visit_l1"] is None


def test_eligible_rows_only_select_physical_non_break_actors():
    basic = {
        "player": 1,
        "state": {"ball_in_hand": False, "break_shot": False},
        "rule_context": {"pending_choice": None},
        "match_context": {"race_to": 9},
        "chosen_action": {"goal_v3": [1.0, 0.0]},
    }
    from copy import deepcopy
    break_row = deepcopy(basic)
    break_row["state"]["break_shot"] = True
    wrong_player = deepcopy(basic)
    wrong_player["player"] = 0
    choice = deepcopy(basic)
    choice["rule_context"]["pending_choice"] = "push_out"
    game = {
        "player_models": ["base", "final"],
        "trajectory": [break_row, wrong_player, choice, basic],
    }
    eligible = _eligible_rows(game, acting_model="final")
    assert len(eligible) == 1
    assert eligible[0][0] == 3


def test_sample_cases_filters_for_games_won_by_base(tmp_path):
    row = {
        "player": 1, "state": {"ball_in_hand": False, "break_shot": False},
        "rule_context": {"pending_choice": None},
        "match_context": {"race_to": 9},
        "chosen_action": {"goal_v3": [1.0, 0.0]},
    }
    for seed, winner in ((1690000, "base"), (1690018, "final")):
        (tmp_path / f"seed_{seed}_break_base.json").write_text(json.dumps({
            "seed": seed, "first_breaker": 0, "winner": winner,
            "match_finished": True, "player_models": ["base", "final"],
            "trajectory": [row],
        }))
    selected = _sample_cases(
        tmp_path, winner="base", acting_model="final", max_states=8,
        random_seed=42,
    )
    assert len(selected) == 1
    assert selected[0]["match_seed"] == 1690000
    assert selected[0]["row_index"] == 0
