"""CPU-only structural tests for controlled Phase31 model x depth audit."""
from copy import deepcopy

import pytest

from scripts.experiments.diagnose_phase31_depth_factorial import (
    _verify_depth2, summarize,
)


def _outcome(*, foul, keeps, potted, kind="DIRECT_ATTACK"):
    return {
        "foul": foul,
        "target_ball_pocketed": potted,
        "exact_target_pocket_success": potted and not foul,
        "rack_winner": None,
        "next_player": 0 if keeps else 1,
        "goal_kind": kind,
    }


def _root(goal, outcome):
    return {
        "goal_v3": goal,
        "root_visits": [6, 5],
        "root_goal_v3": [[1.0, 0.0], [0.0, 1.0]],
        "shot": outcome,
    }


def test_factorial_counts_depth_introduced_foul_only_for_cprime():
    case = {
        "acting_player": 0,
        "results": {
            "base": {
                "depth_1": _root([1.0, 0.0], _outcome(foul=False, keeps=True, potted=True)),
                "depth_2": _root([1.0, 0.0], _outcome(foul=False, keeps=True, potted=True)),
            },
            "final": {
                "depth_1": _root([1.0, 0.0], _outcome(foul=False, keeps=True, potted=True)),
                "depth_2": _root([0.0, 1.0], _outcome(foul=True, keeps=False, potted=False)),
            },
        },
    }
    report = summarize([case], expected=1, invalid=[])
    assert report["compared_boards"] == 1
    assert report["conditions"]["base_depth_2"]["fouls"] == 0
    assert report["conditions"]["final_depth_2"]["fouls"] == 1
    assert report["conditions"]["final_depth_1"]["shooter_keeps_turn"] == 1
    assert report["depth_effects"]["base"]["choice_changed_depth_1_to_2"] == 0
    assert report["depth_effects"]["final"]["foul_introduced_at_depth_2"] == 1
    assert report["disagreements"]["depth_1"] == 0
    assert report["disagreements"]["depth_2"] == 1


def test_verify_depth2_chosen_goal_and_candidate_alignment():
    result = {
        **_root([1.0, 0.0], {
            "physical_shot": {
                "power": 1.5, "angle": 40.0,
                "cue_elevation_deg": 10.0, "tip_x_mm": 0.0, "tip_y_mm": 0.0,
            },
            "first_contact_ball": 3, "foul": False,
            "newly_pocketed": [3], "rack_winner": None, "turn_changed": False,
        }),
    }
    expected = {
        "chosen_goal_v3": [1.0, 0.0],
        "candidates": [
            {"goal_v3": [1.0, 0.0], "visits": 6},
            {"goal_v3": [0.0, 1.0], "visits": 5},
        ],
    }
    _verify_depth2(
        result, expected,
        original_shot=result["shot"]["physical_shot"],
        original_physical={"first_contact_ball": 3},
        original_rules={
            "foul": False, "newly_pocketed": [3],
            "rack_winner": None, "turn_changed": False,
        },
    )
    broken = deepcopy(result)
    broken["root_visits"][0] = 4
    with pytest.raises(ValueError, match="root visits"):
        _verify_depth2(broken, expected)
    broken = deepcopy(result)
    broken["shot"]["foul"] = True
    with pytest.raises(ValueError, match="rules result"):
        _verify_depth2(
            broken, expected,
            original_rules={
                "foul": False, "newly_pocketed": [3],
                "rack_winner": None, "turn_changed": False,
            },
        )
