"""CPU-only test cases for the isolated Phase31 ball-in-hand placement trial."""
from dataclasses import replace

import pytest

from scripts.experiments.diagnose_phase31_bih_placement import (
    _summarize, quality_tuple,
)
from src.game.nine_ball_agents import GeometryNineBallAgent
from src.game.nine_ball_cue_placement_experiment import (
    geometry_placement_ranking, sample_placements,
)
from src.game.nine_ball_session import create_nine_ball_rack


def test_placement_ranking_is_deterministic_and_does_not_mutate_state():
    board = create_nine_ball_rack(breaker=0, seed=320)
    state = replace(board, ball_in_hand=True)
    original = state.ball_positions.copy()
    candidates = GeometryNineBallAgent().cue_placement_candidates(state)
    ranked = geometry_placement_ranking(state, candidates)
    repeat = geometry_placement_ranking(state, candidates)
    assert ranked == repeat
    assert len(ranked) == len(candidates)
    assert (state.ball_positions == original).all()
    assert ranked[0]["clear_pocket_count"] >= ranked[-1]["clear_pocket_count"]
    assert len({tuple(x["xy"]) for x in ranked}) == len(ranked)


def test_sample_includes_original_baseline_and_best_geometry_and_budget():
    board = create_nine_ball_rack(breaker=0, seed=320)
    state = replace(board, ball_in_hand=True)
    candidates = GeometryNineBallAgent().cue_placement_candidates(state)
    rank = geometry_placement_ranking(state, candidates)
    picks = sample_placements(candidates, rank, max_placements=17)
    assert len(picks) == 17
    assert {tuple(p["xy"]) for p in picks} >= {
        tuple(candidates[0]), tuple(rank[0]["xy"])
    }
    assert len({tuple(p["xy"]) for p in picks}) == len(picks)
    with pytest.raises(ValueError, match="max_placements"):
        sample_placements(candidates, rank, max_placements=1)


def test_trial_quality_orders_legal_pot_above_foul_and_turn():
    def metric(*, exact, potted, foul, next_player):
        return {
            "exact_target_pocket_success": exact,
            "target_ball_pocketed": potted,
            "foul": foul, "rack_winner": None,
            "next_player": next_player,
        }
    legal_pot = metric(exact=True, potted=True, foul=False, next_player=0)
    illegal_pot = metric(exact=False, potted=True, foul=True, next_player=1)
    clean_miss = metric(exact=False, potted=False, foul=False, next_player=1)
    assert quality_tuple(legal_pot, 0) > quality_tuple(clean_miss, 0)
    assert quality_tuple(clean_miss, 0) > quality_tuple(illegal_pot, 0)


def test_summary_counts_placement_outcomes_with_no_invented_wins():
    def trial(success, foul, retains):
        return {
            "trial": {
                "valid": True,
                "quality": [int(success), int(success), int(retains), int(not foul)],
                "metrics": {
                    "exact_target_pocket_success": success, "foul": foul,
                },
            },
        }
    row = {
        "selected": {
            "baseline": trial(False, True, False),
            "geometry": trial(True, False, True),
            "sampled_oracle": trial(True, False, True),
        },
    }
    result = _summarize([row])
    assert result["baseline"]["fouls"] == 1
    assert result["geometry"]["declared_legal_target_pots"] == 1
    assert result["sampled_oracle"]["retained_turn_or_rack_win"] == 1
