"""GPU-free decoding/sampling tests for holdout Micro audit."""
from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.evaluation.audit_nine_ball_micro_generalization import (
    goal_from_vector, sample_from_trajectory, select_cases, all_micro_board_hashes,
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
    a, boards_a, meta_a = select_cases(train, max_matches=1, per_goal=5, seed=1, goal_source="logged")
    b, boards_b, meta_b = select_cases(eval_folder, max_matches=1, per_goal=5, seed=2, goal_source="logged")
    assert len(a["DIRECT_ATTACK"]) == 1
    assert len(b["DIRECT_ATTACK"]) == 1
    assert boards_a == boards_b
    blacklist = all_micro_board_hashes(eval_folder, goal_source="logged")
    assert blacklist == boards_a
    filtered, _, stats = select_cases(
        train, max_matches=1, per_goal=5, seed=1,
        goal_source="logged", excluded_board_hashes=blacklist,
    )
    assert not filtered
    assert stats["excluded_holdout_overlap_cases"] == 1
    assert meta_a["sampled_match_files"] == meta_b["sampled_match_files"] == 1


def test_rejects_nonmicro_pocket_data():
    v = encode_goal_v3(NineBallGoal("DIRECT_ATTACK", 1, 0)).copy()
    v[21] = 1  # Invalid: both pocket 0 and pocket 1 are active
    with pytest.raises(ValueError):
        goal_from_vector(v)



def test_can_read_all_root_candidates_and_validate_training_provenance(tmp_path):
    from types import SimpleNamespace
    from scripts.evaluation.audit_nine_ball_micro_generalization import _check_provenance

    rack = replace(create_nine_ball_rack(breaker=0, seed=8), break_shot=False)
    attack = NineBallGoal("DIRECT_ATTACK", 1, 0)
    safety = NineBallGoal("SAFETY", 1)
    row = {
        "state": rack_state_json(rack),
        "rule_context": rule_context_json(NineBallContext()),
        "chosen_action": {
            "goal_v3": encode_goal_v3(attack).tolist(),
            "macro_search": {"goal_v3": [
                encode_goal_v3(attack).tolist(),
                encode_goal_v3(safety).tolist(),
            ]},
        },
    }
    training_folder = tmp_path / "training"
    heldout_folder = tmp_path / "holdout"
    training_folder.mkdir()
    heldout_folder.mkdir()
    (training_folder / "match_seed_100.json").write_text(json.dumps({"trajectory": [row]}))
    found, _, _ = select_cases(
        training_folder, max_matches=1, per_goal=10, seed=5, goal_source="candidates"
    )
    assert set(found) == {"DIRECT_ATTACK", "SAFETY"}

    summary = tmp_path / "micro_summary.json"
    summary.write_text(json.dumps({
        "training_data": {
            "rollout_dirs": [str(training_folder)],
            "holdout_rollout_dir": str(heldout_folder),
        },
        "goal_source": "candidates",
        "wpa_break_rules": True,
        "outcome_position": True,
        "training_steps_this_run": 100000,
    }))
    args = SimpleNamespace(
        training_summary=summary, train_dir=training_folder,
        holdout_dir=heldout_folder, goal_source="candidates", outcome_position=True,
    )
    assert _check_provenance(args)["training_steps_this_run"] == 100000
    args.holdout_dir = training_folder
    with pytest.raises(ValueError, match="Holdout directory"):
        _check_provenance(args)


def test_position_decomposition_separates_continuation_and_stop_distance():
    from scripts.evaluation.audit_nine_ball_micro_generalization import (
        _new_position_details, summarize_position_details,
    )
    details = _new_position_details()
    details["exact_pots"] = 4
    details["continued_after_pot"] = 3
    details["within_goal_radius_after_pot"] = 2
    details["continued_and_within_goal_radius"] = 1
    details["stop_distances_after_pot_m"] = [0.03, 0.08, 0.25, 0.4]
    details["stop_distances_after_continued_pot_m"] = [0.03, 0.25, 0.4]
    details["relative_stop_distances_after_pot"] = [0.3, 0.8, 2.5, 4]
    details["relative_stop_distances_after_continued_pot"] = [0.3, 2.5, 4]
    result = summarize_position_details(details)
    assert result["legal_exact_pots"] == 4
    assert result["continued_after_exact_pot"] == 3
    assert result["within_goal_radius_after_exact_pot"] == 2
    assert result["continued_and_within_goal_radius"] == 1
    assert result["continuation_rate_given_exact_pot"] == 0.75
    assert result["within_radius_rate_given_continued_pot"] == pytest.approx(1 / 3)
    assert result["goal_success_rate_given_exact_pot"] == 0.25
    assert result["failed_continuation_after_pot"] == 1
    assert result["continued_but_missed_goal_radius"] == 2
    assert result["within_absolute_radius_m_after_continued_pot"] == {
        "0.1": 1, "0.2": 1, "0.3": 2,
    }
    assert result["stopping_distance_m_after_continued_pot"]["median"] == 0.25


@pytest.mark.parametrize(
    "stopped_xy,turn_changed,expected_success",
    [
        ((0.52, 0.60), False, True),
        ((0.90, 0.60), False, False),
        ((0.52, 0.60), True, False),
    ],
)
def test_record_trial_captures_position_failure_without_second_simulation(
    monkeypatch, stopped_xy, turn_changed, expected_success
):
    from collections import Counter
    from types import SimpleNamespace
    from scripts.evaluation.audit_nine_ball_micro_generalization import _new_position_details
    from scripts.eval_nine_ball_micro_v3 import record_trial
    from scripts.train_nine_ball_micro_sac import TrainingSample
    import scripts.eval_nine_ball_micro_v3 as evaluation

    state = replace(create_nine_ball_rack(breaker=0, seed=7), break_shot=False)
    goal = NineBallGoal("POSITION_ATTACK", 1, 0, (0.5, 0.6), 0.10)
    context = NineBallContext()
    sample = TrainingSample(state, context, goal)

    pockets = state.ball_pocket_indices.copy()
    pockets[state.array_index_for_ball(1)] = 0
    positions = state.ball_positions.copy()
    positions[0, :] = stopped_xy
    physics = SimpleNamespace(
        pocket_indices=pockets, positions=positions, first_contact_ball=1,
    )
    outcome = SimpleNamespace(
        foul=False, illegal_break=False, turn_changed=turn_changed,
        rack_winner=None, next_state=state, newly_pocketed=(1,),
    )
    called = []
    def fake_execute(*args, **kwargs):
        called.append(1)
        return physics, outcome

    monkeypatch.setattr(evaluation, "execute_valid_training_shot", fake_execute)
    agent = SimpleNamespace(plan_goal=lambda *args: SimpleNamespace(shot=object()))
    details = _new_position_details()
    counts = Counter()
    assert record_trial(
        agent, sample, object(), object(), counts, Counter(),
        outcome_position=True, cue_clearance_retry=False,
        position_details=details,
    ) is expected_success
    assert called == [1]
    assert counts["exact_target_pot"] == 1
    assert details["exact_pots"] == 1
    assert details["continued_after_pot"] == int(not turn_changed)
    assert details["within_goal_radius_after_pot"] == int(stopped_xy[0] == 0.52)
    assert details["continued_and_within_goal_radius"] == int(expected_success)
    assert counts["goal_success"] == int(expected_success)
