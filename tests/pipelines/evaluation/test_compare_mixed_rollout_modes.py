"""Paired-run comparison reads saved experience without FastFiz or training."""

import json

import numpy as np
import pytest

from scripts.compare_mixed_rollout_modes import (
    compare_paired_rollouts,
    discover_seeded_rollouts,
    format_comparison,
    main,
    parse_args,
)


def write_rollout(folder, seed, *, goal="DIRECT_ATTACK", reward=0.0,
                  initial_state=None, score0=0.0, score1=0.0,
                  completed=False, stop="max_turns",
                  target_pocket=None, actual_pocket=None,
                  first_root_candidates=None, first_root_visits=None,
                  first_root_selected=None):
    path = folder / f"rollout_seed_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = (
        np.zeros(12, dtype=np.float32) if initial_state is None
        else np.asarray(initial_state, dtype=np.float32)
    )
    chosen = np.zeros(31, dtype=np.float32)
    chosen[1 if goal == "SAFETY" else 0] = 1
    chosen[{"DIRECT_ATTACK": 2, "POSITION_ATTACK": 3, "SAFETY": 4}[goal]] = 1
    if target_pocket is not None:
        chosen[5] = 1  # object ball 0
        chosen[7 + target_pocket] = 1
    before = np.full((1, 3), -1, dtype=np.int32)
    after = before.copy()
    if actual_pocket is not None:
        after[0, 1] = actual_pocket
    if first_root_candidates is None:
        first_root_candidates = chosen.reshape(1, 31)
    if first_root_visits is None:
        first_root_visits = np.ones(len(first_root_candidates), dtype=np.int32)
    if first_root_selected is None:
        first_root_selected = np.asarray(
            [True] + [False] * (len(first_root_candidates) - 1), dtype=np.bool_
        )
    np.savez_compressed(
        path,
        format_version=np.asarray(1),
        candidate_encoding_version=np.asarray(2),
        data_kind=np.asarray("executed_mixed_rollout"),
        seed=np.asarray(seed),
        stop_reason=np.asarray(stop),
        rollout_terminated=np.asarray(completed),
        state=state.reshape(1, 12),
        candidate=chosen.reshape(1, 31),
        state_pocket_indices=before,
        next_state_pocket_indices=after,
        score_delta=np.asarray([[score0, score1]], dtype=np.float32),
        macro_local_reward=np.asarray([reward], dtype=np.float32),
        player_before=np.asarray([0], dtype=np.int32),
        player_after=np.asarray([0], dtype=np.int32),
        root_turn_index=np.zeros(len(first_root_visits), dtype=np.int32),
        root_candidate=np.asarray(first_root_candidates, dtype=np.float32),
        root_visit_count=np.asarray(first_root_visits, dtype=np.int32),
        root_selected=np.asarray(first_root_selected, dtype=np.bool_),
    )
    return path


def test_paired_by_seed_and_report_first_turn_difference(tmp_path):
    a = tmp_path / "first"
    b = tmp_path / "empirical"
    write_rollout(a, 12, goal="DIRECT_ATTACK", reward=-0.5, score0=1)
    write_rollout(b, 12, goal="POSITION_ATTACK", reward=1.25, score0=1)
    write_rollout(a, 11, reward=0.5, completed=True, stop="game_terminated")
    write_rollout(b, 11, reward=0.5, completed=True, stop="game_terminated")
    result = compare_paired_rollouts(a, b)
    s = result["summary"]
    assert [x["seed"] for x in result["per_seed"]] == [11, 12]
    assert s["paired_seeds"] == 2
    assert s["pairs_with_first_actions"] == 2
    assert s["first_action_changed"] == 1
    assert s["mean_first_reward_first"] == pytest.approx(0.0)
    assert s["mean_first_reward_empirical_q"] == pytest.approx(0.875)
    assert s["mean_paired_first_reward_difference"] == pytest.approx(0.875)
    assert s["completed_games_first"] == s["completed_games_empirical_q"] == 1
    assert "NOT win rate" in format_comparison(result)


def test_refuses_unmatched_seed_set(tmp_path):
    a = tmp_path / "first"
    b = tmp_path / "empirical"
    write_rollout(a, 123)
    write_rollout(b, 124)
    with pytest.raises(ValueError, match="Unpaired seeds"):
        compare_paired_rollouts(a, b)


def test_refuses_same_seed_with_different_initial_states(tmp_path):
    a = tmp_path / "first"
    b = tmp_path / "empirical"
    write_rollout(a, 123)
    different = np.zeros(12, dtype=np.float32)
    different[0] = 0.25
    write_rollout(b, 123, initial_state=different)
    with pytest.raises(ValueError, match="initial GameState differs"):
        compare_paired_rollouts(a, b)


def test_refuses_seed_filename_mismatch(tmp_path):
    a = tmp_path / "first"
    b = tmp_path / "empirical"
    write_rollout(a, 123)
    path = write_rollout(b, 124)
    path.rename(b / "rollout_seed_123.npz")
    with pytest.raises(ValueError, match="seed metadata"):
        compare_paired_rollouts(a, b)


def test_cli_saves_json_without_overwriting(tmp_path, capsys):
    a = tmp_path / "first"
    b = tmp_path / "empirical"
    write_rollout(a, 1)
    write_rollout(b, 1, goal="SAFETY")
    output = tmp_path / "summary.json"
    args = parse_args([
        "--first-dir", str(a), "--empirical-dir", str(b),
        "--json-output", str(output),
    ])
    result = main(args)
    assert "paired seeds: 1" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(args)


def test_rejects_empty_rollout_directory(tmp_path):
    a = tmp_path / "first"
    a.mkdir()
    with pytest.raises(ValueError, match="No rollout"):
        discover_seeded_rollouts(a)


def test_pair_reports_exact_target_pot_separately_from_any_pocket(tmp_path):
    first = tmp_path / "first"
    empirical = tmp_path / "empirical"
    write_rollout(
        first, 94100, goal="DIRECT_ATTACK", target_pocket=2,
        actual_pocket=3, reward=-0.5, score0=1.0,
    )
    write_rollout(
        empirical, 94100, goal="POSITION_ATTACK", target_pocket=3,
        actual_pocket=3, reward=1.25, score0=1.0,
    )
    report = compare_paired_rollouts(first, empirical)
    s = report["summary"]
    assert s["first_target_pot_observed_first"] == 1
    assert s["first_target_pot_observed_empirical_q"] == 1
    assert s["first_target_pot_success_first"] == 0
    assert s["first_target_pot_success_empirical_q"] == 1
    assert s["first_pocketed_count_first"] == 1
    assert s["first_pocketed_count_empirical_q"] == 1
    assert "exact target pocket successes" in format_comparison(report)
    assert report["per_seed"][0]["first_target_pot_first"] is False
    assert report["per_seed"][0]["first_target_pot_empirical_q"] is True


def test_first_turn_safety_has_no_target_pocket(tmp_path):
    first = tmp_path / "first"
    empirical = tmp_path / "empirical"
    write_rollout(first, 101, goal="SAFETY")
    write_rollout(empirical, 101, goal="DIRECT_ATTACK", target_pocket=0)
    report = compare_paired_rollouts(first, empirical)
    assert report["summary"]["first_target_pot_observed_first"] == 0
    assert report["summary"]["first_target_pot_observed_empirical_q"] == 1
    assert report["per_seed"][0]["first_target_pot_first"] is None



def test_matched_first_root_ties_identical_search_but_different_winner(tmp_path):
    """A paired initial board is the fair denominator, not all later roots."""
    base = tmp_path / "first"
    alt = tmp_path / "empirical"
    direct = np.zeros(31, dtype=np.float32)
    direct[0] = direct[2] = 1.0
    position = np.zeros(31, dtype=np.float32)
    position[0] = position[3] = 1.0
    roots = np.stack([direct, position])
    write_rollout(
        base, 95000, goal="DIRECT_ATTACK", reward=0.0,
        first_root_candidates=roots, first_root_visits=[1, 1],
        first_root_selected=[True, False],
    )
    write_rollout(
        alt, 95000, goal="POSITION_ATTACK", reward=1.25,
        first_root_candidates=roots, first_root_visits=[1, 1],
        first_root_selected=[False, True],
    )
    report = compare_paired_rollouts(base, alt)
    s = report["summary"]
    assert s["matched_initial_roots"] == 1
    assert s["matched_initial_roots_with_same_search"] == 1
    assert s["initial_root_ties_first"] == 1
    assert s["initial_root_ties_empirical_q"] == 1
    assert s["initial_root_tied_both"] == 1
    assert s["initial_root_nonfirst_winner_empirical_q"] == 1
    assert s["first_action_changed"] == 1
    assert report["per_seed"][0]["first_root_candidate_count_first"] == 2
    assert report["per_seed"][0]["first_root_tie_count_empirical_q"] == 2
    assert "same-root-search=True" in format_comparison(report)


def test_detects_different_first_root_search_without_claiming_same_search(tmp_path):
    base = tmp_path / "first"
    alt = tmp_path / "empirical"
    direct = np.zeros(31, dtype=np.float32)
    direct[0] = direct[2] = 1.0
    position = np.zeros(31, dtype=np.float32)
    position[0] = position[3] = 1.0
    write_rollout(base, 9, goal="DIRECT_ATTACK")
    write_rollout(
        alt, 9, goal="DIRECT_ATTACK",
        first_root_candidates=np.stack([direct, position]),
        first_root_visits=[1, 1], first_root_selected=[True, False],
    )
    report = compare_paired_rollouts(base, alt)
    assert report["summary"]["matched_initial_roots"] == 1
    assert report["summary"]["matched_initial_roots_with_same_search"] == 0
    assert report["summary"]["initial_root_ties_first"] == 0
    assert report["summary"]["initial_root_ties_empirical_q"] == 1


def test_first_root_absent_is_reported_not_imputed(tmp_path):
    base = tmp_path / "first"
    alt = tmp_path / "empirical"
    write_rollout(
        base, 9, first_root_candidates=np.zeros((0, 31)),
        first_root_visits=[], first_root_selected=[],
    )
    write_rollout(alt, 9)
    report = compare_paired_rollouts(base, alt)
    assert report["summary"]["matched_initial_roots"] == 0
    assert report["summary"]["initial_root_ties_first"] == 0
    assert report["per_seed"][0]["first_root_search_identical"] is None
    assert report["per_seed"][0]["first_root_tie_count_first"] is None
