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
                  completed=False, stop="max_turns"):
    path = folder / f"rollout_seed_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = (
        np.zeros(12, dtype=np.float32) if initial_state is None
        else np.asarray(initial_state, dtype=np.float32)
    )
    chosen = np.zeros(31, dtype=np.float32)
    chosen[1 if goal == "SAFETY" else 0] = 1
    chosen[{"DIRECT_ATTACK": 2, "POSITION_ATTACK": 3, "SAFETY": 4}[goal]] = 1
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
        state_pocket_indices=np.full((1, 3), -1, dtype=np.int32),
        next_state_pocket_indices=np.full((1, 3), -1, dtype=np.int32),
        score_delta=np.asarray([[score0, score1]], dtype=np.float32),
        macro_local_reward=np.asarray([reward], dtype=np.float32),
        player_before=np.asarray([0], dtype=np.int32),
        player_after=np.asarray([0], dtype=np.int32),
        root_turn_index=np.asarray([0], dtype=np.int32),
        root_candidate=chosen.reshape(1, 31),
        root_visit_count=np.asarray([1], dtype=np.int32),
        root_selected=np.asarray([True], dtype=np.bool_),
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
