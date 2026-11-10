"""Read-only analysis of versioned mixed rollout data (no FastFiz needed)."""

import json

import numpy as np
import pytest

from scripts.analyze_mixed_rollouts import (
    format_summary,
    main,
    parse_args,
    resolve_paths,
)
from src.pipelines.rollout_analysis import read_rollout, summarize_rollouts


def _choice(goal):
    row = np.zeros(31, dtype=np.float32)
    row[1 if goal == "SAFETY" else 0] = 1
    row[{"DIRECT_ATTACK": 2, "POSITION_ATTACK": 3, "SAFETY": 4}[goal]] = 1
    return row


def _write(path, *, goals=(), stop="max_turns", completed=False,
           before=None, after=None, score=None, reward=None,
           players_before=None, players_after=None, root_turn=(),
           root_visits=(), root_selected=(), **overrides):
    n = len(goals)
    before = np.full((n, 3), -1, dtype=np.int32) if before is None else np.asarray(before, dtype=np.int32)
    after = np.full((n, 3), -1, dtype=np.int32) if after is None else np.asarray(after, dtype=np.int32)
    score = np.zeros((n, 2), dtype=np.float32) if score is None else np.asarray(score, dtype=np.float32)
    reward = np.zeros(n, dtype=np.float32) if reward is None else np.asarray(reward, dtype=np.float32)
    players_before = np.zeros(n, dtype=np.int32) if players_before is None else np.asarray(players_before, dtype=np.int32)
    players_after = np.zeros(n, dtype=np.int32) if players_after is None else np.asarray(players_after, dtype=np.int32)
    root_turn = np.asarray(root_turn, dtype=np.int32)
    root_visits = np.asarray(root_visits, dtype=np.int32)
    root_selected = np.asarray(root_selected, dtype=np.bool_)
    payload = {
        "format_version": np.asarray(1, dtype=np.int32),
        "candidate_encoding_version": np.asarray(2, dtype=np.int32),
        "data_kind": np.asarray("executed_mixed_rollout"),
        "stop_reason": np.asarray(stop),
        "rollout_terminated": np.asarray(completed),
        "state": np.zeros((n, 12), dtype=np.float32),
        "candidate": np.stack([_choice(g) for g in goals]) if n else np.zeros((0, 31), dtype=np.float32),
        "state_pocket_indices": before,
        "next_state_pocket_indices": after,
        "score_delta": score,
        "macro_local_reward": reward,
        "player_before": players_before,
        "player_after": players_after,
        "root_turn_index": root_turn,
        "root_candidate": np.tile(_choice("DIRECT_ATTACK"), (len(root_turn), 1)),
        "root_visit_count": root_visits,
        "root_selected": root_selected,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def test_completed_partial_empty_game_and_distinct_rewards(tmp_path):
    a = _write(
        tmp_path / "rollout_seed_1.npz",
        goals=("DIRECT_ATTACK", "SAFETY"),
        stop="game_terminated", completed=True,
        before=[[-1, -1, -1], [-1, 4, -1]],
        after=[[-1, 4, -1], [1, 4, -1]],
        score=[[1, 0], [0, 0]], reward=[1.0, 0.0],
        players_before=[0, 0], players_after=[0, 1],
        root_turn=[0, 0, 1], root_visits=[3, 1, 2],
        root_selected=[True, False, True],
    )
    b = _write(
        tmp_path / "rollout_seed_2.npz",
        goals=("POSITION_ATTACK",), reward=[-0.5],
        players_before=[1], players_after=[0],
    )
    c = _write(tmp_path / "rollout_seed_3.npz",
               stop="no_feasible_macro_candidates")
    summary = summarize_rollouts([a, b, c])
    assert summary["games"] == 3
    assert summary["completed_games"] == 1
    assert summary["unfinished_games"] == 2
    assert summary["completion_rate"] == pytest.approx(1 / 3)
    assert summary["stop_reasons"] == {
        "game_terminated": 1, "max_turns": 1,
        "no_feasible_macro_candidates": 1,
    }
    assert summary["total_turns"] == 3
    assert summary["goal_counts"] == {
        "DIRECT_ATTACK": 1, "POSITION_ATTACK": 1, "SAFETY": 1,
    }
    assert summary["pocketed_object_balls"] == 1
    assert summary["turns_with_pocket"] == 1
    assert summary["scratch_turns"] == 1
    assert summary["turn_switches"] == 2
    assert summary["macro_local_reward_sum"] == pytest.approx(0.5)
    assert summary["mean_local_reward_per_turn"] == pytest.approx(1 / 6)
    assert summary["score_delta_by_player"] == [1.0, 0.0]
    assert summary["mcts_roots"] == 2
    assert summary["policy_usable_roots"] == 1
    report = format_summary(summary)
    assert "NOT win rate" in report
    assert "complete: 1 (33.3%)" in report
    assert "SAFETY=1" in report
    assert "Game SCORE" in report and "Macro LOCAL" in report


def test_zero_turns_has_no_fake_reward_mean(tmp_path):
    path = _write(tmp_path / "empty.npz", stop="initial_terminal_state", completed=True)
    result = summarize_rollouts([path])
    assert result["completed_games"] == 0  # not the game_terminated condition
    assert result["mean_local_reward_per_turn"] is None
    assert "N/A (no turns)" in format_summary(result)


def test_rejects_wrong_format_and_bad_encoding(tmp_path):
    wrong = _write(tmp_path / "wrong.npz", goals=("DIRECT_ATTACK",),
                   candidate_encoding_version=np.asarray(1))
    with pytest.raises(ValueError, match="expected executed mixed rollout"):
        read_rollout(wrong)
    invalid = _write(tmp_path / "invalid.npz", goals=("DIRECT_ATTACK",),
                     candidate=np.zeros((1, 31), dtype=np.float32))
    with pytest.raises(ValueError, match="one-hot"):
        read_rollout(invalid)
    wrong_shape = _write(tmp_path / "shape.npz", goals=("DIRECT_ATTACK",),
                         score_delta=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="array shapes"):
        read_rollout(wrong_shape)


def test_rejects_bad_mcts_root_and_duplicate_paths(tmp_path):
    bad = _write(tmp_path / "bad.npz", goals=("SAFETY",),
                 root_turn=[0, 0], root_visits=[1, 2],
                 root_selected=[True, True])
    with pytest.raises(ValueError, match="needs one selected"):
        read_rollout(bad)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_rollouts([bad, bad])


def test_discovery_recursive_and_no_unrelated_npz(tmp_path):
    d = tmp_path / "rollouts"
    direct = _write(d / "rollout_seed_1.npz")
    nested = _write(d / "iteration_000" / "rollouts" / "rollout_seed_2.npz")
    _write(d / "dataset.npz")  # not a rollout filename
    assert resolve_paths(directories=[d]) == [direct]
    assert set(resolve_paths(directories=[d], recursive=True)) == {direct, nested}
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_paths(rollouts=[direct], directories=[d])
    with pytest.raises(ValueError, match="No mixed rollout"):
        resolve_paths(directories=[tmp_path])


def test_cli_summary_json_and_no_overwrite(tmp_path, capsys):
    path = _write(tmp_path / "rollout_seed_7.npz", goals=("SAFETY",))
    output = tmp_path / "report" / "summary.json"
    args = parse_args(["--rollout", str(path), "--json-output", str(output)])
    report = main(args)
    assert "observed statistics" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(args)
