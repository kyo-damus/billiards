"""Phase 12: 9-ball turn-aware match returns and strict v3 Q/V regressors."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.build_nine_ball_macro_returns_v3 import (
    build, parse_args as parse_build_args, turn_aware_returns,
)
from scripts.train_nine_ball_macro_qv_v3 import (
    load_returns, parse_args as parse_train_args, train,
)
from src.game.nine_ball_encoding_v3 import (
    NineBallGoal, encode_goal_v3,
)
from src.game.nine_ball_session import create_nine_ball_rack
from src.game.nine_ball_value_v3 import load_macro_q_v3, load_state_v_v3


def test_turn_aware_return_from_each_actors_perspective():
    actual = turn_aware_returns([0, 0, 1, 0], winner=1, gamma=0.5)
    np.testing.assert_allclose(actual, [-0.125, -0.25, 0.5, -1.0])
    np.testing.assert_allclose(
        turn_aware_returns([1, 1, 1], winner=1, gamma=0.9),
        [0.81, 0.9, 1.0],
    )
    with pytest.raises(ValueError):
        turn_aware_returns([0, 1], 1, gamma=1.1)


def make_row(actor, *, chosen_goal=True):
    state = replace(create_nine_ball_rack(breaker=actor, seed=4), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", target_ball=1, target_pocket=0)
    return {
        "player": actor,
        "state": {
            "ball_numbers": list(state.config.ball_numbers),
            "positions": state.ball_positions.tolist(),
            "pocket_indices": state.ball_pocket_indices.tolist(),
            "current_player": actor,
            "ball_in_hand": False,
            "break_shot": False,
            "turn_index": 2,
        },
        "rule_context": {
            "consecutive_fouls": [0, 1],
            "push_out_available": False,
        },
        "chosen_action": {
            "goal_label": "DIRECT_ATTACK_MICRO_SAC5D_MACRO_POLICY",
            "target_ball": 1,
            "target_pocket": 0,
            "goal_v3": encode_goal_v3(goal).tolist() if chosen_goal else None,
        },
        "rule_result": {
            "rack_winner": None,
            "foul": False,
            "turn_changed": True,
            "newly_pocketed": [],
        },
    }


def write_match(folder, seed, *, winner=1, finished=True, goal=True):
    rows = [make_row(0, chosen_goal=goal), make_row(1, chosen_goal=goal)]
    rows[-1]["rule_result"]["rack_winner"] = winner if finished else None
    payload = {
        "seed": seed,
        "race_to": 1,
        "racks_completed": 1 if finished else 0,
        "match_finished": finished,
        "match_winner": winner if finished else None,
        "trajectory": rows,
    }
    (folder / f"match_seed_{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def test_build_return_dataset_and_fit_strict_q_v(tmp_path):
    folder = tmp_path / "rollouts"
    folder.mkdir()
    write_match(folder, 10, winner=1)
    write_match(folder, 20, winner=0)
    write_match(folder, 30, winner=1)
    write_match(folder, 40, finished=False)  # excluded, never zero-bootstrap

    dataset = tmp_path / "returns" / "selected_mc.npz"
    report = build(parse_build_args([
        "--rollout-dir", str(folder),
        "--output", str(dataset), "--gamma", "0.5",
    ]))
    assert report["completed_matches_used"] == 3
    assert report["unfinished_matches_skipped"] == 1
    assert report["executed_actions_labeled"] == 6
    with np.load(dataset, allow_pickle=False) as d:
        assert d["states"].shape == (6, 72)
        assert d["candidates"].shape == (6, 36)
        np.testing.assert_array_equal(d["match_seeds"], [10, 10, 20, 20, 30, 30])
        # Both matches are labeled by actor; no opponent reward sign leakage.
        np.testing.assert_allclose(d["q_target"], [-0.5, 1, 0.5, -1, -0.5, 1])
        assert d["turn_aware"].item()
    states, goals, targets, groups = load_returns(dataset)
    assert states.shape[1] == 72 and goals.shape[1] == 36
    assert len(targets) == len(groups) == 6

    output = tmp_path / "trained_qv"
    training = train(parse_train_args([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--device", "cpu", "--epochs", "2",
        "--hidden-dim", "16", "--batch-size", "4",
        "--validation-fraction", "0.34",
    ]))
    assert not (
        set(training["train_matches"]) & set(training["validation_matches"])
    )
    q, q_meta = load_macro_q_v3(output / "candidate_q_v3.pt")
    v, v_meta = load_state_v_v3(output / "state_v_v3.pt")
    assert q_meta["turn_aware"] is True and v_meta["turn_aware"] is True
    with torch.no_grad():
        q_value = q(
            torch.as_tensor(states[0]),
            torch.as_tensor(goals[0]),
        )
        v_value = v(torch.as_tensor(states[0])).flatten()
    assert q_value.numel() == 1 and v_value.numel() == 1


def test_reject_legacy_logs_without_selected_goal(tmp_path):
    folder = tmp_path / "rollouts"
    folder.mkdir()
    write_match(folder, 10, goal=False)
    write_match(folder, 20)
    with pytest.raises(ValueError, match="Missing exact chosen_action.goal_v3"):
        build(parse_build_args([
            "--rollout-dir", str(folder),
            "--output", str(tmp_path / "out.npz"),
        ]))


def test_refuse_to_learn_from_unfinished_only(tmp_path):
    folder = tmp_path / "rollouts"
    folder.mkdir()
    write_match(folder, 10, finished=False)
    with pytest.raises(ValueError, match="Need >=2 complete"):
        build(parse_build_args([
            "--rollout-dir", str(folder),
            "--output", str(tmp_path / "out.npz"),
        ]))


def test_q_v_loader_rejects_unversioned_or_wrong_target(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({
        "format_version": 2,
        "model_kind": "nine_ball_macro_selected_action_q_v3",
        "state_dim": 72,
        "candidate_dim": 36,
        "hidden_dim": 16,
        "target_kind": "one_shot_physics_oracle",
        "reward_kind": "one_step_score",
        "turn_aware": False,
        "candidate_value_network": {},
    }, path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_macro_q_v3(path)


def test_match_returns_use_final_winner_across_racks_and_validate_scores(tmp_path):
    from src.game.match import MatchPosition

    folder = tmp_path / "matches"
    folder.mkdir()
    rows = []
    position = MatchPosition(2, (0, 0), 0)
    for i, winner in enumerate((0, 1, 1)):
        row = make_row(position.breaker)
        row["rack_index"] = i
        row["match_context"] = {
            "race_to": 2, "rack_wins": list(position.rack_wins),
            "breaker": position.breaker, "break_policy": "alternate",
        }
        row["rule_result"]["rack_winner"] = winner
        rows.append(row)
        position = position.after_rack(winner)
    payload = {
        "race_to": 2, "racks_completed": 3, "rack_wins": [1, 2],
        "match_finished": True, "match_winner": 1, "trajectory": rows,
    }
    for seed in (10, 20):
        (folder / f"match_seed_{seed}.json").write_text(json.dumps({**payload, "seed": seed}))
    dataset = tmp_path / "returns.npz"
    build(parse_build_args([
        "--rollout-dir", str(folder), "--output", str(dataset), "--match-aware", "--gamma", "1",
    ]))
    states, _, targets, _ = load_returns(dataset)
    assert states.shape == (6, 78)
    np.testing.assert_allclose(states[:3, 72:74], [[1, 1], [1, 0.5], [0.5, 0.5]])
    np.testing.assert_array_equal(targets[:3], [-1, 1, -1])

    rows[1]["match_context"]["rack_wins"] = [0, 0]
    (folder / "match_seed_20.json").write_text(json.dumps({**payload, "seed": 20}))
    with pytest.raises(ValueError, match="Inconsistent pre-shot match score"):
        build(parse_build_args([
            "--rollout-dir", str(folder), "--output", str(tmp_path / "bad.npz"), "--match-aware",
        ]))
