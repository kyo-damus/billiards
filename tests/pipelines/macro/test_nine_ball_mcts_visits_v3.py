"""Phase 14: physical-root MCTS visit dataset and Macro Policy fine-tune."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from scripts.build_nine_ball_mcts_visit_dataset_v3 import (
    TARGET_KIND, build, parse_args as parse_build, read_root,
)
from scripts.train_nine_ball_macro_visits_v3 import (
    load_visits, parse_args as parse_train, train,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal, encode_goal_v3
from src.game.nine_ball_session import create_nine_ball_rack
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.macro.candidate_policy import MacroCandidatePolicyNetwork
from src.micro.actor import GaussianActor


def make_row(seed, *, visits=(6, 2), chosen=0, break_shot=False):
    rack = replace(
        create_nine_ball_rack(breaker=0, seed=seed), break_shot=break_shot,
    )
    goals = (
        NineBallGoal("DIRECT_ATTACK", target_ball=1, target_pocket=0),
        NineBallGoal("SAFETY", target_ball=1),
    )
    encoded = [encode_goal_v3(g).tolist() for g in goals]
    return {
        "player": 0,
        "state": {
            "ball_numbers": list(rack.config.ball_numbers),
            "positions": rack.ball_positions.tolist(),
            "pocket_indices": rack.ball_pocket_indices.tolist(),
            "current_player": 0,
            "ball_in_hand": False,
            "break_shot": break_shot,
            "turn_index": 2,
        },
        "rule_context": {
            "consecutive_fouls": [0, 1],
            "push_out_available": False,
        },
        "chosen_action": {
            "goal_v3": encoded[chosen],
            "target_ball": 1,
            "macro_search": {
                "simulations": sum(visits),
                "root_candidates": 2,
                "chosen_index": chosen,
                "root_visits": list(visits),
                "root_priors": [0.5, 0.5],
                "goal_v3": encoded,
            },
        },
    }


def write_match(folder, seed, *, visits=(6, 2), chosen=0, complete=False):
    data = {
        "seed": seed, "race_to": 1,
        "match_finished": complete,
        "match_winner": 0 if complete else None,
        "trajectory": [make_row(seed, visits=visits, chosen=chosen)],
    }
    (folder / f"match_seed_{seed}.json").write_text(
        json.dumps(data), encoding="utf-8",
    )


def test_visit_dataset_uses_unfinished_roots_and_normalizes_per_board(tmp_path):
    directory = tmp_path / "matches"
    directory.mkdir()
    write_match(directory, 101, visits=(6, 2), chosen=0, complete=True)
    write_match(directory, 202, visits=(1, 7), chosen=1, complete=False)
    path = tmp_path / "visits" / "roots.npz"
    report = build(parse_build([
        "--rollout-dir", str(directory), "--output", str(path),
    ]))
    assert report["policy_training_roots"] == 2
    assert report["total_candidate_rows"] == 4
    assert report["unfinished_matches"] == 1
    assert report["completed_matches"] == 1
    assert report["match_seeds"] == [101, 202]
    state, goal, counts, offsets, seeds, chosen = load_visits(path)
    assert state.shape == (2, 72)
    assert goal.shape == (4, 36)
    np.testing.assert_array_equal(counts, [6, 2, 1, 7])
    np.testing.assert_array_equal(offsets, [0, 2, 4])
    np.testing.assert_array_equal(seeds, [101, 202])
    np.testing.assert_array_equal(chosen, [0, 1])


def test_reject_mismatched_selected_goal_or_visits():
    row = make_row(1)
    row["chosen_action"]["goal_v3"] = encode_goal_v3(
        NineBallGoal("DIRECT_ATTACK", 1, 4),
    ).tolist()
    with pytest.raises(ValueError, match="differs"):
        read_root(row)

    row = make_row(2, visits=(6, 2))
    row["chosen_action"]["macro_search"]["simulations"] = 9
    with pytest.raises(ValueError, match="Inconsistent"):
        read_root(row)

    row = make_row(3, visits=(6, 2))
    row["chosen_action"]["macro_search"]["chosen_index"] = 1
    row["chosen_action"]["goal_v3"] = row["chosen_action"]["macro_search"]["goal_v3"][1]
    with pytest.raises(ValueError, match="maximum-visit"):
        read_root(row)


def test_legacy_invalidated_candidate_is_removed_and_chosen_index_is_remapped():
    row = make_row(1, visits=(6, 2), chosen=1)
    root = row["chosen_action"]["macro_search"]
    root["root_invalid_reasons"] = ["no executable descendant", None]
    _, candidates, visits, chosen = read_root(row)
    np.testing.assert_array_equal(candidates, np.asarray([root["goal_v3"][1]], dtype=np.float32))
    np.testing.assert_array_equal(visits, [2])
    assert chosen == 0

    root["root_invalid_reasons"] = [None, "no executable descendant"]
    with pytest.raises(ValueError, match="was invalidated"):
        read_root(row)


def test_match_policy_initialization_preserves_rack_only_logits(tmp_path):
    from src.game.nine_ball_policy_v3 import load_macro_policy

    old = MacroCandidatePolicyNetwork(state_dim=72, candidate_dim=36, hidden_dim=16)
    path = tmp_path / "old.pt"
    torch.save({
        "format_version": 3, "model_kind": "nine_ball_macro_candidate_policy_v3",
        "state_dim": 72, "candidate_dim": 36, "hidden_dim": 16,
        "candidate_policy_network": old.state_dict(),
    }, path)
    model, _ = load_macro_policy(path, device="cpu", state_dim=78)
    rack, match, candidates = torch.randn(72), torch.randn(6), torch.randn(4, 36)
    torch.testing.assert_close(model(torch.cat((rack, match)), candidates), old(rack, candidates))
    assert torch.count_nonzero(model.network[0].weight[:, 72:78]) == 0


def test_policy_visit_training_splits_matches_and_loads_checkpoint(tmp_path):
    directory = tmp_path / "matches"
    directory.mkdir()
    write_match(directory, 101, visits=(6, 2), chosen=0)
    write_match(directory, 202, visits=(1, 7), chosen=1)
    write_match(directory, 303, visits=(5, 3), chosen=0)
    dataset = tmp_path / "roots.npz"
    build(parse_build([
        "--rollout-dir", str(directory), "--output", str(dataset),
    ]))

    base = tmp_path / "base.pt"
    model = MacroCandidatePolicyNetwork(
        state_dim=72, candidate_dim=36, hidden_dim=16,
    )
    torch.save({
        "format_version": 3,
        "model_kind": "nine_ball_macro_candidate_policy_v3",
        "state_dim": 72, "candidate_dim": 36, "hidden_dim": 16,
        "candidate_policy_network": model.state_dict(),
    }, base)
    output = tmp_path / "visit_policy.pt"
    report = train(parse_train([
        "--dataset", str(dataset),
        "--base-policy-checkpoint", str(base),
        "--output", str(output),
        "--epochs", "2",
        "--device", "cpu",
        "--validation-fraction", "0.34",
    ]))
    assert set(report["train_match_seeds"]).isdisjoint(report["validation_match_seeds"])
    assert report["best_validation"]["visit_target_ce"] <= (
        report["initial_validation"]["visit_target_ce"] + 1e-6
    )
    ckpt = torch.load(output, map_location="cpu", weights_only=True)
    assert ckpt["format_version"] == 3
    assert ckpt["model_kind"] == "nine_ball_macro_candidate_policy_v3"
    assert ckpt["teacher_kind"] == TARGET_KIND
    assert ckpt["hidden_dim"] == 16
    micro = tmp_path / "micro.pt"
    actor = GaussianActor(state_dim=108, action_dim=5, hidden_dim=16)
    torch.save({
        "format_version": 3,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": 108, "action_dim": 5, "hidden_dim": 16,
        "actor": actor.state_dict(),
    }, micro)
    agent = NineBallMicroSACAgent(
        micro_checkpoint=micro, macro_policy_checkpoint=output,
    )
    assert agent.macro_policy is not None


def test_reject_one_source_match(tmp_path):
    directory = tmp_path / "matches"
    directory.mkdir()
    write_match(directory, 101)
    with pytest.raises(ValueError, match="Need >=2 different"):
        build(parse_build([
            "--rollout-dir", str(directory), "--output", str(tmp_path / "fail.npz"),
        ]))
