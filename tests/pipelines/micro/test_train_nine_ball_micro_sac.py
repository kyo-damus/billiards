"""5D-SAC training entry contracts without a GPU or actual simulator."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.train_nine_ball_micro_sac import (
    load_training_states, micro_reward, parse_args, resolve_device,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal
from src.game.nine_ball_session import create_nine_ball_rack
from src.game.nine_ball_rules import NineBallContext


def _rollout(directory):
    directory.mkdir(parents=True)
    state = replace(
        create_nine_ball_rack(breaker=0, seed=1),
        break_shot=False,
    )
    row = {
        "state": {
            "ball_numbers": list(state.config.ball_numbers),
            "positions": state.ball_positions.tolist(),
            "pocket_indices": state.ball_pocket_indices.tolist(),
            "current_player": state.current_player,
            "ball_in_hand": False,
            "break_shot": False,
            "turn_index": 1,
        },
        "rule_context": {
            "consecutive_fouls": [1, 2],
            "push_out_available": False,
        },
    }
    (directory / "match_seed_0.json").write_text(json.dumps({
        "run_type": "nonlearned_geometry_self_play",
        "trajectory": [row],
    }), encoding="utf-8")


def test_training_loader_reads_versioned_state_and_rule_context(tmp_path):
    folder = tmp_path / "matches"
    _rollout(folder)
    samples = load_training_states(folder)
    assert len(samples) == 1
    assert samples[0].state.ball_count == 10
    assert samples[0].context.consecutive_fouls == (1, 2)


def test_training_loader_rejects_empty_rollouts(tmp_path):
    with pytest.raises(ValueError, match="No match_seed"):
        load_training_states(tmp_path)


def test_foul_and_exact_target_pot_have_different_one_shot_rewards():
    rack = replace(create_nine_ball_rack(breaker=0, seed=1), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 2)
    pockets = rack.ball_pocket_indices.copy()
    pockets[1] = 2
    physical = type("Physical", (), {
        "pocket_indices": pockets, "positions": rack.ball_positions.copy(),
        "first_contact_ball": 1,
    })()
    good = type("Outcome", (), {"foul": False, "newly_pocketed": (1,)})()
    bad = type("Outcome", (), {"foul": True, "newly_pocketed": (1,)})()
    r_good, correct = micro_reward(goal, rack, physical, good)
    r_bad, not_correct = micro_reward(goal, rack, physical, bad)
    assert correct and not not_correct
    assert r_good > r_bad


def test_cuda_allocation_requires_explicit_single_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        resolve_device("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="exactly one"):
        resolve_device("cuda")
    assert resolve_device("cpu") == "cpu"


def test_training_defaults_are_cpu_and_separate_output():
    args = parse_args([
        "--rollout-dir", "input", "--output-dir", "fresh_output",
    ])
    assert args.steps == 5000
    assert args.device == "cpu"
    assert args.output_dir == Path("fresh_output")



def test_training_loader_accepts_new_micro_self_play_source(tmp_path):
    folder = tmp_path / "matches"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    payload["run_type"] = "learned_micro_self_play"
    path.write_text(json.dumps(payload))
    samples = load_training_states(folder)
    assert len(samples) == 1
