"""5D-SAC training entry contracts without a GPU or actual simulator."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.train_nine_ball_micro_sac import (
    load_training_states, micro_reward, parse_args, resolve_device,
    restore_micro_checkpoint,
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


def test_position_attack_reward_requires_correct_pocket_in_addition_to_cue_stop():
    from src.game.nine_ball_encoding_v3 import NineBallGoal
    rack = replace(create_nine_ball_rack(breaker=0, seed=2), break_shot=False)
    stop = tuple(float(x) for x in rack.ball_positions[0])
    goal = NineBallGoal("POSITION_ATTACK", 1, 2, stop, 0.25)
    pockets = rack.ball_pocket_indices.copy()
    physical = type("Physical", (), {
        "pocket_indices": pockets,
        "positions": rack.ball_positions.copy(),
        "first_contact_ball": 1,
    })()
    missed = type("Outcome", (), {"foul": False, "newly_pocketed": ()})()
    missed_reward, missed_exact = micro_reward(goal, rack, physical, missed)
    direct_goal = NineBallGoal("DIRECT_ATTACK", 1, 2)
    direct_reward, _ = micro_reward(direct_goal, rack, physical, missed)
    assert not missed_exact
    assert missed_reward == pytest.approx(direct_reward)
    pockets[1] = 2
    potted = type("Outcome", (), {"foul": False, "newly_pocketed": (1,)})()
    success_reward, success_exact = micro_reward(goal, rack, physical, potted)
    assert success_exact
    assert success_reward > missed_reward + 2.0


def test_logged_macro_goal_training_uses_exact_selected_goal(tmp_path):
    from src.game.nine_ball_encoding_v3 import (
        encode_goal_v3, generate_nine_ball_goals,
    )
    folder = tmp_path / "rollouts"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    rack = replace(create_nine_ball_rack(breaker=0, seed=1), break_shot=False)
    chosen = next(g for g in generate_nine_ball_goals(rack)
                  if g.kind in ("DIRECT_ATTACK", "POSITION_ATTACK"))
    payload["trajectory"][0]["chosen_action"] = {
        "goal_v3": encode_goal_v3(chosen).tolist()
    }
    path.write_text(json.dumps(payload))
    logged = load_training_states(folder, goal_source="logged")
    assert len(logged) == 1
    np.testing.assert_allclose(
        encode_goal_v3(logged[0].selected_goal), encode_goal_v3(chosen)
    )
    # A mismatched saved target cannot be silently remapped to a new goal.
    payload["trajectory"][0]["chosen_action"]["goal_v3"][4:20] = (
        [0.] * 15 + [1.]
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cannot be recovered"):
        load_training_states(folder, goal_source="logged")


def test_logged_goal_training_rejects_legacy_unlabeled_source(tmp_path):
    folder = tmp_path / "old"
    _rollout(folder)
    with pytest.raises(ValueError, match="No eligible"):
        load_training_states(folder, goal_source="logged")


def test_pretrained_micro_restores_weights_and_entropy_without_legacy_2d(tmp_path):
    import torch
    from src.micro.sac import SACAgent
    initial = SACAgent(state_dim=108, action_dim=5, hidden_dim=16, device="cpu")
    current = SACAgent(state_dim=108, action_dim=5, hidden_dim=16, device="cpu")
    with torch.no_grad():
        initial.actor.fc1.weight.fill_(0.123)
        initial.log_alpha.fill_(-0.9)
    path = tmp_path / "initial.pt"
    checkpoint = {
        "format_version": 3,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": 108, "action_dim": 5, "hidden_dim": 16,
        "trained_steps": 5000,
        "actor": initial.actor.state_dict(),
        "critic": initial.critic.state_dict(),
        "critic_target": initial.critic_target.state_dict(),
        "log_alpha": initial.log_alpha.detach().cpu(),
    }
    torch.save(checkpoint, path)
    previous_steps = restore_micro_checkpoint(current, path)
    assert previous_steps == 5000
    assert torch.allclose(current.actor.fc1.weight, initial.actor.fc1.weight)
    assert current.alpha.item() == pytest.approx(initial.alpha.item())
    checkpoint["action_dim"] = 2
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Incompatible Micro"):
        restore_micro_checkpoint(current, path)


def test_micro_training_native_nonexecution_is_not_sent_to_rules():
    from types import SimpleNamespace
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    from scripts.train_nine_ball_micro_sac import execute_valid_training_shot

    rack = replace(create_nine_ball_rack(breaker=0, seed=42), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)

    class Physics:
        def restore(self, state):
            assert state is rack

        def step(self, **kwargs):
            return SimpleNamespace(precondition_code=2, placement_code=0)

    class Rules:
        def apply(self, *args):
            raise AssertionError("Unexecuted native shot must not reach rules")

    with pytest.raises(InvalidPhysicalTransition, match="precondition=2"):
        execute_valid_training_shot(
            Physics(), Rules(), rack, goal, NineBallContext(), object(),
        )


def test_micro_training_bad_shot_is_rejected_but_unrelated_errors_propagate():
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    from scripts.train_nine_ball_micro_sac import execute_valid_training_shot

    rack = replace(create_nine_ball_rack(breaker=0, seed=43), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)

    class Physics:
        error = "Bad Shot Exception"

        def restore(self, state):
            pass

        def step(self, **kwargs):
            raise RuntimeError(self.error)

    sim = Physics()
    with pytest.raises(InvalidPhysicalTransition, match="Bad Shot Exception"):
        execute_valid_training_shot(
            sim, object(), rack, goal, NineBallContext(), object(),
        )
    sim.error = "Unexpected C++ error"
    with pytest.raises(RuntimeError, match="Unexpected C\\+\\+ error"):
        execute_valid_training_shot(
            sim, object(), rack, goal, NineBallContext(), object(),
        )


def test_micro_training_post_rule_overlaps_are_not_valid_experience():
    from types import SimpleNamespace
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    from scripts.train_nine_ball_micro_sac import execute_valid_training_shot

    rack = replace(create_nine_ball_rack(breaker=0, seed=44), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)
    positions = rack.ball_positions.copy()
    positions[2] = positions[5]
    bad = replace(rack, ball_positions=positions)

    class Physics:
        def restore(self, state):
            pass

        def step(self, **kwargs):
            return SimpleNamespace(precondition_code=0, placement_code=0)

    class Rules:
        def apply(self, *args):
            return SimpleNamespace(next_state=bad)

    with pytest.raises(InvalidPhysicalTransition, match="overlapping balls"):
        execute_valid_training_shot(
            Physics(), Rules(), rack, goal, NineBallContext(), object(),
        )
