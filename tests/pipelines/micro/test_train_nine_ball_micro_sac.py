"""5D-SAC training entry contracts without a GPU or actual simulator."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.train_nine_ball_micro_sac import (
    load_training_states, micro_reward, parse_args, prepare_training_data,
    resolve_device, restore_micro_checkpoint,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal, encode_goal_v3, generate_nine_ball_goals
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


def _candidate_rollout(directory, *, cue_x=0.52):
    _rollout(directory)
    path = directory / "match_seed_0.json"
    payload = json.loads(path.read_text())
    row = payload["trajectory"][0]
    positions = np.zeros((10, 2), dtype=np.float32)
    positions[[0, 1, 9]] = [(cue_x, 1.68), (0.28, 1.94), (0.75, 1.2)]
    pockets = np.zeros(10, dtype=np.int32)
    pockets[[0, 1, 9]] = -1
    row["state"].update(positions=positions.tolist(), pocket_indices=pockets.tolist())
    state = replace(
        create_nine_ball_rack(breaker=0, seed=1),
        break_shot=False, ball_positions=positions, ball_pocket_indices=pockets,
    )
    goals = generate_nine_ball_goals(state, learned_safety=True)
    candidates = [encode_goal_v3(goal).tolist() for goal in goals]
    row["chosen_action"] = {
        "goal_v3": candidates[0], "macro_search": {"goal_v3": candidates, "chosen_index": 0},
    }
    path.write_text(json.dumps(payload))
    return len(goals)


def test_candidate_goals_include_unselected_and_exclude_heldout_boards(tmp_path):
    first, second, heldout = (tmp_path / name for name in ("first", "second", "heldout"))
    count = _candidate_rollout(first)
    _candidate_rollout(second)
    heldout_count = _candidate_rollout(heldout, cue_x=0.55)
    path = first / "match_seed_0.json"
    payload = json.loads(path.read_text())
    other = json.loads((heldout / path.name).read_text())["trajectory"][0]
    payload["trajectory"].append(other)
    path.write_text(json.dumps(payload))
    args = parse_args([
        "--rollout-dir", str(first), "--extra-rollout-dir", str(second),
        "--holdout-rollout-dir", str(heldout), "--learned-safety", "--goal-source", "candidates",
    ])
    samples, report = prepare_training_data(args)
    assert len(samples) == count
    assert {sample.selected_goal.kind for sample in samples} == {
        "SAFETY", "POSITION_ATTACK", "DIRECT_ATTACK",
    }
    assert report["excluded_overlapping_samples"] == heldout_count
    assert report["removed_duplicate_samples"] == count
    assert report["unique_boards_by_goal"] == dict.fromkeys(
        ("DIRECT_ATTACK", "POSITION_ATTACK", "SAFETY"), 1,
    )


def test_periodic_checkpoint_can_initialize_the_next_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import train_nine_ball_micro_sac as training

    folder = tmp_path / "data"
    _candidate_rollout(folder)
    out = tmp_path / "training"

    def shot(_, rules, state, goal, context, physical_shot):
        physical = SimpleNamespace(
            first_contact_ball=goal.target_ball, pocket_indices=state.ball_pocket_indices,
        )
        return physical, SimpleNamespace(foul=True, newly_pocketed=())

    monkeypatch.setattr(training, "execute_valid_training_shot", shot)
    args = parse_args([
        "--rollout-dir", str(folder), "--learned-safety", "--goal-source", "candidates",
        "--balance-goals", "--output-dir", str(out), "--checkpoint-interval", "2",
        "--steps", "4", "--batch-size", "2", "--hidden-dim", "8",
    ])
    summary = training.train(args, simulator_factory=lambda _: object())
    path = out / "checkpoints" / "step_00000002.pt"
    args.init_checkpoint = path
    args.output_dir = tmp_path / "continuation"
    args.steps = 1
    continued = training.train(args, simulator_factory=lambda _: object())
    assert continued["previous_trained_steps"] == 2
    assert sum(summary["accepted_steps_by_goal"].values()) == 4
    assert (out / "micro_sac_5d.pt").is_file()


def test_frozen_break_is_excluded_and_preserved_across_micro_training(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from scripts import train_nine_ball_micro_sac as training
    from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
    from src.micro.sac import SACAgent

    folder = tmp_path / "data"
    _candidate_rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    break_state = create_nine_ball_rack(breaker=0, seed=1)
    break_goal = NineBallGoal("BREAK", 1)
    raw_goal = encode_goal_v3(break_goal).tolist()
    payload["learned_break"] = True
    payload["trajectory"].append({
        "state": {
            **payload["trajectory"][0]["state"],
            "positions": break_state.ball_positions.tolist(),
            "pocket_indices": break_state.ball_pocket_indices.tolist(),
            "break_shot": True,
        },
        "chosen_action": {
            "goal_v3": raw_goal, "macro_search": {"goal_v3": [raw_goal], "chosen_index": 0},
        },
    })
    path.write_text(json.dumps(payload))
    out = tmp_path / "training"
    base_path = tmp_path / "base.pt"
    args = parse_args([
        "--rollout-dir", str(folder), "--learned-safety", "--learned-break",
        "--goal-source", "candidates", "--output-dir", str(out),
        "--init-checkpoint", str(base_path), "--frozen-break-checkpoint", str(base_path),
        "--steps", "4", "--batch-size", "2", "--hidden-dim", "8",
    ])
    initial = SACAgent(state_dim=108, action_dim=5, hidden_dim=8, device="cpu")
    training.save_micro_checkpoint(initial, args, 0, 0, base_path)

    def ordinary_shot(_, rules, state, goal, context, physical_shot):
        assert goal.kind != "BREAK"
        return SimpleNamespace(
            first_contact_ball=goal.target_ball, pocket_indices=state.ball_pocket_indices,
        ), SimpleNamespace(foul=True, newly_pocketed=())

    monkeypatch.setattr(training, "execute_valid_training_shot", ordinary_shot)
    report = training.train(args, simulator_factory=lambda _: object())
    assert report["training_data"]["excluded_frozen_break_samples"] == 1
    saved_path = out / "micro_sac_5d.pt"
    base, trained = (NineBallMicroSACAgent(p) for p in (base_path, saved_path))
    assert trained.plan_goal(break_state, break_goal).shot == base.plan_goal(
        break_state, break_goal,
    ).shot
    assert not torch.equal(trained.actor.mu_head.weight, base.actor.mu_head.weight)
    args.init_checkpoint = saved_path
    args.frozen_break_checkpoint = None
    assert training.load_frozen_break(args)["source_checkpoint"] == str(base_path)


def test_paired_micro_evaluation_preserves_physics_rejections_in_denominator(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import eval_nine_ball_micro_v3 as evaluation
    from src.game.nine_ball_physics_guard import NativeShotRejected

    folder = tmp_path / "heldout"
    _candidate_rollout(folder)

    def execute(_, rules, state, goal, context, model):
        if model.name == "base.pt":
            raise NativeShotRejected(128, 0)
        pockets = state.ball_pocket_indices.copy()
        if goal.target_pocket is not None:
            pockets[goal.target_ball] = goal.target_pocket
        return SimpleNamespace(
            pocket_indices=pockets, first_contact_ball=goal.target_ball, positions=state.ball_positions,
        ), SimpleNamespace(
            foul=goal.kind == "SAFETY", newly_pocketed=(goal.target_ball,), turn_changed=True,
            rack_winner=None, next_state=replace(state, ball_pocket_indices=pockets),
        )

    monkeypatch.setattr(evaluation, "execute_valid_training_shot", execute)
    args = evaluation.parse_args([
        "--rollout-dir", str(folder), "--base-checkpoint", "base.pt", "--checkpoint", "candidate.pt",
        "--learned-safety", "--max-cases-per-goal", "1", "--output", str(tmp_path / "eval.json"),
    ])
    report = evaluation.evaluate(
        args, simulator_factory=lambda _: object(),
        agent_factory=lambda path, **_: SimpleNamespace(plan_goal=lambda *x: SimpleNamespace(shot=path)),
    )
    result = report["goals"]["DIRECT_ATTACK"]
    assert result["base"]["physics_rejected"] == 1
    assert result["base"]["physics_rejection_reasons"] == {"native precondition=128, placement=0": 1}
    assert result["base"]["goal_success_rate"] == 0
    assert result["candidate"]["goal_success_rate"] == 1
    assert result["paired_goal_success"]["candidate_only"] == 1


def test_logged_break_only_ignores_non_break_boards(tmp_path):
    folder = tmp_path / "matches"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    attack_row = payload["trajectory"][0]
    break_row = {
        **attack_row,
        "state": {**attack_row["state"], "break_shot": True, "turn_index": 0},
        "chosen_action": {"goal_v3": encode_goal_v3(NineBallGoal("BREAK", 1)).tolist()},
    }
    payload.update(learned_break=True, wpa_break_rules=True)
    payload["trajectory"].append(break_row)
    path.write_text(json.dumps(payload))

    sample, = load_training_states(
        folder, goal_source="logged", learned_break=True, wpa_break_rules=True, break_only=True,
    )
    assert sample.state.break_shot
    assert sample.selected_goal.kind == "BREAK"


def test_break_teacher_omits_racks_without_a_witness(monkeypatch):
    from types import SimpleNamespace

    from scripts import bootstrap_nine_ball_break_v3 as teacher

    simulator = SimpleNamespace(require_head_string_crossings=lambda: None)
    weaker = (0.85, 0.0, -0.6, 0.0, 0.0)
    stronger = (0.95, 0.0, -0.6, 0.0, 0.0)
    monkeypatch.setattr(teacher, "teacher_actions", lambda: (weaker, stronger))

    def fake_trial(simulator, rules, state, context, shot):
        if state.current_player == 1:
            return "physics_rejected"
        return "legal_with_pot" if shot.speed_m_s > 5.3 else "legal_without_pot"

    monkeypatch.setattr(teacher, "break_trial", fake_trial)
    observations, pairs, outcomes = teacher.create_teacher(920000, 2, lambda cfg: simulator)

    assert observations.shape == (2, 108)
    assert pairs == [(0, stronger)]
    assert outcomes == {"legal_without_pot": 1, "legal_with_pot": 1, "physics_rejected": 2}


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


def test_logged_safety_uses_only_micro_controlled_goal(tmp_path):
    from src.game.nine_ball_encoding_v3 import encode_goal_v3, generate_nine_ball_goals

    folder = tmp_path / "rollouts"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    rack = replace(create_nine_ball_rack(breaker=0, seed=1), break_shot=False)
    for enabled in (False, True):
        goal = generate_nine_ball_goals(rack, learned_safety=enabled)[-1]
        payload["trajectory"][0]["chosen_action"] = {"goal_v3": encode_goal_v3(goal).tolist()}
        path.write_text(json.dumps(payload))
        if enabled:
            samples = load_training_states(folder, goal_source="logged", learned_safety=True)
            assert samples[0].selected_goal == goal
        else:
            with pytest.raises(ValueError, match="No eligible"):
                load_training_states(folder, goal_source="logged", learned_safety=True)


def test_logged_outcome_position_reuses_exact_selected_candidate(tmp_path):
    from src.game.nine_ball_encoding_v3 import encode_goal_v3
    from src.game.nine_ball_position import POSITION_RADIUS_M

    folder = tmp_path / "rollouts"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    row = payload["trajectory"][0]
    row["state"]["positions"] = [(0.0, 0.0)] * 10
    row["state"]["pocket_indices"] = [0] * 10
    for ball, xy in {0: (0.52, 1.68), 1: (0.28, 1.94), 9: (0.75, 1.2)}.items():
        row["state"]["positions"][ball] = xy
        row["state"]["pocket_indices"][ball] = -1
    goal = NineBallGoal("POSITION_ATTACK", 1, 2, (0.847203, 1.383479), POSITION_RADIUS_M)
    encoded = encode_goal_v3(goal).tolist()
    action = {
        "goal_v3": encoded,
        "macro_search": {"chosen_index": 0, "goal_v3": [encoded.copy()]},
    }
    payload["outcome_position"] = True
    row["chosen_action"] = action
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mode differs"):
        load_training_states(folder, goal_source="logged")
    sample, = load_training_states(folder, goal_source="logged", outcome_position=True)
    np.testing.assert_array_equal(encode_goal_v3(sample.selected_goal), encoded)

    action["macro_search"]["goal_v3"][0][26] += 0.01
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from selected MCTS candidate"):
        load_training_states(folder, goal_source="logged", outcome_position=True)


def test_logged_break_requires_the_micro_controlled_goal(tmp_path):
    from src.game.nine_ball_encoding_v3 import encode_goal_v3, generate_nine_ball_goals

    folder = tmp_path / "rollouts"
    _rollout(folder)
    path = folder / "match_seed_0.json"
    payload = json.loads(path.read_text())
    row = payload["trajectory"][0]
    row["state"]["break_shot"] = True
    payload["learned_break"] = True
    rack = create_nine_ball_rack(breaker=0, seed=1)
    goal, = generate_nine_ball_goals(rack, learned_break=True)
    row["chosen_action"] = {"goal_v3": encode_goal_v3(goal).tolist()}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mode differs"):
        load_training_states(folder, goal_source="logged")
    sample, = load_training_states(folder, goal_source="logged", learned_break=True)
    assert sample.state.break_shot
    assert sample.selected_goal == goal
    assert load_training_states(folder, learned_break=True)[0].state.break_shot

    legacy_goal, = generate_nine_ball_goals(rack)
    row["chosen_action"]["goal_v3"] = encode_goal_v3(legacy_goal).tolist()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cannot be recovered"):
        load_training_states(folder, goal_source="logged", learned_break=True)
    row["state"]["break_shot"] = False
    row["chosen_action"]["goal_v3"] = encode_goal_v3(goal).tolist()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match rack break phase"):
        load_training_states(folder, goal_source="logged", learned_break=True)


@pytest.mark.parametrize(
    "foul,pocketed,winner,expected",
    [(True, (9,), 1, (-1.2, False)), (False, (), None, (0.1, False)),
     (False, (3, 4), None, (1.1, True)), (False, (9,), 0, (2.1, True))],
)
def test_break_reward_uses_legal_pots_and_rack_winner(foul, pocketed, winner, expected):
    from types import SimpleNamespace

    rack = create_nine_ball_rack(breaker=0, seed=1)
    outcome = SimpleNamespace(foul=foul, rack_winner=winner, newly_pocketed=pocketed)
    assert micro_reward(NineBallGoal("BREAK", 1), rack, None, outcome) == expected


def test_outcome_position_bonus_requires_the_next_ball_to_remain(tmp_path):
    from types import SimpleNamespace

    folder = tmp_path / "rollouts"
    _rollout(folder)
    rack = load_training_states(folder)[0].state
    goal = NineBallGoal("POSITION_ATTACK", 1, 2, tuple(rack.ball_positions[0]), 0.10)
    pockets = rack.ball_pocket_indices.copy()
    pockets[1] = 2
    physical = SimpleNamespace(
        pocket_indices=pockets, positions=rack.ball_positions, first_contact_ball=1,
    )
    after = replace(rack, ball_pocket_indices=pockets)
    outcome = SimpleNamespace(
        next_state=after, foul=False, turn_changed=False, rack_winner=None,
        newly_pocketed=(1,),
    )
    reward, exact = micro_reward(goal, rack, physical, outcome, outcome_position=True)
    assert exact and reward == pytest.approx(2.6)
    pockets[2] = 2
    outcome.next_state = replace(rack, ball_pocket_indices=pockets)
    outcome.newly_pocketed = (1, 2)
    reward, exact = micro_reward(goal, rack, physical, outcome, outcome_position=True)
    assert exact and reward == pytest.approx(2.1)


def test_safety_reward_requires_legal_handoff_and_reduced_opponent_options():
    from types import SimpleNamespace
    from src.game.nine_ball_encoding_v3 import direct_pocket_candidates

    rack = replace(create_nine_ball_rack(breaker=0, seed=1), break_shot=False)
    positions = rack.ball_positions.copy()
    pockets = rack.ball_pocket_indices.copy()
    pockets[3:] = 0
    positions[:3] = [[0.5, 0.6], [0.5, 1.4], [0.3, 1.8]]
    before = replace(rack, ball_positions=positions, ball_pocket_indices=pockets)
    positions = positions.copy()
    positions[2] = [0.5, 1.0]
    after = replace(before, current_player=1, ball_positions=positions)
    assert len(tuple(direct_pocket_candidates(before, 1))) > len(tuple(direct_pocket_candidates(after, 1)))
    goal = NineBallGoal("SAFETY", 1)
    outcome = SimpleNamespace(foul=False, turn_changed=True, next_state=after)
    reward, success = micro_reward(goal, before, None, outcome)
    assert success and reward > 0.1
    outcome.next_state = before
    assert micro_reward(goal, before, None, outcome) == (0.1, False)
    outcome.turn_changed = False
    assert micro_reward(goal, before, None, outcome) == (-0.5, False)
    outcome.foul = True
    assert micro_reward(goal, before, None, outcome) == (-1.2, False)
