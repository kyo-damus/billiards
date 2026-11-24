"""Plan-only and data-provenance tests for Phase31 Policy-frozen V continuation."""
import hashlib
import json

import pytest

from scripts.experiments.run_phase31_value_only import (
    make_child_args, main, parse_args,
)
from src.game.nine_ball_rack_split import legacy_rack_order


def _write_parent(tmp_path):
    source = {}
    for name in ("micro", "policy", "state_v"):
        p = tmp_path / f"phase28b_{name}.pt"
        p.write_bytes(name.encode("utf-8"))
        source[name] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    ending_v = tmp_path / "phase29_state_v.pt"
    ending_v.write_bytes(b"phase29-final-v")
    ending_policy = tmp_path / "phase29_policy.pt"
    ending_policy.write_bytes(b"phase29-final-policy")
    split = tmp_path / "layout_split.json"
    split.write_text(json.dumps({
        "kind": "nine_ball_rack_order_split_v1",
        "evaluation_orders": [list(legacy_rack_order(1234))],
    }))
    parent = {
        "kind": "nine_ball_macro_micro_alternating_v3",
        "status": "completed",
        "source_checkpoints": source,
        "source_git_commit": "phase29-original-revision",
        "rack_layout_split": {
            "path": str(split),
            "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        },
        "settings": {
            "micro_steps": 0, "race_to": 9, "wpa_break_rules": True,
            "break_policy": "alternate", "max_shots_per_rack": 120,
            "mcts_simulations": 12, "mcts_max_depth": 2,
            "mcts_root_min_visits": 1, "mcts_c_puct": 1.5, "gamma": 1.0,
            "learned_safety": True, "learned_break": True,
            "outcome_position": True,
        },
        "rounds": [{
            "start_seed": 1430648, "matches": 512, "micro_trained": False,
            "micro": source["micro"]["path"],
            "policy": str(ending_policy),
            "state_v": str(ending_v),
        }],
    }
    manifest = tmp_path / "phase29.json"
    manifest.write_text(json.dumps(parent))
    return manifest, parent, ending_v


def _args(tmp_path, manifest):
    return parse_args([
        "--phase29-manifest", str(manifest),
        "--output-root", str(tmp_path / "phase31_new"),
        "--iterations", "2",
        "--matches-per-iteration", "8",
        "--plan-only",
    ])


def test_launcher_fixes_phase28b_policy_and_phase29_final_v(tmp_path):
    manifest, parent, ending_v = _write_parent(tmp_path)
    args = _args(tmp_path, manifest)
    child, metadata = make_child_args(args)
    assert child.base_policy_checkpoint == parent["source_checkpoints"]["policy"]["path"] or (
        str(child.base_policy_checkpoint) == parent["source_checkpoints"]["policy"]["path"]
    )
    assert str(child.base_state_v_checkpoint) == str(ending_v)
    assert child.freeze_policy
    assert child.warm_start_state_v
    assert child.micro_steps == 0
    assert child.qv_epochs == 2
    assert child.race_to == 9
    assert child.wpa_break_rules
    assert metadata["starting_variant"].startswith("C")

    plan = main(args)
    assert plan["planned_stage_names"] == [
        "000_self_play", "000_return_dataset", "000_macro_qv",
        "001_self_play", "001_return_dataset", "001_macro_qv",
    ]
    assert plan["planned_rounds"][0]["policy"] == str(child.base_policy_checkpoint)
    assert plan["planned_rounds"][1]["policy"] == str(child.base_policy_checkpoint)
    assert not args.output_root.exists()


@pytest.mark.parametrize("corruption", ("micro", "policy", "state_v", "split", "changed_micro"))
def test_rejects_corrupt_or_changed_models_before_creating_run(tmp_path, corruption):
    manifest, parent, ending_v = _write_parent(tmp_path)
    args = _args(tmp_path, manifest)
    if corruption in ("micro", "policy"):
        path = parent["source_checkpoints"][corruption]["path"]
        with open(path, "wb") as stream:
            stream.write(b"corrupted")
    elif corruption == "state_v":
        ending_v.unlink()
    elif corruption == "split":
        with open(parent["rack_layout_split"]["path"], "a") as stream:
            stream.write("\n")
    else:
        new_micro = tmp_path / "changed_micro.pt"
        new_micro.write_bytes(b"new-micro")
        parent["rounds"][-1]["micro"] = str(new_micro)
        manifest.write_text(json.dumps(parent))
    with pytest.raises((ValueError, FileNotFoundError)):
        main(args)
    assert not args.output_root.exists()


def test_launcher_executes_build_plan_exactly_before_training(tmp_path, monkeypatch):
    import scripts.experiments.run_phase31_value_only as module

    manifest, parent, _ = _write_parent(tmp_path)
    args = _args(tmp_path, manifest)
    args.plan_only = False
    observed = []

    def fake_run(child):
        observed.append(child)
    monkeypatch.setattr(module, "run_iteration", fake_run)
    result = main(args)
    assert len(observed) == 1
    assert observed[0].freeze_policy is True
    assert observed[0].warm_start_state_v is True
    assert observed[0].device == "cpu"
    assert result["frozen_micro"] is True
