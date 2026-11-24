"""Macro 2x2: isolate Policy and State V and validate comparison provenance."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.comparisons.eval_nine_ball_macro_ablation import (
    _variant_models, evaluate, parse_args,
)


def _fake_run(tmp_path):
    paths = {}
    for key in ("micro", "policy", "state_v"):
        p = tmp_path / ("phase28b_" + key + ".pt")
        p.write_bytes(("phase28b_" + key).encode())
        paths[key] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    updated = {}
    for key in ("policy", "state_v"):
        p = tmp_path / ("phase29_" + key + ".pt")
        p.write_bytes(("phase29_" + key).encode())
        updated[key] = str(p)

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({
        "kind": "nine_ball_rack_order_split_v1",
        "evaluation_orders": [[2, 3, 4, 5, 6, 7, 8]],
    }))
    training = {
        "kind": "nine_ball_macro_micro_alternating_v3",
        "status": "completed",
        "source_checkpoints": paths,
        "rounds": [
            {
                "start_seed": 1400000, "race_to": 9, "matches": 512,
                "micro_trained": False, "micro": paths["micro"]["path"],
                "policy": updated["policy"], "state_v": updated["state_v"],
            }
        ],
        "settings": {
            "race_to": 9, "break_policy": "alternate",
            "wpa_break_rules": True,
            "mcts_simulations": 24, "mcts_max_depth": 2, "mcts_c_puct": 1.5,
            "mcts_root_min_visits": 1, "gamma": 1.0,
            "max_shots_per_rack": 200, "learned_safety": True,
            "learned_break": True, "outcome_position": True,
        },
        "rack_layout_split": {
            "path": str(split_path),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(training))
    return path, training, paths, updated


def test_four_variants_share_micro_and_correct_policy_value(tmp_path):
    _, training, original, updated = _fake_run(tmp_path)
    v = _variant_models(training)
    assert set(v) == {"A", "B", "C", "D"}
    assert {v[k]["micro"]["sha256"] for k in v} == {original["micro"]["sha256"]}
    assert v["A"]["policy"]["path"] == v["C"]["policy"]["path"] == original["policy"]["path"]
    assert v["B"]["policy"]["path"] == v["D"]["policy"]["path"] == updated["policy"]
    assert v["A"]["state_v"]["path"] == v["B"]["state_v"]["path"] == original["state_v"]["path"]
    assert v["C"]["state_v"]["path"] == v["D"]["state_v"]["path"] == updated["state_v"]


def test_rejects_changed_micro_or_corrupted_starting_checkpoints(tmp_path):
    _, training, original, _ = _fake_run(tmp_path)
    trained = json.loads(json.dumps(training))
    trained["rounds"][0]["micro_trained"] = True
    with pytest.raises(ValueError, match="fixed Micro"):
        _variant_models(trained)
    alternate = tmp_path / "new_micro.pt"
    alternate.write_bytes(b"changed")
    changed = json.loads(json.dumps(training))
    changed["rounds"][0]["micro"] = str(alternate)
    with pytest.raises(ValueError, match="Final Micro differs"):
        _variant_models(changed)
    Path(original["policy"]["path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Starting checkpoint differs"):
        _variant_models(training)


def test_run_config_identifies_variants_and_resume_checks_identity(tmp_path, monkeypatch):
    import scripts.comparisons.eval_nine_ball_macro_ablation as module

    path, _, original, updated = _fake_run(tmp_path)
    args = parse_args([
        "--training-manifest", str(path),
        "--base-variant", "C", "--final-variant", "D",
        "--pairs", "2", "--start-seed", "1450000",
        "--output-dir", str(tmp_path / "eval"),
    ])
    observed = {}
    expected = {"overall": {"matches": 4, "completed": 4}, "paired_results": {}}

    def fake_play(out, config, seeds, agents, physics, split):
        observed["config"] = config
        observed["seeds"] = list(seeds)
        observed["agents"] = agents
        return expected

    def fake_agent(micro, **kwargs):
        return SimpleNamespace(micro=micro, **kwargs)

    monkeypatch.setattr(module, "_play_pairs", fake_play)
    assert evaluate(args, agent_factory=fake_agent, simulator_factory=object) == expected
    saved = json.loads((args.output_dir / "evaluation_config.json").read_text())
    assert saved["variants"] == {"base": "C", "final": "D"}
    assert saved["checkpoints"]["base"]["state_v"]["path"] == updated["state_v"]
    assert saved["checkpoints"]["final"]["state_v"]["path"] == updated["state_v"]
    assert saved["checkpoints"]["base"]["policy"]["path"] == original["policy"]["path"]
    assert saved["checkpoints"]["final"]["policy"]["path"] == updated["policy"]
    assert observed["seeds"] == [1450000, 1450018]
    assert observed["agents"]["base"].micro == observed["agents"]["final"].micro

    with pytest.raises(FileExistsError, match="Refusing existing"):
        evaluate(args, agent_factory=fake_agent, simulator_factory=object)
    args.resume = True
    assert evaluate(args, agent_factory=fake_agent, simulator_factory=object) == expected
    args.pairs = 3
    with pytest.raises(ValueError, match="resume config"):
        evaluate(args, agent_factory=fake_agent, simulator_factory=object)


def test_rejects_evaluation_overlap(tmp_path):
    path, _, _, _ = _fake_run(tmp_path)
    args = parse_args([
        "--training-manifest", str(path),
        "--base-variant", "A", "--final-variant", "B",
        "--pairs", "1", "--start-seed", "1400000",
        "--output-dir", str(tmp_path / "eval"),
    ])
    with pytest.raises(ValueError, match="overlaps training rack seeds"):
        evaluate(args)
    assert not args.output_dir.exists()
