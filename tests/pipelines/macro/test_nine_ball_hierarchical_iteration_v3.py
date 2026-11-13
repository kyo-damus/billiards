"""Phase15 executable Macro/Micro alternation plan (no GPU or FastFiz)."""

from pathlib import Path

import pytest
import torch

from scripts.iterate_nine_ball_hierarchical_v3 import (
    build_plan, parse_args, run,
)


def _args(tmp_path, extra=()):
    files = [tmp_path / name for name in ("micro.pt", "policy.pt", "state_v.pt")]
    for path in files:
        path.write_bytes(b"test-checkpoint-identity-only")
    return parse_args([
        "--base-micro-checkpoint", str(files[0]),
        "--base-policy-checkpoint", str(files[1]),
        "--base-state-v-checkpoint", str(files[2]),
        "--output-root", str(tmp_path / "runs"),
        "--iterations", "2",
        "--matches-per-iteration", "4",
        "--micro-steps", "10",
        "--qv-epochs", "3",
        *extra,
    ])


def test_plan_alternates_complete_macro_micro_loop_and_is_seed_disjoint(tmp_path):
    args = _args(tmp_path)
    stages, rounds = build_plan(args)
    assert len(rounds) == 2
    assert [x.name for x in stages] == [
        "000_self_play", "000_visit_dataset", "000_macro_policy",
        "000_return_dataset", "000_macro_qv", "000_micro_finetune",
        "001_self_play", "001_visit_dataset", "001_macro_policy",
        "001_return_dataset", "001_macro_qv", "001_micro_finetune",
    ]
    assert rounds[1]["start_seed"] > rounds[0]["start_seed"] + 8
    second_collect = stages[6].argv
    assert str(tmp_path / "runs" / "iteration_000" / "macro_policy_v3.pt") in second_collect
    assert str(tmp_path / "runs" / "iteration_000" / "micro" / "micro_sac_5d.pt") in second_collect
    assert str(tmp_path / "runs" / "iteration_000" / "qv" / "state_v_v3.pt") in second_collect
    assert second_collect[second_collect.index("--mcts-gamma")+1] == str(args.gamma)
    assert "--agent-kind" in second_collect
    assert second_collect[second_collect.index("--agent-kind")+1] == "micro_mcts"
    micro = stages[5].argv
    assert micro[micro.index("--goal-source")+1] == "logged"
    assert micro[micro.index("--warmup-steps")+1] == "0"
    assert "--init-checkpoint" in micro


def test_plan_can_hold_micro_and_qv_fixed_but_still_update_macro(tmp_path):
    args = _args(tmp_path, ("--micro-steps", "0", "--qv-epochs", "0"))
    stages, rounds = build_plan(args)
    assert [x.name for x in stages] == [
        "000_self_play", "000_visit_dataset", "000_macro_policy",
        "001_self_play", "001_visit_dataset", "001_macro_policy",
    ]
    assert rounds[1]["micro"] == str(args.base_micro_checkpoint)
    assert rounds[1]["state_v"] == str(args.base_state_v_checkpoint)


def test_iteration_preflight_never_overwrites_previous_results(tmp_path):
    args = _args(tmp_path)
    args.output_root.mkdir()
    with pytest.raises(FileExistsError, match="Refusing existing"):
        run(args)
    assert list(args.output_root.iterdir()) == []


def test_missing_checkpoint_and_gpu_guard_fails_without_creating_output(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.base_micro_checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        run(args)
    assert not args.output_root.exists()

    args.base_micro_checkpoint.write_bytes(b"replaced")
    args.device = "cuda"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError, match="exactly one"):
        run(args)
    assert not args.output_root.exists()


def test_resume_requires_explicit_acknowledgement_when_source_revision_unknown(
    tmp_path, monkeypatch,
):
    import json
    import scripts.iterate_nine_ball_hierarchical_v3 as runner

    args = _args(tmp_path)
    def interrupt(*unused, **kwargs):
        raise RuntimeError("synthetic interruption before first collector")
    monkeypatch.setattr(runner, "_run_logged", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(args)
    manifest_path = args.output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["stages"][0]["status"] == "running"

    # This is how the original Phase15 interrupted run is recorded:
    # it predates source_git_commit metadata.
    manifest.pop("source_git_commit", None)
    manifest_path.write_text(json.dumps(manifest))
    args.resume = True
    with pytest.raises(ValueError, match="Git revision differs"):
        run(args)
    args.accept_unverified_code_change = True
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(args)
    resumed = json.loads(manifest_path.read_text())
    assert resumed["status"] == "interrupted"
    assert resumed["cross_revision_resume"] is True
    assert "debug-only" in resumed["warning"]


def test_stage_expected_outputs_never_treat_partial_collector_as_done(tmp_path):
    from scripts.iterate_nine_ball_hierarchical_v3 import _stage_outputs
    args = _args(tmp_path)
    stages, _ = build_plan(args)
    assert _stage_outputs(stages[0]) == (
        tmp_path / "runs" / "iteration_000" / "rollouts" / "summary.json",
    )
    assert _stage_outputs(stages[2]) == (
        tmp_path / "runs" / "iteration_000" / "macro_policy_v3.pt",
    )


def test_learned_safety_is_passed_to_collection_and_micro_training(tmp_path):
    stages, _ = build_plan(_args(tmp_path, ("--learned-safety",)))
    enabled_stages = {stage.name for stage in stages if "--learned-safety" in stage.argv}
    assert enabled_stages == {
        "000_self_play", "000_micro_finetune", "001_self_play", "001_micro_finetune",
    }
