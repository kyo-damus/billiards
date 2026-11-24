"""Continue Phase31 from C: Phase28b Policy + final Phase29 State V.

Read completed Phase29 manifest, verify original Micro/Policy checkpoint
provenance, and start a separate *development* self-play loop that keeps
both the Macro Policy and the 5D Micro actor frozen. Each round warms
State V network weights from the previous V (optimizer is reset). The
Q model is trained from scratch only as auxiliary selected-action data.
No official comparison seeds are touched.

Use --plan-only to inspect full commands without making files.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.iterate_nine_ball_hierarchical_v3 import (
    build_plan, parse_args as parse_iteration_args, run as run_iteration,
)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_child_args(args):
    parent_path = args.phase29_manifest
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("kind") != "nine_ball_macro_micro_alternating_v3"
        or parent.get("status") != "completed"
        or parent.get("cross_revision_resume")
        or not parent.get("rounds")
    ):
        raise ValueError("Expected completed, single-revision Phase29 manifest")
    settings = parent["settings"]
    if (
        settings.get("micro_steps") != 0
        or settings.get("race_to", 1) <= 1
        or not settings.get("wpa_break_rules")
    ):
        raise ValueError("Phase31 requires fixed-Micro WPA race-to-N Phase29 run")
    source = parent["source_checkpoints"]
    micro_meta = source["micro"]
    policy_meta = source["policy"]
    final = parent["rounds"][-1]
    for name, record in (("micro", micro_meta), ("policy", policy_meta)):
        if not record.get("path") or not record.get("sha256"):
            raise ValueError(f"Missing original {name} checksum")
        if _sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Original {name} checkpoint differs from manifest")
    final_micro_hash = _sha256(final["micro"])
    if final_micro_hash != micro_meta["sha256"]:
        raise ValueError("Micro changed in Phase29; cannot isolate Macro")
    if not final.get("state_v"):
        raise ValueError("Phase29 final State V checkpoint missing")
    v_path = Path(final["state_v"])
    final_v_hash = _sha256(v_path)
    rack_split = parent.get("rack_layout_split")
    if not rack_split or not Path(rack_split["path"]).is_file():
        raise ValueError("Recorded held-out rack layout split required")
    if _sha256(rack_split["path"]) != rack_split["sha256"]:
        raise ValueError("Rack split checksum mismatch")
    if args.iterations < 1 or args.matches_per_iteration < 2:
        raise ValueError("Iterations must be positive, matches-per-iteration >= 2")

    argv = [
        "--base-micro-checkpoint", str(micro_meta["path"]),
        "--base-policy-checkpoint", str(policy_meta["path"]),
        "--base-state-v-checkpoint", str(v_path),
        "--output-root", str(args.output_root),
        "--iterations", str(args.iterations),
        "--matches-per-iteration", str(args.matches_per_iteration),
        "--start-seed", str(args.start_seed),
        "--race-to", str(settings["race_to"]),
        "--break-policy", str(settings.get("break_policy", "alternate")),
        "--max-shots-per-rack", str(settings["max_shots_per_rack"]),
        "--mcts-simulations", str(settings["mcts_simulations"]),
        "--mcts-max-depth", str(settings["mcts_max_depth"]),
        "--mcts-root-min-visits", str(settings["mcts_root_min_visits"]),
        "--mcts-c-puct", str(settings["mcts_c_puct"]),
        "--gamma", str(settings["gamma"]),
        "--rack-layout-split", str(rack_split["path"]),
        "--policy-epochs", "1",  # Unused, Policy training is disabled.
        "--qv-epochs", str(args.qv_epochs),
        "--micro-steps", "0",
        "--freeze-policy",
        "--warm-start-state-v",
        "--train-seed", str(args.train_seed),
        "--device", args.device,
    ]
    for flag, enabled in (
        ("--learned-safety", settings.get("learned_safety", False)),
        ("--learned-break", settings.get("learned_break", False)),
        ("--outcome-position", settings.get("outcome_position", False)),
        ("--wpa-break-rules", settings.get("wpa_break_rules", False)),
    ):
        if enabled:
            argv.append(flag)
    if args.resume:
        argv.append("--resume")
    child = parse_iteration_args(argv)
    metadata = {
        "parent_phase29_manifest": str(parent_path),
        "parent_phase29_source_git_commit": parent.get("source_git_commit"),
        "starting_variant": "C (Phase28b Policy + Phase29 final State V)",
        "starting_policy_sha256": policy_meta["sha256"],
        "starting_state_v_sha256": final_v_hash,
        "starting_micro_sha256": micro_meta["sha256"],
        "frozen_macro_policy": True,
        "frozen_micro": True,
        "state_v_optimizer_resumed": False,
        "macro_q_learnt_from_scratch": True,
        "evaluation_status": "development_only",
    }
    return child, metadata


def main(args):
    child, metadata = make_child_args(args)
    stages, rounds = build_plan(child)
    plan = {
        **metadata,
        "planned_rounds": rounds,
        "planned_stage_names": [stage.name for stage in stages],
        "output_root": str(child.output_root),
        "training_seed_start": child.start_seed,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.plan_only:
        run_iteration(child)
    return plan


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase29-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--matches-per-iteration", type=int, default=8)
    p.add_argument("--qv-epochs", type=int, default=2)
    p.add_argument("--start-seed", type=int, default=1600000)
    p.add_argument("--train-seed", type=int, default=930000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--plan-only", action="store_true",
                   help="Validate checkpoint identities and show planned stages without writing.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
