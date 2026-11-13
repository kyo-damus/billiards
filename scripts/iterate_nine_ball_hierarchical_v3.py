"""Run bounded 9-ball hierarchical improvement iterations without overwrites.

This alternates MCTS self-play -> Macro visit imitation -> behavior-return
Q/V -> goal-conditioned Micro fine-tuning.  It is NOT an independent
strength evaluation, converged joint RL, or WPA-compliant official match.

Existing checkpoints are read-only. Each round has distinct match seeds and
a complete command log. Failure preserves every completed artifact for
inspection; restarting the same run ID deliberately refuses to overwrite.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_options(args):
    if (
        args.iterations < 1
        or args.matches_per_iteration < 2
        or args.race_to < 1
        or args.start_seed < 0
        or args.max_shots_per_rack < 1
        or args.mcts_simulations < 1
        or args.mcts_max_depth < 1
        or args.mcts_root_min_visits < 0
        or args.mcts_c_puct < 0
        or args.policy_epochs < 1
        or args.qv_epochs < 0
        or args.micro_steps < 0
        or not 0.0 < args.gamma <= 1.0
    ):
        raise ValueError("Invalid hierarchical iteration parameters")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 1 or not visible.split(",")[0].strip():
            raise ValueError("Use exactly one existing CUDA_VISIBLE_DEVICES GPU ID")
    for name in (
        "base_micro_checkpoint", "base_policy_checkpoint",
        "base_state_v_checkpoint",
    ):
        value = getattr(args, name, None)
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"Missing {name}: {value}")


def build_plan(args):
    """Pure planned commands; no subprocess, GPU allocation, or filesystem writes."""
    _ensure_options(args)
    root = Path(args.output_root)
    initial_micro = Path(args.base_micro_checkpoint)
    initial_policy = Path(args.base_policy_checkpoint)
    state_v = (
        Path(args.base_state_v_checkpoint)
        if args.base_state_v_checkpoint else None
    )
    stages = []
    iteration_summaries = []
    python = sys.executable
    match_flags = ("--match-aware",) if args.race_to > 1 else ()
    break_flags = ("--break-policy", args.break_policy) if args.break_policy != "alternate" else ()
    goal_flags = tuple(
        flag for flag, enabled in (
            ("--learned-safety", args.learned_safety),
            ("--learned-break", args.learned_break),
            ("--outcome-position", args.outcome_position),
        ) if enabled
    )
    for index in range(args.iterations):
        folder = root / f"iteration_{index:03d}"
        rollouts = folder / "rollouts"
        visits = folder / "visits.npz"
        policy = folder / "macro_policy_v3.pt"
        returns = folder / "selected_returns.npz"
        qv = folder / "qv"
        micro_dir = folder / "micro"
        # 2N-1 racks max per race-to-N; collector reserves 2N seeds per match.
        # Leave 1000 additional seed IDs between separate iterations.
        seed = args.start_seed + index * (2 * args.race_to * args.matches_per_iteration + 1000)
        collect = [
            python, "-m", "scripts.collect_nine_ball_self_play",
            "--matches", str(args.matches_per_iteration),
            "--race-to", str(args.race_to),
            "--start-seed", str(seed),
            "--max-shots-per-rack", str(args.max_shots_per_rack),
            "--agent-kind", "micro_mcts",
            "--micro-checkpoint", str(initial_micro),
            "--macro-policy-checkpoint", str(initial_policy),
            "--mcts-simulations", str(args.mcts_simulations),
            "--mcts-max-depth", str(args.mcts_max_depth),
            "--mcts-root-min-visits", str(args.mcts_root_min_visits),
            "--mcts-c-puct", str(args.mcts_c_puct),
            "--mcts-gamma", str(args.gamma),
            "--device", args.device,
            "--output-dir", str(rollouts),
            *break_flags,
        ]
        if state_v is not None:
            collect.extend(["--state-value-checkpoint", str(state_v)])
        collect.extend(goal_flags)
        stages.append(Stage(f"{index:03d}_self_play", tuple(collect)))
        stages.append(Stage(f"{index:03d}_visit_dataset", (
            python, "-m", "scripts.build_nine_ball_mcts_visit_dataset_v3",
            "--rollout-dir", str(rollouts), "--output", str(visits),
            *match_flags,
        )))
        stages.append(Stage(f"{index:03d}_macro_policy", (
            python, "-m", "scripts.train_nine_ball_macro_visits_v3",
            "--dataset", str(visits),
            "--base-policy-checkpoint", str(initial_policy),
            "--output", str(policy),
            "--epochs", str(args.policy_epochs),
            "--device", args.device,
            "--seed", str(args.train_seed + index),
        )))
        if args.qv_epochs:
            stages.append(Stage(f"{index:03d}_return_dataset", (
                python, "-m", "scripts.build_nine_ball_macro_returns_v3",
                "--rollout-dir", str(rollouts), "--output", str(returns),
                "--gamma", str(args.gamma),
                *match_flags,
            )))
            stages.append(Stage(f"{index:03d}_macro_qv", (
                python, "-m", "scripts.train_nine_ball_macro_qv_v3",
                "--dataset", str(returns), "--output-dir", str(qv),
                "--epochs", str(args.qv_epochs),
                "--device", args.device,
                "--seed", str(args.train_seed + 10000 + index),
            )))
            state_v = qv / "state_v_v3.pt"
        if args.micro_steps:
            stages.append(Stage(f"{index:03d}_micro_finetune", (
                python, "-m", "scripts.train_nine_ball_micro_sac",
                "--rollout-dir", str(rollouts),
                "--init-checkpoint", str(initial_micro),
                "--goal-source", "logged",
                "--warmup-steps", "0",
                "--output-dir", str(micro_dir),
                "--steps", str(args.micro_steps),
                "--seed", str(args.train_seed + 20000 + index),
                "--device", args.device,
                *goal_flags,
            )))
            initial_micro = micro_dir / "micro_sac_5d.pt"
        initial_policy = policy
        iteration_summaries.append({
            "iteration": index,
            "start_seed": seed,
            "matches": args.matches_per_iteration,
            "race_to": args.race_to,
            "rollouts": str(rollouts),
            "policy": str(initial_policy),
            "micro": str(initial_micro),
            "state_v": str(state_v) if state_v else None,
            "qv_trained": bool(args.qv_epochs),
            "micro_trained": bool(args.micro_steps),
        })
    return stages, iteration_summaries


def _git_commit():
    """Record source revision for provenance; no Git modification."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _stage_outputs(stage):
    """Minimal completion artifacts; never trust a completed flag alone."""
    argv = list(stage.argv)
    def argument(name):
        return Path(argv[argv.index(name) + 1])
    if stage.name.endswith("_self_play"):
        return (argument("--output-dir") / "summary.json",)
    if stage.name.endswith("_visit_dataset"):
        return (argument("--output"),)
    if stage.name.endswith("_macro_policy"):
        return (argument("--output"),)
    if stage.name.endswith("_return_dataset"):
        return (argument("--output"),)
    if stage.name.endswith("_macro_qv"):
        directory = argument("--output-dir")
        return (directory / "candidate_q_v3.pt", directory / "state_v_v3.pt")
    if stage.name.endswith("_micro_finetune"):
        return (argument("--output-dir") / "micro_sac_5d.pt",)
    raise ValueError(f"Unknown planned stage: {stage.name}")


def _run_logged(stage, root, *, resume=False):
    log_path = root / "logs" / f"{stage.name}.log"
    if log_path.exists():
        if not resume:
            raise FileExistsError(f"Existing stage log: {log_path}")
        i = 1
        while (root / "logs" / f"{stage.name}_retry_{i:03d}.log").exists():
            i += 1
        log_path = root / "logs" / f"{stage.name}_retry_{i:03d}.log"
    argv = list(stage.argv)
    if resume and stage.name.endswith("_self_play"):
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        if output_dir.is_dir():
            argv.append("--resume")
    print(f"\n=== {stage.name} ===", flush=True)
    print(" ".join(argv), flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"Stage {stage.name} failed (exit {code}); "
            f"logs and prior outputs preserved in {root}"
        )
    return str(log_path)


def run(args):
    _ensure_options(args)
    root = Path(args.output_root)
    if root.exists() and not args.resume:
        raise FileExistsError(f"Refusing existing run directory: {root}")
    if args.resume and not root.is_dir():
        raise FileNotFoundError(f"Cannot resume nonexistent run: {root}")
    stages, rounds = build_plan(args)
    # Resolve model file identities before any output creation or GPU work.
    inputs = {
        name: {
            "path": str(value),
            "sha256": _sha256(value),
        }
        for name, value in (
            ("micro", args.base_micro_checkpoint),
            ("policy", args.base_policy_checkpoint),
            ("state_v", args.base_state_v_checkpoint),
        ) if value is not None
    }
    new_manifest = {
        "kind": "nine_ball_macro_micro_alternating_v3",
        "source_git_commit": _git_commit(),
        "status": "running",
        "source": "version_3_self_play",
        "objective": (
            "MCTS visit Policy imitation + selected-action terminal-return "
            "Q/V regression + fixed-rollout-goal Micro SAC fine-tuning. "
            "Not independent evaluation; partial nine-ball rules."
        ),
        "source_checkpoints": inputs,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rounds": rounds,
        "stages": [
            {"name": stage.name, "argv": list(stage.argv), "status": "pending"}
            for stage in stages
        ],
    }
    location = root / "run_manifest.json"
    if args.resume:
        if not location.is_file():
            raise FileNotFoundError(f"Missing resume manifest: {location}")
        manifest = json.loads(location.read_text(encoding="utf-8"))
        if manifest.get("status") != "interrupted":
            raise ValueError("Only an interrupted run may be resumed")
        if (
            manifest.get("kind") != new_manifest["kind"]
            or manifest.get("source_checkpoints") != inputs
            or len(manifest.get("stages", [])) != len(stages)
            or any(
                saved.get("name") != expected.name
                or saved.get("argv") != list(expected.argv)
                for saved, expected in zip(manifest["stages"], stages)
            )
        ):
            raise ValueError("Resume command/model identity differs from saved manifest")
        previous_settings = {
            k: v for k, v in manifest["settings"].items()
            if k not in ("resume", "accept_unverified_code_change")
        }
        current_settings = {
            k: v for k, v in new_manifest["settings"].items()
            if k not in ("resume", "accept_unverified_code_change")
        }
        previous_settings.setdefault("learned_safety", False)
        previous_settings.setdefault("learned_break", False)
        previous_settings.setdefault("outcome_position", False)
        previous_settings.setdefault("race_to", 1)
        previous_settings.setdefault("break_policy", "alternate")
        if previous_settings != current_settings:
            raise ValueError("Resume settings differ from previous run")
        original_commit = manifest.get("source_git_commit")
        current_commit = new_manifest["source_git_commit"]
        if original_commit != current_commit:
            if not args.accept_unverified_code_change:
                raise ValueError(
                    "Run Git revision differs or original was not recorded; "
                    "explicitly supply --accept-unverified-code-change for "
                    "debug/smoke ONLY, or use a fresh output-root for clean data"
                )
            manifest["cross_revision_resume"] = True
            manifest["warning"] = (
                "Mixed or unknown code versions in one run. Treat as a "
                "debug-only smoke; do not claim clean performance comparisons."
            )
        for saved, expected in zip(manifest["stages"], stages):
            outputs = _stage_outputs(expected)
            if saved["status"] == "completed":
                if not all(p.is_file() for p in outputs):
                    raise FileNotFoundError(
                        f"Completed stage missing outputs: {expected.name}"
                    )
            elif not expected.name.endswith("_self_play"):
                if any(p.exists() for p in outputs):
                    raise FileExistsError(
                        f"Incomplete non-collector stage has existing outputs: "
                        f"{expected.name}; refusing unsafe overwrite"
                    )
        manifest["status"] = "running"
        manifest.pop("error", None)
    else:
        manifest = new_manifest
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()

    def save():
        location.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    save()
    try:
        for i, stage in enumerate(stages):
            record = manifest["stages"][i]
            if record["status"] == "completed":
                continue
            record["status"] = "running"
            save()
            log = _run_logged(stage, root, resume=args.resume)
            record.update(status="completed", log=log)
            save()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print(f"Saved complete iteration manifest: {location}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-micro-checkpoint", type=Path, required=True)
    p.add_argument("--base-policy-checkpoint", type=Path, required=True)
    p.add_argument("--base-state-v-checkpoint", type=Path)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--matches-per-iteration", type=int, default=32)
    p.add_argument("--race-to", type=int, default=1)
    p.add_argument("--break-policy", choices=("alternate", "winner", "loser"), default="alternate")
    p.add_argument("--start-seed", type=int, default=510000)
    p.add_argument("--max-shots-per-rack", type=int, default=80)
    p.add_argument("--mcts-simulations", type=int, default=24)
    p.add_argument("--mcts-max-depth", type=int, default=2)
    p.add_argument("--mcts-root-min-visits", type=int, default=1)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--gamma", type=float)
    p.add_argument("--policy-epochs", type=int, default=10)
    p.add_argument("--qv-epochs", type=int, default=5)
    p.add_argument("--micro-steps", type=int, default=5000)
    p.add_argument("--learned-safety", action="store_true")
    p.add_argument("--learned-break", action="store_true")
    p.add_argument("--outcome-position", action="store_true")
    p.add_argument("--train-seed", type=int, default=610000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run without overwriting completed files")
    p.add_argument("--accept-unverified-code-change", action="store_true",
                   help="Allow mixed/unknown code-revision debug-only continuation")
    args = p.parse_args(argv)
    if args.gamma is None:
        args.gamma = 1.0 if args.race_to > 1 else 0.99
    return args


if __name__ == "__main__":
    run(parse_args())
