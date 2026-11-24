"""Macro Policy/State V 2x2 ablation with one fixed Phase29 training run.

A = starting Policy, starting V (Phase28b)
B = final Policy, starting V    (Policy-only update)
C = starting Policy, final V    (State-V-only update)
D = final Policy, final V       (full Phase29)

The training's starting and final 5D Micro actor must be identical.
This is a *development* evaluation, not the final held-out comparison.
Matches, seeds, initial layouts, break assignments, physics and rules are
identical within a pair. Saved matches can be resumed without overwrite.
"""
import argparse
import json
import os
from pathlib import Path

from scripts.comparisons.eval_nine_ball_baselines import _check_seed_isolation
from scripts.eval_nine_ball_hierarchical_v3 import _checkpoint_info, _play_pairs
from scripts.iterate_nine_ball_hierarchical_v3 import _git_commit
from src.game.agents.proposed.hierarchical import NineBallMCTSAgent
from src.game.nine_ball_rack_split import load_rack_layout_split
from src.pipelines.experiment_io import lock_run, write_experiment_json

VARIANT_KEYS = ("A", "B", "C", "D")


def _variant_models(training):
    """Build all four variants, checking model identity before GPU setup."""
    if (
        training.get("kind") != "nine_ball_macro_micro_alternating_v3"
        or training.get("status") != "completed"
        or training.get("cross_revision_resume")
        or not training.get("rounds")
    ):
        raise ValueError("Expected a completed single-revision nine-ball training run")
    if any(round_info.get("micro_trained") for round_info in training["rounds"]):
        raise ValueError("Ablation requires a fixed Micro across training rounds")

    source = training["source_checkpoints"]
    last = training["rounds"][-1]
    initial = {
        name: _checkpoint_info(
            source[name]["path"], expected_sha256=source[name]["sha256"]
        )
        for name in ("micro", "policy", "state_v")
    }
    final_micro = _checkpoint_info(last["micro"])
    if final_micro["sha256"] != initial["micro"]["sha256"]:
        raise ValueError("Final Micro differs from starting Micro; cannot isolate Macro")
    if last.get("state_v") is None:
        raise ValueError("Final State V checkpoint missing")
    policy_final = _checkpoint_info(last["policy"])
    value_final = _checkpoint_info(last["state_v"])

    shared_micro = initial["micro"]
    return {
        "A": {"micro": shared_micro, "policy": initial["policy"], "state_v": initial["state_v"]},
        "B": {"micro": shared_micro, "policy": policy_final, "state_v": initial["state_v"]},
        "C": {"micro": shared_micro, "policy": initial["policy"], "state_v": value_final},
        "D": {"micro": shared_micro, "policy": policy_final, "state_v": value_final},
    }


def evaluate(args, *, agent_factory=None, simulator_factory=None):
    if args.base_variant not in VARIANT_KEYS or args.final_variant not in VARIANT_KEYS:
        raise ValueError("Unknown Macro variant; expected A/B/C/D")
    if args.base_variant == args.final_variant:
        raise ValueError("Ablation must compare distinct model variants")
    if args.pairs < 1 or args.start_seed < 0:
        raise ValueError("Use positive pairs and nonnegative seed")
    if args.device == "cuda":
        cuda_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cuda_ids or "," in cuda_ids:
            raise ValueError("Set exactly one CUDA_VISIBLE_DEVICES GPU")

    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    variants = _variant_models(training)
    settings = training["settings"]
    if settings.get("race_to", 1) <= 1 or not settings.get("wpa_break_rules", False):
        raise ValueError("Phase31 ablation expects multi-rack WPA nine-ball")
    _check_seed_isolation(training, args.start_seed, args.pairs)
    split_record = training.get("rack_layout_split")
    if split_record is None:
        raise ValueError("Recorded held-out rack layout split is required")
    split = load_rack_layout_split(split_record["path"])
    if split.sha256 != split_record["sha256"]:
        raise ValueError("Recorded rack layout split hash differs")

    checkpoints = {
        "base": variants[args.base_variant],
        "final": variants[args.final_variant],
    }
    search = {
        "simulations": settings["mcts_simulations"],
        "max_depth": settings["mcts_max_depth"],
        "c_puct": settings["mcts_c_puct"],
        "gamma": settings["gamma"],
        "root_min_visits": settings["mcts_root_min_visits"],
        "learned_safety": settings.get("learned_safety", False),
        "learned_break": settings.get("learned_break", False),
        "outcome_position": settings.get("outcome_position", False),
        "match_aware": True,
        "wpa_break_rules": True,
    }
    config = {
        "comparison": "macro_policy_state_v_2x2_ablation",
        "evaluation_git_commit": _git_commit(),
        "training_git_commit": training.get("source_git_commit"),
        "training_manifest": str(args.training_manifest),
        "variants": {"base": args.base_variant, "final": args.final_variant},
        "checkpoints": checkpoints,
        "race_to": settings["race_to"],
        "break_policy": settings.get("break_policy", "alternate"),
        "pairs": args.pairs,
        "start_seed": args.start_seed,
        "max_shots_per_rack": settings["max_shots_per_rack"],
        "search": search,
        "rack_layout_split": split_record,
        "device": args.device,
    }
    root = args.output_dir
    if args.resume:
        if not root.is_dir():
            raise FileNotFoundError(f"Cannot resume missing output: {root}")
    elif root.exists():
        raise FileExistsError(f"Refusing existing ablation output: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)

    with lock_run(root):
        config_path = root / "evaluation_config.json"
        if args.resume:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            expected = dict(config, evaluation_git_commit=saved["evaluation_git_commit"])
            if saved != expected:
                raise ValueError("Ablation resume config/checkpoints mismatch")
        else:
            write_experiment_json(config_path, config)

        summary_path = root / "summary.json"
        if summary_path.exists():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            if agent_factory is None:
                agent_factory = NineBallMCTSAgent
            if simulator_factory is None:
                from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
                simulator_factory = MultiBallFastFizSimulator
            agents = {
                label: agent_factory(
                    model["micro"]["path"],
                    macro_policy_checkpoint=model["policy"]["path"],
                    state_value_checkpoint=model["state_v"]["path"],
                    device=args.device,
                    layout_split=split,
                    **search,
                )
                for label, model in checkpoints.items()
            }
            stride = 2 * settings["race_to"]
            seeds = range(args.start_seed, args.start_seed + args.pairs * stride, stride)
            result = _play_pairs(root, config, seeds, agents, simulator_factory, split)
            write_experiment_json(summary_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--base-variant", choices=VARIANT_KEYS, required=True)
    parser.add_argument("--final-variant", choices=VARIANT_KEYS, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=1450000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
