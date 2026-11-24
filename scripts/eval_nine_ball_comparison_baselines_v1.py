"""Paired evaluation of adapted nine-ball comparison agents vs a learned Macro.

Baselines share the final Micro checkpoint, match rules, rack split, and
physical simulator with the proposal.  No baseline learns from test matches.
The evaluation refuses unsafe overwrite and validates resume identity.
"""
import argparse
import json
import os
from pathlib import Path

from scripts.eval_nine_ball_hierarchical_v3 import _checkpoint_info, _play_pairs
from scripts.iterate_nine_ball_hierarchical_v3 import _git_commit
from src.game.nine_ball_comparison_baselines import PotOnlyGreedyAgent
from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
from src.game.nine_ball_rack_split import load_rack_layout_split
from src.pipelines.experiment_io import lock_run, write_experiment_json


BASELINES = ("pot_greedy", "uniform_mcts")


def _check_seed_isolation(training, start_seed, pairs):
    """A pair uses up to 2*race_to-1 rack seeds; reject training overlap."""
    race_to = training["settings"]["race_to"]
    width = 2 * race_to
    evaluated = [
        (seed, seed + width - 1)
        for seed in range(start_seed, start_seed + pairs * width, width)
    ]
    for round_info in training["rounds"]:
        train_width = 2 * round_info.get("race_to", race_to)
        train_start = round_info["start_seed"]
        for i in range(round_info["matches"]):
            lo = train_start + i * train_width
            hi = lo + train_width - 1
            if any(a < hi and lo < b for a, b in evaluated):
                raise ValueError("Comparison seed interval overlaps training rack seeds")


def _agents(baseline, micro, policy, state_v, *, settings, layout_split, device):
    base = (
        PotOnlyGreedyAgent(micro, device=device)
        if baseline == "pot_greedy"
        else NineBallMCTSAgent(
            micro,
            macro_policy_checkpoint=None,
            state_value_checkpoint=None,
            simulations=settings["mcts_simulations"],
            max_depth=settings["mcts_max_depth"],
            c_puct=settings["mcts_c_puct"],
            gamma=settings["gamma"],
            root_min_visits=settings["mcts_root_min_visits"],
            device=device,
            learned_safety=settings.get("learned_safety", False),
            learned_break=settings.get("learned_break", False),
            outcome_position=settings.get("outcome_position", False),
            match_aware=True,
            wpa_break_rules=settings.get("wpa_break_rules", False),
            layout_split=layout_split,
        )
    )
    final = NineBallMCTSAgent(
        micro,
        macro_policy_checkpoint=policy,
        state_value_checkpoint=state_v,
        simulations=settings["mcts_simulations"],
        max_depth=settings["mcts_max_depth"],
        c_puct=settings["mcts_c_puct"],
        gamma=settings["gamma"],
        root_min_visits=settings["mcts_root_min_visits"],
        device=device,
        learned_safety=settings.get("learned_safety", False),
        learned_break=settings.get("learned_break", False),
        outcome_position=settings.get("outcome_position", False),
        match_aware=True,
        wpa_break_rules=settings.get("wpa_break_rules", False),
        layout_split=layout_split,
    )
    return {"base": base, "final": final}


def evaluate(args):
    if args.baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    if args.pairs < 1 or args.start_seed < 0:
        raise ValueError("pairs must be positive; seed must be nonnegative")
    if args.device == "cuda":
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not gpu or "," in gpu:
            raise ValueError("Choose exactly one CUDA_VISIBLE_DEVICES GPU")
    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if (
        training.get("kind") != "nine_ball_macro_micro_alternating_v3"
        or training.get("status") != "completed"
        or training.get("cross_revision_resume")
        or not training.get("rounds")
    ):
        raise ValueError("Expected a completed, single-revision training manifest")
    settings = training["settings"]
    if settings.get("race_to", 1) <= 1 or not settings.get("wpa_break_rules", False):
        raise ValueError("This baseline evaluation requires WPA nine-ball race-to-N")
    _check_seed_isolation(training, args.start_seed, args.pairs)

    split_record = training.get("rack_layout_split")
    if not split_record:
        raise ValueError("A recorded held-out rack layout split is required")
    split = load_rack_layout_split(split_record["path"])
    if split.sha256 != split_record["sha256"]:
        raise ValueError("Rack split differs from training manifest")

    final = training["rounds"][-1]
    micro = _checkpoint_info(final["micro"])
    policy = _checkpoint_info(final["policy"])
    state_v = _checkpoint_info(final["state_v"])
    # Both sides use this exact Micro, including the frozen-break actor.
    checkpoints = {
        "base": {
            "micro": micro,
            "policy": _checkpoint_info(None),
            "state_v": _checkpoint_info(None),
        },
        "final": {"micro": micro, "policy": policy, "state_v": state_v},
    }
    config = {
        "evaluation_git_commit": _git_commit(),
        "comparison": f"adapted_{args.baseline}_vs_learned_macro",
        "baseline": args.baseline,
        "training_manifest": str(args.training_manifest),
        "training_git_commit": training.get("source_git_commit"),
        "checkpoints": checkpoints,
        "race_to": settings["race_to"],
        "break_policy": settings["break_policy"],
        "pairs": args.pairs,
        "start_seed": args.start_seed,
        "max_shots_per_rack": settings["max_shots_per_rack"],
        "search": {
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
        },
        "rack_layout_split": split_record,
        "device": args.device,
    }
    if not args.resume:
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing existing evaluation output: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    elif not args.output_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing evaluation: {args.output_dir}")

    with lock_run(args.output_dir):
        config_path = args.output_dir / "evaluation_config.json"
        if args.resume:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            expected = dict(config, evaluation_git_commit=saved["evaluation_git_commit"])
            if saved != expected:
                raise ValueError("Resume settings or checkpoints differ from saved evaluation")
        else:
            write_experiment_json(config_path, config)
        summary_path = args.output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator

            agents = _agents(
                args.baseline, micro["path"], policy["path"], state_v["path"],
                settings=settings, layout_split=split, device=args.device,
            )
            seeds = range(
                args.start_seed,
                args.start_seed + args.pairs * (2 * settings["race_to"]),
                2 * settings["race_to"],
            )
            summary = _play_pairs(
                args.output_dir, config, seeds, agents, MultiBallFastFizSimulator, split,
            )
            write_experiment_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-manifest", type=Path, required=True)
    p.add_argument("--baseline", choices=BASELINES, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--start-seed", type=int, default=1460000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
