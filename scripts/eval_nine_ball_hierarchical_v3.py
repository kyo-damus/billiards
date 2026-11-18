"""Run paired 9-ball matches for a training run or with only Micro changed."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.iterate_nine_ball_hierarchical_v3 import _git_commit
from src.pipelines.experiment_io import lock_run, write_experiment_json

MODEL_KEYS = ("micro", "policy", "state_v")


def _checkpoint_info(path, *, expected_sha256=None):
    if path is None:
        return {"path": None, "sha256": None}
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Starting checkpoint differs from the training manifest: {path}")
    return {"path": str(path), "sha256": digest}


def _model_inputs(training, *, candidate_micro_checkpoint=None):
    base = training["source_checkpoints"]
    final = training["rounds"][-1]
    starting_keys = MODEL_KEYS if candidate_micro_checkpoint is None else ("micro",)
    starting = {}
    for key in starting_keys:
        recorded = base.get(key, {})
        path = recorded.get("path")
        if key == "state_v" and path is None:
            starting[key] = _checkpoint_info(None)
        else:
            starting[key] = _checkpoint_info(recorded["path"], expected_sha256=recorded["sha256"])
    macro = {key: _checkpoint_info(final[key]) for key in ("policy", "state_v")}
    if candidate_micro_checkpoint is not None:
        return {
            "base": {**starting, **macro},
            "final": {"micro": _checkpoint_info(candidate_micro_checkpoint), **macro},
        }
    return {
        "base": starting,
        "final": {"micro": _checkpoint_info(final["micro"]), **macro},
    }


def _tally(games):
    wins = Counter(game["winner"] for game in games)
    completed = wins["base"] + wins["final"]
    return {
        "matches": len(games),
        "completed": completed,
        "base_wins": wins["base"],
        "final_wins": wins["final"],
        "unfinished": wins[None],
        "stop_reasons": dict(Counter(game["stop_reason"] for game in games)),
        "final_win_rate_completed": wins["final"] / completed if completed else None,
    }


def _read_saved_match(path, identity):
    game = json.loads(path.read_text(encoding="utf-8"))
    finished = game["match_finished"]
    winner = game["match_winner"]
    players = identity["player_models"]
    if (
        any(game.get(key) != value for key, value in identity.items())
        or type(finished) is not bool
        or (finished and (type(winner) is not int or winner not in (0, 1) or game["winner"] != players[winner]))
        or (not finished and game["winner"] is not None)
        or not isinstance(game.get("stop_reason"), str)
    ):
        raise ValueError(f"Saved match does not match the evaluation schedule: {path}")
    return {key: game[key] for key in ("breaking_model", "winner", "stop_reason")}


def evaluate(args, *, agent_factory=None, simulator_factory=None):
    if args.pairs < 1 or args.start_seed < 0:
        raise ValueError("Use a positive number of pairs and a nonnegative start seed")
    resume = getattr(args, "resume", False)
    if resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Evaluation resume directory absent: {args.output_dir}")
    if not resume and args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse evaluation output: {args.output_dir}")

    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if (
        training.get("kind") != "nine_ball_macro_micro_alternating_v3"
        or training.get("status") != "completed"
        or training.get("cross_revision_resume")
        or not training.get("rounds")
    ):
        raise ValueError("Expected a completed nine-ball v3 training run from one code revision")
    settings = training["settings"]
    race_to = settings.get("race_to", 1)
    break_policy = settings.get("break_policy", "alternate")
    stride = 2 * race_to if race_to > 1 else 1
    seeds = range(args.start_seed, args.start_seed + args.pairs * stride, stride)
    for round_info in training["rounds"]:
        start = round_info["start_seed"]
        training_race = round_info.get("race_to", race_to)
        training_stride = 2 * training_race
        stop = start + training_stride * round_info["matches"]
        for seed in seeds:
            for rack_seed in range(seed, seed + 2 * race_to - 1):
                if start <= rack_seed < stop and (rack_seed - start) % training_stride < training_stride - 1:
                    raise ValueError("Evaluation seeds overlap with training rack seeds")

    from src.game.nine_ball_rack_split import legacy_rack_order, load_rack_layout_split

    recorded_split = training.get("rack_layout_split")
    split_path = args.rack_layout_split or (Path(recorded_split["path"]) if recorded_split else None)
    layout_split = load_rack_layout_split(split_path) if split_path else None
    if recorded_split is not None and layout_split.sha256 != recorded_split["sha256"]:
        raise ValueError("Rack layout split differs from the training manifest")
    if layout_split is not None and recorded_split is None:
        heldout = set(layout_split.evaluation)
        for rnd in training["rounds"]:
            training_stride = 2 * rnd.get("race_to", race_to)
            for i in range(rnd["matches"]):
                start = rnd["start_seed"] + i * training_stride
                if any(legacy_rack_order(seed) in heldout
                       for seed in range(start, start + training_stride - 1)):
                    raise ValueError("Evaluation layouts overlap with legacy training rack orders")

    search = {
        "simulations": settings["mcts_simulations"],
        "max_depth": settings["mcts_max_depth"],
        "c_puct": settings["mcts_c_puct"],
        "gamma": settings["gamma"],
        "root_min_visits": settings["mcts_root_min_visits"],
        "learned_safety": settings.get("learned_safety", False),
        "learned_break": settings.get("learned_break", False),
        "outcome_position": settings.get("outcome_position", False),
        "match_aware": race_to > 1,
        "wpa_break_rules": settings.get("wpa_break_rules", False),
    }
    if layout_split is not None:
        search["layout_split"] = layout_split
    candidate_micro = getattr(args, "candidate_micro_checkpoint", None)
    checkpoints = _model_inputs(training, candidate_micro_checkpoint=candidate_micro)
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Use exactly one existing CUDA_VISIBLE_DEVICES GPU ID")

    if agent_factory is None:
        from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
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
            **search,
        )
        for label, model in checkpoints.items()
    }
    config = {
        "evaluation_git_commit": _git_commit(),
        "comparison": "micro_only" if candidate_micro is not None else "training_base_vs_final",
        "training_manifest": str(args.training_manifest),
        "training_git_commit": training.get("source_git_commit"),
        "checkpoints": checkpoints,
        "race_to": race_to,
        "break_policy": break_policy,
        "pairs": args.pairs,
        "start_seed": args.start_seed,
        "max_shots_per_rack": settings["max_shots_per_rack"],
        "search": {key: value for key, value in search.items() if key != "layout_split"},
        "rack_layout_split": {"path": str(split_path), "sha256": layout_split.sha256} if layout_split else None,
        "device": args.device,
    }
    if not resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with lock_run(args.output_dir):
        config_path = args.output_dir / "evaluation_config.json"
        if resume:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            # Match files written after resume record the new revision; the
            # original config still identifies the revision of legacy files.
            expected = dict(config, evaluation_git_commit=saved["evaluation_git_commit"])
            if saved != expected:
                raise ValueError(f"Resume settings or input checkpoints differ: {config_path}")
        else:
            write_experiment_json(config_path, config)
        summary_path = args.output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = _play_pairs(args.output_dir, config, seeds, agents, simulator_factory, layout_split)
            write_experiment_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _play_pairs(output_dir, config, seeds, agents, simulator_factory, layout_split):
    from scripts.collect_nine_ball_self_play import play_match
    from src.game.match import MatchConfig
    from src.game.nine_ball_session import NineBallMatchSession

    games = []
    pairs = Counter()
    for index, seed in enumerate(seeds):
        # Hold player slots fixed within the pair and balance them across seeds.
        final_player = index % 2
        players = ("base", "final") if final_player else ("final", "base")
        pair = []
        for breaker in (final_player, 1 - final_player):
            identity = {
                "seed": seed,
                "player_models": list(players),
                "first_breaker": breaker,
                "breaking_model": players[breaker],
            }
            path = output_dir / f"seed_{seed}_break_{players[breaker]}.json"
            if path.exists():
                pair.append(_read_saved_match(path, identity))
                print(f"seed={seed} breaker={players[breaker]} reused saved match", flush=True)
                continue
            match = NineBallMatchSession(
                simulator_factory=simulator_factory,
                match_config=MatchConfig(
                    race_to=config["race_to"], rule_id="nine_ball", first_breaker=breaker,
                    break_policy=config["break_policy"],
                ),
                seed=seed,
                wpa_break_rules=config["search"]["wpa_break_rules"],
                layout_split=layout_split,
                evaluation=True,
            )
            outcome = play_match(
                match, tuple(agents[label] for label in players),
                max_shots_per_rack=config["max_shots_per_rack"],
            )
            winner = outcome["match_winner"]
            game = {
                **identity,
                "winner": players[winner] if outcome["match_finished"] else None,
                "stop_reason": outcome["stop_reason"],
                "shots": outcome["shots"],
                "fouls": outcome["fouls"],
            }
            write_experiment_json(
                path, {**outcome, **game, "evaluation_git_commit": config["evaluation_git_commit"]},
            )
            pair.append(game)
            print(
                f"seed={seed} breaker={game['breaking_model']} "
                f"winner={game['winner']} stop={game['stop_reason']}",
                flush=True,
            )
        games.extend(pair)
        wins = tuple(game["winner"] for game in pair)
        if None in wins:
            pairs["unfinished"] += 1
        elif wins[0] == wins[1]:
            pairs[wins[0]] += 1
        else:
            pairs["split"] += 1

    return {
        "overall": _tally(games),
        "by_breaking_model": {
            label: _tally([game for game in games if game["breaking_model"] == label])
            for label in agents
        },
        "paired_results": {
            "final_won_both": pairs["final"],
            "base_won_both": pairs["base"],
            "split": pairs["split"],
            "unfinished": pairs["unfinished"],
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument(
        "--candidate-micro-checkpoint", type=Path,
        help="Compare this Micro with the starting Micro, using the final Macro Policy and State V for both.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Keep saved matches in this output directory and play the missing ones.")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=710000)
    parser.add_argument("--rack-layout-split", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
