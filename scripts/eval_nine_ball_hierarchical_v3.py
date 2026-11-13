"""Evaluate a completed 9-ball v3 run against its starting models on paired racks."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.iterate_nine_ball_hierarchical_v3 import _git_commit

MODEL_KEYS = ("micro", "policy", "state_v")


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _model_inputs(training):
    base = training["source_checkpoints"]
    final = training["rounds"][-1]
    models = {}
    for label, paths in (
        ("base", {key: base[key]["path"] for key in MODEL_KEYS}),
        ("final", {key: final[key] for key in MODEL_KEYS}),
    ):
        models[label] = {}
        for key, path in paths.items():
            with Path(path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if label == "base" and digest != base[key]["sha256"]:
                raise ValueError(f"Starting checkpoint differs from the training manifest: {path}")
            models[label][key] = {"path": path, "sha256": digest}
    return models


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


def evaluate(args, *, agent_factory=None, simulator_factory=None):
    if args.pairs < 1 or args.start_seed < 0:
        raise ValueError("Use a positive number of pairs and a nonnegative start seed")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse evaluation output: {args.output_dir}")

    training = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if (
        training.get("kind") != "nine_ball_macro_micro_alternating_v3"
        or training.get("status") != "completed"
        or training.get("cross_revision_resume")
        or not training.get("rounds")
    ):
        raise ValueError("Expected a completed nine-ball v3 training run from one code revision")
    seeds = range(args.start_seed, args.start_seed + args.pairs)
    for round_info in training["rounds"]:
        start = round_info["start_seed"]
        training_seeds = range(start, start + 2 * round_info["matches"], 2)
        if any(seed in training_seeds for seed in seeds):
            raise ValueError("Evaluation seeds overlap with training matches")

    settings = training["settings"]
    search = {
        "simulations": settings["mcts_simulations"],
        "max_depth": settings["mcts_max_depth"],
        "c_puct": settings["mcts_c_puct"],
        "gamma": settings["gamma"],
        "root_min_visits": settings["mcts_root_min_visits"],
        "learned_safety": settings.get("learned_safety", False),
        "outcome_position": settings.get("outcome_position", False),
    }
    checkpoints = _model_inputs(training)
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Use exactly one existing CUDA_VISIBLE_DEVICES GPU ID")

    from scripts.collect_nine_ball_self_play import play_match
    from src.game.match import MatchConfig
    from src.game.nine_ball_session import NineBallMatchSession

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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "evaluation_config.json", {
        "evaluation_git_commit": _git_commit(),
        "training_manifest": str(args.training_manifest),
        "training_git_commit": training.get("source_git_commit"),
        "checkpoints": checkpoints,
        "race_to": 1,
        "pairs": args.pairs,
        "start_seed": args.start_seed,
        "max_shots_per_rack": settings["max_shots_per_rack"],
        "search": search,
        "device": args.device,
    })

    games = []
    pairs = Counter()
    for index, seed in enumerate(seeds):
        # Hold player slots fixed within the pair and balance them across seeds.
        final_player = index % 2
        players = ("base", "final") if final_player else ("final", "base")
        pair = []
        for breaker in (final_player, 1 - final_player):
            match = NineBallMatchSession(
                simulator_factory=simulator_factory,
                match_config=MatchConfig(race_to=1, rule_id="nine_ball", first_breaker=breaker),
                seed=seed,
            )
            outcome = play_match(
                match, tuple(agents[label] for label in players),
                max_shots_per_rack=settings["max_shots_per_rack"],
            )
            winner = outcome["match_winner"]
            game = {
                "seed": seed,
                "player_models": list(players),
                "first_breaker": breaker,
                "breaking_model": players[breaker],
                "winner": players[winner] if outcome["match_finished"] else None,
                "stop_reason": outcome["stop_reason"],
                "shots": outcome["shots"],
                "fouls": outcome["fouls"],
            }
            _write_json(
                args.output_dir / f"seed_{seed}_break_{players[breaker]}.json",
                {**outcome, **game},
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

    summary = {
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
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=710000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
