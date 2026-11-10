"""Summarize saved mixed rollout NPZ files without running simulation.

This is an observational diagnostic, not a win-rate benchmark.
"""

import argparse
import json
from pathlib import Path

from src.pipelines.rollout_analysis import summarize_rollouts


def resolve_paths(*, rollouts=(), directories=(), recursive=False):
    paths = [Path(p) for p in rollouts]
    pattern = "**/rollout_seed_*.npz" if recursive else "rollout_seed_*.npz"
    for directory in directories:
        folder = Path(directory)
        if not folder.is_dir():
            raise ValueError(f"Not a rollout directory: {folder}")
        paths.extend(sorted(folder.glob(pattern)))
    if not paths:
        raise ValueError("No mixed rollout files found; use --rollout or --rollout-dir.")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Duplicate rollout paths are not allowed.")
    return paths


def format_summary(summary):
    mean = summary["mean_local_reward_per_turn"]
    reward_mean = "N/A (no turns)" if mean is None else f"{mean:+.4f}"
    lines = [
        "=== Mixed rollout observed statistics (NOT win rate) ===",
        f"games: {summary['games']}  complete: {summary['completed_games']} "
        f"({summary['completion_rate']:.1%})  incomplete: {summary['unfinished_games']}",
        f"executed turns: {summary['total_turns']}  "
        f"mean turns/game: {summary['mean_turns_per_game']:.2f}",
        "chosen goals: " + ", ".join(
            f"{name}={count}" for name, count in summary["goal_counts"].items()
        ),
        f"newly pocketed object balls: {summary['pocketed_object_balls']} "
        f"(turns with pocket: {summary['turns_with_pocket']})",
        f"cue scratch turns: {summary['scratch_turns']}  "
        f"turn switches: {summary['turn_switches']}",
        f"Macro LOCAL reward: sum={summary['macro_local_reward_sum']:+.4f}, "
        f"mean/turn={reward_mean}",
        "Game SCORE delta [player 0, player 1]: "
        + str(summary["score_delta_by_player"]),
        f"MCTS roots recorded: {summary['mcts_roots']} "
        f"(usable Policy-visit roots: {summary['policy_usable_roots']})",
        "stop reasons: " + (", ".join(
            f"{reason}={count}" for reason, count in summary["stop_reasons"].items()
        )),
        "Note: simplified two-object-ball rules; physical pocket events do not imply "
        "target-pot success or official legality.",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, nargs="+", default=[])
    parser.add_argument("--rollout-dir", type=Path, action="append", default=[])
    parser.add_argument("--recursive", action="store_true",
                        help="Find rollout_seed_*.npz under subdirectories.")
    parser.add_argument("--json-output", type=Path,
                        help="Optional new JSON summary path (refuse overwrite).")
    return parser.parse_args(argv)


def main(args):
    paths = resolve_paths(rollouts=args.rollout, directories=args.rollout_dir,
                          recursive=args.recursive)
    if args.json_output is not None and args.json_output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.json_output}")
    summary = summarize_rollouts(paths)
    print(format_summary(summary))
    if args.json_output is not None:
        output = args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"JSON summary: {output}")
    return summary


if __name__ == "__main__":
    main(parse_args())
