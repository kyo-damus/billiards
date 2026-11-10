"""Read-only paired comparison of saved mixed MCTS rollout logs.

Compare the SAME seeded initial boards under two tie-break modes.
Scores and local rewards use simplified rules; this is not win rate.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.pipelines.rollout_analysis import GOAL_NAMES, read_rollout


ROLLOUT_NAME = re.compile(r"rollout_seed_([0-9]+)\.npz\Z")


def discover_seeded_rollouts(directory):
    """Require unique, nonempty, accurately named seeded experience logs."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a rollout directory: {directory}")
    files = sorted(directory.glob("rollout_seed_*.npz"))
    if not files:
        raise ValueError(f"No rollout_seed_*.npz in {directory}")

    by_seed = {}
    for path in files:
        match = ROLLOUT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected rollout filename: {path}")
        seed = int(match.group(1))
        if seed in by_seed:
            raise ValueError(f"Duplicate seed {seed} in {directory}")
        by_seed[seed] = path
    return by_seed


def _episode(path, seed):
    """Use the strict rollout decoder, then extract matched-run outcomes."""
    info = read_rollout(path)
    with np.load(path, allow_pickle=False) as f:
        if "seed" not in f or int(f["seed"]) != seed:
            raise ValueError(f"{path}: seed metadata does not match filename.")
        states = np.asarray(f["state"])
        candidates = np.asarray(f["candidate"])
        rewards = np.asarray(f["macro_local_reward"])
        score_delta = np.asarray(f["score_delta"])
        first_action = (
            candidates[0].copy() if len(candidates) else None
        )
        first_goal = (
            GOAL_NAMES[int(np.argmax(first_action[2:5]))]
            if first_action is not None else None
        )
        return {
            "initial_state": states[0].copy() if len(states) else None,
            "first_action": first_action,
            "first_goal": first_goal,
            "first_reward": float(rewards[0]) if len(rewards) else None,
            "first_score_p0": float(score_delta[0, 0]) if len(score_delta) else None,
            "first_score_p1": float(score_delta[0, 1]) if len(score_delta) else None,
            "game_score_p0": float(info["score_delta"][0]),
            "game_score_p1": float(info["score_delta"][1]),
            "completed": bool(info["completed"]),
            "turns": int(info["turns"]),
            "scratch_turns": int(info["scratch_turns"]),
            "local_reward_sum": float(info["local_reward_sum"]),
        }


def _mean(values):
    return sum(values) / len(values) if values else None


def compare_paired_rollouts(first_dir, empirical_dir):
    """Compare two directories only if their seed sets agree exactly."""
    baseline = discover_seeded_rollouts(first_dir)
    alternative = discover_seeded_rollouts(empirical_dir)
    if baseline.keys() != alternative.keys():
        missing_new = sorted(baseline.keys() - alternative.keys())
        missing_first = sorted(alternative.keys() - baseline.keys())
        raise ValueError(
            "Unpaired seeds: missing empirical_q="
            f"{missing_new}; missing first={missing_first}"
        )
    rows = []
    differences = []
    for seed in sorted(baseline):
        a = _episode(baseline[seed], seed)
        b = _episode(alternative[seed], seed)
        if a["initial_state"] is not None and b["initial_state"] is not None:
            if not np.allclose(
                a["initial_state"], b["initial_state"], atol=1e-6, rtol=0
            ):
                raise ValueError(f"seed={seed}: initial GameState differs.")
        changed = None
        if a["first_action"] is not None and b["first_action"] is not None:
            changed = not np.array_equal(a["first_action"], b["first_action"])
            differences.append(b["first_reward"] - a["first_reward"])
        rows.append({
            "seed": seed,
            "first_goal_first": a["first_goal"],
            "first_goal_empirical_q": b["first_goal"],
            "first_candidate_changed": changed,
            "first_local_reward_first": a["first_reward"],
            "first_local_reward_empirical_q": b["first_reward"],
            "first_score_p0_first": a["first_score_p0"],
            "first_score_p0_empirical_q": b["first_score_p0"],
            "turns_first": a["turns"],
            "turns_empirical_q": b["turns"],
            "completed_first": a["completed"],
            "completed_empirical_q": b["completed"],
            "game_score_p0_first": a["game_score_p0"],
            "game_score_p0_empirical_q": b["game_score_p0"],
            "game_score_p1_first": a["game_score_p1"],
            "game_score_p1_empirical_q": b["game_score_p1"],
            "scratch_turns_first": a["scratch_turns"],
            "scratch_turns_empirical_q": b["scratch_turns"],
            "game_local_reward_first": a["local_reward_sum"],
            "game_local_reward_empirical_q": b["local_reward_sum"],
        })
    summary = {
        "paired_seeds": len(rows),
        "pairs_with_first_actions": len(differences),
        "first_action_changed": sum(
            row["first_candidate_changed"] is True for row in rows
        ),
        "mean_first_reward_first": _mean([
            row["first_local_reward_first"] for row in rows
            if row["first_candidate_changed"] is not None
        ]),
        "mean_first_reward_empirical_q": _mean([
            row["first_local_reward_empirical_q"] for row in rows
            if row["first_candidate_changed"] is not None
        ]),
        "mean_paired_first_reward_difference": _mean(differences),
        "completed_games_first": sum(row["completed_first"] for row in rows),
        "completed_games_empirical_q": sum(
            row["completed_empirical_q"] for row in rows
        ),
        "mean_game_score_p0_first": _mean([
            row["game_score_p0_first"] for row in rows
        ]),
        "mean_game_score_p0_empirical_q": _mean([
            row["game_score_p0_empirical_q"] for row in rows
        ]),
        "mean_game_score_p1_first": _mean([
            row["game_score_p1_first"] for row in rows
        ]),
        "mean_game_score_p1_empirical_q": _mean([
            row["game_score_p1_empirical_q"] for row in rows
        ]),
        "total_scratches_first": sum(row["scratch_turns_first"] for row in rows),
        "total_scratches_empirical_q": sum(
            row["scratch_turns_empirical_q"] for row in rows
        ),
    }
    return {"summary": summary, "per_seed": rows}


def format_comparison(report):
    s = report["summary"]
    lines = [
        "=== Paired seeded MCTS rollout comparison (descriptive, NOT win rate) ===",
        f"paired seeds: {s['paired_seeds']} "
        f"first-turn observations: {s['pairs_with_first_actions']} "
        f"first candidate changed: {s['first_action_changed']}",
        f"first-turn Macro LOCAL reward mean: first={s['mean_first_reward_first']} "
        f"empirical_q={s['mean_first_reward_empirical_q']} "
        f"mean paired difference (new-old)={s['mean_paired_first_reward_difference']}",
        f"complete games: first={s['completed_games_first']} "
        f"empirical_q={s['completed_games_empirical_q']}",
        f"mean cumulative SCORE delta for player 0: "
        f"first={s['mean_game_score_p0_first']:.4f} "
        f"empirical_q={s['mean_game_score_p0_empirical_q']:.4f}",
        f"mean cumulative SCORE delta for player 1: "
        f"first={s['mean_game_score_p1_first']:.4f} "
        f"empirical_q={s['mean_game_score_p1_empirical_q']:.4f}",
        f"total cue scratches: first={s['total_scratches_first']} "
        f"empirical_q={s['total_scratches_empirical_q']}",
        "Per-seed first action and result:",
    ]
    for x in report["per_seed"]:
        lines.append(
            f" seed={x['seed']}: "
            f"{x['first_goal_first']} -> {x['first_goal_empirical_q']} "
            f"changed={x['first_candidate_changed']} "
            f"first-local=({x['first_local_reward_first']}, "
            f"{x['first_local_reward_empirical_q']}) "
            f"complete=({x['completed_first']}, {x['completed_empirical_q']}) "
            f"game-score-p0=({x['game_score_p0_first']}, "
            f"{x['game_score_p0_empirical_q']})"
        )
    lines.append(
        "Note: differing trajectories have differing lengths and local shaping; "
        "completion here includes scratches under simplified rules. "
        "Compare held-out starts, not official wins."
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--empirical-dir", type=Path, required=True)
    parser.add_argument(
        "--json-output", type=Path, help="Optional unique JSON output (no overwrite)."
    )
    return parser.parse_args(argv)


def main(args):
    output = args.json_output
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    result = compare_paired_rollouts(args.first_dir, args.empirical_dir)
    print(format_comparison(result))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"Saved paired comparison: {output}")
    return result


if __name__ == "__main__":
    main(parse_args())
