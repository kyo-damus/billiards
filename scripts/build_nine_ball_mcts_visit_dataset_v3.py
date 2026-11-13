"""Convert actual nine-ball v3 MCTS root visits into Macro Policy targets.

Independent of match termination: incomplete games can contribute valid
*root search* visit teachers, unlike terminal-return Q/V datasets.
Do not assume visits are optimal actions or match-level performance labels.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_nine_ball_macro_returns_v3 import logged_match_positions, read_state_and_goal
from src.game.nine_ball_encoding_v3 import MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM

TARGET_KIND = "nine_ball_mcts_root_visit_distribution_v3"


def read_root(row, match_position=None):
    """Read and validate search evidence with exact selected-action identity."""
    action = row.get("chosen_action", {})
    root = action.get("macro_search")
    if not isinstance(root, dict):
        raise ValueError("Expected chosen_action.macro_search from v3 MCTS")
    goal_array = np.asarray(root["goal_v3"], dtype=np.float32)
    visits_raw = root["root_visits"]
    priors = np.asarray(root["root_priors"], dtype=np.float64)
    chosen = int(root["chosen_index"])
    if not isinstance(visits_raw, list) or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 0
        for n in visits_raw
    ):
        raise ValueError("Root visits must be nonnegative integers")
    visits = np.asarray(visits_raw, dtype=np.int64)
    count = len(visits)
    if (
        count < 1
        or goal_array.shape != (count, MACRO_GOAL_V3_DIM)
        or priors.shape != (count,)
        or not np.all(np.isfinite(goal_array))
        or not np.all(np.isfinite(priors))
        or np.any(priors < 0)
        or not np.isclose(priors.sum(), 1.0, rtol=1e-4, atol=1e-5)
        or chosen < 0 or chosen >= count
        or int(root["root_candidates"]) != count
        or int(root["simulations"]) < 1
        or int(visits.sum()) != int(root["simulations"])
        or not np.allclose(goal_array[:, :4].sum(axis=1), 1.0, atol=1e-5)
        or np.any(goal_array[:, :4] < -1e-5)
        or not np.allclose(goal_array[:, 4:20].sum(axis=1), 1.0, atol=1e-5)
        or np.any(goal_array[:, 4:20] < -1e-5)
    ):
        raise ValueError("Inconsistent v3 root candidate/visit/prior metadata")
    state, executed = read_state_and_goal(row, match_position)
    if not np.allclose(executed, goal_array[chosen], rtol=0, atol=1e-5):
        raise ValueError("Selected shot's exact Goal differs from MCTS root choice")
    reasons = root.get("root_invalid_reasons", [None] * count)
    if not isinstance(reasons, list) or len(reasons) != count or any(
        reason is not None and (not isinstance(reason, str) or not reason)
        for reason in reasons
    ):
        raise ValueError("Invalid MCTS root invalidation metadata")
    if reasons[chosen] is not None:
        raise ValueError("Chosen MCTS root candidate was invalidated")
    valid = np.asarray([reason is None for reason in reasons], dtype=bool)
    # Older logs can retain visits on an edge invalidated later in the search.
    # Train only on surviving candidates and remap the executed candidate index.
    chosen = int(valid[:chosen].sum())
    goal_array, visits = goal_array[valid], visits[valid]
    if visits.sum() == 0 or visits[chosen] != visits.max():
        raise ValueError("Chosen root candidate is not a positive maximum-visit action")
    return state, goal_array, visits, chosen


def build(args):
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    files = sorted(args.rollout_dir.glob("match_seed_*.json"))
    if not files:
        raise ValueError(f"No match_seed_*.json in {args.rollout_dir}")

    states, candidates, target_visits, seeds, chosen_indices, offsets = (
        [], [], [], [], [], [0]
    )
    skipped_single, complete, unfinished, boards_total = 0, 0, 0, 0
    state_dim = MATCH_MACRO_STATE_DIM if args.match_aware else MACRO_STATE_V3_DIM
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        seed = int(file.stem.removeprefix("match_seed_"))
        if int(data["seed"]) != seed:
            raise ValueError(f"Match-seed mismatch in {file}")
        if not args.match_aware and int(data.get("race_to", 0)) != 1:
            raise ValueError("Only race-to-1 v3 MCTS visit logs supported")
        if data.get("match_finished"):
            complete += 1
        else:
            unfinished += 1
        rows = data.get("trajectory", [])
        positions = logged_match_positions(data, rows) if args.match_aware and rows else [None] * len(rows)
        for row, position in zip(rows, positions):
            # Each actual root search is a target; the final match winner is
            # deliberately unused. Verify logs before applying filters.
            state, goals, visits, chosen = read_root(row, position)
            boards_total += 1
            if args.exclude_single_candidate and len(visits) == 1:
                skipped_single += 1
                continue
            states.append(state)
            candidates.extend(goals)
            target_visits.extend(visits)
            seeds.append(seed)
            chosen_indices.append(chosen)
            offsets.append(len(candidates))
    if not states:
        raise ValueError("No eligible MCTS roots available for Policy training")
    if len(set(seeds)) < 2:
        raise ValueError(
            "Need >=2 different source matches with eligible root searches "
            "for match-disjoint validation"
        )
    # Every sample is a root: sum of visit targets is one on that root.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        states=np.asarray(states, dtype=np.float32).reshape(-1, state_dim),
        candidates=np.asarray(candidates, dtype=np.float32).reshape(-1, MACRO_GOAL_V3_DIM),
        visits=np.asarray(target_visits, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        match_seeds=np.asarray(seeds, dtype=np.int64),
        chosen_indices=np.asarray(chosen_indices, dtype=np.int64),
        format_version=np.asarray(3),
        match_aware=np.asarray(args.match_aware),
        target_kind=np.asarray(TARGET_KIND),
    )
    summary = {
        "format_version": 3,
        "match_aware": args.match_aware,
        "target_kind": TARGET_KIND,
        "all_logged_roots": boards_total,
        "policy_training_roots": len(states),
        "total_candidate_rows": len(candidates),
        "skipped_single_candidate": skipped_single,
        "completed_matches": complete,
        "unfinished_matches": unfinished,
        "match_seeds": sorted(set(seeds)),
        "note": (
            "MCTS root visits normalized per board, NOT win-rate/return "
            "labels. Unfinished matches allowed for Policy but not MC Q/V. "
            "Search quality depends on fixed Micro and optional State V."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--match-aware", action="store_true")
    p.add_argument(
        "--include-single-candidate", dest="exclude_single_candidate",
        action="store_false",
    )
    p.set_defaults(exclude_single_candidate=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
