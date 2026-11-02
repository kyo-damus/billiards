"""Convert executed rollout MCTS root visits to Candidate Policy targets.

Unlike complete-game return labels, visit distributions exist in
truncated rollouts too. They are a POLICY target (not Q values, and not
evidence of search optimality). Group split should use episode_id.
"""

import argparse
from pathlib import Path

import numpy as np

from src.macro.encoding import MACRO_STATE_DIM
from src.macro.candidate_encoding import MACRO_CANDIDATE_V2_DIM


REQUIRED = (
    "format_version", "candidate_encoding_version", "data_kind",
    "state", "root_turn_index", "root_candidate",
    "root_visit_count", "root_selected",
)


def build_visit_dataset(input_paths, output):
    if not input_paths:
        raise ValueError("At least one rollout file is required.")

    states = []
    candidates = []
    probabilities = []
    state_ids = []
    episode_ids = []
    total_roots = 0
    omitted_roots = 0

    for episode_id, path in enumerate(input_paths):
        with np.load(path, allow_pickle=False) as f:
            missing = [key for key in REQUIRED if key not in f]
            if missing:
                raise ValueError(
                    f"Rollout {path}: missing: {', '.join(missing)}"
                )
            if (
                int(f["format_version"]) != 1
                or int(f["candidate_encoding_version"]) != 2
                or str(f["data_kind"]) != "executed_mixed_rollout"
            ):
                raise ValueError(
                    f"Rollout {path}: expected mixed rollout format v1."
                )

            state = np.asarray(f["state"], dtype=np.float32)
            indexes = np.asarray(f["root_turn_index"], dtype=np.int64)
            cand = np.asarray(f["root_candidate"], dtype=np.float32)
            visits = np.asarray(f["root_visit_count"], dtype=np.int64)
            chosen = np.asarray(f["root_selected"], dtype=np.bool_)

            if state.ndim != 2 or state.shape[1] != MACRO_STATE_DIM:
                raise ValueError(f"Rollout {path}: invalid state shape.")
            count = len(indexes)
            if (
                cand.shape != (count, MACRO_CANDIDATE_V2_DIM)
                or visits.shape != (count,)
                or chosen.shape != (count,)
            ):
                raise ValueError(f"Rollout {path}: invalid MCTS root shapes.")
            if not np.all(np.isfinite(state)) or not np.all(
                np.isfinite(cand)
            ):
                raise ValueError(f"Rollout {path}: nonfinite features.")
            if (
                np.any(indexes < 0)
                or np.any(indexes >= len(state))
                or np.any(visits < 0)
                or (count > 1 and np.any(np.diff(indexes) < 0))
            ):
                raise ValueError(f"Rollout {path}: invalid root index/visits.")

            for turn in np.unique(indexes):
                selected = np.flatnonzero(indexes == turn)
                total_roots += 1
                if chosen[selected].sum() != 1:
                    raise ValueError(
                        f"Rollout {path} turn {turn}: expected one selected root."
                    )
                n = len(selected)
                visit_sum = int(visits[selected].sum())
                # A single action offers no ranking signal.
                if n < 2 or visit_sum <= 0:
                    omitted_roots += 1
                    continue

                target = (
                    visits[selected].astype(np.float32)
                    / float(visit_sum)
                )
                states.append(np.repeat(state[turn:turn + 1], n, axis=0))
                candidates.append(cand[selected].copy())
                probabilities.append(target)
                state_ids.append(np.full(n, len(state_ids), dtype=np.int32))
                episode_ids.append(np.full(n, episode_id, dtype=np.int32))

    if not states:
        raise ValueError(
            "No MCTS roots with at least two candidates and positive visits."
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state=np.concatenate(states).astype(np.float32),
        candidate=np.concatenate(candidates).astype(np.float32),
        policy_target=np.concatenate(probabilities).astype(np.float32),
        state_id=np.concatenate(state_ids),
        episode_id=np.concatenate(episode_ids),
        candidate_encoding_version=np.asarray(2, dtype=np.int32),
        turn_aware=np.asarray(True),
        target_kind=np.asarray("mcts_root_visit_distribution"),
        source_files=np.asarray([str(p) for p in input_paths]),
    )
    return output, len(states), int(sum(len(x) for x in candidates)), omitted_roots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout", nargs="+", type=Path, required=True,
        help="Rollout experience files; incomplete games are allowed.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("macro_candidate_policy_visits_v2.npz"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output, roots, candidates, skipped = build_visit_dataset(
        args.rollout, args.output
    )
    print("=== Mixed MCTS root visits -> Candidate Policy dataset ===")
    print(f"root states: {roots}; candidate rows: {candidates}")
    print(f"single-candidate or zero-visit roots skipped: {skipped}")
    print("Target: root visit distribution (NOT candidate Q).")
    print(f"output: {output}")
