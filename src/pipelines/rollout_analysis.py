"""Read-only diagnostics for saved mixed-strategy rollout experience.

Counts observed events, not win rate, legality or learned policy quality.
The existing 31-D candidate/12-D state/return target contracts are unchanged.
"""

from collections import Counter
from pathlib import Path

import numpy as np

from src.macro.candidate_encoding import MACRO_CANDIDATE_V2_DIM
from src.macro.encoding import MACRO_STATE_DIM


REQUIRED_FIELDS = (
    "format_version", "candidate_encoding_version", "data_kind",
    "stop_reason", "rollout_terminated", "state", "candidate",
    "state_pocket_indices", "next_state_pocket_indices",
    "score_delta", "macro_local_reward", "player_before", "player_after",
    "root_turn_index", "root_candidate", "root_visit_count", "root_selected",
)
GOAL_NAMES = ("DIRECT_ATTACK", "POSITION_ATTACK", "SAFETY")


def _one_hot_index(vector, *, path, field):
    index = int(np.argmax(vector))
    expected = np.zeros(len(vector), dtype=np.float32)
    expected[index] = 1.0
    if not np.allclose(vector, expected, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: invalid one-hot {field}.")
    return index


def _goal_name(row, *, path, field):
    """Decode one 31-D Macro candidate without inferring physical success."""
    strategy = _one_hot_index(row[:2], path=path, field=f"strategy[{field}]")
    goal = _one_hot_index(row[2:5], path=path, field=f"goal[{field}]")
    if (strategy == 1) != (goal == 2):
        raise ValueError(f"{path}: inconsistent strategy and goal[{field}].")
    return GOAL_NAMES[goal]


def read_rollout(path):
    """Return counters for one versioned NPZ; never modify it."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED_FIELDS) - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
        if (
            int(data["format_version"]) != 1
            or int(data["candidate_encoding_version"]) != 2
            or str(data["data_kind"]) != "executed_mixed_rollout"
        ):
            raise ValueError(f"{path}: expected executed mixed rollout v1 / 31-D.")

        state = np.asarray(data["state"])
        choice = np.asarray(data["candidate"])
        before = np.asarray(data["state_pocket_indices"])
        after = np.asarray(data["next_state_pocket_indices"])
        delta = np.asarray(data["score_delta"])
        reward = np.asarray(data["macro_local_reward"])
        player_before = np.asarray(data["player_before"])
        player_after = np.asarray(data["player_after"])
        root_turn = np.asarray(data["root_turn_index"])
        root_candidate = np.asarray(data["root_candidate"])
        root_visits = np.asarray(data["root_visit_count"])
        root_selected = np.asarray(data["root_selected"])
        n = len(state)
        roots = len(root_turn)
        if (
            state.shape != (n, MACRO_STATE_DIM)
            or choice.shape != (n, MACRO_CANDIDATE_V2_DIM)
            or before.shape != (n, 3) or after.shape != (n, 3)
            or delta.shape != (n, 2) or reward.shape != (n,)
            or player_before.shape != (n,) or player_after.shape != (n,)
            or root_turn.shape != (roots,)
            or root_candidate.shape != (roots, MACRO_CANDIDATE_V2_DIM)
            or root_visits.shape != (roots,) or root_selected.shape != (roots,)
        ):
            raise ValueError(f"{path}: invalid rollout array shapes.")
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(choice))
            or not np.all(np.isfinite(delta))
            or not np.all(np.isfinite(reward))
            or np.any((player_before < 0) | (player_before > 1))
            or np.any((player_after < 0) | (player_after > 1))
            or np.any(root_turn < 0) or np.any(root_turn >= n)
            or np.any(root_visits < 0)
        ):
            raise ValueError(f"{path}: invalid/nonfinite rollout values.")
        if n and (np.any(before < -1) or np.any(after < -1)):
            raise ValueError(f"{path}: invalid pocket indices.")

        goals = Counter({name: 0 for name in GOAL_NAMES})
        selected_goals = []
        for i, row in enumerate(choice):
            name = _goal_name(row, path=path, field=f"selected-{i}")
            selected_goals.append(name)
            goals[name] += 1

        # An MCTS root contains the candidates that were actually offered
        # to the search. Their presence must not be conflated with selection.
        root_candidates_by_goal = Counter({name: 0 for name in GOAL_NAMES})
        root_available_by_goal = Counter({name: 0 for name in GOAL_NAMES})
        root_visits_by_goal = Counter({name: 0 for name in GOAL_NAMES})
        non_direct_available_roots = 0
        direct_chosen_with_non_direct_available = 0
        non_direct_chosen_roots = 0
        usable_roots = 0
        for index in np.unique(root_turn):
            rows = root_turn == index
            if int(np.count_nonzero(root_selected[rows])) != 1:
                raise ValueError(f"{path}: MCTS root {index} needs one selected candidate.")
            offered = set()
            for j, (candidate, visits) in enumerate(
                zip(root_candidate[rows], root_visits[rows])
            ):
                name = _goal_name(
                    candidate, path=path, field=f"root-{index}-{j}"
                )
                root_candidates_by_goal[name] += 1
                root_visits_by_goal[name] += int(visits)
                offered.add(name)
            for name in offered:
                root_available_by_goal[name] += 1
            has_non_direct = bool(offered - {"DIRECT_ATTACK"})
            non_direct_available_roots += int(has_non_direct)
            chosen_name = selected_goals[int(index)]
            direct_chosen_with_non_direct_available += int(
                has_non_direct and chosen_name == "DIRECT_ATTACK"
            )
            non_direct_chosen_roots += int(chosen_name != "DIRECT_ATTACK")
            if np.count_nonzero(rows) >= 2 and np.sum(root_visits[rows]) > 0:
                usable_roots += 1

        stop = str(data["stop_reason"])
        terminated = bool(data["rollout_terminated"])
        return {
            "stop_reason": stop,
            "completed": terminated and stop == "game_terminated",
            "turns": n,
            "goals": dict(goals),
            "pocketed_object_balls": int(np.count_nonzero((before[:, 1:] < 0) & (after[:, 1:] >= 0))),
            "turns_with_pocket": int(np.count_nonzero(np.any((before[:, 1:] < 0) & (after[:, 1:] >= 0), axis=1))),
            "scratch_turns": int(np.count_nonzero((before[:, 0] < 0) & (after[:, 0] >= 0))),
            "turn_switches": int(np.count_nonzero(player_before != player_after)),
            "local_reward_sum": float(np.sum(reward, dtype=np.float64)),
            "score_delta": np.sum(delta, axis=0, dtype=np.float64).tolist(),
            "mcts_roots": int(len(np.unique(root_turn))),
            "policy_usable_roots": usable_roots,
            "root_candidate_counts": dict(root_candidates_by_goal),
            "root_goal_available_turns": dict(root_available_by_goal),
            "root_goal_visit_totals": dict(root_visits_by_goal),
            "roots_with_non_direct_available": non_direct_available_roots,
            "roots_direct_chosen_with_non_direct_available": (
                direct_chosen_with_non_direct_available
            ),
            "roots_non_direct_chosen": non_direct_chosen_roots,
        }


def read_root_diagnostics(path):
    """Inspect the actual per-candidate MCTS decision, without simulating.

    The selection algorithm chooses the *first* root candidate tied for
    the maximum individual visit count, not the most-visited goal category.
    A recorded discrepancy is flagged rather than silently repaired.
    """
    path = Path(path)
    read_rollout(path)  # Verify the version, shapes, selection and encoding.
    with np.load(path, allow_pickle=False) as data:
        turns = np.asarray(data["root_turn_index"])
        candidates = np.asarray(data["root_candidate"])
        visits = np.asarray(data["root_visit_count"])
        selected = np.asarray(data["root_selected"])
        priors = (
            np.asarray(data["root_prior"])
            if "root_prior" in data else None
        )
        predictions = (
            np.asarray(data["root_predicted_q"])
            if "root_predicted_q" in data else None
        )

        if priors is not None and priors.shape != visits.shape:
            raise ValueError(f"{path}: invalid root_prior shape.")
        if predictions is not None and predictions.shape != visits.shape:
            raise ValueError(f"{path}: invalid root_predicted_q shape.")

        results = []
        for turn in np.unique(turns):
            indices = np.flatnonzero(turns == turn)
            local_visits = visits[indices]
            selected_indices = np.flatnonzero(selected[indices])
            if len(selected_indices) != 1:
                raise ValueError(f"{path}: invalid root selection for turn {turn}.")
            chosen_index = int(selected_indices[0])
            first_max_index = int(np.argmax(local_visits))
            max_visits = int(local_visits[first_max_index])
            tied = int(np.count_nonzero(local_visits == max_visits))
            selected_row = int(indices[chosen_index])

            counts = Counter({name: 0 for name in GOAL_NAMES})
            visit_totals = Counter({name: 0 for name in GOAL_NAMES})
            for candidate, count in zip(candidates[indices], local_visits):
                name = _goal_name(candidate, path=path, field=f"root-{turn}")
                counts[name] += 1
                visit_totals[name] += int(count)

            chosen_name = _goal_name(
                candidates[selected_row], path=path, field=f"chosen-root-{turn}"
            )
            predicted = (
                None if predictions is None
                else float(predictions[selected_row])
            )
            if predicted is not None and not np.isfinite(predicted):
                predicted = None
            results.append({
                "path": str(path),
                "turn_index": int(turn),
                "selected_goal": chosen_name,
                "selected_visits": int(local_visits[chosen_index]),
                "max_candidate_visits": max_visits,
                "max_visit_tie_count": tied,
                "selected_is_first_visit_max": bool(
                    chosen_index == first_max_index
                ),
                "candidate_count": len(indices),
                "candidate_counts_by_goal": dict(counts),
                "visits_by_goal": dict(visit_totals),
                "selected_prior": (
                    None if priors is None else float(priors[selected_row])
                ),
                "selected_predicted_q": predicted,
            })
        return results


def summarize_rollouts(paths):
    """Aggregate per-file observed statistics; rates are descriptive only."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("At least one mixed rollout NPZ is required.")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Duplicate rollout paths are not allowed.")

    totals = {
        "games": len(paths), "completed_games": 0,
        "total_turns": 0, "pocketed_object_balls": 0,
        "turns_with_pocket": 0, "scratch_turns": 0,
        "turn_switches": 0, "macro_local_reward_sum": 0.0,
        "score_delta_by_player": [0.0, 0.0],
        "mcts_roots": 0, "policy_usable_roots": 0,
        "roots_with_non_direct_available": 0,
        "roots_direct_chosen_with_non_direct_available": 0,
        "roots_non_direct_chosen": 0,
    }
    goals = Counter({name: 0 for name in GOAL_NAMES})
    root_candidates_by_goal = Counter({name: 0 for name in GOAL_NAMES})
    root_available_by_goal = Counter({name: 0 for name in GOAL_NAMES})
    root_visits_by_goal = Counter({name: 0 for name in GOAL_NAMES})
    stops = Counter()
    for path in paths:
        game = read_rollout(path)
        stops[game["stop_reason"]] += 1
        totals["completed_games"] += int(game["completed"])
        totals["total_turns"] += game["turns"]
        goals.update(game["goals"])
        root_candidates_by_goal.update(game["root_candidate_counts"])
        root_available_by_goal.update(game["root_goal_available_turns"])
        root_visits_by_goal.update(game["root_goal_visit_totals"])
        for target, source in (
            ("pocketed_object_balls", "pocketed_object_balls"),
            ("turns_with_pocket", "turns_with_pocket"),
            ("scratch_turns", "scratch_turns"),
            ("turn_switches", "turn_switches"),
            ("mcts_roots", "mcts_roots"),
            ("policy_usable_roots", "policy_usable_roots"),
            ("roots_with_non_direct_available", "roots_with_non_direct_available"),
            (
                "roots_direct_chosen_with_non_direct_available",
                "roots_direct_chosen_with_non_direct_available",
            ),
            ("roots_non_direct_chosen", "roots_non_direct_chosen"),
            ("macro_local_reward_sum", "local_reward_sum"),
        ):
            totals[target] += game[source]
        for i in range(2):
            totals["score_delta_by_player"][i] += game["score_delta"][i]

    games, turns = totals["games"], totals["total_turns"]
    totals["unfinished_games"] = games - totals["completed_games"]
    totals["completion_rate"] = totals["completed_games"] / games
    totals["mean_turns_per_game"] = turns / games
    totals["mean_local_reward_per_turn"] = (
        totals["macro_local_reward_sum"] / turns if turns else None
    )
    totals["goal_counts"] = dict(goals)
    totals["root_candidate_counts"] = dict(root_candidates_by_goal)
    totals["root_goal_available_turns"] = dict(root_available_by_goal)
    totals["root_goal_visit_totals"] = dict(root_visits_by_goal)
    totals["stop_reasons"] = dict(sorted(stops.items()))
    return totals
