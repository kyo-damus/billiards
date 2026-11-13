"""Build v3 9-ball selected-action Q and state-V match-return training data.

Only complete matches with exact logged v3 Macro goals are supported.
These labels are realized behavior-policy terminal returns,
NOT unexecuted candidate Q targets and NOT a claim of an optimal value function.
No incomplete matches are zero-bootstrapped.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.game.nine_ball_encoding_v3 import (
    MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM, encode_macro_state,
)
from src.game.match import MatchConfig, RaceToMatch
from src.game.nine_ball_rules import NineBallContext
from src.game.rack_state import RackState, TableGameConfig

TARGET_KIND = "selected_action_terminal_match_return"
REWARD_KIND = "terminal_match_win_loss"


def turn_aware_returns(players, winner, gamma):
    """Discount a terminal +/-1 match result from every acting player's view.

    Equivalent to propagating r + gamma * turn_sign * V(next) backwards
    with r=0 until the terminal shot, and turn_sign=-1 on player change.
    """
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0,1]")
    if winner not in (0, 1):
        raise ValueError("winner must be player 0 or 1")
    if not players or any(p not in (0, 1) for p in players):
        raise ValueError("Need nonempty valid acting-player sequence")
    target = np.empty(len(players), dtype=np.float32)
    target[-1] = 1.0 if players[-1] == winner else -1.0
    for i in range(len(players) - 2, -1, -1):
        sign = 1.0 if players[i] == players[i + 1] else -1.0
        target[i] = gamma * sign * target[i + 1]
    return target


def read_state_and_goal(row, match_position=None):
    """Recover state, rule context, and *exact logged chosen* v3 candidate."""
    raw = row["state"]
    config = TableGameConfig.nine_ball_inventory()
    if tuple(raw["ball_numbers"]) != config.ball_numbers:
        raise ValueError("Only complete 1..9 ball inventories are supported")
    if raw.get("ball_in_hand"):
        raise ValueError("Collector must place the cue before logging pre-shot state")
    state = RackState(
        config=config,
        ball_positions=np.asarray(raw["positions"], dtype=np.float32),
        ball_pocket_indices=np.asarray(raw["pocket_indices"], dtype=np.int32),
        current_player=int(raw["current_player"]),
        score=np.zeros(2, dtype=np.float32),
        ball_in_hand=False,
        break_shot=bool(raw["break_shot"]),
        turn_index=int(raw["turn_index"]),
    )
    if int(row["player"]) != state.current_player:
        raise ValueError("Shot actor does not match logged pre-shot state")
    raw_ctx = row.get("rule_context")
    if raw_ctx is None:
        raise ValueError("Exact rule_context is required for Macro v3 return training")
    context = NineBallContext(
        consecutive_fouls=tuple(int(n) for n in raw_ctx["consecutive_fouls"]),
        push_out_available=bool(raw_ctx["push_out_available"]),
    )
    encoded = row["chosen_action"].get("goal_v3")
    if encoded is None:
        raise ValueError(
            "Missing exact chosen_action.goal_v3. Existing early self-play "
            "files cannot be converted safely; recollect with the new logger."
        )
    goal = np.asarray(encoded, dtype=np.float32)
    if goal.shape != (MACRO_GOAL_V3_DIM,) or not np.all(np.isfinite(goal)):
        raise ValueError("Invalid 36D chosen Macro Goal")
    if not np.isclose(float(goal[:4].sum()), 1.0, atol=1e-5):
        raise ValueError("Chosen Macro Goal kind must be one-hot")
    return encode_macro_state(state, context, match_position), goal


def logged_match_positions(match, rows):
    """Replay the rack-winner evidence to verify each pre-shot match context."""
    first = rows[0].get("match_context")
    if first is None:
        raise ValueError("Match-aware training requires logged pre-shot match context")
    tracker = RaceToMatch(MatchConfig(
        race_to=match["race_to"], rule_id="nine_ball",
        first_breaker=first["breaker"], break_policy=match.get("break_policy", "alternate"),
    ))
    positions = []
    for row in rows:
        position = tracker.position
        raw = row.get("match_context")
        if (
            raw is None
            or position.winner is not None
            or raw["race_to"] != position.race_to
            or tuple(raw["rack_wins"]) != position.rack_wins
            or raw["breaker"] != position.breaker
            or raw["break_policy"] != position.break_policy
            or row["rack_index"] != sum(position.rack_wins)
        ):
            raise ValueError("Inconsistent pre-shot match score or rack order")
        positions.append(position)
        winner = row["rule_result"]["rack_winner"]
        if winner is not None:
            tracker.finish_rack(winner)
    snapshot = tracker.snapshot()
    if (
        tuple(match["rack_wins"]) != snapshot.rack_wins
        or match["racks_completed"] != snapshot.racks_completed
        or match["match_finished"] != snapshot.finished
        or match["match_winner"] != snapshot.match_winner
    ):
        raise ValueError("Logged match result does not match the rack winner sequence")
    return positions


def build(args):
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0,1]")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    files = sorted(args.rollout_dir.glob("match_seed_*.json"))
    if not files:
        raise ValueError(f"No match_seed_*.json in {args.rollout_dir}")
    states, goals, returns, groups, step_indices, winner_labels = [], [], [], [], [], []
    completed, unfinished, unsupported = 0, 0, 0
    state_dim = MATCH_MACRO_STATE_DIM if args.match_aware else MACRO_STATE_V3_DIM
    for file in files:
        match = json.loads(file.read_text(encoding="utf-8"))
        if not match.get("match_finished"):
            unfinished += 1
            continue
        if not args.match_aware and (
            int(match.get("race_to", -1)) != 1 or int(match.get("racks_completed", -1)) != 1
        ):
            unsupported += 1
            continue
        winner = match["match_winner"]
        rows = match.get("trajectory", [])
        if not rows or rows[-1]["rule_result"]["rack_winner"] != winner:
            raise ValueError(f"Missing terminal winner evidence in {file}")
        players = [int(row["player"]) for row in rows]
        targets = turn_aware_returns(players, winner, args.gamma)
        seed = int(match["seed"])
        if seed != int(file.stem.removeprefix("match_seed_")):
            raise ValueError(f"Match seed mismatch in {file}")
        positions = logged_match_positions(match, rows) if args.match_aware else [None] * len(rows)
        for i, (row, position) in enumerate(zip(rows, positions)):
            state, goal = read_state_and_goal(row, position)
            states.append(state)
            goals.append(goal)
            returns.append(float(targets[i]))
            groups.append(seed)
            step_indices.append(i)
            winner_labels.append(winner)
        completed += 1

    if completed < 2:
        raise ValueError(
            "Need >=2 complete matches with logged v3 goals "
            "for disjoint train/validation splitting."
        )
    arrays = {
        "states": np.asarray(states, dtype=np.float32).reshape(-1, state_dim),
        "candidates": np.asarray(goals, dtype=np.float32).reshape(-1, MACRO_GOAL_V3_DIM),
        "q_target": np.asarray(returns, dtype=np.float32),
        "match_seeds": np.asarray(groups, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
        "winners": np.asarray(winner_labels, dtype=np.int64),
        "format_version": np.array(3, dtype=np.int64),
        "match_aware": np.array(args.match_aware),
        "turn_aware": np.array(True),
        "target_kind": np.array(TARGET_KIND),
        "reward_kind": np.array(REWARD_KIND),
        "gamma": np.array(args.gamma),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "format_version": 3,
        "match_aware": args.match_aware,
        "target_kind": TARGET_KIND,
        "reward_kind": REWARD_KIND,
        "turn_aware": True,
        "gamma": args.gamma,
        "completed_matches_used": completed,
        "unfinished_matches_skipped": unfinished,
        "non_race_to_1_matches_skipped": unsupported,
        "executed_actions_labeled": len(returns),
        "unique_match_seeds": sorted(set(groups)),
        "interpretation": (
            "Executed-action Monte Carlo terminal match return (+/-1), "
            "not a label for unexecuted candidates. State V is on-policy "
            "for the recorded behavior mixture. Match outcome labels "
            "include the effects of current nine-ball foul rules."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--match-aware", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
