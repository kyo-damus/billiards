"""Audit deterministic 5D Micro on separate training and held-out match boards.

Read-only: no SAC optimization, no changed checkpoints. Replays the *same
recorded goal* through the actor on both dataset groups and reports goal-
specific outcomes, legal shots, fouls and native physics rejections.

This detects possible generalization gaps, not a definitive diagnosis
of overfitting: training/heldout matches may present different goal difficulty
distributions. Pair with checkpoint histories and controlled board splits.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.eval_nine_ball_micro_v3 import record_trial
from scripts.train_nine_ball_micro_sac import TrainingSample
from src.game.nine_ball_encoding_v3 import (
    GOAL_KINDS, MACRO_GOAL_V3_DIM, NineBallGoal, encode_rack_v3, encode_goal_v3,
)
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.rack_state import RackState, TableGameConfig
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER
from src.common.table import TABLE_LENGTH, TABLE_WIDTH
from src.pipelines.experiment_io import write_experiment_json


def goal_from_vector(raw):
    v = np.asarray(raw, dtype=np.float32)
    if v.shape != (MACRO_GOAL_V3_DIM,) or not np.all(np.isfinite(v)):
        raise ValueError("Invalid stored Macro Goal vector")
    kind_index = int(np.argmax(v[:4]))
    if not np.array_equal(v[:4], np.eye(4, dtype=np.float32)[kind_index]):
        raise ValueError("Unrecognized stored Macro Goal kind")
    # Skip rule actions, fixed shots, and anything not controlled by Micro.
    if v[28] < 0 or v[29] != 0:
        return None
    target = int(np.argmax(v[4:20]))
    if target < 1 or v[4 + target] != 1 or v[4:20].sum() != 1:
        raise ValueError("Invalid target-ball encoding")
    pocket = int(np.argmax(v[20:26])) if v[20:26].sum() else None
    if pocket is not None and (v[20 + pocket] != 1 or v[20:26].sum() != 1):
        raise ValueError("Invalid pocket encoding")
    kind = GOAL_KINDS[kind_index]
    stop_xy, stop_radius = None, None
    if kind == "POSITION_ATTACK":
        from math import hypot
        stop_xy = (float(v[26] * TABLE_WIDTH), float(v[27] * TABLE_LENGTH))
        stop_radius = float(v[28] * hypot(TABLE_WIDTH, TABLE_LENGTH))
        if stop_radius <= 0:
            raise ValueError("Invalid position goal")
    goal = NineBallGoal(kind, target, pocket, stop_xy, stop_radius)
    if not np.allclose(encode_goal_v3(goal), v, atol=2e-6, rtol=0):
        raise ValueError("Goal encoding not round-trippable")
    return goal


def sample_from_trajectory(row, cfg, *, goal_vector=None):
    raw_state, raw_ctx = row.get("state"), row.get("rule_context")
    if raw_state is None or raw_ctx is None or raw_ctx.get("pending_choice") is not None:
        return None
    selected = goal_vector if goal_vector is not None else (row.get("chosen_action") or {}).get("goal_v3")
    if selected is None or raw_state.get("ball_in_hand"):
        return None
    goal = goal_from_vector(selected)
    if goal is None:
        return None
    if tuple(raw_state["ball_numbers"]) != cfg.ball_numbers:
        raise ValueError("Expected numbered nine-ball inventory")
    state = RackState(
        config=cfg,
        ball_positions=np.asarray(raw_state["positions"], dtype=np.float32),
        ball_pocket_indices=np.asarray(raw_state["pocket_indices"], dtype=np.int32),
        current_player=int(raw_state["current_player"]),
        score=np.zeros(2, dtype=np.float32),
        ball_in_hand=False,
        break_shot=bool(raw_state["break_shot"]),
        turn_index=int(raw_state["turn_index"]),
    )
    context = NineBallContext(
        consecutive_fouls=tuple(raw_ctx["consecutive_fouls"]),
        push_out_available=bool(raw_ctx["push_out_available"]),
    )
    if not state.on_table_object_balls() or state.break_shot != (goal.kind == "BREAK"):
        return None
    if goal.target_ball != min(state.on_table_object_balls()):
        raise ValueError("Recorded goal does not target the lowest legal ball")
    return TrainingSample(state, context, goal)


def _match_files(folder):
    files = sorted(folder.glob("match_seed_*.json"))
    if not files:
        files = sorted(folder.glob("seed_*_break_*.json"))
    if not files:
        raise ValueError(f"No supported match logs in {folder}")
    return files


def select_cases(folder, *, max_matches, per_goal, seed, goal_source="candidates", excluded_board_hashes=None):
    if max_matches < 1 or per_goal < 1:
        raise ValueError("max_matches and per_goal must be >= 1")
    rng = np.random.default_rng(seed)
    files = _match_files(folder)
    file_selection = sorted(rng.choice(
        len(files), min(max_matches, len(files)), replace=False,
    ))
    samples = defaultdict(list)
    encountered = Counter()
    unique = set()
    boards = set()
    excluded_board_hashes = excluded_board_hashes or set()
    excluded_overlaps = 0
    cfg = TableGameConfig.nine_ball_inventory()
    for index in file_selection:
        path = files[int(index)]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("trajectory", []):
            action = row.get("chosen_action") or {}
            if goal_source == "candidates":
                root = action.get("macro_search") or {}
                vectors = root.get("goal_v3") or []
            elif goal_source == "logged":
                vectors = (action.get("goal_v3"),)
            else:
                raise ValueError("Unknown --goal-source")
            for vector in vectors:
                if vector is None:
                    continue
                case = sample_from_trajectory(row, cfg, goal_vector=vector)
                if case is None:
                    continue
                kind = case.selected_goal.kind
                encountered[kind] += 1
                rack = encode_rack_v3(case.state, case.context)
                board = hashlib.sha256(rack[:64].tobytes()).hexdigest()
                if board in excluded_board_hashes:
                    excluded_overlaps += 1
                    continue
                key = (rack.tobytes(), encode_goal_v3(case.selected_goal).tobytes())
                if key in unique or len(samples[kind]) >= per_goal:
                    continue
                unique.add(key)
                boards.add(board)
                samples[kind].append(case)
    return dict(samples), boards, {
        "available_match_files": len(files),
        "sampled_match_files": len(file_selection),
        "eligible_goals_seen": dict(encountered),
        "unique_selected_goals": {k: len(v) for k, v in samples.items()},
        "sampled_board_hashes": len(boards),
        "excluded_holdout_overlap_cases": excluded_overlaps,
    }


def all_micro_board_hashes(folder, *, goal_source="candidates"):
    """Full holdout board set, matching Phase28a pretraining exclusion."""
    cfg = TableGameConfig.nine_ball_inventory()
    result = set()
    for path in _match_files(folder):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("trajectory", []):
            action = row.get("chosen_action") or {}
            vectors = (
                (action.get("macro_search") or {}).get("goal_v3") or []
                if goal_source == "candidates"
                else (action.get("goal_v3"),)
            )
            for vector in vectors:
                if vector is None:
                    continue
                sample = sample_from_trajectory(row, cfg, goal_vector=vector)
                if sample is not None:
                    rack = encode_rack_v3(sample.state, sample.context)
                    result.add(hashlib.sha256(rack[:64].tobytes()).hexdigest())
                    break
    return result


def _check_provenance(args):
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    training = summary.get("training_data") or {}
    sources = {Path(p).resolve() for p in training.get("rollout_dirs", [])}
    expected_holdout = training.get("holdout_rollout_dir")
    if args.train_dir.resolve() not in sources:
        raise ValueError("Train directory is not recorded as Micro training input")
    if expected_holdout is None or args.holdout_dir.resolve() != Path(expected_holdout).resolve():
        raise ValueError("Holdout directory does not match Micro training exclusion")
    if summary.get("goal_source") != args.goal_source:
        raise ValueError("Micro training goal-source differs from audit")
    if not summary.get("wpa_break_rules", False):
        raise ValueError("This audit requires WPA training rules")
    if summary.get("outcome_position", False) != args.outcome_position:
        raise ValueError("Outcome position setting differs from Micro training")
    if summary.get("training_steps_this_run", 0) <= 0:
        raise ValueError("Invalid saved Micro training summary")
    return summary


def audit(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.train_dir.resolve() == args.holdout_dir.resolve():
        raise ValueError("Train and heldout must be different directories")
    training_summary = _check_provenance(args)
    # Phase28a trainer excludes *all* heldout boards from its replay inputs.
    # Apply the same exclusion when auditing original input rollouts.
    all_holdout_boards = all_micro_board_hashes(
        args.holdout_dir, goal_source=args.goal_source,
    )
    train, train_boards, train_meta = select_cases(
        args.train_dir, max_matches=args.max_matches_per_split,
        per_goal=args.max_cases_per_goal, seed=args.seed, goal_source=args.goal_source,
        excluded_board_hashes=all_holdout_boards,
    )
    heldout, heldout_boards, heldout_meta = select_cases(
        args.holdout_dir, max_matches=args.max_matches_per_split,
        per_goal=args.max_cases_per_goal, seed=args.seed + 1,
        goal_source=args.goal_source,
    )
    if train_boards & heldout_boards:
        raise ValueError("Training/holdout contain identical sampled boards: leakage")
    if not train or not heldout:
        raise ValueError("No eligible Micro-controlled goals in one of the two groups")

    agents = {"candidate": NineBallMicroSACAgent(args.checkpoint, device="cpu")}
    if args.base_checkpoint is not None:
        agents["base"] = NineBallMicroSACAgent(args.base_checkpoint, device="cpu")
    for agent in agents.values():
        agent.break_decoder = WPA_NINE_BALL_BREAK_DECODER
    rules = NineBallRules(wpa_break_rules=True)
    simulator = MultiBallFastFizSimulator(TableGameConfig.nine_ball_inventory())
    simulator.require_head_string_crossings()

    result = {}
    for group_name, grouped in (("train", train), ("heldout", heldout)):
        subset = {}
        for kind, cases in grouped.items():
            by_model = {}
            for name, agent in agents.items():
                counts = Counter()
                reasons = Counter()
                for case in cases:
                    record_trial(
                        agent, case, simulator, rules, counts, reasons,
                        outcome_position=args.outcome_position,
                        cue_clearance_retry=True,
                    )
                count = len(cases)
                by_model[name] = {
                    **dict(counts),
                    "cases": count,
                    "goal_success_rate": counts["goal_success"] / count,
                    "legal_rate": counts["legal"] / count,
                    "physics_rejection_rate": counts["physics_rejected"] / count,
                    "foul_rate": counts["fouls"] / count,
                    "physical_rejections": dict(reasons),
                }
            subset[kind] = by_model
        result[group_name] = subset
    output = {
        "kind": "nine_ball_micro_generalization_audit_v1",
        "checkpoint": str(args.checkpoint),
        "training_summary": str(args.training_summary),
        "training_steps_this_run": training_summary["training_steps_this_run"],
        "goal_source": args.goal_source,
        "base_checkpoint": str(args.base_checkpoint) if args.base_checkpoint else None,
        "train_dir": str(args.train_dir),
        "holdout_dir": str(args.holdout_dir),
        "seed": args.seed,
        "max_matches_per_split": args.max_matches_per_split,
        "max_cases_per_goal": args.max_cases_per_goal,
        "train_sample": train_meta,
        "heldout_sample": heldout_meta,
        "sampled_board_overlap": 0,
        "all_holdout_boards_excluded_from_train": len(all_holdout_boards),
        "wpa_break_rules": True,
        "cue_clearance_retry": True,
        "by_goal": result,
        "limitations": (
            "Observed-goal difficulty and distribution can differ by split. "
            "Train-vs-heldout success gaps alone do not prove overfitting. "
            "Use fixed independent board/goal sets, learning curves, and "
            "targeted teacher-oracle gap checks for diagnosis."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_json(args.output, output)
    return output


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-summary", type=Path, required=True)
    p.add_argument("--goal-source", choices=("candidates", "logged"), default="candidates")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--holdout-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path)
    p.add_argument("--max-matches-per-split", type=int, default=24)
    p.add_argument("--max-cases-per-goal", type=int, default=48)
    p.add_argument("--seed", type=int, default=320001)
    p.add_argument("--outcome-position", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))
