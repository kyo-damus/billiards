"""One-step physical counterfactuals for Phase31 C versus C-prime.

Read Phase31's prior reproducible same-state MCTS diagnostic, rerun both
policies on each *choice-disagreement* position, and execute their selected
shots once from the identical original board using the same FastFiz and
nine-ball rules. This is a DEVELOPMENT diagnostic, not causal proof of a
match result or a replacement for paired matches. No training/checkpoint
files or existing evaluation files are modified.

Examples: docs/PHASE31_SHOT_OUTCOMES.md
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _digest, _goal_key, _rebuild_state, _root_result, _validate_evaluation,
)


def load_cases(root_diagnostic, evaluation_dir):
    """Require prior root replay, not arbitrary unverified physical positions."""
    summary = json.loads((root_diagnostic / "summary.json").read_text(encoding="utf-8"))
    details = json.loads((root_diagnostic / "details.json").read_text(encoding="utf-8"))
    if summary["diagnostic_kind"] != "phase31_identical_state_root_mcts_replay":
        raise ValueError("Expected Phase31 same-state root diagnostic")
    if Path(summary["evaluation_dir"]).resolve() != evaluation_dir.resolve():
        raise ValueError("Root diagnostic was not generated from this evaluation")
    if len(details["cases"]) != summary["compared_states"]:
        raise ValueError("Root diagnostic case count mismatch")
    cases = []
    for row in details["cases"]:
        if not row["comparison"]["chosen_different"]:
            continue
        if not row["logged_choice_reproduced"] or not row["comparison"]["candidate_sets_equal"]:
            raise ValueError("Root diagnostic contains non-reproducible or misaligned case")
        source = Path(row["source_file"]).resolve()
        if not source.is_relative_to(evaluation_dir.resolve()):
            raise ValueError(f"Source file outside evaluation directory: {source}")
        game = json.loads(source.read_text(encoding="utf-8"))
        index = row["row_index"]
        original = game["trajectory"][index]
        if game["seed"] != row["match_seed"] or game["winner"] != "base":
            raise ValueError("Source game metadata differs from prior diagnostic")
        if game["player_models"][original["player"]] != "final":
            raise ValueError("Expected an action by C-prime in a C-winning game")
        if _goal_key(original["chosen_action"]["goal_v3"]) != _goal_key(row["final"]["chosen_goal_v3"]):
            raise ValueError("Original log C-prime shot differs from root replay")
        cases.append({
            "source_file": str(source),
            "match_seed": row["match_seed"],
            "row_index": index,
            "state": original["state"],
            "rule_context": original["rule_context"],
            "match_context": original["match_context"],
            "logged_action": original["chosen_action"],
            "expected": row,
        })
    if not cases:
        raise ValueError("No disagreed choices in root diagnostic")
    return cases


def shot_metrics(state, context, plan, physical, outcome):
    """Rule and physics facts only. One shot is not an estimate of winning."""
    from src.game.nine_ball_encoding_v3 import direct_pocket_candidates
    from src.game.nine_ball_encoding_v3 import TABLE_CHOICES

    kind = plan.macro_goal.kind
    if kind in TABLE_CHOICES:
        raise ValueError("Cannot evaluate a nonphysical accept/pass as a shot")
    cue_id = outcome.next_state.array_index_for_ball(0)
    cue_active = outcome.next_state.ball_pocket_indices[cue_id] == -1
    xy = (
        [float(x) for x in outcome.next_state.ball_positions[cue_id]]
        if cue_active else None
    )
    next_balls = outcome.next_state.on_table_object_balls()
    target_id = plan.macro_goal.target_ball
    pocketed = target_id in outcome.newly_pocketed
    pocket_id = (
        int(outcome.next_state.ball_pocket_indices[
            outcome.next_state.array_index_for_ball(target_id)
        ])
        if pocketed else None
    )
    expected_pocket = plan.macro_goal.target_pocket
    exact_target_pot = bool(
        pocketed and expected_pocket is not None
        and pocket_id == expected_pocket and not outcome.foul
    )
    cue_stop_distance_m = None
    cue_stop_success = None
    if plan.macro_goal.cue_stop_xy is not None:
        if cue_active:
            cue_stop_distance_m = float(np.linalg.norm(
                np.asarray(xy) - np.asarray(plan.macro_goal.cue_stop_xy)
            ))
        cue_stop_success = bool(
            cue_stop_distance_m is not None and
            cue_stop_distance_m <= plan.macro_goal.cue_stop_radius
            and exact_target_pot
        )
    next_direct_options = None
    if (outcome.rack_winner is None and next_balls
            and not outcome.next_state.ball_in_hand
            and outcome.next_context.pending_choice is None):
        next_direct_options = list(direct_pocket_candidates(
            outcome.next_state, min(next_balls)
        ))
    return {
        "goal_kind": kind,
        "declared_target_ball": int(target_id),
        "declared_target_pocket": expected_pocket,
        "physical_shot": plan.shot.to_fastfiz_kwargs(),
        "first_contact_ball": physical.first_contact_ball,
        "foul": bool(outcome.foul),
        "foul_reasons": list(outcome.foul_reasons),
        "newly_pocketed": list(outcome.newly_pocketed),
        "target_ball_pocketed": bool(pocketed),
        "target_pocket_actual": pocket_id,
        "exact_target_pocket_success": exact_target_pot,
        "rack_winner": outcome.rack_winner,
        "turn_changed": bool(outcome.turn_changed),
        "next_player": int(outcome.next_state.current_player),
        "ball_in_hand_next": bool(outcome.next_state.ball_in_hand),
        "cue_xy_m": xy,
        "cue_stop_goal_xy_m": (
            list(plan.macro_goal.cue_stop_xy) if plan.macro_goal.cue_stop_xy else None
        ),
        "cue_stop_radius_m": plan.macro_goal.cue_stop_radius,
        "cue_stop_distance_m": cue_stop_distance_m,
        "full_position_goal_success": cue_stop_success,
        "next_lowest_ball": min(next_balls) if next_balls else None,
        "next_direct_pocket_options": next_direct_options,
        "next_direct_pocket_count": len(next_direct_options) if next_direct_options is not None else None,
    }


def replay_and_evaluate(models, simulator, rules, case):
    from src.game.nine_ball_physics_guard import simulate_nine_ball_shot
    state, context, position = _rebuild_state(case)
    outputs = {}
    for side in ("base", "final"):
        root = _root_result(models[side], state, context, position)
        recorded = case["expected"][side]
        if (_goal_key(root["chosen_goal_v3"]) != _goal_key(recorded["chosen_goal_v3"])
                or [c["visits"] for c in root["candidates"]] !=
                [c["visits"] for c in recorded["candidates"]]
                or any(_goal_key(c["goal_v3"]) != _goal_key(d["goal_v3"])
                       for c, d in zip(root["candidates"], recorded["candidates"], strict=True))):
            raise ValueError(f"{side} root MCTS no longer reproduces previous diagnostic")
        plan = models[side].searcher.last_plan
        if plan.shot is None:
            raise ValueError(f"{side}: chosen rule action has no physical shot")
        if side == "final":
            logged = case["logged_action"]
            actual = plan.shot.to_fastfiz_kwargs()
            if any(not np.isclose(actual[name], logged[name], rtol=1e-6, atol=1e-5)
                   for name in actual):
                raise ValueError("C-prime planned physical shot differs from original match log")
        physical, outcome = simulate_nine_ball_shot(
            simulator, rules, state, context, target_ball=plan.target_ball,
            shot=plan.shot, declared_push_out=(plan.macro_goal.kind == "PUSH_OUT"),
        )
        outputs[side] = shot_metrics(state, context, plan, physical, outcome)
    return outputs


def summarize(records, errors, total):
    success = [item for item in records if "base" in item and "final" in item]
    counts = {}
    for side in ("base", "final"):
        rows = [case[side] for case in success]
        counts[side] = {
            "fouls": sum(x["foul"] for x in rows),
            "target_ball_potted": sum(x["target_ball_pocketed"] for x in rows),
            "exact_target_pocket_success": sum(x["exact_target_pocket_success"] for x in rows),
            "rack_wins": sum(x["rack_winner"] is not None for x in rows),
            "shooter_keeps_turn": sum(
                x["rack_winner"] is None and x["next_player"] == case["acting_player"]
                for x, case in zip(rows, success)
            ),
            "position_goals_attempted": sum(x["full_position_goal_success"] is not None for x in rows),
            "position_goals_fully_satisfied": sum(x["full_position_goal_success"] is True for x in rows),
        }
    return {
        "diagnostic_kind": "phase31_same_board_one_step_physics",
        "requested_disagreements": total,
        "compared": len(success),
        "invalid_or_nonreproducible": len(errors),
        "base": counts["base"],
        "final": counts["final"],
        "interpretation": (
            "Deterministic one-shot physics facts on C-winning source games, "
            "not independent win probabilities, a causal explanation, "
            "or full-position play-out. Position goal success means a legal "
            "declared-target pot AND cue-ball inside its requested region. "
            "Next direct-pocket count is a geometrical proxy only."
        ),
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Set exactly one CUDA_VISIBLE_DEVICES GPU ID")
    config = _validate_evaluation(args.evaluation_dir)
    cases = load_cases(args.root_diagnostic_dir, args.evaluation_dir)
    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    from src.game.nine_ball_rack_split import load_rack_layout_split
    from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
    from src.game.nine_ball_rules import NineBallRules
    from src.game.rack_state import TableGameConfig

    split_record = config.get("rack_layout_split")
    search = dict(config["search"])
    if split_record:
        if _digest(split_record["path"]) != split_record["sha256"]:
            raise ValueError("Rack layout split hash mismatch")
        search["layout_split"] = load_rack_layout_split(split_record["path"])
    models = {
        side: NineBallMCTSAgent(
            config["checkpoints"][side]["micro"]["path"],
            macro_policy_checkpoint=config["checkpoints"][side]["policy"]["path"],
            state_value_checkpoint=config["checkpoints"][side]["state_v"]["path"],
            device=args.device,
            **search,
        )
        for side in ("base", "final")
    }
    simulator = MultiBallFastFizSimulator(TableGameConfig.nine_ball_inventory())
    rules = NineBallRules(wpa_break_rules=config["search"]["wpa_break_rules"])
    if config["search"]["wpa_break_rules"]:
        simulator.require_head_string_crossings()
    results, errors = [], []
    for case in cases:
        try:
            sides = replay_and_evaluate(models, simulator, rules, case)
        except (InvalidPhysicalTransition, ValueError) as exc:
            errors.append({
                "seed": case["match_seed"], "row": case["row_index"],
                "error_type": type(exc).__name__, "error": str(exc),
            })
            print(f"seed={case['match_seed']} row={case['row_index']} SKIP {exc}", flush=True)
            continue
        result = {
            "seed": case["match_seed"], "row": case["row_index"],
            "source_file": case["source_file"],
            "acting_player": case["state"]["current_player"],
            "base": sides["base"], "final": sides["final"],
        }
        results.append(result)
        print(
            f"seed={case['match_seed']} row={case['row_index']} "
            f"C={sides['base']['goal_kind']}/{sides['base']['foul']}/"
            f"{sides['base']['newly_pocketed']} "
            f"Cprime={sides['final']['goal_kind']}/{sides['final']['foul']}/"
            f"{sides['final']['newly_pocketed']}",
            flush=True,
        )
    if not results:
        raise RuntimeError("No comparable physically executable shot pairs")
    summary = summarize(results, errors, len(cases))
    summary.update(
        evaluation_dir=str(args.evaluation_dir),
        root_diagnostic_dir=str(args.root_diagnostic_dir),
        checkpoint_hashes={
            side: config["checkpoints"][side]["state_v"]["sha256"]
            for side in ("base", "final")
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "details.json").write_text(
        json.dumps({"cases": results, "errors": errors}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--root-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
