"""Three-board frozen Macro/Micro ball-in-hand placement comparison.

This is an OFFLINE diagnostic, not a new deployed player. It replays
the original three C-prime scratch shots, then compares one deterministic
frozen Policy + Micro physical shot at each of a bounded set of legal
candidate cue placements.

Methods:
- baseline: first native-legal cue position in legacy distance ordering.
- geometry: best native-legal position by visible target pockets and path.
- sampled_oracle: hindsight-best of the same frozen shots, from bounded
  sampled positions. It is an unattainable outcome-aware UPPER BOUND,
  never a real policy to compare match strength against.

No model weights, MCTS, match outcomes or source artifacts are changed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _rebuild_state, _validate_evaluation,
)
from scripts.experiments.diagnose_phase31_shot_outcomes import (
    shot_metrics,
)
from src.game.nine_ball_cue_placement_experiment import (
    geometry_placement_ranking, sample_placements,
)

DEFAULT_TARGETS = ("1690522:197", "1690144:40", "1690324:190")


def _native_place(simulator, rules, ball_in_hand_state, xy):
    from src.game.nine_ball_session import NineBallRackSession
    session = NineBallRackSession(simulator, ball_in_hand_state, rules=rules)
    try:
        session.place_cue_ball(*xy)
    except ValueError as exc:
        if "overlaps" in str(exc) or "outside playing surface" in str(exc):
            return None
        raise
    return session.state


def _first_native_legal(ranked, simulator, rules, state):
    for item in ranked:
        placed = _native_place(simulator, rules, state, item["xy"])
        if placed is not None:
            return item
    raise ValueError("No native-legal cue location from candidate grid")


def quality_tuple(metrics, shooter):
    """Declared target legality outranks mere retention/no-foul."""
    return (
        int(metrics["exact_target_pocket_success"]),
        int(metrics["target_ball_pocketed"] and not metrics["foul"]),
        int(metrics["rack_winner"] == shooter or (
            metrics["rack_winner"] is None and metrics["next_player"] == shooter
        )),
        int(not metrics["foul"]),
    )


def _one_micro_shot(agent, simulator, rules, context, position, state):
    from src.game.nine_ball_physics_guard import (
        InvalidPhysicalTransition, simulate_nine_ball_shot,
        simulate_with_cue_clearance,
    )
    plan = agent.choose_match_shot(state, context, position)
    if plan.shot is None:
        return {
            "valid": False, "error": "nonphysical_rule_action",
            "goal_kind": plan.macro_goal.kind,
        }
    try:
        executed, physical, outcome, clearance_trials = simulate_with_cue_clearance(
            lambda shot: simulate_nine_ball_shot(
                simulator, rules, state, context,
                target_ball=plan.target_ball, shot=shot,
                declared_push_out=plan.macro_goal.kind == "PUSH_OUT",
            ),
            plan.shot,
        )
    except InvalidPhysicalTransition as exc:
        return {
            "valid": False, "error": str(exc),
            "goal_kind": plan.macro_goal.kind,
        }
    metrics = shot_metrics(state, context, plan, physical, outcome)
    return {
        "valid": True,
        "goal_kind": plan.macro_goal.kind,
        "declared_pocket": plan.target_pocket,
        "quality": list(quality_tuple(metrics, state.current_player)),
        "cue_clearance_trials": int(clearance_trials),
        "executed_physical_shot": executed.to_fastfiz_kwargs(),
        "metrics": metrics,
    }


def _summarize(rows):
    output = {}
    for label in ("baseline", "geometry", "sampled_oracle"):
        selected = [row["selected"][label] for row in rows]
        output[label] = {
            "boards": len(rows),
            "available_physical_shots": sum(t["trial"]["valid"] for t in selected),
            "declared_legal_target_pots": sum(
                t["trial"].get("metrics", {}).get("exact_target_pocket_success", False)
                for t in selected
            ),
            "fouls": sum(
                t["trial"].get("metrics", {}).get("foul", False) for t in selected
            ),
            "retained_turn_or_rack_win": sum(
                t["trial"].get("quality", [0, 0, 0, 0])[2] for t in selected
            ),
        }
    return output


def _select(rows, id_target):
    return next((row for row in rows if row["key"] == id_target), None)


def _evaluate_board(case, details, second_ply, agent, simulator, rules, max_placements):
    from src.game.nine_ball_physics_guard import simulate_nine_ball_shot
    from src.micro.shot_control import PhysicalShot5D

    state, context, position = _rebuild_state(case)
    key = f"{case['match_seed']}:{case['row_index']}"
    original = details[key]
    shot = original["final"]["physical_shot"]
    physical_shot = PhysicalShot5D(
        speed_m_s=float(shot["power"]),
        azimuth_deg=float(shot["angle"]),
        elevation_deg=float(shot["cue_elevation_deg"]),
        tip_x_mm=float(shot["tip_x_mm"]),
        tip_y_mm=float(shot["tip_y_mm"]),
    )
    physical, transition = simulate_nine_ball_shot(
        simulator, rules, state, context,
        target_ball=original["final"]["declared_target_ball"],
        shot=physical_shot,
    )
    if not transition.foul or not transition.next_state.ball_in_hand:
        raise ValueError(f"Expected original C-prime scratch, got {key}")
    if (list(transition.foul_reasons) != original["final"]["foul_reasons"]
            or list(transition.newly_pocketed) != original["final"]["newly_pocketed"]):
        raise ValueError(f"Original scratch could not be physically reproduced: {key}")
    # Opponent's ball-in-hand next shot. No policy/model changes.
    after = transition.next_state
    after_context = transition.next_context
    candidates = agent.cue_placement_candidates(after)
    geometric = geometry_placement_ranking(after, candidates)
    original_order = sorted(geometric, key=lambda x: x["original_index"])
    baseline = _first_native_legal(original_order, simulator, rules, after)
    geometry = _first_native_legal(geometric, simulator, rules, after)
    previous = second_ply.get(key)
    if previous is not None:
        prior_xy = previous["models"]["final"]["Cprime_choice"][
            "second_shooter_cue_xy_after_placement_m"
        ]
        if not np.allclose(baseline["xy"], prior_xy, rtol=0, atol=1e-6):
            raise ValueError(
                f"Baseline placement changed relative to original MCTS child: {key} "
                f"{baseline['xy']} vs {prior_xy}"
            )
    pool = sample_placements(candidates, geometric, max_placements=max_placements)
    # Ensure both actual methods are evaluated, even if geometry's
    # best native-legal candidate was just beyond the sample budget.
    pool = [baseline, geometry] + [
        item for item in pool
        if tuple(item["xy"]) not in (tuple(baseline["xy"]), tuple(geometry["xy"]))
    ]
    pool = pool[:max_placements]
    evaluated = []
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    for item in pool:
        placed = _native_place(simulator, rules, after, item["xy"])
        if placed is None:
            evaluated.append({
                "xy": item["xy"], "geometry_clear_pockets": item["clear_pocket_count"],
                "trial": {"valid": False, "error": "native_illegal_cue_placement"},
            })
            continue
        trial = _one_micro_shot(
            agent, simulator, rules, after_context, position, placed
        )
        evaluated.append({
            "xy": item["xy"],
            "geometry_clear_pockets": item["clear_pocket_count"],
            "geometry_clear_pocket_ids": item["clear_pocket_ids"],
            "trial": trial,
        })
    by_xy = {tuple(item["xy"]): item for item in evaluated}
    selected = {
        "baseline": by_xy[tuple(baseline["xy"])],
        "geometry": by_xy[tuple(geometry["xy"])],
    }
    feasible = [item for item in evaluated if item["trial"]["valid"]]
    if not feasible:
        raise ValueError(f"All candidate placement/shot combinations failed: {key}")
    selected["sampled_oracle"] = max(
        feasible,
        key=lambda item: tuple(item["trial"]["quality"])
    )
    return {
        "key": key,
        "match_rack_wins": list(position.rack_wins),
        "opponent": int(after.current_player),
        "source_first_shot_foul_reasons": list(transition.foul_reasons),
        "grid_candidates": len(candidates),
        "geometry_ranked": len(geometric),
        "physically_tested_placements": len(evaluated),
        "selected": selected,
        "all_trials": evaluated,
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if args.max_placements < 3 or args.max_placements > 64:
        raise ValueError("--max-placements must be in 3..64")
    if args.device == "cuda":
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not gpu or "," in gpu:
            raise ValueError("Set exactly one CUDA_VISIBLE_DEVICES device")
    cfg = _validate_evaluation(args.evaluation_dir)
    quality = json.loads((args.shot_quality_dir / "details.json").read_text())["cases"]
    quality_by_key = {f"{c['seed']}:{c['row']}": c for c in quality}
    prior_physics = json.loads((args.second_ply_dir / "details.json").read_text())
    prior_by_key = {f"{c['seed']}:{c['row']}": c for c in prior_physics}
    from scripts.experiments.diagnose_phase31_shot_outcomes import load_cases

    roots = load_cases(args.root_diagnostic_dir, args.evaluation_dir)
    targets = set(args.seed_row or DEFAULT_TARGETS)
    for key in targets:
        values = key.split(":")
        if len(values) != 2 or not all(v.isdecimal() for v in values):
            raise ValueError(f"Bad --seed-row {key}")
    cases = [
        case for case in roots
        if f"{case['match_seed']}:{case['row_index']}" in targets
    ]
    if len(cases) != len(targets):
        raise ValueError("Not all seed/row targets found in existing root diagnostic")
    for case in cases:
        key = f"{case['match_seed']}:{case['row_index']}"
        if key not in quality_by_key or key not in prior_by_key:
            raise ValueError(f"Missing prerequisite physical evidence for {key}")
        if not quality_by_key[key]["final"]["foul"]:
            raise ValueError(f"Expected C-prime foul in shot-quality record: {key}")

    from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
    from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
    from src.game.nine_ball_rules import NineBallRules
    from src.game.rack_state import TableGameConfig

    ck = cfg["checkpoints"]["base"]
    agent = NineBallMicroSACAgent(
        ck["micro"]["path"],
        macro_policy_checkpoint=ck["policy"]["path"],
        deterministic=True, device=args.device,
        learned_safety=True, learned_break=True,
        outcome_position=False, match_aware=cfg["search"]["match_aware"],
    )
    simulator = MultiBallFastFizSimulator(TableGameConfig.nine_ball_inventory())
    rules = NineBallRules(wpa_break_rules=cfg["search"]["wpa_break_rules"])
    if cfg["search"]["wpa_break_rules"]:
        simulator.require_head_string_crossings()
    records = []
    for case in cases:
        item = _evaluate_board(
            case, quality_by_key, prior_by_key,
            agent, simulator, rules, args.max_placements,
        )
        records.append(item)
        print(
            f"board={item['key']} sampled_positions={item['physically_tested_placements']} "
            f"baseline={item['selected']['baseline']['trial'].get('quality')} "
            f"geometry={item['selected']['geometry']['trial'].get('quality')} "
            f"sampled_oracle={item['selected']['sampled_oracle']['trial'].get('quality')}",
            flush=True,
        )
    summary = {
        "diagnostic_kind": "phase31_bih_placement_frozen_micro_ablations",
        "compared_boards": len(records),
        "max_sampled_placements_per_board": args.max_placements,
        "per_method": _summarize(records),
        "micro_sha256": ck["micro"]["sha256"],
        "policy_sha256": ck["policy"]["sha256"],
        "note": (
            "FAILURE-FOCUSED 3 C-winning-game boards only. Frozen Macro Policy "
            "and Micro, a single deterministic shot per placement; direct candidate "
            "generation and Macro Policy do not use MCTS/position outcome here. "
            "Sampled oracle uses real physical outcomes and is NOT deployable. "
            "Comparison isolates placement geometry under this fixed policy, "
            "not the effect on full MCTS or match wins."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    (args.output_dir / "details.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--root-diagnostic-dir", type=Path, required=True)
    p.add_argument("--shot-quality-dir", type=Path, required=True)
    p.add_argument("--second-ply-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-placements", type=int, default=17)
    p.add_argument("--seed-row", action="append")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
