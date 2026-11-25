"""Controlled 2x2 model x MCTS depth diagnostic on Phase31 disagreement boards.

Evaluate (C, C-prime) x (depth 1, depth 2) from the same seven original
positions. This is a fixed, failure-focused DEVELOPMENT diagnostic and
does not change any training or previously stored experimental artifacts.
The originally evaluated model x depth 2 choices must reproduce their
previous root-search records before a physical outcome is accepted.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _goal_key, _rebuild_state, _root_result, _validate_evaluation,
)
from scripts.experiments.diagnose_phase31_shot_outcomes import (
    _digest, load_cases, shot_metrics,
)

DEPTHS = (1, 2)
SIDES = ("base", "final")


def _plan_and_one_step(agent, simulator, rules, state, context, position):
    from src.game.nine_ball_physics_guard import simulate_nine_ball_shot

    root = _root_result(agent, state, context, position)
    plan = agent.searcher.last_plan
    if plan is None or plan.shot is None:
        raise ValueError("MCTS selected a rule choice rather than a physical shot")
    physical, outcome = simulate_nine_ball_shot(
        simulator, rules, state, context,
        target_ball=plan.target_ball,
        shot=plan.shot,
        declared_push_out=plan.macro_goal.kind == "PUSH_OUT",
    )
    selected = root["candidates"][root["chosen_index"]]
    return {
        "goal_v3": root["chosen_goal_v3"],
        "goal_kind": plan.macro_goal.kind,
        "declared_pocket": plan.target_pocket,
        "selected_q": selected["q"],
        "selected_prior": selected["prior"],
        "selected_visits": selected["visits"],
        "root_visits": [candidate["visits"] for candidate in root["candidates"]],
        "root_goal_v3": [candidate["goal_v3"] for candidate in root["candidates"]],
        "invalid_root_simulations": root["rejected_invalid_simulations"],
        "shot": shot_metrics(state, context, plan, physical, outcome),
    }


def _verify_depth2(result, old, *, original_shot=None, original_physical=None, original_rules=None):
    if _goal_key(result["goal_v3"]) != _goal_key(old["chosen_goal_v3"]):
        raise ValueError("Depth-2 chosen goal does not reproduce prior root analysis")
    if result["root_visits"] != [candidate["visits"] for candidate in old["candidates"]]:
        raise ValueError("Depth-2 root visits do not reproduce prior root analysis")
    if any(
        _goal_key(a) != _goal_key(b["goal_v3"])
        for a, b in zip(result["root_goal_v3"], old["candidates"], strict=True)
    ):
        raise ValueError("Depth-2 candidate goals differ from prior root analysis")
    if original_shot is not None:
        physical = result["shot"]["physical_shot"]
        if any(not np.isclose(physical[name], original_shot[name], rtol=1e-6, atol=1e-5)
               for name in physical):
            raise ValueError("C-prime depth-2 physical controls differ from historical match log")
    if original_physical is not None and result["shot"]["first_contact_ball"] != original_physical.get("first_contact_ball"):
        raise ValueError("C-prime depth-2 physical first contact differs from match log")
    if original_rules is not None:
        current = result["shot"]
        if (current["foul"] != original_rules["foul"]
                or current["newly_pocketed"] != original_rules["newly_pocketed"]
                or current["rack_winner"] != original_rules["rack_winner"]
                or current["turn_changed"] != original_rules["turn_changed"]):
            raise ValueError("C-prime depth-2 rules result differs from historical match log")


def summarize(records, *, expected, invalid):
    summary = {
        "diagnostic_kind": "phase31_model_by_depth_same_board",
        "source_disagreements": expected,
        "compared_boards": len(records),
        "unusable_boards": len(invalid),
        "conditions": {},
        "depth_effects": {},
        "disagreements": {},
    }
    for side in SIDES:
        for depth in DEPTHS:
            label = f"{side}_depth_{depth}"
            entries = [case["results"][side][f"depth_{depth}"]["shot"] for case in records]
            summary["conditions"][label] = {
                "fouls": sum(int(e["foul"]) for e in entries),
                "target_ball_potted": sum(int(e["target_ball_pocketed"]) for e in entries),
                "exact_target_pocket_success": sum(int(e["exact_target_pocket_success"]) for e in entries),
                "shooter_keeps_turn": sum(
                    int(e["rack_winner"] is None
                        and e["next_player"] == case["acting_player"])
                    for e, case in zip(entries, records, strict=True)
                ),
                "position_attack_chosen": sum(int(e["goal_kind"] == "POSITION_ATTACK") for e in entries),
            }
        changed = [
            case for case in records if _goal_key(case["results"][side]["depth_1"]["goal_v3"])
            != _goal_key(case["results"][side]["depth_2"]["goal_v3"])
        ]
        introduced = sum(
            not case["results"][side]["depth_1"]["shot"]["foul"]
            and case["results"][side]["depth_2"]["shot"]["foul"]
            for case in records
        )
        avoided = sum(
            case["results"][side]["depth_1"]["shot"]["foul"]
            and not case["results"][side]["depth_2"]["shot"]["foul"]
            for case in records
        )
        summary["depth_effects"][side] = {
            "choice_changed_depth_1_to_2": len(changed),
            "foul_introduced_at_depth_2": introduced,
            "foul_avoided_at_depth_2": avoided,
        }
    for depth in DEPTHS:
        summary["disagreements"][f"depth_{depth}"] = sum(
            _goal_key(case["results"]["base"][f"depth_{depth}"]["goal_v3"]) !=
            _goal_key(case["results"]["final"][f"depth_{depth}"]["goal_v3"])
            for case in records
        )
    summary["interpretation"] = (
        "All four configurations use the same Macro Policy/Micro/physics and the "
        "same failure-focused C-winning source boards. Model differences affect "
        "only State V; depth changes the planning budget horizon, not the "
        "network. Deterministic one-step results do not estimate match wins "
        "or justify permanently reducing depth. Prior results are read-only."
    )
    return summary


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Select exactly one CUDA_VISIBLE_DEVICES GPU")
    config = _validate_evaluation(args.evaluation_dir)
    if config["search"]["max_depth"] != 2:
        raise ValueError("Prior diagnostic must have used MCTS depth=2")
    cases = load_cases(args.root_diagnostic_dir, args.evaluation_dir)

    from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition
    from src.game.nine_ball_rack_split import load_rack_layout_split
    from src.game.nine_ball_rules import NineBallRules
    from src.game.rack_state import TableGameConfig

    search = dict(config["search"])
    split_record = config.get("rack_layout_split")
    if split_record:
        if _digest(split_record["path"]) != split_record["sha256"]:
            raise ValueError("Rack split SHA256 differs from evaluation")
        search["layout_split"] = load_rack_layout_split(split_record["path"])
    agents = {}
    for side in SIDES:
        ckpts = config["checkpoints"][side]
        for depth in DEPTHS:
            agents[(side, depth)] = NineBallMCTSAgent(
                ckpts["micro"]["path"],
                macro_policy_checkpoint=ckpts["policy"]["path"],
                state_value_checkpoint=ckpts["state_v"]["path"],
                device=args.device,
                **{**search, "max_depth": depth},
            )
    simulator = MultiBallFastFizSimulator(TableGameConfig.nine_ball_inventory())
    rules = NineBallRules(wpa_break_rules=search["wpa_break_rules"])
    if search["wpa_break_rules"]:
        simulator.require_head_string_crossings()

    records = []
    invalid = []
    for case in cases:
        state, context, position = _rebuild_state(case)
        board = {
            "match_seed": case["match_seed"],
            "row_index": case["row_index"],
            "source_file": case["source_file"],
            "acting_player": int(state.current_player),
            "results": {"base": {}, "final": {}},
        }
        try:
            for side in SIDES:
                for depth in DEPTHS:
                    result = _plan_and_one_step(
                        agents[(side, depth)], simulator, rules,
                        state, context, position,
                    )
                    if depth == 2:
                        _verify_depth2(
                            result, case["expected"][side],
                            original_shot=case["logged_action"] if side == "final" else None,
                            original_physical=case.get("logged_physical") if side == "final" else None,
                            original_rules=case.get("logged_rule_result") if side == "final" else None,
                        )
                    board["results"][side][f"depth_{depth}"] = result
        except (InvalidPhysicalTransition, ValueError) as exc:
            invalid.append({
                "seed": case["match_seed"], "row": case["row_index"],
                "error": str(exc), "type": type(exc).__name__,
            })
            print(f"seed={case['match_seed']} row={case['row_index']} SKIP: {exc}", flush=True)
            continue
        records.append(board)
        mini = [
            f"{side}d{depth}:{'FOUL' if board['results'][side][f'depth_{depth}']['shot']['foul'] else 'legal'}"
            for side in SIDES for depth in DEPTHS
        ]
        print(
            f"seed={case['match_seed']} row={case['row_index']} " + " ".join(mini),
            flush=True,
        )
    if not records:
        raise RuntimeError("No fully comparable 2x2 replay boards")
    summary = summarize(records, expected=len(cases), invalid=invalid)
    summary.update({
        "evaluation_dir": str(args.evaluation_dir),
        "root_diagnostic_dir": str(args.root_diagnostic_dir),
        "checkpoint_state_v_sha256": {
            side: config["checkpoints"][side]["state_v"]["sha256"]
            for side in SIDES
        },
        "mcts_simulations": search["simulations"],
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "details.json").write_text(
        json.dumps({"cases": records, "errors": invalid}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--root-diagnostic-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
