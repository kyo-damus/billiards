"""Read-only physical audit of second-ply replies in Phase31 C/C-prime MCTS.

Question: when an initial candidate scratches and hands the opponent
ball-in-hand, what *actual FastFiz shot* does the opponent's second-ply
search attempt, and was that shot legal/successful?

Do NOT interpret this as independent match-win prediction, or as proof
of erroneous PUCT/backup. The C-winning boards are failure-focused.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _goal_key, _rebuild_state, _validate_evaluation,
)
from scripts.experiments.diagnose_phase31_shot_outcomes import (
    _digest, load_cases, shot_metrics,
)
from scripts.experiments.diagnose_phase31_two_ply_values import (
    _match_target, _reproduce,
)
from src.game.nine_ball_encoding_v3 import TABLE_CHOICES, encode_goal_v3


def physical_reply_records(searcher, simulator, rules, edge, *, root_player):
    """Re-execute the visited second-ply physical actions from the saved tree.

    The searcher's child is already cue-placed by the exact same
    deterministic ball-in-hand placement used in the original MCTS.
    Each physical reply restores this identical child rack snapshot.
    """
    from src.game.nine_ball_physics_guard import (
        InvalidPhysicalTransition, simulate_nine_ball_shot,
    )

    child = edge.child
    if child is None:
        raise ValueError("Unvisited first-ply edge")
    data = []
    for response in child.edges:
        if response.visits <= 0 or response.invalid_reason is not None:
            continue
        goal = response.goal
        info = {
            "goal_v3": encode_goal_v3(goal).tolist(),
            "goal_kind": goal.kind,
            "target_ball": int(goal.target_ball),
            "declared_pocket": goal.target_pocket,
            "visits": int(response.visits),
            "q_second_player": float(response.q),
            "q_root_perspective": float(
                searcher.gamma * (1 if child.state.current_player == root_player else -1)
                * response.q
            ),
            "second_shooter": int(child.state.current_player),
            "first_shot_had_ball_in_hand_after_prepare": bool(child.state.ball_in_hand),
            "second_shooter_cue_xy_m": [
                float(x) for x in child.state.ball_positions[0]
            ],
        }
        if goal.kind in TABLE_CHOICES:
            info["physical_result"] = {
                "available": False, "reason": "nonphysical_rule_choice",
            }
        elif response.plan is None or response.plan.shot is None:
            info["physical_result"] = {
                "available": False, "reason": "missing_recorded_second_shot",
            }
        else:
            try:
                physical, outcome = simulate_nine_ball_shot(
                    simulator, rules, child.state, child.context,
                    target_ball=response.plan.target_ball,
                    shot=response.plan.shot,
                    declared_push_out=goal.kind == "PUSH_OUT",
                )
                if response.child is not None:
                    if not np.array_equal(
                        outcome.next_state.ball_pocket_indices[1:],
                        response.child.state.ball_pocket_indices[1:],
                    ):
                        raise ValueError(
                            "Second-ply object-ball pocket state does not reproduce search"
                        )
                info["physical_result"] = {
                    "available": True,
                    **shot_metrics(
                        child.state, child.context,
                        response.plan, physical, outcome,
                    ),
                }
            except InvalidPhysicalTransition as exc:
                info["physical_result"] = {
                    "available": False,
                    "reason": "invalid_native_transition",
                    "error": str(exc),
                }
        data.append(info)
    return sorted(data, key=lambda x: (-x["visits"], -x["q_second_player"]))


def summarize(records):
    aggregate = {}
    for side in ("base", "final"):
        for label in ("C_choice", "Cprime_choice"):
            rows = [r["models"][side][label] for r in records]
            replies = [
                x for row in rows
                for x in row["second_replies"]
                if x["physical_result"]["available"]
            ]
            aggregate[f"{side}:{label}"] = {
                "boards": len(rows),
                "first_shot_fouls": sum(row["first_foul"] for row in rows),
                "second_reply_candidates_simulated": len(replies),
                "second_shot_fouls": sum(x["physical_result"]["foul"] for x in replies),
                "second_shot_target_pots": sum(
                    x["physical_result"]["target_ball_pocketed"] for x in replies
                ),
                "second_shot_keeps_turn": sum(
                    x["physical_result"]["next_player"] == x["second_shooter"]
                    for x in replies
                ),
                "warning": "Second-shot candidates are correlated search branches; not independent shot attempts.",
            }
    return aggregate


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Select one CUDA_VISIBLE_DEVICES GPU")
    config = _validate_evaluation(args.evaluation_dir)
    if config["search"]["max_depth"] != 2:
        raise ValueError("The original comparison must have been depth 2")
    cases = load_cases(args.root_diagnostic_dir, args.evaluation_dir)
    targets = set(args.seed_row or [
        "1690522:197", "1690144:40", "1690324:190",
    ])
    for token in targets:
        parts = token.split(":")
        if len(parts) != 2 or not all(p.isdecimal() for p in parts):
            raise ValueError(f"Invalid --seed-row {token}; expected SEED:ROW")
    cases = [
        case for case in cases
        if f"{case['match_seed']}:{case['row_index']}" in targets
    ]
    if len(cases) != len(targets):
        raise ValueError("Requested seed/row not found in recorded root audit")

    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    from src.game.nine_ball_rack_split import load_rack_layout_split
    from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
    from src.game.nine_ball_rules import NineBallRules
    from src.game.nine_ball_physics_guard import simulate_nine_ball_shot
    from src.game.rack_state import TableGameConfig

    search = dict(config["search"])
    split_record = config.get("rack_layout_split")
    if split_record:
        if _digest(split_record["path"]) != split_record["sha256"]:
            raise ValueError("Rack layout split hash mismatch")
        search["layout_split"] = load_rack_layout_split(split_record["path"])
    models = {
        side: NineBallMCTSAgent(
            config["checkpoints"][side]["micro"]["path"],
            macro_policy_checkpoint=config["checkpoints"][side]["policy"]["path"],
            state_value_checkpoint=config["checkpoints"][side]["state_v"]["path"],
            device=args.device, **search,
        )
        for side in ("base", "final")
    }
    simulator = MultiBallFastFizSimulator(TableGameConfig.nine_ball_inventory())
    rules = NineBallRules(wpa_break_rules=search["wpa_break_rules"])
    if search["wpa_break_rules"]:
        simulator.require_head_string_crossings()

    results = []
    for case in cases:
        state, context, position = _rebuild_state(case)
        result = {
            "seed": case["match_seed"],
            "row": case["row_index"],
            "root_player": int(state.current_player),
            "rack_wins": list(position.rack_wins),
            "race_to": int(position.race_to),
            "pre_shot_fair_match_value_root": float(position.fair_match_value(state.current_player)),
            "models": {},
        }
        for side in ("base", "final"):
            agent = models[side]
            agent.choose_match_shot(state, context, position)
            searcher = agent.searcher
            _reproduce(searcher.last_root, case["expected"][side])
            result["models"][side] = {}
            for choice_name, origin in (("C_choice", "base"), ("Cprime_choice", "final")):
                edge = _match_target(
                    searcher.last_root,
                    case["expected"][origin]["chosen_goal_v3"],
                )
                if edge.plan is None or edge.plan.shot is None:
                    raise ValueError("Expected a visited root physical shot")
                physical, outcome = simulate_nine_ball_shot(
                    simulator, rules, state, context,
                    target_ball=edge.plan.target_ball,
                    shot=edge.plan.shot,
                    declared_push_out=edge.goal.kind == "PUSH_OUT",
                )
                if edge.child is None:
                    raise ValueError("Expected root edge child")
                if not np.array_equal(
                    outcome.next_state.ball_pocket_indices[1:],
                    edge.child.state.ball_pocket_indices[1:],
                ):
                    raise ValueError("Root edge object-ball pocket state does not reproduce")
                root_result = {
                    "root_q": float(edge.q),
                    "root_visits": int(edge.visits),
                    "first_foul": bool(outcome.foul),
                    "first_foul_reasons": list(outcome.foul_reasons),
                    "first_potted": list(outcome.newly_pocketed),
                    "first_next_player": int(outcome.next_state.current_player),
                    "first_ball_in_hand": bool(outcome.next_state.ball_in_hand),
                    "second_shooter": int(edge.child.state.current_player),
                    "second_shooter_cue_xy_after_placement_m": [
                        float(x) for x in edge.child.state.ball_positions[0]
                    ],
                    "second_replies": physical_reply_records(
                        searcher, simulator, rules, edge,
                        root_player=state.current_player,
                    ),
                }
                result["models"][side][choice_name] = root_result
                print(
                    f"seed={case['match_seed']} row={case['row_index']} "
                    f"V={side} first={choice_name} "
                    f"foul={outcome.foul} "
                    f"second_shooter={root_result['second_shooter']} "
                    f"replies={len(root_result['second_replies'])}",
                    flush=True,
                )
        results.append(result)

    summary = {
        "diagnostic_kind": "phase31_second_ply_physical_reply_audit",
        "compared_boards": len(results),
        "seed_rows": [f"{r['seed']}:{r['row']}" for r in results],
        "aggregate": summarize(results),
        "note": (
            "Observed second-ply search replies on failure-focused boards, "
            "not complete game continuations or estimated opponent strength. "
            "Replays held-out training artifacts unchanged."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "details.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--root-diagnostic-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-row", action="append",
                   help="Repeat SEED:ROW. Default: original three scratch boards.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
