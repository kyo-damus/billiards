"""Inspect *why* depth-2 MCTS reverses Phase31 C/C-prime first-shot rankings.

This script does not change MCTS behavior, reward, State V, or training
checkpoints. It loads previously reproduced disagreement boards and
compares, for both candidate goals in both State V models:
  - immediate FastFiz shot outcome and first successor's State V,
  - root MCTS visits/Q, and
  - actual explored child (second-shot) candidates and child Q values.
The first successor is value-encoded *after* MCTS's normal, deterministic
cue-ball placement when ball-in-hand, matching MCTS's leaf evaluation.

This is an observational, failure-focused diagnostic, not proof of a bug,
or proof which candidate wins an entire match.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.diagnose_phase31_mcts_choices import (
    _goal_key, _rebuild_state, _validate_evaluation,
)
from scripts.experiments.diagnose_phase31_shot_outcomes import (
    _digest, load_cases,
)
from src.game.nine_ball_encoding_v3 import TABLE_CHOICES, encode_goal_v3, encode_macro_state


def _match_target(root, encoded):
    edges = [e for e in root.edges if _goal_key(encode_goal_v3(e.goal)) == _goal_key(encoded)]
    if len(edges) != 1:
        raise ValueError("Reference chosen goal must identify exactly one root candidate")
    return edges[0]


def _reproduce(root, prior):
    from src.game.nine_ball_encoding_v3 import encode_goal_v3
    chosen = max(
        (edge for edge in root.edges if edge.invalid_reason is None),
        key=lambda e: (e.visits, e.q, e.prior),
    )
    if _goal_key(encode_goal_v3(chosen.goal)) != _goal_key(prior["chosen_goal_v3"]):
        raise ValueError("MCTS chosen goal changed relative to previous diagnosis")
    if [e.visits for e in root.edges] != [e["visits"] for e in prior["candidates"]]:
        raise ValueError("MCTS root visits changed relative to previous diagnosis")
    if any(
        _goal_key(encode_goal_v3(a.goal)) != _goal_key(b["goal_v3"])
        for a, b in zip(root.edges, prior["candidates"], strict=True)
    ):
        raise ValueError("MCTS root candidate list changed relative to previous diagnosis")


def _first_successor_value(searcher, root_player, edge):
    """Root-perspective one-shot leaf estimate; not a stored root Q."""
    child = edge.child
    if child is None:
        return {
            "available": False,
            "reason": "not_visited_in_root_search",
        }
    if child.winner is not None:
        return {
            "available": True,
            "kind": "rack_terminal",
            "root_value": float(edge.terminal_reward),
            "child_player": int(child.state.current_player),
        }
    if child.context.pending_choice is not None:
        return {
            "available": False,
            "reason": "pending_nonphysical_table_choice",
            "child_player": int(child.state.current_player),
        }
    if searcher.value is None:
        return {
            "available": False, "reason": "no_state_value_network",
        }
    state = child.state
    encoded = torch.as_tensor(
        encode_macro_state(
            state, child.context, None if searcher.rack_value else child.match_position
        ), dtype=torch.float32, device=searcher.device,
    ).unsqueeze(0)
    with torch.no_grad():
        raw = float(searcher.value(encoded).reshape(-1)[0].item())
    clipped = float(np.clip(raw, -1.0, 1.0))
    sign = 1 if state.current_player == root_player else -1
    return {
        "available": True,
        "kind": "one_shot_value",
        "child_player": int(state.current_player),
        "child_ball_in_hand_after_preparation": bool(state.ball_in_hand),
        "value_raw_child_player": raw,
        "value_clipped_child_player": clipped,
        "root_value": float(searcher.gamma * sign * clipped),
    }


def _visited_responses(searcher, root_player, edge):
    child = edge.child
    if child is None or not child.edges:
        return []
    parent_sign = 1 if root_player == child.state.current_player else -1
    responses = []
    for response in child.edges:
        if response.visits == 0:
            continue
        item = {
            "goal_v3": encode_goal_v3(response.goal).tolist(),
            "kind": response.goal.kind,
            "target_ball": int(response.goal.target_ball),
            "target_pocket": response.goal.target_pocket,
            "visits": int(response.visits),
            "q_from_second_shooters_perspective": float(response.q),
            "q_from_root_perspective_after_first_shot": float(
                searcher.gamma * parent_sign * response.q
            ),
            "invalid_reason": response.invalid_reason,
            "second_shot_player": int(child.state.current_player),
        }
        if response.child is not None:
            item["next_player_after_second_shot"] = int(response.child.state.current_player)
            item["second_shot_rack_terminal"] = response.child.winner is not None
        responses.append(item)
    return sorted(responses, key=lambda item: (-item["visits"], -item["q_from_second_shooters_perspective"]))


def analyze_edge(searcher, simulator, rules, state, context, root_player, encoded):
    """Read the existing tree and independently re-run only its fixed root action."""
    from src.game.nine_ball_physics_guard import simulate_nine_ball_shot

    edge = _match_target(searcher.last_root, encoded)
    record = {
        "goal_v3": encode_goal_v3(edge.goal).tolist(),
        "goal_kind": edge.goal.kind,
        "target_ball": int(edge.goal.target_ball),
        "target_pocket": edge.goal.target_pocket,
        "root_visits": int(edge.visits),
        "root_q": float(edge.q) if edge.visits else None,
        "root_prior": float(edge.prior),
        "first_successor_value": _first_successor_value(searcher, root_player, edge),
        "visited_second_step_responses": _visited_responses(searcher, root_player, edge),
    }
    if edge.plan is None or edge.plan.shot is None or edge.goal.kind in TABLE_CHOICES:
        record["first_physics"] = {"available": False, "reason": "no_executed_physical_edge"}
        return record
    physical, outcome = simulate_nine_ball_shot(
        simulator, rules, state, context,
        target_ball=edge.plan.target_ball,
        shot=edge.plan.shot,
        declared_push_out=edge.goal.kind == "PUSH_OUT",
    )
    record["first_physics"] = {
        "available": True,
        "foul": bool(outcome.foul),
        "foul_reasons": list(outcome.foul_reasons),
        "newly_pocketed": list(outcome.newly_pocketed),
        "turn_changed": bool(outcome.turn_changed),
        "root_player": int(root_player),
        "next_player": int(outcome.next_state.current_player),
        "ball_in_hand_before_MCTS_cue_placement": bool(outcome.next_state.ball_in_hand),
        "rack_winner": outcome.rack_winner,
        "first_contact_ball": physical.first_contact_ball,
    }
    return record


def _rank_summary(root_a, root_b):
    """Distinguish one-step Value ranking from final backed-up MCTS ranking."""
    first_a = root_a["first_successor_value"]
    first_b = root_b["first_successor_value"]
    one_step_margin = (
        first_a["root_value"] - first_b["root_value"]
        if first_a["available"] and first_b["available"] else None
    )
    backed_up_margin = (
        root_a["root_q"] - root_b["root_q"]
        if root_a["root_q"] is not None and root_b["root_q"] is not None
        else None
    )
    return {
        "one_step_margin_C_choice_minus_Cprime_choice": one_step_margin,
        "depth2_root_q_margin_C_choice_minus_Cprime_choice": backed_up_margin,
        "rank_reversed": (
            one_step_margin * backed_up_margin < 0
            if one_step_margin is not None and backed_up_margin is not None else None
        ),
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Select exactly one visible GPU")
    config = _validate_evaluation(args.evaluation_dir)
    if config["search"]["max_depth"] != 2:
        raise ValueError("Prior MCTS root diagnostic must have depth=2")
    cases = load_cases(args.root_diagnostic_dir, args.evaluation_dir)
    targets = set(args.seed_row or ["1690522:197", "1690144:40", "1690324:190"])
    for token in targets:
        values = token.split(":")
        if len(values) != 2 or not all(v.isdecimal() for v in values):
            raise ValueError(f"Invalid --seed-row: {token}; expected nonnegative SEED:ROW")
    cases = [
        case for case in cases
        if f"{case['match_seed']}:{case['row_index']}" in targets
    ]
    if len(cases) != len(targets):
        raise ValueError("Some requested seed/row pairs are absent from earlier diagnostic")

    from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    from src.game.nine_ball_rack_split import load_rack_layout_split
    from src.game.nine_ball_rules import NineBallRules
    from src.game.rack_state import TableGameConfig

    search = dict(config["search"])
    split_record = config.get("rack_layout_split")
    if split_record:
        if _digest(split_record["path"]) != split_record["sha256"]:
            raise ValueError("Evaluation rack split modified")
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
    records = []
    for case in cases:
        state, context, position = _rebuild_state(case)
        by_model = {}
        for side in ("base", "final"):
            model = models[side]
            model.choose_match_shot(state, context, position)
            searcher = model.searcher
            _reproduce(searcher.last_root, case["expected"][side])
            root_choices = {}
            for origin in ("base", "final"):
                root_choices[origin] = analyze_edge(
                    searcher, simulator, rules, state, context, state.current_player,
                    case["expected"][origin]["chosen_goal_v3"],
                )
            by_model[side] = {
                "C_choice": root_choices["base"],
                "Cprime_choice": root_choices["final"],
                "relative_ranking": _rank_summary(
                    root_choices["base"], root_choices["final"]
                ),
            }
        record = {
            "seed": case["match_seed"],
            "row": case["row_index"],
            "acting_player": int(state.current_player),
            "models": by_model,
        }
        records.append(record)
        for side in ("base", "final"):
            rank = by_model[side]["relative_ranking"]
            print(
                f"seed={case['match_seed']} row={case['row_index']} "
                f"model={side} one_step_margin={rank['one_step_margin_C_choice_minus_Cprime_choice']} "
                f"depth2_q_margin={rank['depth2_root_q_margin_C_choice_minus_Cprime_choice']} "
                f"reversed={rank['rank_reversed']}",
                flush=True,
            )
    if not records:
        raise ValueError("No comparable cases")
    report = {
        "diagnostic_kind": "phase31_two_ply_value_and_response_audit",
        "compared_states": len(records),
        "seed_rows": [f"{case['seed']}:{case['row']}" for case in records],
        "state_v_models": {
            side: config["checkpoints"][side]["state_v"]["sha256"]
            for side in ("base", "final")
        },
        "rank_reversal_count": {
            side: sum(r["models"][side]["relative_ranking"]["rank_reversed"] is True
                      for r in records)
            for side in ("base", "final")
        },
        "warning": (
            "First-step Value uses the child state AFTER deterministic MCTS "
            "ball-in-hand placement; root Q mixes first-step Value backups "
            "with deeper continuation samples and is not a pure first-shot score. "
            "All states are selected from C-winning games; no match-strength claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "details.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--root-diagnostic-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-row", action="append",
                   help="Repeat SEED:ROW to audit particular recorded cases. "
                        "Defaults to the three previously observed scratches.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
