"""Replay identical Phase31 evaluation positions with C and C-prime MCTS.

This is a read-only *development diagnostic*, NOT a new paired win-rate
evaluation. Both agents receive exactly the same logged pre-shot state.
Records root candidate priors/visits/Q and disagreement; it does not
retrain, alter results, or claim causal proof from observational traces.

Example:
    python -m scripts.experiments.diagnose_phase31_mcts_choices \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --max-states 16 --device cuda \
      --output-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diagnostic_v1
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_evaluation(directory):
    config_path = directory / "evaluation_config.json"
    summary_path = directory / "summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config.get("comparison") != "state_v_only":
        raise ValueError("Expected a completed --state-v-only evaluation directory")
    if summary.get("overall", {}).get("completed") != summary.get("overall", {}).get("matches"):
        raise ValueError("Evaluation must be completed before diagnostic")
    ckpts = config["checkpoints"]
    for name in ("micro", "policy"):
        if ckpts["base"][name]["sha256"] != ckpts["final"][name]["sha256"]:
            raise ValueError(f"{name} changed; not a State-V-only comparison")
    if ckpts["base"]["state_v"]["sha256"] == ckpts["final"]["state_v"]["sha256"]:
        raise ValueError("State V checkpoints are byte-identical")
    for side in ("base", "final"):
        for name in ("micro", "policy", "state_v"):
            metadata = ckpts[side][name]
            if not metadata.get("path") or _digest(metadata["path"]) != metadata["sha256"]:
                raise ValueError(f"Checkpoint mismatch: {side}/{name}")
    return config


def _eligible_rows(game, *, acting_model):
    players = game["player_models"]
    if len(players) != 2 or acting_model not in players:
        raise ValueError("Evaluation game has unexpected player_models")
    slot = players.index(acting_model)
    return [
        (i, row) for i, row in enumerate(game["trajectory"])
        if row.get("player") == slot
        and row.get("state", {}).get("ball_in_hand") is False
        and not row["state"].get("break_shot")
        and row.get("rule_context") is not None
        and row["rule_context"].get("pending_choice") is None
        and row.get("match_context") is not None
        and row.get("chosen_action", {}).get("goal_v3") is not None
    ]


def _sample_cases(directory, *, winner, acting_model, max_states, random_seed):
    rng = random.Random(random_seed)
    files = sorted(directory.glob("seed_*_break_*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluated match files in {directory}")
    rng.shuffle(files)
    cases = []
    for path in files:
        game = json.loads(path.read_text(encoding="utf-8"))
        if game.get("winner") != winner or not game.get("match_finished"):
            continue
        options = _eligible_rows(game, acting_model=acting_model)
        if not options:
            continue
        index, row = rng.choice(options)
        cases.append({
            "file": str(path),
            "match_seed": game["seed"],
            "first_breaker": game["first_breaker"],
            "row_index": index,
            "state": row["state"],
            "rule_context": row["rule_context"],
            "match_context": row["match_context"],
            "logged_goal_v3": row["chosen_action"]["goal_v3"],
        })
        if len(cases) >= max_states:
            break
    if not cases:
        raise ValueError("No eligible non-break physical states in selected evaluation games")
    return cases


def _rebuild_state(case):
    from src.game.match import MatchPosition
    from src.game.nine_ball_rules import NineBallContext
    from src.game.rack_state import RackState, TableGameConfig

    raw = case["state"]
    config = TableGameConfig.nine_ball_inventory()
    if tuple(raw["ball_numbers"]) != config.ball_numbers:
        raise ValueError("Not a nine-ball board")
    state = RackState(
        config=config,
        ball_positions=np.asarray(raw["positions"], dtype=np.float32),
        ball_pocket_indices=np.asarray(raw["pocket_indices"], dtype=np.int32),
        current_player=int(raw["current_player"]),
        score=np.zeros(2, dtype=np.float32),
        ball_in_hand=bool(raw["ball_in_hand"]),
        break_shot=bool(raw["break_shot"]),
        turn_index=int(raw["turn_index"]),
    )
    rules = case["rule_context"]
    context = NineBallContext(
        consecutive_fouls=tuple(rules["consecutive_fouls"]),
        push_out_available=bool(rules["push_out_available"]),
        pending_choice=rules["pending_choice"],
    )
    m = case["match_context"]
    position = MatchPosition(
        race_to=int(m["race_to"]),
        rack_wins=tuple(m["rack_wins"]),
        breaker=int(m["breaker"]),
        break_policy=m["break_policy"],
    )
    return state, context, position


def _root_result(agent, state, context, position):
    from src.game.nine_ball_encoding_v3 import encode_goal_v3

    plan = agent.choose_match_shot(state, context, position)
    report = agent.searcher.last_report
    candidates = [
        {
            "goal_v3": goal,
            "prior": float(prior),
            "visits": int(visits),
            "q": float(q),
            "invalid_reason": invalid,
        }
        for goal, prior, visits, q, invalid in zip(
            report["goal_v3"], report["root_priors"],
            report["root_visits"], report["root_empirical_q"],
            report["root_invalid_reasons"],
            strict=True,
        )
    ]
    return {
        "chosen_index": int(report["chosen_index"]),
        "chosen_goal_v3": encode_goal_v3(plan.macro_goal).tolist(),
        "chosen_kind": plan.macro_goal.kind,
        "chosen_ball": int(plan.macro_goal.target_ball),
        "chosen_pocket": plan.macro_goal.target_pocket,
        "rejected_invalid_simulations": int(report["rejected_invalid_simulations"]),
        "candidates": candidates,
    }


def _goal_key(goal):
    return tuple(float(x) for x in goal)


def compare_roots(base, final):
    """Numerical root comparison; only align candidates with equal goal IDs."""
    original = {_goal_key(c["goal_v3"]): c for c in base["candidates"]}
    updated = {_goal_key(c["goal_v3"]): c for c in final["candidates"]}
    aligned = set(original) == set(updated) and len(original) == len(base["candidates"]) \
        and len(updated) == len(final["candidates"])
    different = _goal_key(base["chosen_goal_v3"]) != _goal_key(final["chosen_goal_v3"])
    out = {
        "chosen_different": different,
        "candidate_sets_equal": aligned,
        "base_selected_visits": base["candidates"][base["chosen_index"]]["visits"],
        "final_selected_visits": final["candidates"][final["chosen_index"]]["visits"],
        "base_selected_q": base["candidates"][base["chosen_index"]]["q"],
        "final_selected_q": final["candidates"][final["chosen_index"]]["q"],
    }
    if aligned:
        keys = sorted(original)
        a = np.array([original[k]["visits"] for k in keys], dtype=float)
        b = np.array([updated[k]["visits"] for k in keys], dtype=float)
        pa = np.array([original[k]["prior"] for k in keys])
        pb = np.array([updated[k]["prior"] for k in keys])
        out["normalized_visit_l1"] = float(
            np.abs(a / max(a.sum(), 1) - b / max(b.sum(), 1)).sum() / 2.0
        )
        out["maximum_prior_difference"] = float(np.max(np.abs(pa - pb)))
        out["maximum_root_q_difference"] = float(
            max(abs(original[k]["q"] - updated[k]["q"]) for k in keys)
        )
    else:
        out["normalized_visit_l1"] = None
        out["maximum_prior_difference"] = None
        out["maximum_root_q_difference"] = None
    return out


def run(args):
    if args.max_states < 1:
        raise ValueError("--max-states must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise ValueError("Set exactly one CUDA_VISIBLE_DEVICES GPU ID")

    config = _validate_evaluation(args.evaluation_dir)
    cases = _sample_cases(
        args.evaluation_dir, winner=args.winner, acting_model=args.acting_model,
        max_states=args.max_states, random_seed=args.random_seed,
    )
    from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
    from src.game.nine_ball_rack_split import load_rack_layout_split
    from src.game.nine_ball_physics_guard import InvalidPhysicalTransition

    split_record = config.get("rack_layout_split")
    split = None
    if split_record:
        if _digest(split_record["path"]) != split_record["sha256"]:
            raise ValueError("Evaluation rack layout split changed")
        split = load_rack_layout_split(split_record["path"])
    search = dict(config["search"])
    if split is not None:
        search["layout_split"] = split
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
    details = []
    errors = []
    for case in cases:
        state, context, position = _rebuild_state(case)
        roots = {}
        try:
            for side, model in models.items():
                roots[side] = _root_result(model, state, context, position)
        except InvalidPhysicalTransition as exc:
            errors.append({
                "file": case["file"], "row_index": case["row_index"],
                "error": str(exc),
            })
            continue
        comparison = compare_roots(roots["base"], roots["final"])
        replay = roots[args.acting_model]
        logged_matches = _goal_key(replay["chosen_goal_v3"]) == \
            _goal_key(case["logged_goal_v3"])
        record = {
            "source_file": case["file"],
            "match_seed": case["match_seed"],
            "row_index": case["row_index"],
            "acting_player": state.current_player,
            "match_rack_wins": list(position.rack_wins),
            "logged_choice_reproduced": logged_matches,
            "comparison": comparison,
            "base": roots["base"],
            "final": roots["final"],
        }
        details.append(record)
        print(
            f"seed={case['match_seed']} row={case['row_index']} "
            f"choice_changed={comparison['chosen_different']} "
            f"visit_L1={comparison['normalized_visit_l1']} "
            f"logged_choice_reproduced={logged_matches}",
            flush=True,
        )
    if not details:
        raise RuntimeError("Every replay failed; diagnostic has no comparable roots")
    aligned = [row["comparison"] for row in details if row["comparison"]["candidate_sets_equal"]]
    summary = {
        "diagnostic_kind": "phase31_identical_state_root_mcts_replay",
        "evaluation_dir": str(args.evaluation_dir),
        "evaluation_config": str(args.evaluation_dir / "evaluation_config.json"),
        "source_winner": args.winner,
        "acting_model": args.acting_model,
        "random_seed": args.random_seed,
        "requested_states": args.max_states,
        "compared_states": len(details),
        "physics_failures": len(errors),
        "chosen_different": sum(row["comparison"]["chosen_different"] for row in details),
        "chosen_disagreement_fraction": sum(
            row["comparison"]["chosen_different"] for row in details
        ) / len(details),
        "aligned_candidate_roots": len(aligned),
        "mean_normalized_visit_l1": (
            float(np.mean([x["normalized_visit_l1"] for x in aligned])) if aligned else None
        ),
        "max_prior_difference": (
            max(x["maximum_prior_difference"] for x in aligned) if aligned else None
        ),
        "logged_choice_reproduced": sum(
            row["logged_choice_reproduced"] for row in details
        ),
        "interpretation": (
            "Development-only same-state search analysis. Differences in root "
            "action/visits are not by themselves proof of why the match was lost. "
            "Original match trajectories are not modified."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "details.json").write_text(
        json.dumps({"cases": details, "errors": errors}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--winner", choices=("base", "final"), default="base",
                        help="Select matches won by this variant (base=C)")
    parser.add_argument("--acting-model", choices=("base", "final"), default="final",
                        help="Replay states where this variant acted (final=C-prime)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
