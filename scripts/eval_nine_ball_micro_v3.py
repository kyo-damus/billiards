"""Compare two deterministic nine-ball Micro actors on the same held-out goals."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_nine_ball_micro_sac import (
    execute_valid_training_shot, load_training_states, micro_reward,
    position_goal_success, sample_keys, _continues_position_goal, _cue_stop_distance,
)
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_physics_guard import (
    CueClearanceExhausted, InvalidPhysicalTransition, NativeShotRejected,
    is_cue_stick_collision, simulate_with_cue_clearance,
)
from src.game.nine_ball_rules import NineBallRules
from src.game.rack_state import TableGameConfig
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER
from src.pipelines.experiment_io import write_experiment_json


def evaluation_cases(samples, max_cases_per_goal, seed):
    cases, seen = {}, set()
    for sample in samples:
        _, key = sample_keys(sample)
        if key not in seen:
            seen.add(key)
            cases.setdefault(sample.selected_goal.kind, []).append(sample)
    rng = np.random.default_rng(seed)
    return {
        kind: entries if len(entries) <= max_cases_per_goal else [
            entries[index] for index in sorted(
                rng.choice(len(entries), max_cases_per_goal, replace=False)
            )
        ]
        for kind, entries in cases.items()
    }


def record_trial(agent, sample, simulator, rules, counts, rejections, *, outcome_position, cue_clearance_retry, position_details=None):
    state, goal, context = sample.state, sample.selected_goal, sample.context
    plan = agent.plan_goal(state, goal, context)
    counts["attempted"] += 1
    simulate = partial(execute_valid_training_shot, simulator, rules, state, goal, context)
    trials = 0
    try:
        if cue_clearance_retry:
            _, physical, outcome, trials = simulate_with_cue_clearance(simulate, plan.shot)
        else:
            physical, outcome = simulate(plan.shot)
    except InvalidPhysicalTransition as exc:
        counts["physics_rejected"] += 1
        initial = exc.rejected if isinstance(exc, CueClearanceExhausted) else exc
        if is_cue_stick_collision(initial):
            counts["initial_cue_obstructions"] += 1
        reason = str(initial).split(":", 1)[0]
        if isinstance(initial, NativeShotRejected):
            reason = f"native precondition={initial.precondition}, placement={initial.placement}"
        rejections[reason] += 1
        return False
    counts["executed"] += 1
    counts["fouls"] += int(outcome.foul)
    legal = not outcome.foul and not getattr(outcome, "illegal_break", False)
    counts["legal"] += int(legal)
    if trials:
        counts["initial_cue_obstructions"] += 1
        counts["recovered_executable"] += 1
        counts["recovered_legal"] += int(legal)
    _, success = micro_reward(goal, state, physical, outcome, outcome_position=outcome_position)
    if goal.kind in ("DIRECT_ATTACK", "POSITION_ATTACK"):
        counts["exact_target_pot"] += int(success)
    if goal.kind == "POSITION_ATTACK":
        if position_details is not None and success:
            # This block only runs after a *legal, exact-pocket pot*.
            # Failures before that are accounted for in existing counts.
            continued = _continues_position_goal(state, goal, outcome)
            distance_m = _cue_stop_distance(goal, physical)
            within_goal = distance_m <= goal.cue_stop_radius
            position_details["exact_pots"] += 1
            position_details["continued_after_pot"] += int(continued)
            position_details["within_goal_radius_after_pot"] += int(within_goal)
            position_details["continued_and_within_goal_radius"] += int(continued and within_goal)
            position_details["stop_distances_after_pot_m"].append(distance_m)
            position_details["relative_stop_distances_after_pot"].append(
                distance_m / goal.cue_stop_radius
            )
            if continued:
                position_details["stop_distances_after_continued_pot_m"].append(distance_m)
                position_details["relative_stop_distances_after_continued_pot"].append(
                    distance_m / goal.cue_stop_radius
                )
        success = position_goal_success(state, goal, physical, outcome, exact_pot=success)
    counts["goal_success"] += int(success)
    return success


def evaluate(args, *, simulator_factory=MultiBallFastFizSimulator, agent_factory=NineBallMicroSACAgent):
    if args.max_cases_per_goal < 1:
        raise ValueError("--max-cases-per-goal must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    samples = load_training_states(
        args.rollout_dir, goal_source=args.goal_source, learned_safety=args.learned_safety,
        learned_break=args.learned_break, outcome_position=args.outcome_position,
        wpa_break_rules=args.wpa_break_rules,
    )
    cases = evaluation_cases(samples, args.max_cases_per_goal, args.seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    agents = {
        "base": agent_factory(args.base_checkpoint, device="cpu"),
        "candidate": agent_factory(args.checkpoint, device="cpu"),
    }
    simulator = simulator_factory(TableGameConfig.nine_ball_inventory())
    if args.wpa_break_rules:
        simulator.require_head_string_crossings()
        for agent in agents.values():
            agent.break_decoder = WPA_NINE_BALL_BREAK_DECODER
    rules = NineBallRules(wpa_break_rules=args.wpa_break_rules)
    results = {}
    for kind, entries in cases.items():
        columns = (
            "attempted", "executed", "physics_rejected", "fouls", "legal", "goal_success",
            "initial_cue_obstructions", "recovered_executable", "recovered_legal",
        )
        if kind in ("DIRECT_ATTACK", "POSITION_ATTACK"):
            columns += ("exact_target_pot",)
        counts = {name: Counter(dict.fromkeys(columns, 0)) for name in agents}
        rejections = {name: Counter() for name in agents}
        paired_success = Counter(both=0, base_only=0, candidate_only=0, neither=0)
        for sample in entries:
            base, candidate = (
                record_trial(agent, sample, simulator, rules, counts[name], rejections[name],
                             outcome_position=args.outcome_position,
                             cue_clearance_retry=args.cue_clearance_retry)
                for name, agent in agents.items()
            )
            paired_success[{
                (True, True): "both", (True, False): "base_only",
                (False, True): "candidate_only", (False, False): "neither",
            }[base, candidate]] += 1
        results[kind] = {
            "cases": len(entries),
            **{
                name: {
                    **metrics,
                    "physics_rejection_reasons": dict(rejections[name]),
                    "goal_success_rate": metrics["goal_success"] / len(entries),
                    "legal_rate": metrics["legal"] / len(entries),
                }
                for name, metrics in counts.items()
            },
            "paired_goal_success": dict(paired_success),
        }
    summary = {
        "rollout_dir": str(args.rollout_dir),
        "base_checkpoint": str(args.base_checkpoint),
        "candidate_checkpoint": str(args.checkpoint),
        "goal_source": args.goal_source,
        "max_cases_per_goal": args.max_cases_per_goal,
        "sample_seed": args.seed,
        "wpa_break_rules": args.wpa_break_rules,
        "outcome_position": args.outcome_position,
        "cue_clearance_retry": args.cue_clearance_retry,
        "success_definition": {
            "DIRECT_ATTACK": "Legal exact target pot.",
            "POSITION_ATTACK": "Legal exact target pot, continuation to the next ball, and cue-ball stop within the target radius.",
            "SAFETY": "Legal handoff that reduces geometric direct-pot options.",
            "BREAK": "Legal break with a pot.",
        },
        "rate_denominator": "All board/goal cases, including unresolved physical rejections.",
        "goals": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_json(args.output, summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goal-source", choices=("candidates", "logged"), default="candidates")
    parser.add_argument("--max-cases-per-goal", type=int, default=256)
    parser.add_argument("--seed", type=int, default=280001)
    parser.add_argument("--learned-safety", action="store_true")
    parser.add_argument("--learned-break", action="store_true")
    parser.add_argument("--wpa-break-rules", action="store_true")
    parser.add_argument("--outcome-position", action="store_true")
    parser.add_argument("--cue-clearance-retry", action="store_true",
                        help="Use the same cue-obstruction retry as MCTS")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2))
