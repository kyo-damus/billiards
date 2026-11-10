"""Build version-3 nine-ball Macro candidate labels by one-shot FastFiz evaluation.

This is a fixed-Micro, one-step *physics teacher*, NOT a match-return/Q target.
It uses only observed pre-shot states from match trajectories; never uses
recorded next_state or the action outcome to label counterfactual candidates.
No candidate strategy is quota-balanced. Re-running a candidate always starts
from the identical saved state and rules context.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM,
    encode_goal_v3, encode_rack_v3, generate_nine_ball_goals,
)
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.rack_state import RackState, TableGameConfig


def observed_samples(rollout_dir: Path, max_states_per_match: int = 0):
    """Yield (source match seed, pre-shot state, rule context).

    Each input JSON is a *single* collected match. Its match seed is a group
    identifier used for held-out splitting later; never split individual
    candidates from one board between training and validation.
    """
    files = sorted(Path(rollout_dir).glob("match_seed_*.json"))
    if not files:
        raise ValueError(f"No match_seed_*.json in {rollout_dir}")
    config = TableGameConfig.nine_ball_inventory()
    for file in files:
        match_seed = int(file.stem.removeprefix("match_seed_"))
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("run_type") not in (
            "nonlearned_geometry_self_play", "learned_micro_self_play",
        ):
            raise ValueError(f"Unexpected run_type in {file}")
        seen = 0
        for record in data.get("trajectory", []):
            raw = record["state"]
            if tuple(raw["ball_numbers"]) != config.ball_numbers:
                raise ValueError(f"Non-nine-ball inventory in {file}")
            # The collector records state AFTER any legal cue placement.
            if raw["break_shot"] or raw["ball_in_hand"]:
                continue
            if int(raw["pocket_indices"][0]) >= 0:
                continue
            state = RackState(
                config=config,
                ball_positions=np.asarray(raw["positions"], dtype=np.float32),
                ball_pocket_indices=np.asarray(
                    raw["pocket_indices"], dtype=np.int32,
                ),
                current_player=int(raw["current_player"]),
                score=np.zeros(2, dtype=np.float32),
                ball_in_hand=False,
                break_shot=False,
                turn_index=int(raw["turn_index"]),
            )
            context_raw = record.get("rule_context")
            context = NineBallContext() if context_raw is None else NineBallContext(
                consecutive_fouls=tuple(
                    int(n) for n in context_raw["consecutive_fouls"]
                ),
                push_out_available=bool(context_raw["push_out_available"]),
            )
            yield match_seed, state, context
            seen += 1
            if max_states_per_match and seen >= max_states_per_match:
                break


def one_step_score(goal, state, physical, result):
    """Declared immediate teacher; not a probability, state V, or match Q.

    Legal correct-pocket pots > legal wrong-pocket pots > legal misses.
    POSITION receives a small cue-region bonus *only after* a correct pot.
    SAFETY remains a neutral legal-shot baseline until opponent threat
    and continuation values are integrated.
    """
    if result.foul:
        # Three-foul losses are worse than other fouls.
        return -5.0 if result.rack_winner is not None else -2.0

    if result.rack_winner == state.current_player:
        return 5.0

    if goal.kind == "SAFETY":
        return 0.0

    if goal.kind not in ("DIRECT_ATTACK", "POSITION_ATTACK"):
        return 0.0

    made_target = goal.target_ball in result.newly_pocketed
    if not made_target:
        return 0.0
    ball_index = state.array_index_for_ball(goal.target_ball)
    correct_pocket = (
        goal.target_pocket is not None
        and int(physical.pocket_indices[ball_index]) == goal.target_pocket
    )
    score = 2.0 if correct_pocket else 0.4
    if correct_pocket and goal.cue_stop_xy is not None:
        distance = float(np.linalg.norm(
            np.asarray(physical.positions[0], dtype=np.float64)
            - np.asarray(goal.cue_stop_xy, dtype=np.float64)
        ))
        score += 0.25 * max(
            0.0, 1.0 - distance / float(goal.cue_stop_radius)
        )
    return score


def evaluate_goal(state, context, goal, *, agent, simulator, rules):
    """Predict one physical outcome for one candidate using fixed 5D Micro."""
    plan = agent.plan_goal(state, goal, context)
    simulator.restore(state)  # isolation is essential: no candidate leakage
    physical = simulator.step(target_ball=plan.target_ball, shot=plan.shot)
    result = rules.apply(state, physical, context)
    idx = state.array_index_for_ball(goal.target_ball)
    exact = (
        not result.foul
        and goal.target_pocket is not None
        and goal.target_ball in result.newly_pocketed
        and int(physical.pocket_indices[idx]) == goal.target_pocket
    )
    return {
        "score": float(one_step_score(goal, state, physical, result)),
        "foul": bool(result.foul),
        "exact_target_pot": bool(exact),
        "rack_win": result.rack_winner == state.current_player,
        "turn_changed": bool(result.turn_changed),
    }


def build(args, *, agent=None, simulator=None, rules=None):
    if args.max_states_per_match < 0:
        raise ValueError("--max-states-per-match must be >= 0")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    agent = agent or NineBallMicroSACAgent(
        args.micro_checkpoint, device=args.device, deterministic=True,
    )
    simulator = simulator or MultiBallFastFizSimulator(
        TableGameConfig.nine_ball_inventory()
    )
    rules = rules or NineBallRules()
    states, goals, scores, match_ids, offsets = [], [], [], [], [0]
    foul, exact, win, changed = [], [], [], []
    sample_count = 0
    for match_seed, state, context in observed_samples(
        args.rollout_dir, args.max_states_per_match
    ):
        candidates = generate_nine_ball_goals(state)
        features = [encode_goal_v3(g) for g in candidates]
        evaluations = [
            evaluate_goal(
                state, context, g, agent=agent, simulator=simulator, rules=rules,
            )
            for g in candidates
        ]
        states.append(encode_rack_v3(state, context))
        goals.extend(features)
        scores.extend(e["score"] for e in evaluations)
        match_ids.append(match_seed)
        offsets.append(len(goals))
        foul.extend(e["foul"] for e in evaluations)
        exact.extend(e["exact_target_pot"] for e in evaluations)
        win.extend(e["rack_win"] for e in evaluations)
        changed.extend(e["turn_changed"] for e in evaluations)
        sample_count += 1
    if not states:
        raise ValueError("No usable non-break pre-shot states")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=np.asarray(states, dtype=np.float32).reshape(-1, MACRO_STATE_V3_DIM),
        candidates=np.asarray(goals, dtype=np.float32).reshape(-1, MACRO_GOAL_V3_DIM),
        teacher_scores=np.asarray(scores, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        match_seeds=np.asarray(match_ids, dtype=np.int64),
        foul=np.asarray(foul, dtype=np.bool_),
        exact_target_pot=np.asarray(exact, dtype=np.bool_),
        rack_win=np.asarray(win, dtype=np.bool_),
        turn_changed=np.asarray(changed, dtype=np.bool_),
    )
    summary = {
        "format_version": 3,
        "teacher": "fixed_micro_one_step_fastfiz",
        "not_a": ["MCTS", "match_return", "learned_Q", "win_rate"],
        "micro_checkpoint": str(args.micro_checkpoint),
        "rollout_dir": str(args.rollout_dir),
        "states": sample_count,
        "candidates": len(goals),
        "match_seeds": sorted(set(match_ids)),
        "foul_rate_across_candidates": float(np.mean(foul)),
        "exact_pot_rate_across_candidates": float(np.mean(exact)),
        "score_contract": (
            "foul -2, three-foul loss -5, legal rack win +5, "
            "correct target pocket +2, other target pocket +0.4, "
            "correct-pot POSITION cue stop up to +0.25; legal SAFETY 0. "
            "No opponent follow-up or learned match value."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--micro-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-states-per-match", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
