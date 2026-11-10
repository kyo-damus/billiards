"""Collect actual 9-ball physical self-play trajectories, no learned policy yet.

This is a baseline wiring run with a geometric placeholder, not a final
hierarchical RL self-play result. Unfinished racks/matches are never wins.
"""

import argparse
import json
from pathlib import Path
from functools import partial

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.match import MatchConfig
from src.game.nine_ball_agents import GeometryNineBallAgent
from src.game.nine_ball_encoding_v3 import encode_goal_v3
from src.game.nine_ball_session import NineBallMatchSession


def rack_state_json(state):
    """Keep enough state for future supervised/return dataset conversion."""
    return {
        "ball_numbers": list(state.config.ball_numbers),
        "positions": state.ball_positions.tolist(),
        "pocket_indices": state.ball_pocket_indices.tolist(),
        "current_player": int(state.current_player),
        "ball_in_hand": bool(state.ball_in_hand),
        "break_shot": bool(state.break_shot),
        "turn_index": int(state.turn_index),
    }


def play_match(
    match,
    agents,
    *,
    max_shots_per_rack: int,
):
    """Consume one match with interchangeable policies and bounded rack length."""
    if max_shots_per_rack < 1:
        raise ValueError("max_shots_per_rack must be positive")
    if len(agents) != 2:
        raise ValueError("Expected exactly two player agents.")
    records = []
    rack_shots = 0
    while not match.match.snapshot().finished:
        if rack_shots >= max_shots_per_rack:
            break  # No invented winner for a truncated rack.
        before = match.state
        player = int(before.current_player)
        agent = agents[player]
        cue_placement = None
        if before.ball_in_hand:
            for candidate in agent.cue_placement_candidates(before):
                try:
                    match.place_cue_ball(*candidate)
                    cue_placement = list(candidate)
                    break
                except ValueError as exc:
                    if (
                        "overlaps" not in str(exc)
                        and "outside playing surface" not in str(exc)
                    ):
                        raise
            if cue_placement is None:
                raise RuntimeError("No valid cue placement found for ball-in-hand.")
        state = match.state
        context = getattr(match.rack, "context", None)
        context_before = {
            "consecutive_fouls": list(context.consecutive_fouls),
            "push_out_available": bool(context.push_out_available),
        } if context is not None else None
        plan = agent.choose_shot(state, context=context)
        result = match.step(target_ball=plan.target_ball, shot=plan.shot)
        physical = getattr(
            match, "last_physical",
            getattr(match.rack, "last_physical", None),
        )
        records.append({
            "rack_index": int(match.match.snapshot().racks_completed)
                - (1 if result.rack_winner is not None else 0),
            "player": player,
            "state": rack_state_json(state),
            "rule_context": context_before,
            "cue_placement": cue_placement,
            "chosen_action": {
                "goal_label": plan.goal_label,
                "goal_v3": (
                    encode_goal_v3(plan.macro_goal).tolist()
                    if plan.macro_goal is not None else None
                ),
                "target_ball": int(plan.target_ball),
                "target_pocket": plan.target_pocket,
                **plan.shot.to_fastfiz_kwargs(),
            },
            "physical": {
                "first_contact_ball": (
                    physical.first_contact_ball if physical is not None else None
                ),
                "precondition_code": (
                    int(physical.precondition_code) if physical is not None else None
                ),
                "events": [
                    {
                        "time_s": float(e.time_s),
                        "kind": e.kind,
                        "ball1": int(e.ball1),
                        "ball2": e.ball2,
                    }
                    for e in physical.events
                ] if physical is not None else [],
            },
            "rule_result": {
                "foul": bool(result.foul),
                "foul_reasons": list(result.foul_reasons),
                "newly_pocketed": list(result.newly_pocketed),
                "rack_winner": result.rack_winner,
                "turn_changed": bool(result.turn_changed),
            },
            "next_state": rack_state_json(result.next_state),
            "next_rule_context": (
                {
                    "consecutive_fouls": list(result.next_context.consecutive_fouls),
                    "push_out_available": bool(result.next_context.push_out_available),
                }
                if hasattr(result, "next_context") else None
            ),
        })
        rack_shots += 1
        if result.rack_winner is not None:
            rack_shots = 0

    snapshot = match.match.snapshot()
    return {
        "match_finished": bool(snapshot.finished),
        "match_winner": snapshot.match_winner,
        "rack_wins": list(snapshot.rack_wins),
        "racks_completed": int(snapshot.racks_completed),
        "stop_reason": "race_to_reached" if snapshot.finished else "rack_shot_limit",
        "shots": len(records),
        "fouls": sum(int(row["rule_result"]["foul"]) for row in records),
        "rack_history": list(match.rack_history),
        "trajectory": records,
    }


def collect_matches(
    *,
    matches: int,
    race_to: int,
    start_seed: int,
    max_shots_per_rack: int,
    output_dir: Path,
    simulator_factory=MultiBallFastFizSimulator,
    agent_factory=GeometryNineBallAgent,
    match_factory=None,
):
    """Write each match once; fail if output directory already exists."""
    if matches < 1 or race_to < 1 or max_shots_per_rack < 1:
        raise ValueError("matches, race_to and max_shots_per_rack must be >=1")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    agents = (agent_factory(), agent_factory())
    players = [type(agent).__name__ for agent in agents]
    if players == ["GeometryNineBallAgent", "GeometryNineBallAgent"]:
        run_type = "nonlearned_geometry_self_play"
    else:
        run_type = "learned_micro_self_play"
    summaries = []
    for index in range(matches):
        # Reserve 2*race_to seeds per match. A race-to-N lasts at most
        # 2*N-1 racks, so independently collected matches do not share racks.
        seed = start_seed + index * (2 * race_to)
        if match_factory is None:
            match = NineBallMatchSession(
                simulator_factory=simulator_factory,
                match_config=MatchConfig(
                    race_to=race_to,
                    rule_id="nine_ball",
                    first_breaker=index % 2,
                ),
                seed=seed,
            )
        else:
            match = match_factory(seed)
        outcome = play_match(
            match, agents, max_shots_per_rack=max_shots_per_rack,
        )
        payload = {
            "format_version": 1,
            "run_type": run_type,
            "seed": seed,
            "race_to": race_to,
            "max_shots_per_rack": max_shots_per_rack,
            "players": players,
            **outcome,
        }
        file = output_dir / f"match_seed_{seed}.json"
        with file.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        summaries.append({
            k: v for k, v in payload.items()
            if k not in ("trajectory", "rack_history")
        })
        print(
            f"seed={seed} finished={outcome['match_finished']} "
            f"winner={outcome['match_winner']} "
            f"rack_wins={outcome['rack_wins']} "
            f"shots={outcome['shots']} fouls={outcome['fouls']}"
        )
    result = {
        "run_type": run_type,
        "matches": matches,
        "finished_matches": sum(s["match_finished"] for s in summaries),
        "unfinished_matches": sum(not s["match_finished"] for s in summaries),
        "racks_completed": sum(s["racks_completed"] for s in summaries),
        "shots": sum(s["shots"] for s in summaries),
        "fouls": sum(s["fouls"] for s in summaries),
        "per_match": summaries,
        "note": (
            "Partial nine-ball rules. Model kind is recorded in run_type/players. "
            "Do not infer final hierarchical-RL performance or official win rate."
        ),
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--race-to", type=int, default=2)
    p.add_argument("--start-seed", type=int, default=310000)
    p.add_argument("--max-shots-per-rack", type=int, default=60)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--agent-kind", choices=("geometry", "micro_sac"), default="geometry")
    p.add_argument("--micro-checkpoint", type=Path)
    p.add_argument("--macro-policy-checkpoint", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    agent_factory = GeometryNineBallAgent
    if args.agent_kind == "micro_sac":
        if args.micro_checkpoint is None:
            raise ValueError("--micro-checkpoint is required for micro_sac")
        if args.device == "cuda":
            import os
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible or len(visible.split(",")) != 1:
                raise ValueError("Shared GPU: set exactly one CUDA_VISIBLE_DEVICES ID")
        from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
        agent_factory = partial(
            NineBallMicroSACAgent,
            micro_checkpoint=args.micro_checkpoint,
            macro_policy_checkpoint=args.macro_policy_checkpoint,
            device=args.device,
        )
    elif args.micro_checkpoint is not None or args.macro_policy_checkpoint is not None:
        raise ValueError("Checkpoints require --agent-kind micro_sac")
    summary = collect_matches(
        matches=args.matches,
        race_to=args.race_to,
        start_seed=args.start_seed,
        max_shots_per_rack=args.max_shots_per_rack,
        output_dir=args.output_dir,
        agent_factory=agent_factory,
    )
    print(
        "=== Completed nine-ball data collection ===\n"
        f"finished={summary['finished_matches']}/{summary['matches']} "
        f"unfinished={summary['unfinished_matches']} "
        f"racks={summary['racks_completed']} shots={summary['shots']}"
    )
