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
from src.game.nine_ball_physics_guard import InvalidPhysicalTransition


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
    physics_failure = None
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
        try:
            plan = agent.choose_shot(state, context=context)
            result = match.step(target_ball=plan.target_ball, shot=plan.shot)
        except InvalidPhysicalTransition as exc:
            # This is a simulator failure, NOT a billiards foul or winner.
            # Keep earlier valid turns, flag the match as unfinished, and
            # proceed to the next independent seed.
            physics_failure = {
                "error": str(exc),
                "player": player,
                "state": rack_state_json(state),
                "rule_context": context_before,
                "shot_index": len(records),
            }
            break
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
                "macro_search": plan.macro_search,
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
        "stop_reason": (
            "invalid_physics" if physics_failure is not None
            else "race_to_reached" if snapshot.finished
            else "rack_shot_limit"
        ),
        "physics_failure": physics_failure,
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
    resume: bool = False,
    learned_safety: bool = False,
    outcome_position: bool = False,
):
    """Write each match once; optional validated completion of interrupted batch.

    This method cannot verify source code/checkpoint history by itself.
    Use the higher-level hierarchical runner for provenance-aware recovery.
    """
    if matches < 1 or race_to < 1 or max_shots_per_rack < 1:
        raise ValueError("matches, race_to and max_shots_per_rack must be >=1")
    output_dir = Path(output_dir)
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory absent: {output_dir}")
        if (output_dir / "summary.json").exists():
            raise FileExistsError("Refusing to resume completed collector summary")
    else:
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
    allowed_seeds = {
        start_seed + index * (2 * race_to) for index in range(matches)
    }
    existing = {}
    if resume:
        for file in output_dir.glob("match_seed_*.json"):
            try:
                seed_from_name = int(file.stem.removeprefix("match_seed_"))
            except ValueError as exc:
                raise ValueError(f"Malformed seed filename: {file}") from exc
            if seed_from_name not in allowed_seeds:
                raise ValueError(f"Unexpected seed in resumed run: {file}")
            payload = json.loads(file.read_text(encoding="utf-8"))
            if (
                payload.get("seed") != seed_from_name
                or payload.get("race_to") != race_to
                or payload.get("max_shots_per_rack") != max_shots_per_rack
                or payload.get("run_type") != run_type
                or payload.get("players") != players
                or payload.get("learned_safety", False) != learned_safety
                or payload.get("outcome_position", False) != outcome_position
                or payload.get("format_version") != 1
                or not isinstance(payload.get("trajectory"), list)
                or not isinstance(payload.get("match_finished"), bool)
                or payload.get("shots") != len(payload["trajectory"])
            ):
                raise ValueError(f"Resume file identity/content mismatch: {file}")
            existing[seed_from_name] = payload
    for index in range(matches):
        # Reserve 2*race_to seeds per match; no independent match shares a rack seed.
        seed = start_seed + index * (2 * race_to)
        if seed in existing:
            payload = existing[seed]
            summaries.append({
                k: v for k, v in payload.items()
                if k not in ("trajectory", "rack_history")
            })
            print(f"seed={seed} reused=existing (read-only)", flush=True)
            continue
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
            "learned_safety": learned_safety,
            "outcome_position": outcome_position,
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
        "learned_safety": learned_safety,
        "outcome_position": outcome_position,
        "finished_matches": sum(s["match_finished"] for s in summaries),
        "unfinished_matches": sum(not s["match_finished"] for s in summaries),
        "racks_completed": sum(s["racks_completed"] for s in summaries),
        "shots": sum(s["shots"] for s in summaries),
        "fouls": sum(s["fouls"] for s in summaries),
        "invalid_physics_matches": sum(
            s.get("stop_reason") == "invalid_physics" for s in summaries
        ),
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
    p.add_argument("--resume", action="store_true",
                   help="Complete only missing match_seed files in an incomplete directory")
    p.add_argument("--agent-kind", choices=("geometry", "micro_sac", "micro_mcts"), default="geometry")
    p.add_argument("--learned-safety", action="store_true")
    p.add_argument("--outcome-position", action="store_true")
    p.add_argument("--micro-checkpoint", type=Path)
    p.add_argument("--macro-policy-checkpoint", type=Path)
    p.add_argument("--state-value-checkpoint", type=Path)
    p.add_argument("--mcts-simulations", type=int, default=24)
    p.add_argument("--mcts-max-depth", type=int, default=2)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--mcts-gamma", type=float, default=0.99)
    p.add_argument("--mcts-root-min-visits", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    agent_factory = GeometryNineBallAgent
    if args.agent_kind in ("micro_sac", "micro_mcts"):
        if args.micro_checkpoint is None:
            raise ValueError("--micro-checkpoint is required for learned agents")
        if args.device == "cuda":
            import os
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible or len(visible.split(",")) != 1:
                raise ValueError("Shared GPU: set exactly one CUDA_VISIBLE_DEVICES ID")
        if args.agent_kind == "micro_sac":
            if args.state_value_checkpoint is not None:
                raise ValueError("State V checkpoint requires --agent-kind micro_mcts")
            from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
            agent_factory = partial(
                NineBallMicroSACAgent,
                micro_checkpoint=args.micro_checkpoint,
                macro_policy_checkpoint=args.macro_policy_checkpoint,
                learned_safety=args.learned_safety,
                outcome_position=args.outcome_position,
                device=args.device,
            )
        else:
            from src.game.nine_ball_mcts_agent import NineBallMCTSAgent
            agent_factory = partial(
                NineBallMCTSAgent,
                micro_checkpoint=args.micro_checkpoint,
                macro_policy_checkpoint=args.macro_policy_checkpoint,
                state_value_checkpoint=args.state_value_checkpoint,
                simulations=args.mcts_simulations,
                max_depth=args.mcts_max_depth,
                c_puct=args.mcts_c_puct,
                gamma=args.mcts_gamma,
                root_min_visits=args.mcts_root_min_visits,
                learned_safety=args.learned_safety,
                outcome_position=args.outcome_position,
                device=args.device,
            )
    elif any(v is not None for v in (
        args.micro_checkpoint, args.macro_policy_checkpoint,
        args.state_value_checkpoint,
    )) or args.learned_safety or args.outcome_position:
        raise ValueError("Checkpoints and learned goal options require a learned --agent-kind")
    summary = collect_matches(
        matches=args.matches,
        race_to=args.race_to,
        start_seed=args.start_seed,
        max_shots_per_rack=args.max_shots_per_rack,
        output_dir=args.output_dir,
        agent_factory=agent_factory,
        resume=args.resume,
        learned_safety=args.learned_safety,
        outcome_position=args.outcome_position,
    )
    print(
        "=== Completed nine-ball data collection ===\n"
        f"finished={summary['finished_matches']}/{summary['matches']} "
        f"unfinished={summary['unfinished_matches']} "
        f"racks={summary['racks_completed']} shots={summary['shots']}"
    )
