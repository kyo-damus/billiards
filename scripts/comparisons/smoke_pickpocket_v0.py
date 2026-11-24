"""One decision, not a whole match: DGX integration smoke for PickPocket-v0.

Read a saved physical post-break board from Phase29; select one shot
using noisy sampled physical shots, then run that selected shot through
the same FastFiz + WPA rules. No data or checkpoints are modified.
"""
import argparse
import json
from pathlib import Path

from scripts.evaluation.audit_nine_ball_micro_generalization import sample_from_trajectory
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.agents.baselines.pickpocket import PickPocketAdaptedAgent
from src.game.nine_ball_physics_guard import InvalidPhysicalTransition, simulate_nine_ball_shot
from src.game.nine_ball_rules import NineBallRules
from src.game.rack_state import TableGameConfig
from src.pipelines.experiment_io import write_experiment_json


def first_postbreak_case(folder, *, max_files=8):
    files = sorted(folder.glob("match_seed_*.json"))
    if not files:
        raise ValueError(f"No Phase29 self-play files in {folder}")
    cfg = TableGameConfig.nine_ball_inventory()
    for path in files[:max_files]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("trajectory", []):
            case = sample_from_trajectory(row, cfg)
            if case is None or case.state.break_shot:
                continue
            return path.name, case
    raise ValueError("No Micro-controlled postbreak board in selected saved matches")


def smoke(args):
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    filename, case = first_postbreak_case(args.rollout_dir, max_files=args.max_files)
    agent = PickPocketAdaptedAgent(
        args.micro_checkpoint, device=args.device, samples=args.samples,
        max_candidates=args.max_candidates,
    )
    state, context = case.state, case.context
    plan = agent.choose_match_shot(state, context, None)
    simulator = MultiBallFastFizSimulator(state.config)
    simulator.require_head_string_crossings()
    rules = NineBallRules(wpa_break_rules=True)
    result = {
        "kind": "pickpocket_v0_one_board_smoke",
        "source": filename,
        "break_shot": bool(state.break_shot),
        "planned_goal": plan.macro_goal.kind,
        "target_ball": plan.target_ball,
        "target_pocket": plan.target_pocket,
        "search": agent.last_diagnostics,
        "executed": False,
        "physics_rejected": None,
    }
    try:
        _, outcome = simulate_nine_ball_shot(
            simulator, rules, state, context, target_ball=plan.target_ball, shot=plan.shot
        )
    except InvalidPhysicalTransition as exc:
        result["physics_rejected"] = str(exc)
    else:
        result.update(
            executed=True,
            foul=bool(outcome.foul),
            illegal_break=bool(outcome.illegal_break),
            newly_pocketed=list(outcome.newly_pocketed),
            rack_winner=outcome.rack_winner,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_experiment_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--micro-checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--max-files", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path)
    return p.parse_args(argv)


if __name__ == "__main__":
    smoke(parse_args())
