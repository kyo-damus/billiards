"""Train a *new*, versioned 5D goal-conditioned SAC on real 9-ball shots.

Each RL episode is one physical shot from an observed 9-ball state; the
candidate goal is sampled for Micro pretraining and NOT the Macro policy.
This does not train Macro Policy/Q/V or demonstrate game-level competence.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MICRO_CONDITIONED_V3_DIM, conditioned_observation_v3,
    generate_nine_ball_goals, reference_angle_deg,
)
from src.game.nine_ball_rules import NineBallContext, NineBallRules, TABLE_LENGTH_M, TABLE_WIDTH_M
from src.game.rack_state import RackState, TableGameConfig
from src.micro.replay_buffer import ReplayBuffer
from src.micro.sac import SACAgent
from src.micro.shot_control import Shot5DDecoder


@dataclass(frozen=True)
class TrainingSample:
    state: RackState
    context: NineBallContext


def load_training_states(rollout_dir: Path) -> tuple[TrainingSample, ...]:
    """Load only observed pre-shot states; reject simulated target leakage."""
    rollout_dir = Path(rollout_dir)
    files = sorted(rollout_dir.glob("match_seed_*.json"))
    if not files:
        raise ValueError(f"No match_seed_*.json files in {rollout_dir}")
    training = []
    cfg = TableGameConfig.nine_ball_inventory()
    for path in files:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("run_type") not in (
            "nonlearned_geometry_self_play",
            "learned_micro_self_play",
        ):
            raise ValueError(f"Unexpected rollout kind in {path}")
        for row in data.get("trajectory", []):
            raw = row["state"]
            if tuple(raw["ball_numbers"]) != cfg.ball_numbers:
                raise ValueError(f"Expected cue + numbered 1..9 balls in {path}")
            if raw["ball_in_hand"]:
                # The collector logs the board after cue placement, so
                # normally this does not occur; no implicit placement here.
                continue
            state = RackState(
                config=cfg,
                ball_positions=np.asarray(raw["positions"], dtype=np.float32),
                ball_pocket_indices=np.asarray(raw["pocket_indices"], dtype=np.int32),
                current_player=int(raw["current_player"]),
                score=np.zeros(2, dtype=np.float32),
                ball_in_hand=False,
                break_shot=bool(raw["break_shot"]),
                turn_index=int(raw["turn_index"]),
            )
            if state.ball_pocket_indices[0] >= 0:
                continue
            raw_ctx = row.get("rule_context")
            if raw_ctx is None:
                context = NineBallContext()
            else:
                context = NineBallContext(
                    consecutive_fouls=tuple(
                        int(n) for n in raw_ctx["consecutive_fouls"]
                    ),
                    push_out_available=bool(raw_ctx["push_out_available"]),
                )
            if not state.break_shot and state.on_table_object_balls():
                training.append(TrainingSample(state, context))
    if not training:
        raise ValueError("No usable non-break pre-shot states in the rollouts")
    return tuple(training)


def micro_reward(goal, before, physical, outcome) -> tuple[float, bool]:
    """Physics-grounded one-shot reward (not the final match objective)."""
    goal_id = before.array_index_for_ball(goal.target_ball)
    pocket = int(physical.pocket_indices[goal_id])
    exact = bool(
        goal.target_pocket is not None
        and before.ball_pocket_indices[goal_id] == -1
        and pocket == goal.target_pocket
    )
    reward = -1.2 if outcome.foul else 0.1
    if exact and not outcome.foul:
        reward += 2.0
    elif goal.target_ball in outcome.newly_pocketed and not outcome.foul:
        reward += 0.4
    if physical.first_contact_ball != min(before.on_table_object_balls()):
        reward -= 0.5
    if goal.cue_stop_xy is not None and not outcome.foul:
        cue = np.asarray(physical.positions[0], dtype=np.float64)
        desired = np.asarray(goal.cue_stop_xy, dtype=np.float64)
        d = float(np.linalg.norm(cue - desired))
        reward += 0.5 * max(0.0, 1.0 - d / goal.cue_stop_radius)
    return float(reward), exact and not outcome.foul


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError(
                "Shared server: explicitly set CUDA_VISIBLE_DEVICES to one free GPU"
            )
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
            raise ValueError("Micro training must use exactly one visible GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda:0"
    raise ValueError("Select --device cpu or cuda explicitly")


def train(args, *, simulator_factory=MultiBallFastFizSimulator):
    """Run one-step off-policy SAC with an explicitly bounded GPU allocation."""
    if args.steps <= 0 or args.batch_size <= 0 or args.warmup_steps < 0:
        raise ValueError("Invalid steps, batch size or warmup")
    if args.updates_per_step < 1 or args.replay_capacity < args.batch_size:
        raise ValueError("Invalid replay capacity or update count")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    samples = load_training_states(args.rollout_dir)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    rng = np.random.default_rng(args.seed)
    agent = SACAgent(
        state_dim=MICRO_CONDITIONED_V3_DIM,
        action_dim=5,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        device=device,
    )
    replay = ReplayBuffer(capacity=args.replay_capacity)
    simulator = simulator_factory(TableGameConfig.nine_ball_inventory())
    rules = NineBallRules()
    decoder = Shot5DDecoder()
    logs = []
    window_rewards, window_success = [], []
    print(
        f"=== Nine-ball 5D goal-conditioned Micro SAC ===\n"
        f"device={device}, training_states={len(samples)}, steps={args.steps}, "
        "action_dim=5, version=3 (NOT old 2D Micro)"
    )
    for step in range(1, args.steps + 1):
        sample = samples[int(rng.integers(0, len(samples)))]
        state, context = sample.state, sample.context
        candidate_goals = tuple(
            g for g in generate_nine_ball_goals(state)
            if g.kind in ("DIRECT_ATTACK", "POSITION_ATTACK")
        )
        if not candidate_goals:
            continue
        goal = candidate_goals[int(rng.integers(0, len(candidate_goals)))]
        encoded = conditioned_observation_v3(state, goal, context)
        if step <= args.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        else:
            action = agent.select_action(encoded, deterministic=False)
        shot = decoder.decode(action, reference_angle_deg(state, goal))
        simulator.restore(state)
        physical = simulator.step(target_ball=goal.target_ball, shot=shot)
        outcome = rules.apply(state, physical, context)
        reward, exact = micro_reward(goal, state, physical, outcome)
        replay.push(
            encoded, action, reward, np.zeros_like(encoded), True
        )
        losses = None
        if len(replay) >= args.batch_size:
            for _ in range(args.updates_per_step):
                losses = agent.update(replay.sample(args.batch_size, device))
        window_rewards.append(reward)
        window_success.append(float(exact))
        if step == args.steps or step % args.report_interval == 0:
            log = {
                "step": step,
                "mean_reward": float(np.mean(window_rewards)),
                "exact_target_pot_rate": float(np.mean(window_success)),
                "replay_size": len(replay),
                "alpha": float(agent.alpha.item()),
                "actor_loss": losses["actor_loss"] if losses else None,
                "critic_loss": losses["critic_loss"] if losses else None,
            }
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False))
            window_rewards.clear()
            window_success.clear()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "format_version": ENCODING_VERSION,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": MICRO_CONDITIONED_V3_DIM,
        "action_dim": 5,
        "hidden_dim": args.hidden_dim,
        "trained_steps": args.steps,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }
    torch.save(checkpoint, args.output_dir / "micro_sac_5d.pt")
    summary = {
        "version": ENCODING_VERSION,
        "method": "one_shot_goal_conditioned_sac",
        "training_states": len(samples),
        "episodes": args.steps,
        "source_rollout_dir": str(args.rollout_dir),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "reward_note": (
            "One-shot goal reward, not self-play game outcome. "
            "Macro Policy/Q/V and match-level value have NOT been trained."
        ),
        "logs": logs,
    }
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved: {args.output_dir / 'micro_sac_5d.pt'}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
