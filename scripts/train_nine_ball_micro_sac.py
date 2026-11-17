"""Train or fine-tune versioned 5D goal-conditioned SAC on nine-ball shots.

Each episode is one physical shot from an observed board. With --goal-source
logged, learn against the exact selected Micro-controlled Goal of self-play Macro.
Candidates uses every supported saved root goal; uniform generates goals afresh.
This script
never updates Macro Policy/Q/V or claims gameplay strength.
"""

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from src.common.table import TABLE_LENGTH, TABLE_WIDTH
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, GOAL_KINDS, MACRO_GOAL_V3_DIM, MICRO_CONDITIONED_V3_DIM,
    NineBallGoal, conditioned_observation_v3, direct_pocket_candidates, encode_goal_v3,
    encode_rack_v3,
    generate_nine_ball_goals, reference_angle_deg,
)
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, simulate_nine_ball_shot,
)
from src.game.nine_ball_position import (
    POSITION_RADIUS_M, continues_to_next_ball, generate_outcome_position_goals,
)
from src.game.rack_state import RackState, TableGameConfig
from src.micro.replay_buffer import ReplayBuffer
from src.micro.sac import SACAgent
from src.micro.shot_control import NINE_BALL_BREAK_DECODER, WPA_NINE_BALL_BREAK_DECODER, Shot5DDecoder
from src.pipelines.experiment_io import write_experiment_json


@dataclass(frozen=True)
class TrainingSample:
    state: RackState
    context: NineBallContext
    selected_goal: object | None = None


def _logged_outcome_position(state, selected, pockets):
    """Read the recorded condition without rerunning FastFiz during SAC loading."""
    target = int(np.argmax(selected[4:20]))
    pocket = int(np.argmax(selected[20:26]))
    x, y = selected[26:28] * np.array([TABLE_WIDTH, TABLE_LENGTH])
    goal = NineBallGoal("POSITION_ATTACK", target, pocket, (float(x), float(y)), POSITION_RADIUS_M)
    remaining = state.on_table_object_balls()
    if (
        len(remaining) < 2 or target != min(remaining)
        or not (0 < x < TABLE_WIDTH and 0 < y < TABLE_LENGTH)
        or pocket not in pockets
        or not np.array_equal(encode_goal_v3(goal), selected)
    ):
        raise ValueError("Invalid logged outcome POSITION_ATTACK Goal")
    return goal


def _logged_micro_goals(state, row, *, goal_source, learned_safety, learned_break, outcome_position):
    """Use exact logged conditions without consuming the recorded shot or reward."""
    action = row.get("chosen_action") or {}
    chosen = action.get("goal_v3")
    if chosen is None:
        return
    search = action.get("macro_search")
    candidates = (chosen,)
    chosen_features = np.asarray(chosen, dtype=np.float32)
    is_outcome_position = (
        outcome_position and chosen_features.shape == (MACRO_GOAL_V3_DIM,)
        and chosen_features[GOAL_KINDS.index("POSITION_ATTACK")] == 1
    )
    if search is not None and (goal_source == "candidates" or is_outcome_position):
        root_goals = search["goal_v3"]
        index = search["chosen_index"]
        if not 0 <= index < len(root_goals) or not np.array_equal(
            chosen_features, np.asarray(root_goals[index], dtype=np.float32),
        ):
            raise ValueError("Logged Goal differs from selected MCTS candidate")
        if goal_source == "candidates":
            candidates = root_goals
    elif goal_source == "candidates":
        raise ValueError("Candidate mode requires logged MCTS root goals")
    generated = generate_nine_ball_goals(
        state, learned_safety=learned_safety, learned_break=learned_break,
    )
    encoded = [
        (goal, encode_goal_v3(goal)) for goal in generated if goal.explicit_shot is None
    ]
    pockets = {goal.target_pocket for goal in generated if goal.kind == "DIRECT_ATTACK"}
    for raw in candidates:
        goal = _decode_logged_goal(
            state, raw, encoded, pockets, learned_safety=learned_safety,
            learned_break=learned_break, outcome_position=outcome_position,
        )
        if goal is not None and (
            goal_source != "candidates" or goal.kind != "DIRECT_ATTACK" or goal.target_pocket is not None
        ):
            yield goal


def _decode_logged_goal(state, raw, encoded, pockets, *, learned_safety, learned_break, outcome_position):
    selected = np.asarray(raw, dtype=np.float32)
    if selected.shape != (MACRO_GOAL_V3_DIM,) or not np.all(np.isfinite(selected)):
        raise ValueError("Malformed logged 36D Macro Goal")
    kind_index = int(np.argmax(selected[:len(GOAL_KINDS)]))
    if not np.allclose(selected[:len(GOAL_KINDS)], np.eye(len(GOAL_KINDS))[kind_index], atol=1e-5):
        raise ValueError("Invalid logged Macro Goal kind")
    kind = GOAL_KINDS[kind_index]
    if kind == "SAFETY" and selected[28] < 0:
        return None  # Macro-only table choices and fixed push-out shots.
    if state.break_shot != (kind == "BREAK"):
        raise ValueError("Logged Macro Goal kind does not match rack break phase")
    if kind == "BREAK" and not learned_break:
        return None
    if kind == "SAFETY" and (
        not learned_safety or np.isclose(selected[29], 1.0, atol=1e-5)
    ):
        return None
    if outcome_position and kind == "POSITION_ATTACK":
        return _logged_outcome_position(state, selected, pockets)
    matches = [
        goal for goal, features in encoded
        if np.allclose(features, selected, atol=1e-5, rtol=0)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Logged selected Macro Goal cannot be recovered exactly. "
            "Candidate generator or stored goal encoding may have changed."
        )
    return matches[0]


def load_training_states(
    rollout_dir: Path, *, goal_source: str = "uniform", learned_safety=False,
    learned_break=False, outcome_position=False, wpa_break_rules=False, break_only=False,
) -> tuple[TrainingSample, ...]:
    """Load observed pre-shot states; optionally use exact selected Macro goals."""
    if goal_source not in ("uniform", "logged", "candidates"):
        raise ValueError("goal_source must be uniform, logged or candidates")
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
        if goal_source != "uniform" and data.get("outcome_position", False) != outcome_position:
            raise ValueError(f"Outcome position mode differs from logged collection in {path}")
        if goal_source != "uniform" and data.get("learned_break", False) != learned_break:
            raise ValueError(f"Learned break mode differs from logged collection in {path}")
        if data.get("wpa_break_rules", False) != wpa_break_rules:
            raise ValueError(f"WPA break mode differs from collection in {path}")
        for row in data.get("trajectory", []):
            raw_ctx = row.get("rule_context")
            if raw_ctx and raw_ctx.get("pending_choice") is not None:
                continue
            raw = row["state"]
            if break_only and not raw["break_shot"]:
                continue
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
            if raw_ctx is None:
                context = NineBallContext()
            else:
                context = NineBallContext(
                    consecutive_fouls=tuple(
                        int(n) for n in raw_ctx["consecutive_fouls"]
                    ),
                    push_out_available=bool(raw_ctx["push_out_available"]),
                )
            if (learned_break or not state.break_shot) and state.on_table_object_balls():
                goals = (None,) if goal_source == "uniform" else _logged_micro_goals(
                    state, row, goal_source=goal_source, learned_safety=learned_safety,
                    learned_break=learned_break, outcome_position=outcome_position,
                )
                training.extend(TrainingSample(state, context, goal) for goal in goals)
    if not training:
        raise ValueError(
            "No eligible states with supported Macro goals "
            f"for goal_source={goal_source}"
        )
    return tuple(training)


def sample_keys(sample):
    rack = encode_rack_v3(sample.state, sample.context)
    board = rack[:64].tobytes()  # Ball presence, on-table flags and coordinates.
    goal = encode_goal_v3(sample.selected_goal).tobytes() if sample.selected_goal else b""
    return board, (rack.tobytes(), goal)


def prepare_training_data(args):
    dirs = (args.rollout_dir, *args.extra_rollout_dir)
    paths = [path.resolve() for path in dirs]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated training rollout directory")
    holdout = args.holdout_rollout_dir
    if holdout is not None and holdout.resolve() in paths:
        raise ValueError("Holdout must use a different rollout directory")
    options = {
        "goal_source": args.goal_source,
        "learned_safety": args.learned_safety,
        "learned_break": args.learned_break,
        "outcome_position": args.outcome_position,
        "wpa_break_rules": args.wpa_break_rules,
        "break_only": args.break_only,
    }
    heldout_boards = (
        {sample_keys(sample)[0] for sample in load_training_states(holdout, **options)}
        if holdout is not None else set()
    )
    deduplicate = args.goal_source == "candidates" or len(dirs) > 1 or holdout is not None
    samples, seen, boards_by_goal = [], set(), {}
    overlaps = duplicates = 0
    for directory in dirs:
        for sample in load_training_states(directory, **options):
            board, key = sample_keys(sample)
            if board in heldout_boards:
                overlaps += 1
                continue
            if deduplicate and key in seen:
                duplicates += 1
                continue
            seen.add(key)
            samples.append(sample)
            kind = sample.selected_goal.kind if sample.selected_goal else "generated_at_training"
            boards_by_goal.setdefault(kind, set()).add(board)
    if not samples:
        raise ValueError("No Micro training samples after excluding held-out boards")
    return samples, {
        "rollout_dirs": [str(path) for path in dirs],
        "holdout_rollout_dir": str(holdout) if holdout is not None else None,
        "holdout_boards": len(heldout_boards),
        "excluded_overlapping_samples": overlaps,
        "removed_duplicate_samples": duplicates,
        "training_samples": len(samples),
        "samples_by_goal": dict(Counter(
            sample.selected_goal.kind if sample.selected_goal else "generated_at_training"
            for sample in samples
        )),
        "unique_boards_by_goal": {kind: len(boards) for kind, boards in boards_by_goal.items()},
    }


def micro_reward(goal, before, physical, outcome, *, outcome_position=False) -> tuple[float, bool]:
    """Physics-grounded one-shot reward (not the final match objective)."""
    if goal.kind == "BREAK":
        if outcome.foul:
            return -1.2, False
        if getattr(outcome, "illegal_break", False):
            return -0.5, False
        if outcome.rack_winner == before.current_player:
            return 2.1, True
        pocketed = len(outcome.newly_pocketed)
        return 0.1 + 0.5 * min(pocketed, 3), pocketed > 0
    if goal.kind == "SAFETY":
        if outcome.foul:
            return -1.2, False
        if not outcome.turn_changed:
            return -0.5, False
        after = outcome.next_state
        before_count = len(tuple(direct_pocket_candidates(before, goal.target_ball)))
        after_count = len(tuple(direct_pocket_candidates(
            after, min(after.on_table_object_balls()),
        )))
        reduction = before_count - after_count
        return 0.1 + reduction / 6.0, reduction > 0

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
    # POSITION means pot the specified ball in the specified pocket AND
    # place the cue. Do not reward cue position after a missed pot.
    if goal.cue_stop_xy is not None and exact and not outcome.foul and (
        not outcome_position or _continues_position_goal(before, goal, outcome)
    ):
        d = _cue_stop_distance(goal, physical)
        reward += 0.5 * max(0.0, 1.0 - d / goal.cue_stop_radius)
    return float(reward), exact and not outcome.foul


def _cue_stop_distance(goal, physical):
    return float(np.linalg.norm(
        np.asarray(physical.positions[0], dtype=np.float64) - goal.cue_stop_xy
    ))


def _continues_position_goal(before, goal, outcome):
    next_ball = min(
        (ball for ball in before.on_table_object_balls() if ball != goal.target_ball),
        default=None,
    )
    return continues_to_next_ball(next_ball, outcome)


def position_goal_success(before, goal, physical, outcome, *, exact_pot):
    return (
        exact_pot and _continues_position_goal(before, goal, outcome)
        and _cue_stop_distance(goal, physical) <= goal.cue_stop_radius
    )


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



def restore_micro_checkpoint(agent: SACAgent, checkpoint: Path) -> int:
    """Continue versioned 5D SAC weights without ever loading legacy 2D models.

    Optimizer states and replay are *intentionally reset* on this new run;
    only actor, critics and entropy coefficient are carried forward.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = ("actor", "critic", "critic_target", "log_alpha")
    if (
        ckpt.get("format_version") != ENCODING_VERSION
        or ckpt.get("model_kind") != "nine_ball_goal_conditioned_micro_sac_5d"
        or ckpt.get("state_dim") != MICRO_CONDITIONED_V3_DIM
        or ckpt.get("action_dim") != 5
        or ckpt.get("hidden_dim") != agent.actor.fc1.out_features
        or any(key not in ckpt for key in required)
    ):
        raise ValueError("Incompatible Micro checkpoint: expected strict v3/108D/5D SAC")
    agent.actor.load_state_dict(ckpt["actor"], strict=True)
    agent.critic.load_state_dict(ckpt["critic"], strict=True)
    agent.critic_target.load_state_dict(ckpt["critic_target"], strict=True)
    with torch.no_grad():
        previous_log_alpha = torch.as_tensor(
            ckpt["log_alpha"], dtype=agent.log_alpha.dtype, device=agent.device,
        )
        if previous_log_alpha.shape != agent.log_alpha.shape or not torch.isfinite(previous_log_alpha).all():
            raise ValueError("Invalid saved Micro entropy coefficient")
        agent.log_alpha.copy_(previous_log_alpha)
    previous_steps = int(ckpt.get("trained_steps", 0))
    if previous_steps < 0:
        raise ValueError("Negative previous Micro trained_steps")
    return previous_steps


def execute_valid_training_shot(simulator, rules, state, goal, context, shot):
    """Never label a native non-shot or impossible successor as SAC experience."""
    return simulate_nine_ball_shot(
        simulator, rules, state, context, target_ball=goal.target_ball, shot=shot,
    )


def save_micro_checkpoint(agent, args, initial_steps, step, path):
    checkpoint = {
        "format_version": ENCODING_VERSION,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": MICRO_CONDITIONED_V3_DIM,
        "action_dim": 5,
        "hidden_dim": args.hidden_dim,
        "trained_steps": initial_steps + step,
        "training_steps_this_run": step,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "optimizer_state_carried_over": False,
        "goal_source": args.goal_source,
        "learned_safety": args.learned_safety,
        "learned_break": args.learned_break,
        "break_only": args.break_only,
        "outcome_position": args.outcome_position,
        "wpa_break_rules": args.wpa_break_rules,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
    }
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def train(args, *, simulator_factory=MultiBallFastFizSimulator):
    """Run one-step off-policy SAC with an explicitly bounded GPU allocation."""
    if args.steps <= 0 or args.batch_size <= 0 or args.warmup_steps < 0:
        raise ValueError("Invalid steps, batch size or warmup")
    if args.updates_per_step < 1 or args.replay_capacity < args.batch_size:
        raise ValueError("Invalid replay capacity or update count")
    if args.break_only and not args.learned_break:
        raise ValueError("--break-only requires --learned-break")
    if args.balance_goals and args.goal_source == "uniform":
        raise ValueError("--balance-goals requires logged or candidates goal source")
    if args.report_interval < 1 or args.checkpoint_interval < 0:
        raise ValueError("Invalid report or checkpoint interval")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for training")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    samples, data = prepare_training_data(args)
    buckets = (samples,)
    if args.balance_goals:
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample.selected_goal.kind, []).append(sample)
        buckets = tuple(grouped[kind] for kind in GOAL_KINDS if kind in grouped)
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
    initial_steps = (
        restore_micro_checkpoint(agent, args.init_checkpoint)
        if args.init_checkpoint is not None else 0
    )
    replay = ReplayBuffer(capacity=args.replay_capacity)
    simulator = simulator_factory(TableGameConfig.nine_ball_inventory())
    if args.wpa_break_rules and hasattr(simulator, "require_head_string_crossings"):
        simulator.require_head_string_crossings()
    rules = NineBallRules(wpa_break_rules=args.wpa_break_rules)
    decoder = Shot5DDecoder()
    break_decoder = WPA_NINE_BALL_BREAK_DECODER if args.wpa_break_rules else NINE_BALL_BREAK_DECODER
    if args.checkpoint_interval:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "checkpoints").mkdir()
        write_experiment_json(args.output_dir / "training_data.json", data)
    witnessed_goals = {}
    logs = []
    window_rewards, window_pots, window_safety = [], [], []
    window_breaks = []
    window_positions = []
    accepted_steps_by_goal = Counter()
    print(
        f"=== Nine-ball 5D goal-conditioned Micro SAC ===\n"
        f"device={device}, training_states={len(samples)}, steps={args.steps}, "
        "action_dim=5, version=3 (NOT old 2D Micro)"
    )
    invalid_physics_attempts = 0
    skipped_no_goal_attempts = 0
    consecutive_invalid_physics = 0
    consecutive_no_goal = 0
    for step in range(1, args.steps + 1):
        # "steps" counts accepted physical transitions, not rejected native
        # attempts. Do not push invalid shots or invent a negative reward.
        while True:
            bucket = buckets[int(rng.integers(len(buckets)))] if len(buckets) > 1 else buckets[0]
            sample_index = int(rng.integers(len(bucket)))
            sample = bucket[sample_index]
            state, context = sample.state, sample.context
            if args.goal_source != "uniform":
                goal = sample.selected_goal
                if goal is None:
                    raise RuntimeError("Logged-goal training sample has no selected goal")
            else:
                if args.outcome_position and not state.break_shot:
                    if sample_index not in witnessed_goals:
                        witnessed_goals[sample_index] = generate_outcome_position_goals(
                            state, context, simulator, rules, learned_safety=args.learned_safety,
                        )
                    goals = witnessed_goals[sample_index]
                else:
                    goals = generate_nine_ball_goals(
                        state, learned_safety=args.learned_safety, learned_break=args.learned_break,
                    )
                candidate_goals = tuple(g for g in goals if g.explicit_shot is None)
                if not candidate_goals:
                    skipped_no_goal_attempts += 1
                    consecutive_no_goal += 1
                    if consecutive_no_goal >= 128:
                        raise RuntimeError("No eligible Micro Goal in 128 attempts")
                    continue
                goal = candidate_goals[int(rng.integers(0, len(candidate_goals)))]
            consecutive_no_goal = 0
            encoded = conditioned_observation_v3(state, goal, context)
            if step <= args.warmup_steps:
                action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
            else:
                action = agent.select_action(encoded, deterministic=False)
            shot_decoder = break_decoder if goal.kind == "BREAK" else decoder
            shot = shot_decoder.decode(action, reference_angle_deg(state, goal))
            try:
                physical, outcome = execute_valid_training_shot(
                    simulator, rules, state, goal, context, shot,
                )
            except InvalidPhysicalTransition as exc:
                invalid_physics_attempts += 1
                consecutive_invalid_physics += 1
                if (consecutive_invalid_physics >= 128
                        or invalid_physics_attempts >= max(256, args.steps * 2)):
                    raise RuntimeError(
                        "Too many invalid Micro physical transitions "
                        f"(count={invalid_physics_attempts}, last={exc})"
                    ) from exc
                continue
            consecutive_invalid_physics = 0
            break
        reward, success = micro_reward(
            goal, state, physical, outcome, outcome_position=args.outcome_position,
        )
        replay.push(
            encoded, action, reward, np.zeros_like(encoded), True
        )
        losses = None
        if len(replay) >= args.batch_size:
            for _ in range(args.updates_per_step):
                losses = agent.update(replay.sample(args.batch_size, device))
        window_rewards.append(reward)
        accepted_steps_by_goal[goal.kind] += 1
        if goal.kind == "SAFETY":
            window_safety.append(success)
        elif goal.kind == "BREAK":
            window_breaks.append((not outcome.foul and not getattr(outcome, "illegal_break", False), success))
        else:
            window_pots.append(success)
        if args.outcome_position and goal.kind == "POSITION_ATTACK":
            window_positions.append(
                position_goal_success(state, goal, physical, outcome, exact_pot=success)
            )
        if step == args.steps or step % args.report_interval == 0:
            break_rates = np.mean(window_breaks, axis=0).tolist() if window_breaks else (None, None)
            break_legal_rate, break_legal_pot_rate = break_rates
            log = {
                "step": step,
                "mean_reward": float(np.mean(window_rewards)),
                "exact_target_pot_rate": float(np.mean(window_pots)) if window_pots else None,
                "safety_goal_success_rate": float(np.mean(window_safety)) if window_safety else None,
                "safety_steps": len(window_safety),
                "break_legal_rate": break_legal_rate,
                "break_legal_pot_rate": break_legal_pot_rate,
                "break_steps": len(window_breaks),
                "position_goal_success_rate": float(np.mean(window_positions)) if window_positions else None,
                "position_steps": len(window_positions),
                "replay_size": len(replay),
                "invalid_physics_attempts": invalid_physics_attempts,
                "skipped_no_goal_attempts": skipped_no_goal_attempts,
                "alpha": float(agent.alpha.item()),
                "actor_loss": losses["actor_loss"] if losses else None,
                "critic_loss": losses["critic_loss"] if losses else None,
            }
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False))
            window_rewards.clear()
            window_pots.clear()
            window_safety.clear()
            window_breaks.clear()
            window_positions.clear()
        if args.checkpoint_interval and step % args.checkpoint_interval == 0 and step < args.steps:
            path = args.output_dir / "checkpoints" / f"step_{step:08d}.pt"
            save_micro_checkpoint(agent, args, initial_steps, step, path)
            print(f"Saved: {path}", flush=True)

    if not args.checkpoint_interval:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_experiment_json(args.output_dir / "training_data.json", data)
    save_micro_checkpoint(agent, args, initial_steps, args.steps, args.output_dir / "micro_sac_5d.pt")
    summary = {
        "version": ENCODING_VERSION,
        "method": "one_shot_goal_conditioned_sac",
        "training_states": len(samples),
        "episodes": args.steps,
        "accepted_steps_by_goal": dict(accepted_steps_by_goal),
        "invalid_physics_attempts": invalid_physics_attempts,
        "skipped_no_goal_attempts": skipped_no_goal_attempts,
        "source_rollout_dir": str(args.rollout_dir),
        "training_data": data,
        "goal_source": args.goal_source,
        "balance_goals": args.balance_goals,
        "checkpoint_interval": args.checkpoint_interval,
        "learned_safety": args.learned_safety,
        "learned_break": args.learned_break,
        "break_only": args.break_only,
        "outcome_position": args.outcome_position,
        "wpa_break_rules": args.wpa_break_rules,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "previous_trained_steps": initial_steps,
        "training_steps_this_run": args.steps,
        "optimizer_state_carried_over": False,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "reward_note": (
            "One-shot Micro goal reward, not the self-play match outcome. "
            "This run does not train Macro Policy/Q/V; its logged goals may "
            "originate from a separately trained Macro agent."
        ),
        "logs": logs,
    }
    write_experiment_json(args.output_dir / "summary.json", summary)
    print(f"Saved: {args.output_dir / 'micro_sac_5d.pt'}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--extra-rollout-dir", action="append", type=Path, default=[])
    parser.add_argument("--holdout-rollout-dir", type=Path,
                        help="Exclude matching board observations from training")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Report sample counts without running physics or training")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--goal-source", choices=("uniform", "logged", "candidates"), default="uniform",
        help="fresh goals, the logged selected goal, or all logged Micro-controlled root goals",
    )
    parser.add_argument("--balance-goals", action="store_true",
                        help="Sample a goal type uniformly before sampling its board/goal pair")
    parser.add_argument("--learned-safety", action="store_true")
    parser.add_argument("--learned-break", action="store_true")
    parser.add_argument("--break-only", action="store_true", help="Sample only recorded pre-break boards")
    parser.add_argument("--wpa-break-rules", action="store_true")
    parser.add_argument("--outcome-position", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="Save a weight checkpoint every N accepted steps (0 disables)")
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.inspect_only:
        _, report = prepare_training_data(args)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        train(args)
