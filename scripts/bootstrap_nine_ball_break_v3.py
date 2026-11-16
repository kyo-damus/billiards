"""Warm start the v3 Micro Actor from WPA-legal FastFiz break shots."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.check_nine_ball_break_v3 import break_trial
from scripts.train_nine_ball_micro_sac import load_training_states
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_encoding_v3 import (
    MICRO_CONDITIONED_V3_DIM, NineBallGoal, conditioned_observation_v3, reference_angle_deg,
)
from src.game.nine_ball_learned_agent import MICRO_KIND, _load_versioned
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_session import create_nine_ball_rack
from src.micro.actor import GaussianActor
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER
from src.pipelines.experiment_io import write_experiment_json


def teacher_actions():
    # Interior actions can be represented by a finite tanh Actor logit.
    return tuple(product(
        (0.995, 0.95, 0.85, 0.65), (-0.5, -0.25, 0.0, 0.25, 0.5),
        (-0.8, -0.6, -0.4), (0.0,), (-0.5, 0.0, 0.5),
    ))


def create_teacher(start_seed, racks, simulator_factory):
    context = NineBallContext()
    rules = NineBallRules(wpa_break_rules=True)
    goal = NineBallGoal("BREAK", 1)
    actions = teacher_actions()
    cfg = create_nine_ball_rack(breaker=0, seed=start_seed).config
    simulator = simulator_factory(cfg)
    simulator.require_head_string_crossings()
    observations, labels = [], []
    legal_counts, pot_counts, outcomes = Counter(), Counter(), Counter()
    decoder = WPA_NINE_BALL_BREAK_DECODER

    for index in range(racks):
        state = create_nine_ball_rack(breaker=index % 2, seed=start_seed + index)
        heading = reference_angle_deg(state, goal)
        observations.append(conditioned_observation_v3(state, goal, context))
        legal = {}
        for action in actions:
            result = break_trial(simulator, rules, state, context, decoder.decode(action, heading))
            outcomes[result] += 1
            if result in ("legal_with_pot", "legal_without_pot"):
                potted = int(result == "legal_with_pot")
                legal[action] = potted
                legal_counts[action] += 1
                pot_counts[action] += potted
        labels.append(legal)

    pairs = []
    for index, legal in enumerate(labels):
        if legal:
            # Prefer similarly successful actions across racks to reduce
            # regression toward an average of unrelated legal shots.
            target = max(legal, key=lambda a: (legal[a], legal_counts[a], pot_counts[a]))
            pairs.append((index, target))
    return np.asarray(observations, dtype=np.float32), pairs, dict(outcomes)


def bootstrap(args, *, simulator_factory=MultiBallFastFizSimulator):
    if args.racks < 1 or args.start_seed < 0 or args.steps < 1:
        raise ValueError("Use positive racks and steps and a nonnegative start seed")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    ckpt = _load_versioned(
        args.micro_checkpoint, model_kind=MICRO_KIND, state_dim=MICRO_CONDITIONED_V3_DIM,
        action_dim=5,
    )
    actor = GaussianActor(MICRO_CONDITIONED_V3_DIM, 5, int(ckpt["hidden_dim"]))
    actor.load_state_dict(ckpt["actor"], strict=True)
    observations, pairs, outcomes = create_teacher(args.start_seed, args.racks, simulator_factory)
    if not pairs:
        raise ValueError("No legal break in the teacher search")
    samples = load_training_states(
        args.anchor_rollouts, goal_source="logged", learned_safety=ckpt.get("learned_safety", False),
        learned_break=ckpt.get("learned_break", False),
        outcome_position=ckpt.get("outcome_position", False), wpa_break_rules=True,
    )
    rng = np.random.default_rng(args.seed)
    ordinary = [sample for sample in samples if not sample.state.break_shot]
    if not ordinary:
        raise ValueError("Need non-break rollouts to preserve the other Micro goals")
    if len(ordinary) > 1024:
        ordinary = [ordinary[i] for i in rng.choice(len(ordinary), 1024, replace=False)]
    anchors = torch.from_numpy(np.stack([
        conditioned_observation_v3(sample.state, sample.selected_goal, sample.context)
        for sample in ordinary
    ]))
    indices, targets = zip(*pairs)
    boards = torch.from_numpy(observations[list(indices)])
    # Fit latent means, so high-speed targets do not saturate action gradients.
    logits = torch.atanh(torch.tensor(targets, dtype=torch.float32))
    with torch.no_grad():
        original_mu, original_log_std = actor(anchors)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    for _ in range(args.steps):
        batch = rng.integers(len(boards), size=min(64, len(boards)))
        reference = rng.integers(len(anchors), size=min(64, len(anchors)))
        mean, _ = actor(boards[batch])
        anchor_mu, anchor_std = actor(anchors[reference])
        loss = F.mse_loss(mean, logits[batch]) + (
            F.mse_loss(anchor_mu, original_mu[reference])
            + F.mse_loss(anchor_std, original_log_std[reference])
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        drift = actor.deterministic(anchors) - original_mu.tanh()
        anchor_action_rmse = drift.square().mean().sqrt().item()

    summary = {
        "micro_checkpoint": str(args.micro_checkpoint),
        "anchor_rollouts": str(args.anchor_rollouts),
        "start_seed": args.start_seed,
        "racks": args.racks,
        "teacher_actions_per_rack": len(teacher_actions()),
        "teacher_outcomes": outcomes,
        "labeled_racks": len(pairs),
        "anchor_states": len(anchors),
        "anchor_action_rmse": anchor_action_rmse,
        "behavior_cloning_steps": args.steps,
        "seed": args.seed,
    }
    ckpt["actor"] = actor.state_dict()
    ckpt["learned_break"] = True
    ckpt["wpa_break_rules"] = True
    ckpt["break_bootstrap"] = summary
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(ckpt, args.output_dir / "micro_sac_5d.pt")
    write_experiment_json(args.output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-checkpoint", type=Path, required=True)
    parser.add_argument("--anchor-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--racks", type=int, default=48)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=649000)
    print(json.dumps(bootstrap(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
