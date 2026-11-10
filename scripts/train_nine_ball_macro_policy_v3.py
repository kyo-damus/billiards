"""Train a version-3 nine-ball Candidate Policy from a fixed-Micro oracle.

Supervised listwise ranking from ONE-STEP physics scores, not reinforcement
learning of Macro Q/V or a claimed improvement in nine-ball match win rate.
Train/validation separation is by entire source matches (match seeds).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_STATE_V3_DIM, MACRO_GOAL_V3_DIM,
)
from src.macro.candidate_policy import MacroCandidatePolicyNetwork


def load_dataset(path: Path):
    with np.load(path, allow_pickle=False) as data:
        names = ("states", "candidates", "teacher_scores", "offsets", "match_seeds")
        if any(k not in data.files for k in names):
            raise ValueError("Missing required dataset arrays")
        states = np.asarray(data["states"], dtype=np.float32)
        goals = np.asarray(data["candidates"], dtype=np.float32)
        scores = np.asarray(data["teacher_scores"], dtype=np.float32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        seeds = np.asarray(data["match_seeds"], dtype=np.int64)
    if (
        states.ndim != 2 or states.shape[1] != MACRO_STATE_V3_DIM
        or goals.ndim != 2 or goals.shape[1] != MACRO_GOAL_V3_DIM
        or scores.shape != (len(goals),)
        or offsets.shape != (len(states) + 1,)
        or seeds.shape != (len(states),)
        or len(states) < 2
        or offsets[0] != 0 or offsets[-1] != len(goals)
        or np.any(np.diff(offsets) < 1)
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(goals))
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("Invalid Macro v3 dataset shapes, offsets, or finite values")
    return states, goals, scores, offsets, seeds


def split_by_match(seeds, *, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be in (0,1)")
    matches = np.unique(seeds)
    if len(matches) < 2:
        raise ValueError("Need >=2 distinct source matches for held-out evaluation")
    rng = np.random.default_rng(seed)
    matches = rng.permutation(matches)
    n_val = max(1, min(len(matches) - 1, round(len(matches) * validation_fraction)))
    val = set(matches[:n_val].tolist())
    train = np.asarray([i for i, x in enumerate(seeds) if x not in val], dtype=int)
    validation = np.asarray([i for i, x in enumerate(seeds) if x in val], dtype=int)
    if not len(train) or not len(validation):
        raise ValueError("Empty train or validation split")
    return train, validation, sorted(val)


def evaluate(model, states, goals, scores, offsets, indices, device, temperature):
    model.eval()
    regrets, agreements, losses = [], [], []
    with torch.no_grad():
        for i in indices:
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            s = torch.as_tensor(states[i], device=device)
            c = torch.as_tensor(goals[lo:hi], device=device)
            logits = model(s, c)
            teacher = torch.as_tensor(scores[lo:hi], device=device)
            weights = torch.softmax(teacher / temperature, dim=0)
            losses.append(float(-(weights * F.log_softmax(logits, dim=0)).sum().item()))
            predicted = int(torch.argmax(logits).item())
            best = float(np.max(scores[lo:hi]))
            regrets.append(best - float(scores[lo + predicted]))
            agreements.append(float(scores[lo + predicted] >= best - 1e-6))
    return {
        "soft_label_ce": float(np.mean(losses)),
        "mean_one_step_teacher_regret": float(np.mean(regrets)),
        "top_choice_teacher_tie_agreement": float(np.mean(agreements)),
        "boards": int(len(indices)),
    }


def train(args):
    if args.epochs < 1 or args.hidden_dim < 1 or args.temperature <= 0:
        raise ValueError("Invalid epochs, hidden_dim or teacher temperature")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if len(visible.split(",")) != 1 or not visible:
            raise ValueError("Specify exactly one CUDA_VISIBLE_DEVICES on shared GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    states, goals, scores, offsets, seeds = load_dataset(args.dataset)
    training, validation, val_matches = split_by_match(
        seeds, validation_fraction=args.validation_fraction, seed=args.seed,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))

    model = MacroCandidatePolicyNetwork(
        state_dim=MACRO_STATE_V3_DIM,
        candidate_dim=MACRO_GOAL_V3_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i in rng.permutation(training):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            s = torch.as_tensor(states[i], device=device)
            c = torch.as_tensor(goals[lo:hi], device=device)
            teacher = torch.as_tensor(scores[lo:hi], device=device)
            weights = torch.softmax(teacher / args.temperature, dim=0)
            logits = model(s, c)
            loss = -(weights * F.log_softmax(logits, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(
            f"epoch={epoch + 1} train_soft_label_ce={np.mean(losses):.5f}"
        )

    heldout = evaluate(
        model, states, goals, scores, offsets, validation,
        device, args.temperature,
    )
    # Baseline is the uniform-candidate teacher distribution, not an
    # independent physical baseline. This metric is merely a sanity check.
    heldout["uniform_candidate_teacher_regret"] = float(np.mean([
        float(np.max(scores[int(offsets[i]):int(offsets[i + 1])]))
        - float(np.mean(scores[int(offsets[i]):int(offsets[i + 1])]))
        for i in validation
    ]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": ENCODING_VERSION,
        "model_kind": "nine_ball_macro_candidate_policy_v3",
        "state_dim": MACRO_STATE_V3_DIM,
        "candidate_dim": MACRO_GOAL_V3_DIM,
        "hidden_dim": args.hidden_dim,
        "candidate_policy_network": model.cpu().state_dict(),
        "teacher_kind": "fixed_micro_one_step_fastfiz",
        "training_boards": int(len(training)),
        "heldout_boards": int(len(validation)),
    }
    torch.save(checkpoint, args.output)
    report = {
        "model_kind": checkpoint["model_kind"],
        "teacher_kind": checkpoint["teacher_kind"],
        "not_a": ["MCTS", "learned_Q", "match_return", "match_win_rate"],
        "dataset": str(args.dataset),
        "training_boards": int(len(training)),
        "validation_boards": int(len(validation)),
        "train_match_seeds": sorted(set(seeds[training].tolist())),
        "validation_match_seeds": val_matches,
        "teacher_temperature": args.temperature,
        "heldout": heldout,
        "checkpoint": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=410100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
