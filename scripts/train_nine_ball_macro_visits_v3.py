"""Fine-tune versioned 9-ball Macro Candidate Policy on physical MCTS visits.

Listwise visit-distribution cross entropy; hold out complete source matches.
This is policy distillation from the existing search, not improvement proof.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.build_nine_ball_mcts_visit_dataset_v3 import TARGET_KIND
from scripts.train_nine_ball_macro_policy_v3 import split_by_match
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM,
)
from src.game.nine_ball_policy_v3 import load_macro_policy
from src.macro.candidate_policy import MacroCandidatePolicyNetwork


def load_visits(path):
    with np.load(path, allow_pickle=False) as d:
        required = (
            "states", "candidates", "visits", "offsets",
            "match_seeds", "chosen_indices", "format_version", "target_kind",
        )
        missing = [key for key in required if key not in d.files]
        if missing:
            raise ValueError("Missing v3 visit dataset arrays: " + ", ".join(missing))
        if (
            int(d["format_version"]) != ENCODING_VERSION
            or str(d["target_kind"]) != TARGET_KIND
        ):
            raise ValueError("Not a nine-ball version-3 MCTS visit dataset")
        state = np.asarray(d["states"], dtype=np.float32)
        state_dim = MATCH_MACRO_STATE_DIM if bool(d.get("match_aware", False)) else MACRO_STATE_V3_DIM
        goal = np.asarray(d["candidates"], dtype=np.float32)
        visits = np.asarray(d["visits"])
        offsets = np.asarray(d["offsets"], dtype=np.int64)
        seeds = np.asarray(d["match_seeds"], dtype=np.int64)
        chosen = np.asarray(d["chosen_indices"], dtype=np.int64)
    n = len(state)
    if (
        n < 2 or state.shape != (n, state_dim)
        or goal.ndim != 2 or goal.shape[1] != MACRO_GOAL_V3_DIM
        or visits.shape != (len(goal),)
        or not np.issubdtype(visits.dtype, np.integer)
        or np.any(visits < 0)
        or offsets.shape != (n + 1,) or offsets[0] != 0
        or offsets[-1] != len(goal) or np.any(np.diff(offsets) < 1)
        or seeds.shape != (n,) or chosen.shape != (n,)
        or not np.all(np.isfinite(state)) or not np.all(np.isfinite(goal))
    ):
        raise ValueError("Invalid nine-ball MCTS visit dataset")
    visits = visits.astype(np.int64)
    for i in range(n):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if (
            int(visits[lo:hi].sum()) < 1
            or chosen[i] < 0 or chosen[i] >= hi - lo
            or visits[lo + chosen[i]] != visits[lo:hi].max()
        ):
            raise ValueError(f"Invalid visit totals/selected action at root {i}")
    if len(np.unique(seeds)) < 2:
        raise ValueError("Need >=2 distinct match seeds for held-out validation")
    return state, goal, visits, offsets, seeds, chosen


def evaluate(model, data, indices, device):
    states, goals, visits, offsets, _, chosen = data
    model.eval()
    ce, kl, match, entropy = [], [], [], []
    with torch.no_grad():
        for index in indices:
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            state = torch.as_tensor(states[index], dtype=torch.float32, device=device)
            goal = torch.as_tensor(goals[lo:hi], dtype=torch.float32, device=device)
            target = torch.as_tensor(visits[lo:hi], dtype=torch.float32, device=device)
            target /= target.sum()
            logits = model(state, goal)
            logprob = F.log_softmax(logits, dim=0)
            ce_val = float(-(target * logprob).sum().item())
            positive = target > 0
            entropy_val = float(-(target[positive] * torch.log(target[positive])).sum().item())
            ce.append(ce_val)
            entropy.append(entropy_val)
            kl.append(max(0., ce_val - entropy_val))
            chosen_index = int(torch.argmax(logits).item())
            best_visits = int(np.max(visits[lo:hi]))
            match.append(float(visits[lo + chosen_index] == best_visits))
    return {
        "roots": int(len(indices)),
        "visit_target_ce": float(np.mean(ce)),
        "visit_target_kl": float(np.mean(kl)),
        "visit_target_entropy": float(np.mean(entropy)),
        "argmax_max_visit_agreement": float(np.mean(match)),
    }


def train(args):
    if (
        args.epochs < 1 or args.learning_rate <= 0
        or not np.isfinite(args.learning_rate)
        or args.hidden_dim < 1
    ):
        raise ValueError("Invalid training parameters")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 1:
            raise ValueError("Choose one CUDA_VISIBLE_DEVICES GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    data = load_visits(args.dataset)
    states, goals, visits, offsets, seeds, chosen = data
    state_dim = states.shape[1]
    train_ids, val_ids, val_matches = split_by_match(
        seeds, validation_fraction=args.validation_fraction, seed=args.seed,
    )
    if args.base_policy_checkpoint is not None:
        model, hidden_dim = load_macro_policy(args.base_policy_checkpoint, device=device, state_dim=state_dim)
    else:
        hidden_dim = args.hidden_dim
        model = MacroCandidatePolicyNetwork(
            state_dim=state_dim,
            candidate_dim=MACRO_GOAL_V3_DIM,
            hidden_dim=hidden_dim,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    before = evaluate(model, data, val_ids, device)
    best_loss = before["visit_target_ce"]
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for index in rng.permutation(train_ids):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            state = torch.as_tensor(states[index], dtype=torch.float32, device=device)
            goal = torch.as_tensor(goals[lo:hi], dtype=torch.float32, device=device)
            target = torch.as_tensor(visits[lo:hi], dtype=torch.float32, device=device)
            target /= target.sum()
            logits = model(state, goal)
            loss = -(target * F.log_softmax(logits, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        heldout = evaluate(model, data, val_ids, device)
        if heldout["visit_target_ce"] < best_loss:
            best_loss = heldout["visit_target_ce"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch in (1, args.epochs):
            print(
                f"epoch={epoch} train_visit_ce={np.mean(losses):.6f} "
                f"val_visit_ce={heldout['visit_target_ce']:.6f}"
            )
    model.load_state_dict(best_weights)
    after = evaluate(model, data, val_ids, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": ENCODING_VERSION,
        "model_kind": "nine_ball_macro_candidate_policy_v3",
        "state_dim": state_dim,
        "candidate_dim": MACRO_GOAL_V3_DIM,
        "hidden_dim": hidden_dim,
        "candidate_policy_network": model.cpu().state_dict(),
        "teacher_kind": TARGET_KIND,
        "best_epoch": best_epoch,
        "base_policy_checkpoint": (
            str(args.base_policy_checkpoint) if args.base_policy_checkpoint else None
        ),
    }, args.output)
    summary = {
        "model_kind": "nine_ball_macro_candidate_policy_v3",
        "teacher_kind": TARGET_KIND,
        "dataset": str(args.dataset),
        "train_roots": len(train_ids),
        "validation_roots": len(val_ids),
        "train_match_seeds": sorted(set(seeds[train_ids].tolist())),
        "validation_match_seeds": val_matches,
        "initial_validation": before,
        "best_validation": after,
        "best_epoch": best_epoch,
        "output_checkpoint": str(args.output),
        "warning": (
            "MCTS visit imitation only. No independent match improvement "
            "demonstrated. Q/V not updated; do not train on validation matches."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base-policy-checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--validation-fraction", type=float, default=0.34)
    p.add_argument("--seed", type=int, default=420200)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
