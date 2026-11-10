"""Train 9-ball v3 Candidate Q and State V on SELECTED-action match returns.

On-policy/behavior-policy Monte Carlo regression. Does not assign targets to
unselected candidate goals, and does not imply the learned Q is optimal.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.build_nine_ball_macro_returns_v3 import REWARD_KIND, TARGET_KIND
from scripts.train_nine_ball_macro_policy_v3 import split_by_match
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM,
)
from src.macro.candidate_value import MacroCandidateValueNetwork
from src.macro.value import MacroValueNetwork


def load_returns(path):
    with np.load(path, allow_pickle=False) as d:
        needed = (
            "states", "candidates", "q_target", "match_seeds",
            "format_version", "turn_aware", "target_kind", "reward_kind",
        )
        missing = [k for k in needed if k not in d.files]
        if missing:
            raise ValueError("Missing v3 return fields: " + ", ".join(missing))
        if (
            int(d["format_version"]) != ENCODING_VERSION
            or not bool(d["turn_aware"])
            or str(d["target_kind"]) != TARGET_KIND
            or str(d["reward_kind"]) != REWARD_KIND
        ):
            raise ValueError("Wrong v3 turn-aware terminal-return dataset contract")
        states = np.asarray(d["states"], dtype=np.float32)
        goals = np.asarray(d["candidates"], dtype=np.float32)
        targets = np.asarray(d["q_target"], dtype=np.float32)
        match_ids = np.asarray(d["match_seeds"], dtype=np.int64)
    n = len(targets)
    if (
        n < 2 or states.shape != (n, MACRO_STATE_V3_DIM)
        or goals.shape != (n, MACRO_GOAL_V3_DIM)
        or match_ids.shape != (n,)
        or not all(np.all(np.isfinite(x)) for x in (states, goals, targets))
        or np.any(np.abs(targets) > 1.00001)
    ):
        raise ValueError("Invalid v3 selected-action return shapes or values")
    if len(np.unique(match_ids)) < 2:
        raise ValueError("Require at least two distinct complete matches")
    return states, goals, targets, match_ids


def model_predictions(model, state, goal, indices, model_kind):
    if model_kind == "q":
        return model(state[indices], goal[indices]).reshape(-1)
    return model(state[indices]).reshape(-1)


def fit_model(kind, model, state, goals, targets, train_ids, val_ids, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + (1 if kind == "q" else 2))
    best_loss, best_epoch, best_state = float("inf"), None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        groups = np.array_split(
            rng.permutation(train_ids),
            max(1, int(np.ceil(len(train_ids) / args.batch_size))),
        )
        train_losses = []
        for batch in groups:
            pred = model_predictions(model, state, goals, batch, kind)
            loss = F.mse_loss(pred, targets[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_pred = model_predictions(model, state, goals, val_ids, kind)
            val_loss = float(F.mse_loss(val_pred, targets[val_ids]).item())
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch in (1, args.epochs):
            print(
                f"{kind} epoch={epoch} train_mse={np.mean(train_losses):.6f} "
                f"val_mse={val_loss:.6f}"
            )
    if best_state is None:
        raise RuntimeError(f"{kind}: no validation checkpoint")
    return best_state, best_epoch, best_loss


def train(args):
    if (
        args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1
        or args.learning_rate <= 0 or not np.isfinite(args.learning_rate)
    ):
        raise ValueError("Invalid Q/V training hyperparameters")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or len(visible.split(",")) != 1:
            raise ValueError("Choose one CUDA_VISIBLE_DEVICES GPU on shared server")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    states, goals, y, match_ids = load_returns(args.dataset)
    train_ids, val_ids, val_matches = split_by_match(
        match_ids, validation_fraction=args.validation_fraction, seed=args.seed,
    )
    print(
        f"=== 9-ball v3 selected-action Macro Q/V learning ===\n"
        f"device={device} matches={len(np.unique(match_ids))} "
        f"train_shots={len(train_ids)} val_shots={len(val_ids)}"
    )
    state = torch.as_tensor(states, device=device)
    goal = torch.as_tensor(goals, device=device)
    target = torch.as_tensor(y, device=device)
    q = MacroCandidateValueNetwork(
        state_dim=MACRO_STATE_V3_DIM,
        candidate_dim=MACRO_GOAL_V3_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    v = MacroValueNetwork(
        state_dim=MACRO_STATE_V3_DIM, hidden_dim=args.hidden_dim,
    ).to(device)
    q_weights, q_epoch, q_mse = fit_model(
        "q", q, state, goal, target, train_ids, val_ids, args, device,
    )
    v_weights, v_epoch, v_mse = fit_model(
        "v", v, state, goal, target, train_ids, val_ids, args, device,
    )
    common = {
        "format_version": ENCODING_VERSION,
        "state_dim": MACRO_STATE_V3_DIM,
        "hidden_dim": args.hidden_dim,
        "target_kind": TARGET_KIND,
        "reward_kind": REWARD_KIND,
        "turn_aware": True,
        "dataset": str(args.dataset),
    }
    q_ckpt = {
        **common,
        "model_kind": "nine_ball_macro_selected_action_q_v3",
        "candidate_dim": MACRO_GOAL_V3_DIM,
        "candidate_value_network": q_weights,
        "best_epoch": q_epoch,
        "best_val_mse": q_mse,
    }
    v_ckpt = {
        **common,
        "model_kind": "nine_ball_macro_state_v_v3",
        "state_value_network": v_weights,
        "best_epoch": v_epoch,
        "best_val_mse": v_mse,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(q_ckpt, args.output_dir / "candidate_q_v3.pt")
    torch.save(v_ckpt, args.output_dir / "state_v_v3.pt")
    baseline = float(np.mean((y[val_ids] - np.mean(y[train_ids])) ** 2))
    report = {
        "target_kind": TARGET_KIND,
        "reward_kind": REWARD_KIND,
        "train_matches": sorted(set(match_ids[train_ids].tolist())),
        "validation_matches": val_matches,
        "train_shots": len(train_ids),
        "validation_shots": len(val_ids),
        "q": {"best_epoch": q_epoch, "validation_mse": q_mse},
        "v": {"best_epoch": v_epoch, "validation_mse": v_mse},
        "constant_train_mean_baseline_val_mse": baseline,
        "warning": (
            "Behavior-policy terminal MC returns only, selected action for Q; "
            "cannot rank arbitrary counterfactual actions reliably yet. "
            "Small smoke runs do not establish game strength."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--validation-fraction", type=float, default=0.34)
    p.add_argument("--seed", type=int, default=420100)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
