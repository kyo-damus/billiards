"""Train 31-D Candidate Policy on MCTS root visit distributions.

This is separate from softmax(oracle Q) supervision. Accepts
incomplete rollouts, as long as the saved search root contains
candidate visit counts. Small runs only verify the training path.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.training_utils import make_group_split, make_groups, resolve_device
from src.macro.candidate_policy import MacroCandidatePolicyNetwork
from src.macro.candidate_encoding import MACRO_CANDIDATE_V2_DIM
from src.macro.encoding import MACRO_STATE_DIM


def load_visit_dataset(path):
    required = (
        "state", "candidate", "policy_target", "state_id",
        "episode_id", "candidate_encoding_version",
        "turn_aware", "target_kind",
    )
    with np.load(path, allow_pickle=False) as f:
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(
                "Visit Policy dataset missing: " + ", ".join(missing)
            )
        if (
            int(f["candidate_encoding_version"]) != 2
            or not bool(f["turn_aware"])
            or str(f["target_kind"]) != "mcts_root_visit_distribution"
        ):
            raise ValueError("Expected turn-aware MCTS visit Policy v2.")

        state = np.asarray(f["state"], dtype=np.float32)
        cand = np.asarray(f["candidate"], dtype=np.float32)
        target = np.asarray(f["policy_target"], dtype=np.float32)
        groups = np.asarray(f["state_id"], dtype=np.int64)
        episode = np.asarray(f["episode_id"], dtype=np.int64)
        n = len(target)
        if (
            state.shape != (n, MACRO_STATE_DIM)
            or cand.shape != (n, MACRO_CANDIDATE_V2_DIM)
            or groups.shape != (n,)
            or episode.shape != (n,)
            or n == 0
        ):
            raise ValueError("Invalid visit Policy dataset shapes.")
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(cand))
            or not np.all(np.isfinite(target))
            or np.any(target < 0.0)
        ):
            raise ValueError("Invalid visit Policy values.")
        for group in np.unique(groups):
            selected = groups == group
            if selected.sum() < 2 or not np.isclose(
                target[selected].sum(), 1.0, atol=1e-5
            ):
                raise ValueError("Each MCTS root must be a distribution.")
            if not np.all(episode[selected] == episode[selected][0]):
                raise ValueError("A root must belong to one episode.")
        return state, cand, target, groups, episode


def policy_loss(network, states, candidates, targets, indices):
    index = torch.as_tensor(
        indices, dtype=torch.long, device=states.device
    )
    logits = network(states[index[0]], candidates[index])
    return -(targets[index] * F.log_softmax(logits, dim=0)).sum()


def evaluate_visits(network, states, candidates, targets, groups):
    network.eval()
    ce = []
    correct = 0
    with torch.no_grad():
        for indices in groups:
            index = torch.as_tensor(
                indices, dtype=torch.long, device=states.device
            )
            logits = network(states[index[0]], candidates[index])
            prob = targets[index]
            ce.append(
                float(-(prob * F.log_softmax(logits, dim=0)).sum().item())
            )
            max_prob = prob.max()
            correct += int(
                torch.isclose(
                    prob[torch.argmax(logits)], max_prob, atol=1e-6
                )
            )
    return {
        "cross_entropy": float(np.mean(ce)),
        "top1_accuracy": float(correct / len(groups)),
    }


def train(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_np, cand_np, target_np, group_ids, episode_ids = (
        load_visit_dataset(args.dataset)
    )
    # Split by WHOLE EPISODE, never by individual turns in the
    # same game. This avoids putting adjacent states in train+val.
    training_idx, validation_idx = make_group_split(
        episode_ids, args.validation_fraction, args.seed
    )
    training = make_groups(training_idx, group_ids)
    validation = make_groups(validation_idx, group_ids)
    if not training or not validation:
        raise ValueError("Need at least two episodes for train/validation.")

    states = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    candidates = torch.as_tensor(
        cand_np, dtype=torch.float32, device=device
    )
    targets = torch.as_tensor(
        target_np, dtype=torch.float32, device=device
    )
    network = MacroCandidatePolicyNetwork(
        candidate_dim=MACRO_CANDIDATE_V2_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = Adam(network.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best_ce = float("inf")
    best = None
    best_metrics = None
    best_epoch = 0

    print("=== Mixed Candidate Policy via MCTS visits ===")
    print(f"device: {device}")
    print(
        f"episodes={len(np.unique(episode_ids))} "
        f"root_states={len(np.unique(group_ids))} "
        f"candidate_rows={len(target_np)}"
    )
    for epoch in range(1, args.epochs + 1):
        network.train()
        losses = []
        for position in rng.permutation(len(training)):
            group = training[int(position)]
            loss = policy_loss(
                network, states, candidates, targets, group
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate_visits(
            network, states, candidates, targets, validation
        )
        if metrics["cross_entropy"] < best_ce:
            best_ce = metrics["cross_entropy"]
            best = copy.deepcopy(network.state_dict())
            best_metrics = dict(metrics)
            best_epoch = epoch
        if epoch == 1 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_ce={np.mean(losses):.5f} "
                f"val_ce={metrics['cross_entropy']:.5f}"
            )

    if best is None:
        raise RuntimeError("No usable training checkpoint.")
    network.load_state_dict(best)
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_policy_network": network.state_dict(),
            "state_dim": MACRO_STATE_DIM,
            "candidate_dim": MACRO_CANDIDATE_V2_DIM,
            "candidate_encoding_version": 2,
            "turn_aware": True,
            "hidden_dim": args.hidden_dim,
            "target_kind": "mcts_root_visit_distribution",
            "dataset": str(args.dataset),
            "best_epoch": best_epoch,
            "best_val_cross_entropy": best_ce,
            "best_val_metrics": best_metrics,
        },
        path,
    )
    print(f"saved Policy checkpoint: {path} (best epoch={best_epoch})")
    print(
        "MCTS imitation smoke only. Not evidence of better "
        "gameplay or unbiased policy quality."
    )
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint", default="macro_candidate_policy_visits_v2.pt"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=93000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.hidden_dim < 1
        or args.learning_rate <= 0.0
        or not 0.0 < args.validation_fraction < 1.0
    ):
        parser.error("Invalid training settings.")
    return args


if __name__ == "__main__":
    train(parse_args())
