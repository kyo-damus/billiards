"""Train a state-value V(s) on COMPLETE executed-game MC returns.

The targets are per-player discounted *Macro local rewards* on actual
states visited by the behavior policy. This is a policy value, not an
all-candidate optimal max-Q label. Incomplete rollouts cannot enter
the completed-only converter used to create this dataset.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.training_utils import make_group_split, resolve_device
from src.macro.encoding import MACRO_STATE_DIM
from src.macro.value import MacroValueNetwork


def load_state_value_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        required = (
            "state", "q_target", "episode_id", "target_kind",
            "reward_kind", "turn_aware", "candidate_encoding_version",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                "State V training data missing: " + ", ".join(missing)
            )
        if (
            str(data["target_kind"]) != "selected_action_mc_return"
            or str(data["reward_kind"]) != "macro_local_reward"
            or not bool(data["turn_aware"])
            or int(data["candidate_encoding_version"]) != 2
        ):
            raise ValueError(
                "State V requires completed turn-aware v2 "
                "selected-action Macro local MC returns."
            )
        state = np.asarray(data["state"], dtype=np.float32)
        target = np.asarray(data["q_target"], dtype=np.float32).reshape(-1)
        episode = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
        n = len(target)
        if (
            n < 2 or state.shape != (n, MACRO_STATE_DIM)
            or episode.shape != (n,)
            or not np.all(np.isfinite(state))
            or not np.all(np.isfinite(target))
            or np.any(episode < 0)
        ):
            raise ValueError("State V dataset has invalid shapes/values.")
        if len(np.unique(episode)) < 2:
            raise ValueError(
                "At least two completed episodes are required for "
                "episode-disjoint validation."
            )
        return state.copy(), target.copy(), episode.copy()


def train(args):
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state, targets, episode = load_state_value_dataset(args.dataset)
    train_indices, validation_indices = make_group_split(
        episode, args.validation_fraction, args.seed
    )

    network = MacroValueNetwork(
        hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = Adam(
        network.parameters(), lr=args.learning_rate
    )
    state_tensor = torch.as_tensor(
        state, device=device, dtype=torch.float32
    )
    target_tensor = torch.as_tensor(
        targets, device=device, dtype=torch.float32
    )
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_weights = None
    best_epoch = -1

    print("=== Mixed turn-aware state V learning smoke ===")
    print(
        f"device={device} complete_episodes={len(np.unique(episode))} "
        f"transitions={len(targets)} "
        f"train={len(train_indices)} val={len(validation_indices)}"
    )
    for epoch in range(1, args.epochs + 1):
        network.train()
        train_error = []
        for batch in np.array_split(
            rng.permutation(train_indices),
            max(1, int(np.ceil(len(train_indices) / args.batch_size))),
        ):
            prediction = network(state_tensor[batch]).squeeze(-1)
            loss = F.mse_loss(prediction, target_tensor[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_error.append(float(loss.item()))

        network.eval()
        with torch.no_grad():
            predicted = network(
                state_tensor[validation_indices]
            ).squeeze(-1)
            validation_loss = float(
                F.mse_loss(
                    predicted, target_tensor[validation_indices]
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(network.state_dict())
        if epoch == 1 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_mse={np.mean(train_error):.6f} "
                f"val_mse={validation_loss:.6f}"
            )

    if best_weights is None:
        raise RuntimeError("State V did not produce a checkpoint.")
    checkpoint = {
        "state_value_network": best_weights,
        "state_dim": MACRO_STATE_DIM,
        "hidden_dim": int(args.hidden_dim),
        "target_kind": "selected_action_mc_return",
        "reward_kind": "macro_local_reward",
        "turn_aware": True,
        "dataset": str(args.dataset),
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "validation_fraction": args.validation_fraction,
    }
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"state V checkpoint saved: {output} (best epoch={best_epoch})")
    print("Integration-only: no performance inference from small returns.")
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint", default="macro_state_value_mixed_v2.pt"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=94200)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.hidden_dim < 1
        or args.batch_size < 1
        or not 0.0 < args.validation_fraction < 1.0
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
    ):
        parser.error("Invalid state V training arguments.")
    return args


if __name__ == "__main__":
    train(parse_args())
