import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
)
from src.macro.candidate_value import (
    MacroCandidateValueNetwork,
)
from src.macro.encoding import (
    MACRO_STATE_DIM,
)


REQUIRED_KEYS = (
    "state",
    "candidate",
    "q_target",
    "state_id",
)


def resolve_device(
    requested,
):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def load_dataset(
    path,
):
    data = np.load(
        path
    )

    missing = [
        key
        for key in REQUIRED_KEYS
        if key not in data
    ]

    if missing:
        raise ValueError(
            "Candidate value dataset "
            "is missing keys: "
            + ", ".join(missing)
        )

    states = np.asarray(
        data["state"],
        dtype=np.float32,
    )
    candidates = np.asarray(
        data["candidate"],
        dtype=np.float32,
    )
    targets = np.asarray(
        data["q_target"],
        dtype=np.float32,
    ).reshape(-1)

    state_ids = np.asarray(
        data["state_id"],
        dtype=np.int64,
    ).reshape(-1)

    sample_count = len(
        targets
    )

    if states.shape != (
        sample_count,
        MACRO_STATE_DIM,
    ):
        raise ValueError(
            "state must have shape "
            f"({sample_count}, "
            f"{MACRO_STATE_DIM})."
        )

    if candidates.shape != (
        sample_count,
        MACRO_CANDIDATE_DIM,
    ):
        raise ValueError(
            "candidate must have shape "
            f"({sample_count}, "
            f"{MACRO_CANDIDATE_DIM})."
        )

    if state_ids.shape != (
        sample_count,
    ):
        raise ValueError(
            "state_id length does not "
            "match q_target."
        )

    return (
        states,
        candidates,
        targets,
        state_ids,
    )


def make_group_split(
    state_ids,
    validation_fraction,
    seed,
):
    unique_ids = np.unique(
        state_ids
    )

    if len(unique_ids) < 2:
        raise ValueError(
            "At least two distinct states "
            "are required."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled = rng.permutation(
        unique_ids
    )

    val_state_count = max(
        1,
        int(
            round(
                len(unique_ids)
                * validation_fraction
            )
        ),
    )
    val_state_count = min(
        val_state_count,
        len(unique_ids) - 1,
    )

    val_ids = set(
        shuffled[
            :val_state_count
        ].tolist()
    )

    val_mask = np.asarray(
        [
            state_id in val_ids
            for state_id in state_ids
        ],
        dtype=bool,
    )

    train_indices = np.flatnonzero(
        ~val_mask
    )
    val_indices = np.flatnonzero(
        val_mask
    )

    return (
        train_indices,
        val_indices,
    )


def predict(
    network,
    states,
    candidates,
    batch_size,
):
    outputs = []

    network.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(states),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(states),
            )

            output = network(
                states[
                    start:end
                ],
                candidates[
                    start:end
                ],
            )

            outputs.append(
                output
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

    return np.concatenate(
        outputs,
        axis=0,
    )


def calculate_metrics(
    prediction,
    target,
    state_ids,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )
    target = np.asarray(
        target,
        dtype=np.float32,
    )

    error = (
        prediction
        - target
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )
    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )
    rmse = float(
        np.sqrt(
            mse
        )
    )

    top1_correct = 0
    top1_total = 0

    for state_id in np.unique(
        state_ids
    ):
        indices = np.flatnonzero(
            state_ids
            == state_id
        )

        if len(indices) < 2:
            continue

        group_targets = target[
            indices
        ]

        best_target = float(
            np.max(
                group_targets
            )
        )

        predicted_best = indices[
            int(
                np.argmax(
                    prediction[
                        indices
                    ]
                )
            )
        ]

        # 教師Qが同点の候補は
        # どれを選んでもtop-1正解とする。
        top1_correct += int(
            np.isclose(
                target[
                    predicted_best
                ],
                best_target,
                atol=1e-6,
            )
        )
        top1_total += 1

    top1_accuracy = (
        float(
            top1_correct
            / top1_total
        )
        if top1_total > 0
        else float("nan")
    )

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "top1_accuracy": (
            top1_accuracy
        ),
    }


def evaluate(
    network,
    states,
    candidates,
    targets,
    state_ids,
    batch_size,
):
    prediction = predict(
        network,
        states,
        candidates,
        batch_size,
    )

    return calculate_metrics(
        prediction,
        targets,
        state_ids,
    )


def train(
    args,
):
    device = resolve_device(
        args.device
    )

    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )

    (
        states_np,
        candidates_np,
        targets_np,
        state_ids_np,
    ) = load_dataset(
        args.dataset
    )

    (
        train_indices,
        val_indices,
    ) = make_group_split(
        state_ids_np,
        args.validation_fraction,
        args.seed,
    )

    network = (
        MacroCandidateValueNetwork(
            hidden_dim=args.hidden_dim,
        )
        .to(
            device
        )
    )

    optimizer = Adam(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    states = torch.as_tensor(
        states_np,
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.as_tensor(
        candidates_np,
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(
        targets_np,
        dtype=torch.float32,
        device=device,
    )

    train_states = states[
        train_indices
    ]
    train_candidates = candidates[
        train_indices
    ]
    train_targets = targets[
        train_indices
    ]

    val_states = states[
        val_indices
    ]
    val_candidates = candidates[
        val_indices
    ]
    val_targets_np = targets_np[
        val_indices
    ]
    val_state_ids = state_ids_np[
        val_indices
    ]

    rng = np.random.default_rng(
        args.seed
    )

    best_val_mse = float(
        "inf"
    )
    best_epoch = 0
    best_state = None
    best_metrics = None

    initial_metrics = evaluate(
        network,
        val_states,
        val_candidates,
        val_targets_np,
        val_state_ids,
        args.batch_size,
    )

    print(
        f"device: {device}"
    )
    print(
        f"dataset: {args.dataset}"
    )
    print(
        f"samples: {len(states_np)} "
        f"(train={len(train_indices)}, "
        f"val={len(val_indices)})"
    )
    print(
        "states: "
        f"{len(np.unique(state_ids_np))}"
    )
    print(
        "Initial validation MSE: "
        f"{initial_metrics['mse']:.6f}"
    )
    print(
        "Initial validation top-1: "
        f"{initial_metrics['top1_accuracy']:.3f}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        network.train()

        order = rng.permutation(
            len(train_indices)
        )

        train_loss_sum = 0.0
        train_items = 0

        for start in range(
            0,
            len(order),
            args.batch_size,
        ):
            batch_order = order[
                start:
                start + args.batch_size
            ]

            prediction = network(
                train_states[
                    batch_order
                ],
                train_candidates[
                    batch_order
                ],
            )

            loss = F.mse_loss(
                prediction,
                train_targets[
                    batch_order
                ],
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            batch_items = len(
                batch_order
            )

            train_loss_sum += (
                float(
                    loss.item()
                )
                * batch_items
            )
            train_items += (
                batch_items
            )

        metrics = evaluate(
            network,
            val_states,
            val_candidates,
            val_targets_np,
            val_state_ids,
            args.batch_size,
        )

        train_mse = (
            train_loss_sum
            / max(
                train_items,
                1,
            )
        )

        if (
            metrics["mse"]
            < best_val_mse
        ):
            best_val_mse = (
                metrics["mse"]
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                network.state_dict()
            )
            best_metrics = dict(
                metrics
            )

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch
            % args.log_interval
            == 0
        ):
            print(
                f"epoch={epoch:3d} "
                f"train_mse={train_mse:.6f} "
                f"val_mse={metrics['mse']:.6f} "
                f"val_top1="
                f"{metrics['top1_accuracy']:.3f}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce "
            "a checkpoint."
        )

    network.load_state_dict(
        best_state
    )

    output_path = Path(
        args.checkpoint
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "candidate_value_network": (
            network.state_dict()
        ),
        "state_dim": int(
            MACRO_STATE_DIM
        ),
        "candidate_dim": int(
            MACRO_CANDIDATE_DIM
        ),
        "hidden_dim": int(
            args.hidden_dim
        ),
        "dataset": str(
            args.dataset
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_val_mse": float(
            best_val_mse
        ),
        "best_val_metrics": (
            best_metrics
        ),
        "validation_fraction": float(
            args.validation_fraction
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )

    print()
    print(
        "=== Best validation result ==="
    )
    print(
        f"best epoch: {best_epoch}"
    )
    print(
        f"MSE: {best_metrics['mse']:.6f}"
    )
    print(
        f"MAE: {best_metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {best_metrics['rmse']:.4f}"
    )
    print(
        "top-1 candidate accuracy: "
        f"{best_metrics['top1_accuracy']:.3f}"
    )
    print(
        f"checkpoint: {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "macro_candidate_value.pt"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=92000,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be >= 1"
        )

    if args.hidden_dim < 1:
        parser.error(
            "--hidden-dim must be >= 1"
        )

    if not (
        0.0
        < args.validation_fraction
        < 1.0
    ):
        parser.error(
            "--validation-fraction must be "
            "between 0 and 1"
        )

    return args


if __name__ == "__main__":
    train(
        parse_args()
    )
