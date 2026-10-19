import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.micro.position_residual_policy import (
    PositionResidualNetwork,
)


REQUIRED_KEYS = (
    "conditioned_state",
    "base_action",
    "witness_action",
    "residual_target",
)


def resolve_device(requested):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def load_residual_dataset(path):
    data = np.load(path)

    missing = [
        key
        for key in REQUIRED_KEYS
        if key not in data
    ]

    if missing:
        raise ValueError(
            "Residual dataset is missing keys: "
            + ", ".join(missing)
        )

    states = np.asarray(
        data["conditioned_state"],
        dtype=np.float32,
    )
    base_actions = np.asarray(
        data["base_action"],
        dtype=np.float32,
    )
    witness_actions = np.asarray(
        data["witness_action"],
        dtype=np.float32,
    )
    residual_targets = np.asarray(
        data["residual_target"],
        dtype=np.float32,
    )

    if states.ndim != 2:
        raise ValueError(
            "conditioned_state must be 2-D."
        )

    expected_action_shape = (
        len(states),
        2,
    )

    for name, array in (
        ("base_action", base_actions),
        ("witness_action", witness_actions),
        ("residual_target", residual_targets),
    ):
        if array.shape != expected_action_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_action_shape}, "
                f"got {array.shape}."
            )

    reconstructed = (
        witness_actions
        - base_actions
    )

    if not np.allclose(
        reconstructed,
        residual_targets,
        atol=1e-6,
    ):
        raise ValueError(
            "residual_target does not match "
            "witness_action - base_action."
        )

    return (
        states,
        base_actions,
        witness_actions,
        residual_targets,
    )


def make_split(
    sample_count,
    validation_fraction,
    seed,
):
    if sample_count < 2:
        raise ValueError(
            "At least two samples are required."
        )

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must be "
            "between 0 and 1."
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        sample_count
    )

    val_count = max(
        1,
        int(
            round(
                sample_count
                * validation_fraction
            )
        ),
    )
    val_count = min(
        val_count,
        sample_count - 1,
    )

    val_indices = indices[
        :val_count
    ]
    train_indices = indices[
        val_count:
    ]

    return (
        train_indices,
        val_indices,
    )


def predict_residuals(
    network,
    states,
    base_actions,
    batch_size,
):
    predictions = []

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

            prediction = network(
                states[start:end],
                base_actions[start:end],
            )

            predictions.append(
                prediction
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return np.concatenate(
        predictions,
        axis=0,
    )


def calculate_metrics(
    prediction,
    target,
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

    mae_per_dim = np.mean(
        np.abs(error),
        axis=0,
    )

    rmse_per_dim = np.sqrt(
        np.mean(
            error ** 2,
            axis=0,
        )
    )

    target_norm = np.linalg.norm(
        target,
        axis=1,
    )
    error_norm = np.linalg.norm(
        error,
        axis=1,
    )

    return {
        "mse": mse,
        "mae_per_dim": (
            mae_per_dim.astype(
                np.float32
            )
        ),
        "rmse_per_dim": (
            rmse_per_dim.astype(
                np.float32
            )
        ),
        "mean_target_norm": float(
            np.mean(
                target_norm
            )
        ),
        "mean_error_norm": float(
            np.mean(
                error_norm
            )
        ),
    }


def residual_loss(
    prediction,
    target,
    mode,
):
    if mode == "full":
        return F.mse_loss(
            prediction,
            target,
        )

    if mode == "power_only":
        return F.mse_loss(
            prediction[:, 0],
            target[:, 0],
        )

    raise ValueError(
        f"Unsupported residual mode: {mode}"
    )


def optimization_mse(
    prediction,
    target,
    mode,
):
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )
    target = np.asarray(
        target,
        dtype=np.float32,
    )

    if mode == "full":
        error = prediction - target

    elif mode == "power_only":
        error = (
            prediction[:, 0]
            - target[:, 0]
        )

    else:
        raise ValueError(
            f"Unsupported residual mode: {mode}"
        )

    return float(
        np.mean(
            error ** 2
        )
    )


def evaluate(
    network,
    states,
    base_actions,
    targets,
    batch_size,
    mode,
):
    prediction = predict_residuals(
        network,
        states,
        base_actions,
        batch_size,
    )

    target_np = (
        targets
        .detach()
        .cpu()
        .numpy()
    )

    metrics = calculate_metrics(
        prediction,
        target_np,
    )

    metrics["optimization_mse"] = (
        optimization_mse(
            prediction,
            target_np,
            mode,
        )
    )

    return metrics


def train(args):
    device = resolve_device(
        args.device
    )

    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    (
        states_np,
        base_actions_np,
        witness_actions_np,
        residual_targets_np,
    ) = load_residual_dataset(
        args.dataset
    )

    (
        train_indices,
        val_indices,
    ) = make_split(
        len(states_np),
        args.validation_fraction,
        args.seed,
    )

    state_dim = int(
        states_np.shape[1]
    )
    action_dim = int(
        base_actions_np.shape[1]
    )

    network = PositionResidualNetwork(
        conditioned_state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        zero_init_output=True,
    ).to(
        device
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
    base_actions = torch.as_tensor(
        base_actions_np,
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(
        residual_targets_np,
        dtype=torch.float32,
        device=device,
    )

    train_states = states[
        train_indices
    ]
    train_base_actions = base_actions[
        train_indices
    ]
    train_targets = targets[
        train_indices
    ]

    val_states = states[
        val_indices
    ]
    val_base_actions = base_actions[
        val_indices
    ]
    val_targets = targets[
        val_indices
    ]

    initial_metrics = evaluate(
        network,
        val_states,
        val_base_actions,
        val_targets,
        args.batch_size,
        args.residual_mode,
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
        f"state dim: {state_dim}"
    )
    print(
        f"hidden dim: {args.hidden_dim}"
    )
    print(
        f"residual mode: "
        f"{args.residual_mode}"
    )
    print()
    print(
        "Initial validation "
        "optimization MSE: "
        f"{initial_metrics['optimization_mse']:.6f}"
    )
    print(
        "Initial validation mean error norm: "
        f"{initial_metrics['mean_error_norm']:.4f}"
    )

    rng = np.random.default_rng(
        args.seed
    )

    best_val_mse = float(
        "inf"
    )
    best_epoch = 0
    best_state = None
    best_metrics = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        network.train()

        order = rng.permutation(
            len(train_indices)
        )

        total_loss = 0.0
        total_items = 0

        for start in range(
            0,
            len(order),
            args.batch_size,
        ):
            batch_order = order[
                start:
                start + args.batch_size
            ]

            batch_states = (
                train_states[
                    batch_order
                ]
            )
            batch_base = (
                train_base_actions[
                    batch_order
                ]
            )
            batch_targets = (
                train_targets[
                    batch_order
                ]
            )

            prediction = network(
                batch_states,
                batch_base,
            )

            loss = residual_loss(
                prediction,
                batch_targets,
                args.residual_mode,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            batch_items = int(
                len(batch_order)
            )
            total_loss += (
                float(
                    loss.item()
                )
                * batch_items
            )
            total_items += batch_items

        val_metrics = evaluate(
            network,
            val_states,
            val_base_actions,
            val_targets,
            args.batch_size,
            args.residual_mode,
        )

        train_mse = (
            total_loss
            / max(
                total_items,
                1,
            )
        )

        if (
            val_metrics["optimization_mse"]
            < best_val_mse
        ):
            best_val_mse = (
                val_metrics["optimization_mse"]
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                network.state_dict()
            )
            best_metrics = (
                val_metrics
            )

        if (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.log_interval == 0
        ):
            print(
                f"epoch={epoch:3d} "
                f"train_mse={train_mse:.6f} "
                f"val_mse="
                f"{val_metrics['optimization_mse']:.6f} "
                f"val_error_norm="
                f"{val_metrics['mean_error_norm']:.4f}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a checkpoint."
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
        "residual_network": (
            network.state_dict()
        ),
        "conditioned_state_dim": (
            state_dim
        ),
        "action_dim": action_dim,
        "hidden_dim": int(
            args.hidden_dim
        ),
        "base_checkpoint": (
            args.base_checkpoint
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
        "validation_fraction": float(
            args.validation_fraction
        ),
        "seed": int(
            args.seed
        ),
        "residual_mode": (
            args.residual_mode
        ),
        "initial_val_metrics": {
            "mse": float(
                initial_metrics["mse"]
            ),
            "optimization_mse": float(
                initial_metrics[
                    "optimization_mse"
                ]
            ),
            "mean_error_norm": float(
                initial_metrics[
                    "mean_error_norm"
                ]
            ),
        },
        "best_val_metrics": {
            "mse": float(
                best_metrics["mse"]
            ),
            "optimization_mse": float(
                best_metrics[
                    "optimization_mse"
                ]
            ),
            "mean_error_norm": float(
                best_metrics[
                    "mean_error_norm"
                ]
            ),
            "power_mae": float(
                best_metrics[
                    "mae_per_dim"
                ][0]
            ),
            "angle_mae": float(
                best_metrics[
                    "mae_per_dim"
                ][1]
            ),
            "power_rmse": float(
                best_metrics[
                    "rmse_per_dim"
                ][0]
            ),
            "angle_rmse": float(
                best_metrics[
                    "rmse_per_dim"
                ][1]
            ),
        },
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
        "optimization MSE: "
        f"{best_metrics['optimization_mse']:.6f}"
    )
    print(
        "mean residual error norm: "
        f"{best_metrics['mean_error_norm']:.4f}"
    )
    print(
        "MAE power / angle: "
        f"{best_metrics['mae_per_dim'][0]:.4f} / "
        f"{best_metrics['mae_per_dim'][1]:.4f}"
    )
    print(
        "RMSE power / angle: "
        f"{best_metrics['rmse_per_dim'][0]:.4f} / "
        f"{best_metrics['rmse_per_dim'][1]:.4f}"
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
        "--base-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "position_residual_mlp.pt"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
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
        "--residual-mode",
        type=str,
        default="full",
        choices=[
            "full",
            "power_only",
        ],
        help=(
            "full learns both power and angle "
            "residuals. power_only learns only "
            "the power residual; the angle output "
            "remains zero from initialization."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261019,
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
