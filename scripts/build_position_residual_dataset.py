import argparse

import numpy as np
import torch

from scripts.train_outcome_witness_bc import (
    deterministic_action,
    generate_dataset as generate_witness_dataset,
    load_agent,
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


def compute_base_actions(
    actor,
    conditioned_states,
    batch_size,
    device,
):
    actor.eval()

    outputs = []

    states = torch.as_tensor(
        conditioned_states,
        dtype=torch.float32,
        device=device,
    )

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

            action = deterministic_action(
                actor,
                states[
                    start:end
                ],
            )

            outputs.append(
                action
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return np.concatenate(
        outputs,
        axis=0,
    )


def build_residual_targets(
    base_actions,
    witness_actions,
):
    base_actions = np.asarray(
        base_actions,
        dtype=np.float32,
    )

    witness_actions = np.asarray(
        witness_actions,
        dtype=np.float32,
    )

    if (
        base_actions.shape
        != witness_actions.shape
    ):
        raise ValueError(
            "base_actions and witness_actions "
            "must have the same shape."
        )

    return (
        witness_actions
        - base_actions
    ).astype(
        np.float32
    )


def print_residual_stats(
    residual_targets,
):
    residual_targets = np.asarray(
        residual_targets,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        residual_targets,
        axis=1,
    )

    abs_residual = np.abs(
        residual_targets
    )

    print()
    print(
        "=== Position Residual Dataset ==="
    )
    print(
        f"samples: "
        f"{len(residual_targets)}"
    )

    print()
    print(
        "[Residual norm]"
    )
    print(
        f"mean:   "
        f"{np.mean(norms):.4f}"
    )
    print(
        f"median: "
        f"{np.median(norms):.4f}"
    )
    print(
        f"p90:    "
        f"{np.quantile(norms, 0.90):.4f}"
    )
    print(
        f"p95:    "
        f"{np.quantile(norms, 0.95):.4f}"
    )

    names = [
        "power",
        "angle",
    ]

    print()
    print(
        "[Per-dimension residual]"
    )

    for index, name in enumerate(
        names
    ):
        values = residual_targets[
            :,
            index,
        ]

        print(
            f"{name:>5s}: "
            f"mean={np.mean(values): .4f} "
            f"std={np.std(values):.4f} "
            f"mean_abs="
            f"{np.mean(abs_residual[:, index]):.4f}"
        )


def build_dataset(
    args,
):
    device = resolve_device(
        args.device
    )

    print(
        f"device: {device}"
    )
    print(
        f"base checkpoint: "
        f"{args.base_checkpoint}"
    )
    print(
        f"samples: {args.demos}"
    )

    print()
    print(
        "Generating outcome-guided "
        "witness samples..."
    )

    (
        conditioned_states,
        witness_actions,
    ) = generate_witness_dataset(
        args
    )

    base_agent, _ = load_agent(
        args.base_checkpoint,
        device,
    )

    base_actions = compute_base_actions(
        base_agent.agent.actor,
        conditioned_states,
        args.batch_size,
        device,
    )

    residual_targets = (
        build_residual_targets(
            base_actions,
            witness_actions,
        )
    )

    print_residual_stats(
        residual_targets
    )

    np.savez_compressed(
        args.output,
        conditioned_state=(
            conditioned_states
        ),
        base_action=(
            base_actions
        ),
        witness_action=(
            witness_actions
        ),
        residual_target=(
            residual_targets
        ),
    )

    print()
    print(
        f"saved: {args.output}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "position_residual_dataset.npz"
        ),
    )

    parser.add_argument(
        "--demos",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dedup-distance",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
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

    if args.demos < 1:
        parser.error(
            "--demos must be >= 1"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be >= 1"
        )

    return args


if __name__ == "__main__":
    build_dataset(
        parse_args()
    )
