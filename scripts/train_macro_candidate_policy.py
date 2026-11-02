import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.train_macro_candidate_value import (
    load_dataset,
    make_group_split,
    resolve_device,
)
from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
    MACRO_CANDIDATE_V2_DIM,
)
from src.macro.candidate_policy import (
    MacroCandidatePolicyNetwork,
)
from src.macro.encoding import (
    MACRO_STATE_DIM,
)


def target_policy_distribution(
    q_values,
    temperature,
):
    if temperature <= 0.0:
        raise ValueError(
            "temperature must be positive."
        )

    q_values = torch.as_tensor(
        q_values,
        dtype=torch.float32,
    )

    return torch.softmax(
        q_values / float(
            temperature
        ),
        dim=0,
    )


def make_groups(
    indices,
    state_ids,
):
    groups = []

    selected_state_ids = np.unique(
        state_ids[
            indices
        ]
    )

    index_set = set(
        np.asarray(
            indices,
            dtype=np.int64,
        ).tolist()
    )

    for state_id in selected_state_ids:
        group = [
            int(index)
            for index in np.flatnonzero(
                state_ids == state_id
            )
            if int(index) in index_set
        ]

        if group:
            groups.append(
                np.asarray(
                    group,
                    dtype=np.int64,
                )
            )

    return groups


def group_loss(
    network,
    states,
    candidates,
    q_targets,
    indices,
    temperature,
):
    index_tensor = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=states.device,
    )

    group_state = states[
        index_tensor[0]
    ]
    group_candidates = candidates[
        index_tensor
    ]
    group_q = q_targets[
        index_tensor
    ]

    logits = network(
        group_state,
        group_candidates,
    )

    target_prob = torch.softmax(
        group_q
        / float(
            temperature
        ),
        dim=0,
    )

    log_prob = F.log_softmax(
        logits,
        dim=0,
    )

    return -torch.sum(
        target_prob
        * log_prob
    )


def evaluate(
    network,
    states,
    candidates,
    q_targets,
    groups,
    temperature,
):
    losses = []
    top1_correct = 0

    network.eval()

    with torch.no_grad():
        for indices in groups:
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=states.device,
            )

            logits = network(
                states[
                    index_tensor[0]
                ],
                candidates[
                    index_tensor
                ],
            )

            target_prob = torch.softmax(
                q_targets[
                    index_tensor
                ]
                / float(
                    temperature
                ),
                dim=0,
            )

            loss = -torch.sum(
                target_prob
                * F.log_softmax(
                    logits,
                    dim=0,
                )
            )

            losses.append(
                float(
                    loss.item()
                )
            )

            q_group = (
                q_targets[
                    index_tensor
                ]
                .detach()
                .cpu()
                .numpy()
            )

            best_q = float(
                np.max(
                    q_group
                )
            )

            predicted_local = int(
                torch.argmax(
                    logits
                ).item()
            )

            top1_correct += int(
                np.isclose(
                    q_group[
                        predicted_local
                    ],
                    best_q,
                    atol=1e-6,
                )
            )

    return {
        "cross_entropy": float(
            np.mean(
                losses
            )
        ),
        "top1_accuracy": float(
            top1_correct
            / max(
                len(groups),
                1,
            )
        ),
    }


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
        q_targets_np,
        state_ids_np,
    ) = load_dataset(
        args.dataset
    )

    candidate_dim = int(candidates_np.shape[1])
    if candidate_dim not in (MACRO_CANDIDATE_DIM, MACRO_CANDIDATE_V2_DIM):
        raise ValueError("Candidate dimension must be 25 or 31.")

    (
        train_indices,
        val_indices,
    ) = make_group_split(
        state_ids_np,
        args.validation_fraction,
        args.seed,
    )

    train_groups = make_groups(
        train_indices,
        state_ids_np,
    )
    val_groups = make_groups(
        val_indices,
        state_ids_np,
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
    q_targets = torch.as_tensor(
        q_targets_np,
        dtype=torch.float32,
        device=device,
    )

    network = (
        MacroCandidatePolicyNetwork(
            candidate_dim=candidate_dim,
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

    initial_metrics = evaluate(
        network,
        states,
        candidates,
        q_targets,
        val_groups,
        args.temperature,
    )

    print(
        f"device: {device}"
    )
    print(
        f"dataset: {args.dataset}"
    )
    print(
        f"states: "
        f"{len(np.unique(state_ids_np))} "
        f"(train={len(train_groups)}, "
        f"val={len(val_groups)})"
    )
    print(
        f"samples: {len(q_targets_np)}"
    )
    print(f"candidate dimension: {candidate_dim}")
    print(
        f"target temperature: "
        f"{args.temperature:.3f}"
    )
    print(
        "Initial validation CE: "
        f"{initial_metrics['cross_entropy']:.6f}"
    )
    print(
        "Initial validation top-1: "
        f"{initial_metrics['top1_accuracy']:.3f}"
    )

    rng = np.random.default_rng(
        args.seed
    )

    best_val_loss = float(
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
            len(train_groups)
        )

        train_losses = []

        for group_index in order:
            indices = train_groups[
                int(
                    group_index
                )
            ]

            loss = group_loss(
                network,
                states,
                candidates,
                q_targets,
                indices,
                args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            train_losses.append(
                float(
                    loss.item()
                )
            )

        metrics = evaluate(
            network,
            states,
            candidates,
            q_targets,
            val_groups,
            args.temperature,
        )

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        if (
            metrics["cross_entropy"]
            < best_val_loss
        ):
            best_val_loss = (
                metrics[
                    "cross_entropy"
                ]
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
                f"train_ce={train_loss:.6f} "
                f"val_ce="
                f"{metrics['cross_entropy']:.6f} "
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
        "candidate_policy_network": (
            network.state_dict()
        ),
        "state_dim": int(
            MACRO_STATE_DIM
        ),
        "candidate_dim": int(candidate_dim),
        "candidate_encoding_version": int(
            2 if candidate_dim == MACRO_CANDIDATE_V2_DIM else 1
        ),
        "turn_aware": bool(
            candidate_dim == MACRO_CANDIDATE_V2_DIM
        ),
        "hidden_dim": int(
            args.hidden_dim
        ),
        "dataset": str(
            args.dataset
        ),
        "temperature": float(
            args.temperature
        ),
        "best_epoch": int(
            best_epoch
        ),
        "best_val_cross_entropy": float(
            best_val_loss
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
        "cross entropy: "
        f"{best_metrics['cross_entropy']:.6f}"
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
            "macro_candidate_policy.pt"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        "--temperature",
        type=float,
        default=0.5,
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
        default=93000,
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

    if args.hidden_dim < 1:
        parser.error(
            "--hidden-dim must be >= 1"
        )

    if args.temperature <= 0.0:
        parser.error(
            "--temperature must be positive"
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
