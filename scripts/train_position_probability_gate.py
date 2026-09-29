import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.train_position_success_gate import (
    generate_dataset,
    parse_lambdas,
)
from src.micro.position_blend_gate import (
    POSITION_BLEND_GATE_INPUT_DIM,
    PositionBlendGate,
)


def selection_metrics(
    logits,
    success_masks,
    zero_index,
    threshold,
):
    probabilities = torch.sigmoid(
        logits
    )

    confidence, predicted = (
        torch.max(
            probabilities,
            dim=1,
        )
    )

    selected = predicted.clone()

    fallback = (
        confidence < threshold
    )

    selected[
        fallback
    ] = zero_index

    row_index = torch.arange(
        len(selected),
        device=selected.device,
    )

    selected_success = (
        success_masks[
            row_index,
            selected,
        ] > 0
    )

    capable = (
        success_masks.sum(
            dim=1
        ) > 0
    )

    overall = float(
        selected_success
        .float()
        .mean()
        .item()
    )

    if bool(
        capable.any()
    ):
        capable_rate = float(
            selected_success[
                capable
            ]
            .float()
            .mean()
            .item()
        )
    else:
        capable_rate = float(
            "nan"
        )

    return (
        overall,
        capable_rate,
        selected,
    )


def choose_threshold(
    logits,
    success_masks,
    zero_index,
):
    best_threshold = 0.0
    best_overall = -1.0
    best_capable = -1.0

    for threshold in np.linspace(
        0.0,
        0.95,
        20,
    ):
        (
            overall,
            capable_rate,
            _,
        ) = selection_metrics(
            logits,
            success_masks,
            zero_index,
            float(threshold),
        )

        score = (
            overall,
            capable_rate,
        )

        if score > (
            best_overall,
            best_capable,
        ):
            best_threshold = float(
                threshold
            )
            best_overall = overall
            best_capable = capable_rate

    return (
        best_threshold,
        best_overall,
        best_capable,
    )


def main(
    args,
):
    lambdas = parse_lambdas(
        args.lambdas
    )

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(
        f"device: {device}"
    )
    print(
        f"samples: {args.samples}"
    )
    print(
        f"workers: {args.workers}"
    )
    print(
        "lambdas:",
        lambdas,
    )

    print()
    print(
        "Generating per-lambda success dataset..."
    )

    (
        features_np,
        success_np,
    ) = generate_dataset(
        args,
        lambdas,
    )

    capable = (
        success_np.sum(
            axis=1
        ) > 0
    )

    print(
        "success-capable rate: "
        f"{capable.mean():.3f}"
    )

    print(
        "per-lambda success rates:"
    )

    for index, value in enumerate(
        lambdas
    ):
        print(
            f"  lambda={value:.2f}: "
            f"{success_np[:, index].mean():.3f}"
        )

    rng = np.random.default_rng(
        args.seed + 999999
    )

    order = rng.permutation(
        len(features_np)
    )

    val_count = max(
        1,
        int(
            len(order)
            * args.validation_fraction
        ),
    )

    val_indices = order[
        :val_count
    ]

    train_indices = order[
        val_count:
    ]

    train_x = torch.as_tensor(
        features_np[
            train_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    train_y = torch.as_tensor(
        success_np[
            train_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    val_x = torch.as_tensor(
        features_np[
            val_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    val_y = torch.as_tensor(
        success_np[
            val_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    zero_index = lambdas.index(
        0.0
    )

    positive_counts = (
        train_y.sum(
            dim=0
        )
    )

    negative_counts = (
        len(train_y)
        - positive_counts
    )

    pos_weight = (
        negative_counts
        / torch.clamp(
            positive_counts,
            min=1.0,
        )
    )

    pos_weight = torch.clamp(
        pos_weight,
        min=1.0,
        max=args.max_pos_weight,
    )

    print(
        "positive weights:"
    )

    for index, value in enumerate(
        lambdas
    ):
        print(
            f"  lambda={value:.2f}: "
            f"{float(pos_weight[index].item()):.3f}"
        )

    model = PositionBlendGate(
        input_dim=(
            POSITION_BLEND_GATE_INPUT_DIM
        ),
        hidden_dim=(
            args.hidden_dim
        ),
        num_lambdas=len(
            lambdas
        ),
    ).to(
        device
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_threshold = 0.0
    best_overall = -1.0
    best_capable = -1.0
    best_epoch = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        permutation = rng.permutation(
            len(train_indices)
        )

        losses = []

        for start in range(
            0,
            len(permutation),
            args.batch_size,
        ):
            batch_ids = permutation[
                start:
                start + args.batch_size
            ]

            batch_ids = torch.as_tensor(
                batch_ids,
                dtype=torch.long,
                device=device,
            )

            batch_x = train_x[
                batch_ids
            ]

            batch_y = train_y[
                batch_ids
            ]

            logits = model(
                batch_x
            )

            loss_matrix = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    batch_y,
                    pos_weight=pos_weight,
                    reduction="none",
                )
            )

            capable_rows = (
                batch_y.sum(
                    dim=1
                ) > 0
            ).float()

            row_weights = (
                1.0
                + capable_rows
                * (
                    args.capable_weight
                    - 1.0
                )
            )

            loss = (
                loss_matrix.mean(
                    dim=1
                )
                * row_weights
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        model.eval()

        with torch.no_grad():
            val_logits = model(
                val_x
            )

        (
            threshold,
            overall,
            capable_rate,
        ) = choose_threshold(
            val_logits,
            val_y,
            zero_index,
        )

        (
            _,
            _,
            selected,
        ) = selection_metrics(
            val_logits,
            val_y,
            zero_index,
            threshold,
        )

        counts = torch.bincount(
            selected,
            minlength=len(
                lambdas
            ),
        ).detach().cpu().numpy()

        score = (
            overall,
            capable_rate,
        )

        if score > (
            best_overall,
            best_capable,
        ):
            best_overall = overall
            best_capable = capable_rate
            best_threshold = threshold
            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        count_text = " ".join(
            (
                f"{lambdas[index]:.2f}:"
                f"{int(counts[index])}"
            )
            for index in range(
                len(lambdas)
            )
        )

        print(
            f"epoch={epoch:3d} "
            f"loss={np.mean(losses):.4f} "
            f"val_selected_success="
            f"{overall:.3f} "
            f"val_success_when_capable="
            f"{capable_rate:.3f} "
            f"threshold={threshold:.2f} "
            f"counts=[{count_text}]"
        )

    if best_state is None:
        raise RuntimeError(
            "No best model was recorded."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint = {
        "gate": model.state_dict(),
        "lambdas": lambdas,
        "input_dim": (
            POSITION_BLEND_GATE_INPUT_DIM
        ),
        "hidden_dim": (
            args.hidden_dim
        ),
        "base_checkpoint": (
            args.base_checkpoint
        ),
        "bc_checkpoint": (
            args.bc_checkpoint
        ),
        "training_samples": (
            args.samples
        ),
        "training_seed": (
            args.seed
        ),
        "gate_objective": (
            "per_lambda_binary_success"
        ),
        "selection_threshold": (
            best_threshold
        ),
        "best_epoch": (
            best_epoch
        ),
        "validation_selected_success": (
            best_overall
        ),
        "validation_success_when_capable": (
            best_capable
        ),
    }

    torch.save(
        checkpoint,
        args.checkpoint,
    )

    print()
    print(
        "Position probability gate "
        "training completed."
    )
    print(
        f"best epoch: "
        f"{best_epoch}"
    )
    print(
        f"best threshold: "
        f"{best_threshold:.2f}"
    )
    print(
        "best validation selected success: "
        f"{best_overall:.3f}"
    )
    print(
        "best validation success when capable: "
        f"{best_capable:.3f}"
    )
    print(
        f"checkpoint: "
        f"{args.checkpoint}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "position_probability_gate.pt"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--capable-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default=(
            "0,0.1,0.2,0.3,0.4,"
            "0.5,0.75,1.0"
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--outcome-grid-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--outcome-top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--outcome-dedup-distance",
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
        default=40000,
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

    if args.capable_weight < 1.0:
        parser.error(
            "--capable-weight must be >= 1"
        )

    if args.max_pos_weight < 1.0:
        parser.error(
            "--max-pos-weight must be >= 1"
        )

    return args


if __name__ == "__main__":
    main(
        parse_args()
    )
