import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.eval_position_residual import (
    build_game_state,
    evaluate_action,
    load_base_agent,
    load_residual_network,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)
from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
    compose_residual_action,
)
from src.micro.position_residual_scale_gate import (
    POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM,
    PositionResidualScaleGate,
    build_position_residual_scale_features,
)


_WORKER_SAMPLE_ENV = None
_WORKER_EVAL_ENV = None
_WORKER_SAMPLER = None
_WORKER_BASE_AGENT = None
_WORKER_RESIDUAL_POLICY = None
_WORKER_SCALES = None


def parse_scales(text):
    values = [
        float(item.strip())
        for item in text.split(",")
        if item.strip()
    ]

    if not values:
        raise ValueError(
            "At least one residual scale is required."
        )

    if any(
        value < 0.0
        for value in values
    ):
        raise ValueError(
            "Residual scales must be >= 0."
        )

    if 0.0 not in values:
        raise ValueError(
            "Residual scales must include 0.0."
        )

    return values


def _init_worker(
    base_checkpoint,
    residual_checkpoint,
    scales,
    radius,
    outcome_grid_size,
    outcome_top_k,
    outcome_dedup_distance,
):
    global _WORKER_SAMPLE_ENV
    global _WORKER_EVAL_ENV
    global _WORKER_SAMPLER
    global _WORKER_BASE_AGENT
    global _WORKER_RESIDUAL_POLICY
    global _WORKER_SCALES

    torch.set_num_threads(1)

    _WORKER_SAMPLE_ENV = MicroBilliardEnv()
    _WORKER_EVAL_ENV = MicroBilliardEnv()

    _WORKER_SAMPLER = (
        MixedTacticalGoalSampler(
            position_probability=1.0,
            position_mode="outcome",
            outcome_grid_size=(
                outcome_grid_size
            ),
            outcome_top_k=(
                outcome_top_k
            ),
            outcome_dedup_distance=(
                outcome_dedup_distance
            ),
        )
    )

    _WORKER_SAMPLER.position_generator.target_radius = (
        radius
    )

    _WORKER_BASE_AGENT = load_base_agent(
        base_checkpoint,
        "cpu",
    )

    (
        residual_network,
        _,
    ) = load_residual_network(
        residual_checkpoint,
        "cpu",
    )

    _WORKER_RESIDUAL_POLICY = (
        PositionResidualMicroPolicy(
            base_agent=(
                _WORKER_BASE_AGENT
            ),
            residual_network=(
                residual_network
            ),
            device="cpu",
        )
    )

    _WORKER_SCALES = list(
        scales
    )


def _generate_example(seed):
    rng = np.random.default_rng(
        int(seed)
    )

    sample = _WORKER_SAMPLER.sample(
        env=_WORKER_SAMPLE_ENV,
        rng=rng,
        seed=int(seed),
    )

    observation = sample.observation
    goal = sample.goal
    macro_action = sample.macro_action

    game_state = build_game_state(
        _WORKER_SAMPLE_ENV
    )

    _ = (
        _WORKER_RESIDUAL_POLICY
        .select_action(
            observation,
            goal,
            deterministic=True,
        )
    )

    selection = (
        _WORKER_RESIDUAL_POLICY
        .last_selection
    )

    if selection is None:
        raise RuntimeError(
            "POSITION sample expected "
            "a residual selection."
        )

    conditioned_state = (
        _WORKER_BASE_AGENT
        .build_conditioned_state(
            observation,
            goal,
        )
    )

    features = (
        build_position_residual_scale_features(
            conditioned_state,
            selection.base_action,
            selection.residual_action,
        )
    )

    success_mask = []
    pot_mask = []

    for scale in _WORKER_SCALES:
        action = compose_residual_action(
            selection.base_action,
            selection.residual_action
            * float(scale),
        )

        result = evaluate_action(
            _WORKER_EVAL_ENV,
            game_state,
            macro_action,
            observation,
            goal,
            action,
        )

        success_mask.append(
            float(
                result["goal"]
            )
        )
        pot_mask.append(
            float(
                result["pot"]
            )
        )

    return (
        features,
        np.asarray(
            success_mask,
            dtype=np.float32,
        ),
        np.asarray(
            pot_mask,
            dtype=np.float32,
        ),
    )


def generate_dataset(
    args,
    scales,
):
    seeds = [
        args.seed
        + index * 1000
        for index in range(
            args.samples
        )
    ]

    if args.workers <= 0:
        _init_worker(
            args.base_checkpoint,
            args.residual_checkpoint,
            scales,
            args.radius,
            args.outcome_grid_size,
            args.outcome_top_k,
            args.outcome_dedup_distance,
        )

        rows = [
            _generate_example(
                seed
            )
            for seed in seeds
        ]

    else:
        context = mp.get_context(
            "spawn"
        )

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(
                args.base_checkpoint,
                args.residual_checkpoint,
                scales,
                args.radius,
                args.outcome_grid_size,
                args.outcome_top_k,
                args.outcome_dedup_distance,
            ),
        ) as executor:
            rows = list(
                executor.map(
                    _generate_example,
                    seeds,
                    chunksize=args.chunksize,
                )
            )

    features = np.stack(
        [
            row[0]
            for row in rows
        ]
    ).astype(
        np.float32
    )

    success_masks = np.stack(
        [
            row[1]
            for row in rows
        ]
    ).astype(
        np.float32
    )

    pot_masks = np.stack(
        [
            row[2]
            for row in rows
        ]
    ).astype(
        np.float32
    )

    return (
        features,
        success_masks,
        pot_masks,
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

    confidence, selected = torch.max(
        probabilities,
        dim=1,
    )

    selected = selected.clone()

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
            float(
                threshold
            ),
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
            best_capable = (
                capable_rate
            )

    return (
        best_threshold,
        best_overall,
        best_capable,
    )


def train(
    args,
):
    scales = parse_scales(
        args.scales
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
        "scales:",
        scales,
    )
    print()
    print(
        "Generating per-scale Position Goal "
        "success dataset..."
    )

    (
        features_np,
        success_np,
        pot_np,
    ) = generate_dataset(
        args,
        scales,
    )

    capable = (
        success_np.sum(
            axis=1
        ) > 0
    )

    pot_capable = (
        pot_np.sum(
            axis=1
        ) > 0
    )

    print(
        "goal-capable rate: "
        f"{capable.mean():.3f}"
    )
    print(
        "pot-capable rate: "
        f"{pot_capable.mean():.3f}"
    )
    print(
        "per-scale Position Goal rates:"
    )

    for index, scale in enumerate(
        scales
    ):
        print(
            f"  scale={scale:.2f}: "
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

    zero_index = scales.index(
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

    for index, scale in enumerate(
        scales
    ):
        print(
            f"  scale={scale:.2f}: "
            f"{float(pos_weight[index].item()):.3f}"
        )

    model = PositionResidualScaleGate(
        input_dim=(
            POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM
        ),
        hidden_dim=(
            args.hidden_dim
        ),
        num_scales=len(
            scales
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
                scales
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
                key: (
                    value.detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in model.state_dict().items()
            }

        count_text = " ".join(
            (
                f"{scales[index]:.2f}:"
                f"{int(counts[index])}"
            )
            for index in range(
                len(scales)
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
            "No best gate was recorded."
        )

    model.load_state_dict(
        best_state
    )

    checkpoint = {
        "gate": model.state_dict(),
        "scales": scales,
        "input_dim": (
            POSITION_RESIDUAL_SCALE_GATE_INPUT_DIM
        ),
        "hidden_dim": (
            args.hidden_dim
        ),
        "base_checkpoint": (
            args.base_checkpoint
        ),
        "residual_checkpoint": (
            args.residual_checkpoint
        ),
        "training_samples": (
            args.samples
        ),
        "training_seed": (
            args.seed
        ),
        "gate_objective": (
            "per_scale_position_goal_success"
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
        "Position Residual Scale Gate "
        "training completed."
    )
    print(
        f"best epoch: {best_epoch}"
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
        "--residual-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "position_residual_scale_gate.pt"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6000,
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
        default=128,
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
        "--scales",
        type=str,
        default=(
            "0,0.25,0.5,0.75,1.0"
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
        default=60000,
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

    if args.samples < 1:
        parser.error(
            "--samples must be >= 1"
        )

    if args.epochs < 1:
        parser.error(
            "--epochs must be >= 1"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be >= 1"
        )

    if args.workers < 0:
        parser.error(
            "--workers must be >= 0"
        )

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
    train(
        parse_args()
    )
