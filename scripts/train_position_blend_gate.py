import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.common.types import (
    GameState,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.reward import (
    calculate_goal_conditioned_reward,
)
from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)
from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)
from src.micro.position_blend_gate import (
    POSITION_BLEND_GATE_INPUT_DIM,
    PositionBlendGate,
    build_position_blend_gate_features,
    choose_oracle_lambda_index,
)


_WORKER_SAMPLE_ENV = None
_WORKER_EVAL_ENV = None
_WORKER_SAMPLER = None
_WORKER_BASE_AGENT = None
_WORKER_BC_AGENT = None
_WORKER_LAMBDAS = None
_WORKER_ZERO_INDEX = None


def parse_lambdas(
    text,
):
    values = [
        float(
            item.strip()
        )
        for item in text.split(",")
        if item.strip()
    ]

    if not values:
        raise ValueError(
            "At least one lambda is required."
        )

    if any(
        value < 0.0
        or value > 1.0
        for value in values
    ):
        raise ValueError(
            "lambdas must be in [0, 1]."
        )

    if 0.0 not in values:
        raise ValueError(
            "lambda list must include 0.0."
        )

    return values


def load_agent(
    checkpoint_path,
):
    agent = GoalConditionedSACAgent(
        action_dim=2,
        device="cpu",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    agent.agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.agent.actor.eval()

    return agent


def build_game_state(
    env,
):
    snapshot = env.sim.snapshot()

    return GameState(
        ball_positions=(
            snapshot.positions.copy()
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=(
            snapshot.pocket_indices.copy()
        ),
    )


def _init_worker(
    base_checkpoint,
    bc_checkpoint,
    lambdas,
    radius,
    outcome_grid_size,
    outcome_top_k,
    outcome_dedup_distance,
):
    global _WORKER_SAMPLE_ENV
    global _WORKER_EVAL_ENV
    global _WORKER_SAMPLER
    global _WORKER_BASE_AGENT
    global _WORKER_BC_AGENT
    global _WORKER_LAMBDAS
    global _WORKER_ZERO_INDEX

    torch.set_num_threads(
        1
    )

    _WORKER_SAMPLE_ENV = (
        MicroBilliardEnv()
    )

    _WORKER_EVAL_ENV = (
        MicroBilliardEnv()
    )

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

    _WORKER_BASE_AGENT = (
        load_agent(
            base_checkpoint
        )
    )

    _WORKER_BC_AGENT = (
        load_agent(
            bc_checkpoint
        )
    )

    _WORKER_LAMBDAS = list(
        lambdas
    )

    _WORKER_ZERO_INDEX = (
        _WORKER_LAMBDAS.index(
            0.0
        )
    )


def evaluate_action(
    state,
    observation,
    goal,
    macro_action,
    action,
):
    _WORKER_EVAL_ENV.reset_to_game_state(
        state,
        macro_action,
    )

    (
        _,
        base_reward,
        _,
        _,
        info,
    ) = _WORKER_EVAL_ENV.step(
        action
    )

    next_observation = (
        _WORKER_EVAL_ENV
        .get_physical_observation()
    )

    (
        _,
        goal_info,
    ) = calculate_goal_conditioned_reward(
        base_reward=base_reward,
        goal=goal,
        before_observation=observation,
        after_observation=(
            next_observation
        ),
        shot_info=info,
    )

    return (
        bool(
            goal_info.get(
                "goal_achieved",
                False,
            )
        ),
        bool(
            info.get(
                "success",
                False,
            )
        ),
        bool(
            info.get(
                "scratched",
                False,
            )
        ),
        float(
            goal_info.get(
                "cue_distance_after",
                float("nan"),
            )
        ),
    )


def _generate_training_example(
    seed,
):
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
    macro_action = (
        sample.macro_action
    )

    state = build_game_state(
        _WORKER_SAMPLE_ENV
    )

    base_action = (
        _WORKER_BASE_AGENT
        .select_action(
            observation,
            goal,
            deterministic=True,
        )
    )

    bc_action = (
        _WORKER_BC_AGENT
        .select_action(
            observation,
            goal,
            deterministic=True,
        )
    )

    conditioned_state = (
        _WORKER_BASE_AGENT
        .build_conditioned_state(
            observation,
            goal,
        )
    )

    features = (
        build_position_blend_gate_features(
            conditioned_state,
            base_action,
            bc_action,
        )
    )

    goal_success = []
    pot_success = []
    scratched = []
    cue_distances = []

    for blend_lambda in (
        _WORKER_LAMBDAS
    ):
        action = (
            (1.0 - blend_lambda)
            * base_action
            + blend_lambda
            * bc_action
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        (
            goal_ok,
            pot_ok,
            scratch,
            cue_distance,
        ) = evaluate_action(
            state,
            observation,
            goal,
            macro_action,
            action,
        )

        goal_success.append(
            goal_ok
        )
        pot_success.append(
            pot_ok
        )
        scratched.append(
            scratch
        )
        cue_distances.append(
            cue_distance
        )

    label = (
        choose_oracle_lambda_index(
            goal_success,
            pot_success,
            scratched,
            cue_distances,
            zero_index=(
                _WORKER_ZERO_INDEX
            ),
        )
    )

    success_possible = bool(
        any(
            goal_success
        )
    )

    return (
        features,
        int(label),
        float(
            success_possible
        ),
    )


def generate_dataset(
    args,
    lambdas,
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
            args.bc_checkpoint,
            lambdas,
            args.radius,
            args.outcome_grid_size,
            args.outcome_top_k,
            args.outcome_dedup_distance,
        )

        rows = [
            _generate_training_example(
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
                args.bc_checkpoint,
                lambdas,
                args.radius,
                args.outcome_grid_size,
                args.outcome_top_k,
                args.outcome_dedup_distance,
            ),
        ) as executor:
            rows = list(
                executor.map(
                    _generate_training_example,
                    seeds,
                    chunksize=(
                        args.chunksize
                    ),
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

    labels = np.asarray(
        [
            row[1]
            for row in rows
        ],
        dtype=np.int64,
    )

    success_possible = np.asarray(
        [
            row[2]
            for row in rows
        ],
        dtype=np.float32,
    )

    return (
        features,
        labels,
        success_possible,
    )


def accuracy(
    model,
    features,
    labels,
):
    model.eval()

    with torch.no_grad():
        logits = model(
            features
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        return float(
            (
                predictions
                == labels
            )
            .float()
            .mean()
            .item()
        )


def subset_accuracy(
    model,
    features,
    labels,
    mask,
):
    if int(
        mask.sum().item()
    ) == 0:
        return float(
            "nan"
        )

    return accuracy(
        model,
        features[
            mask
        ],
        labels[
            mask
        ],
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
        "Generating oracle-labeled "
        "gate dataset..."
    )

    (
        features_np,
        labels_np,
        success_possible_np,
    ) = generate_dataset(
        args,
        lambdas,
    )

    counts = np.bincount(
        labels_np,
        minlength=len(
            lambdas
        ),
    )

    print(
        "label counts:"
    )

    for index, value in enumerate(
        lambdas
    ):
        print(
            f"  lambda={value:.2f}: "
            f"{counts[index]}"
        )

    print(
        "oracle-success-capable rate: "
        f"{success_possible_np.mean():.3f}"
    )

    rng = np.random.default_rng(
        args.seed
        + 999999
    )

    order = rng.permutation(
        len(
            features_np
        )
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
        labels_np[
            train_indices
        ],
        dtype=torch.long,
        device=device,
    )

    train_success = torch.as_tensor(
        success_possible_np[
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
        labels_np[
            val_indices
        ],
        dtype=torch.long,
        device=device,
    )

    val_success = torch.as_tensor(
        success_possible_np[
            val_indices
        ],
        dtype=torch.bool,
        device=device,
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
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        permutation = rng.permutation(
            len(
                train_indices
            )
        )

        losses = []

        for start in range(
            0,
            len(permutation),
            args.batch_size,
        ):
            batch_index = (
                permutation[
                    start:
                    start
                    + args.batch_size
                ]
            )

            batch_index = (
                torch.as_tensor(
                    batch_index,
                    dtype=torch.long,
                    device=device,
                )
            )

            batch_x = train_x[
                batch_index
            ]

            batch_y = train_y[
                batch_index
            ]

            batch_success = (
                train_success[
                    batch_index
                ]
            )

            logits = model(
                batch_x
            )

            per_example_loss = (
                F.cross_entropy(
                    logits,
                    batch_y,
                    reduction="none",
                )
            )

            weights = (
                1.0
                + batch_success
                * (
                    args.success_weight
                    - 1.0
                )
            )

            loss = (
                per_example_loss
                * weights
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        val_acc = accuracy(
            model,
            val_x,
            val_y,
        )

        success_val_acc = (
            subset_accuracy(
                model,
                val_x,
                val_y,
                val_success,
            )
        )

        print(
            f"epoch={epoch:3d} "
            f"loss={np.mean(losses):.4f} "
            f"val_acc={val_acc:.3f} "
            f"success_val_acc="
            f"{success_val_acc:.3f}"
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
    }

    torch.save(
        checkpoint,
        args.checkpoint,
    )

    print()
    print(
        "Adaptive blend gate training "
        "completed."
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
            "position_blend_gate.pt"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        "--success-weight",
        type=float,
        default=3.0,
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
        default=20000,
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

    if args.samples < 10:
        parser.error(
            "--samples must be >= 10"
        )

    if args.success_weight < 1.0:
        parser.error(
            "--success-weight must be >= 1"
        )

    return args


if __name__ == "__main__":
    main(
        parse_args()
    )
