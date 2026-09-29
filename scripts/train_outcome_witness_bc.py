import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.conditioner import (
    encode_micro_goal,
)
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)
from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)


_WORKER_ENV = None
_WORKER_GENERATOR = None


def _init_worker(
    grid_size,
    radius,
    top_k,
    dedup_distance,
):
    global _WORKER_ENV
    global _WORKER_GENERATOR

    _WORKER_ENV = MicroBilliardEnv()

    _WORKER_GENERATOR = (
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=grid_size,
            target_radius=radius,
            top_k=top_k,
            dedup_distance=dedup_distance,
        )
    )


def _build_game_state(
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


def _conditioned_state(
    observation,
    goal,
):
    goal_vector = encode_micro_goal(
        observation,
        goal,
    )

    return np.concatenate(
        [
            observation,
            goal_vector,
        ]
    ).astype(
        np.float32
    )


def _sample_witness(
    seed,
    max_attempts=100,
):
    if (
        _WORKER_ENV is None
        or _WORKER_GENERATOR is None
    ):
        raise RuntimeError(
            "Worker is not initialized."
        )

    rng = np.random.default_rng(
        int(seed)
    )

    placeholder = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=0,
    )

    for attempt in range(
        max_attempts
    ):
        _WORKER_ENV.set_macro_action(
            placeholder
        )

        _WORKER_ENV.reset(
            seed=(
                int(seed)
                + attempt
            )
        )

        state = _build_game_state(
            _WORKER_ENV
        )

        witnesses = (
            _WORKER_GENERATOR
            .generate_with_witnesses(
                state
            )
        )

        if not witnesses:
            continue

        index = int(
            rng.integers(
                0,
                len(witnesses),
            )
        )

        witness = witnesses[
            index
        ]

        _WORKER_ENV.reset_to_game_state(
            state,
            witness.candidate.action,
        )

        observation = (
            _WORKER_ENV
            .get_physical_observation()
        )

        conditioned = (
            _conditioned_state(
                observation,
                witness.candidate.goal,
            )
        )

        action = np.asarray(
            witness.witness_action,
            dtype=np.float32,
        )

        return (
            conditioned,
            action,
        )

    raise RuntimeError(
        "Could not generate an outcome-guided "
        "witness sample."
    )


def load_agent(
    checkpoint_path,
    device,
):
    agent = GoalConditionedSACAgent(
        action_dim=2,
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    agent.agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.agent.critic.load_state_dict(
        checkpoint["critic"]
    )

    if "critic_target" in checkpoint:
        agent.agent.critic_target.load_state_dict(
            checkpoint[
                "critic_target"
            ]
        )

    if "log_alpha" in checkpoint:
        with torch.no_grad():
            agent.agent.log_alpha.copy_(
                checkpoint[
                    "log_alpha"
                ].to(
                    device=(
                        agent.agent
                        .log_alpha.device
                    ),
                    dtype=(
                        agent.agent
                        .log_alpha.dtype
                    ),
                )
            )

    return agent, checkpoint


def generate_dataset(
    args,
):
    seeds = [
        args.seed
        + index * 1000
        for index in range(
            args.demos
        )
    ]

    if args.workers <= 0:
        _init_worker(
            args.grid_size,
            args.radius,
            args.top_k,
            args.dedup_distance,
        )

        samples = [
            _sample_witness(
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
                args.grid_size,
                args.radius,
                args.top_k,
                args.dedup_distance,
            ),
        ) as executor:
            samples = list(
                executor.map(
                    _sample_witness,
                    seeds,
                    chunksize=(
                        args.chunksize
                    ),
                )
            )

    states = np.stack(
        [
            sample[0]
            for sample in samples
        ]
    ).astype(
        np.float32
    )

    actions = np.stack(
        [
            sample[1]
            for sample in samples
        ]
    ).astype(
        np.float32
    )

    return (
        states,
        actions,
    )


def deterministic_action(
    actor,
    states,
):
    mu, _ = actor(
        states
    )

    return torch.tanh(
        mu
    )


def evaluate_mse(
    actor,
    states,
    actions,
    batch_size,
):
    losses = []

    actor.eval()

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

            prediction = (
                deterministic_action(
                    actor,
                    states[
                        start:end
                    ],
                )
            )

            loss = F.mse_loss(
                prediction,
                actions[
                    start:end
                ],
            )

            losses.append(
                float(
                    loss.item()
                )
            )

    actor.train()

    return float(
        np.mean(
            losses
        )
    )


def save_checkpoint(
    source_checkpoint,
    agent,
    path,
    args,
):
    output = copy.deepcopy(
        source_checkpoint
    )

    output["actor"] = (
        agent.agent.actor.state_dict()
    )

    output[
        "witness_bc"
    ] = {
        "demos": int(
            args.demos
        ),
        "epochs": int(
            args.epochs
        ),
        "learning_rate": float(
            args.learning_rate
        ),
        "grid_size": int(
            args.grid_size
        ),
        "radius": float(
            args.radius
        ),
        "top_k": int(
            args.top_k
        ),
        "dedup_distance": float(
            args.dedup_distance
        ),
    }

    torch.save(
        output,
        path,
    )


def train(
    args,
):
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
        f"checkpoint: "
        f"{args.init_checkpoint}"
    )
    print(
        f"demos: {args.demos}"
    )
    print(
        f"workers: {args.workers}"
    )

    print()
    print(
        "Generating witness demonstrations..."
    )

    (
        states_np,
        actions_np,
    ) = generate_dataset(
        args
    )

    rng = np.random.default_rng(
        args.seed
    )

    indices = rng.permutation(
        len(states_np)
    )

    val_count = max(
        1,
        int(
            len(indices)
            * args.validation_fraction
        ),
    )

    val_indices = indices[
        :val_count
    ]

    train_indices = indices[
        val_count:
    ]

    if len(train_indices) == 0:
        raise ValueError(
            "Training split is empty."
        )

    agent, source_checkpoint = (
        load_agent(
            args.init_checkpoint,
            device,
        )
    )

    actor = agent.agent.actor

    optimizer = Adam(
        actor.parameters(),
        lr=args.learning_rate,
    )

    states = torch.as_tensor(
        states_np,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        actions_np,
        dtype=torch.float32,
        device=device,
    )

    train_states = states[
        train_indices
    ]
    train_actions = actions[
        train_indices
    ]

    val_states = states[
        val_indices
    ]
    val_actions = actions[
        val_indices
    ]

    initial_train_mse = (
        evaluate_mse(
            actor,
            train_states,
            train_actions,
            args.batch_size,
        )
    )

    initial_val_mse = (
        evaluate_mse(
            actor,
            val_states,
            val_actions,
            args.batch_size,
        )
    )

    print(
        "initial train mse: "
        f"{initial_train_mse:.6f}"
    )
    print(
        "initial validation mse: "
        f"{initial_val_mse:.6f}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        order = rng.permutation(
            len(train_indices)
        )

        epoch_losses = []

        actor.train()

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

            batch_actions = (
                train_actions[
                    batch_order
                ]
            )

            prediction = (
                deterministic_action(
                    actor,
                    batch_states,
                )
            )

            loss = F.mse_loss(
                prediction,
                batch_actions,
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        val_mse = evaluate_mse(
            actor,
            val_states,
            val_actions,
            args.batch_size,
        )

        print(
            f"epoch={epoch:3d} "
            f"train_mse="
            f"{np.mean(epoch_losses):.6f} "
            f"val_mse="
            f"{val_mse:.6f}"
        )

    save_checkpoint(
        source_checkpoint,
        agent,
        args.checkpoint,
        args,
    )

    print()
    print(
        "Witness behavior cloning completed."
    )
    print(
        f"checkpoint: "
        f"{args.checkpoint}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "goal_conditioned_micro_witness_bc.pt"
        ),
    )

    parser.add_argument(
        "--demos",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
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
        "--grad-clip",
        type=float,
        default=5.0,
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

    if args.demos < 2:
        parser.error(
            "--demos must be >= 2"
        )

    if not (
        0.0
        < args.validation_fraction
        < 1.0
    ):
        parser.error(
            "--validation-fraction "
            "must be between 0 and 1"
        )

    return args


if __name__ == "__main__":
    train(
        parse_args()
    )
