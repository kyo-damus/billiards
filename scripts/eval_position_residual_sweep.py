import argparse

import numpy as np

from scripts.eval_position_residual import (
    build_game_state,
    build_scaled_residual_action,
    evaluate_action,
    load_base_agent,
    load_residual_network,
    resolve_device,
    summarize,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)
from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
)


METRIC_KEYS = (
    "pot",
    "cue_region",
    "goal",
    "scratch",
    "wrong_pocket",
    "mean_cue_distance",
    "mean_goal_reward",
)


def evaluate_seed(
    *,
    base_agent,
    residual_policy,
    seed,
    episodes,
    radius,
    outcome_grid_size,
    outcome_top_k,
    outcome_dedup_distance,
    residual_scales,
    power_scale,
    angle_scale,
):
    sampler = MixedTacticalGoalSampler(
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

    sampler.position_generator.target_radius = (
        radius
    )

    sample_env = MicroBilliardEnv()
    base_env = MicroBilliardEnv()

    scale_envs = {
        scale: MicroBilliardEnv()
        for scale in residual_scales
    }

    rng = np.random.default_rng(
        seed
    )

    base_results = []
    scale_results = {
        scale: []
        for scale in residual_scales
    }

    residual_norms = {
        scale: []
        for scale in residual_scales
    }

    clip_counts = {
        scale: 0
        for scale in residual_scales
    }

    for episode in range(
        episodes
    ):
        sample = sampler.sample(
            env=sample_env,
            rng=rng,
            seed=(
                seed
                + episode * 1000
            ),
        )

        observation = (
            sample.observation.copy()
        )
        goal = sample.goal
        macro_action = sample.macro_action

        game_state = build_game_state(
            sample_env
        )

        base_action = (
            base_agent.select_action(
                observation,
                goal,
                deterministic=True,
            )
        )

        _ = residual_policy.select_action(
            observation,
            goal,
            deterministic=True,
        )

        selection = (
            residual_policy.last_selection
        )

        if selection is None:
            raise RuntimeError(
                "POSITION evaluation expected "
                "a residual selection."
            )

        base_results.append(
            evaluate_action(
                base_env,
                game_state,
                macro_action,
                observation,
                goal,
                base_action,
            )
        )

        raw_residual = np.asarray(
            selection.residual_action,
            dtype=np.float32,
        )

        for scale in residual_scales:
            scaled_residual = (
                raw_residual.copy()
            )
            scaled_residual[0] *= float(
                power_scale
            )
            scaled_residual[1] *= float(
                angle_scale
            )
            scaled_residual *= float(
                scale
            )

            residual_norms[
                scale
            ].append(
                float(
                    np.linalg.norm(
                        scaled_residual
                    )
                )
            )

            unclipped = (
                selection.base_action
                + scaled_residual
            )

            if np.any(
                np.abs(
                    unclipped
                )
                > 1.0
            ):
                clip_counts[
                    scale
                ] += 1

            action = (
                build_scaled_residual_action(
                    selection,
                    residual_scale=scale,
                    power_scale=(
                        power_scale
                    ),
                    angle_scale=(
                        angle_scale
                    ),
                )
            )

            scale_results[
                scale
            ].append(
                evaluate_action(
                    scale_envs[scale],
                    game_state,
                    macro_action,
                    observation,
                    goal,
                    action,
                )
            )

    base_metrics = summarize(
        base_results
    )

    scale_metrics = {}

    for scale in residual_scales:
        metrics = summarize(
            scale_results[scale]
        )

        metrics[
            "mean_residual_norm"
        ] = float(
            np.mean(
                residual_norms[
                    scale
                ]
            )
        )

        metrics[
            "clip_rate"
        ] = float(
            clip_counts[scale]
            / episodes
        )

        scale_metrics[
            scale
        ] = metrics

    return (
        base_metrics,
        scale_metrics,
    )


def aggregate(
    per_seed,
):
    output = {}

    keys = list(
        METRIC_KEYS
    ) + [
        "mean_residual_norm",
        "clip_rate",
    ]

    for key in keys:
        values = np.asarray(
            [
                metrics[key]
                for metrics
                in per_seed
            ],
            dtype=np.float64,
        )

        output[key] = {
            "mean": float(
                np.mean(
                    values
                )
            ),
            "std": float(
                np.std(
                    values
                )
            ),
        }

    return output


def print_seed_result(
    seed,
    base_metrics,
    scale_metrics,
):
    print()
    print(
        f"=== seed {seed} ==="
    )
    print(
        "Base: "
        f"pot={base_metrics['pot']:.3f} "
        f"goal={base_metrics['goal']:.3f} "
        f"cue_dist="
        f"{base_metrics['mean_cue_distance']:.3f} "
        f"scratch={base_metrics['scratch']:.3f}"
    )

    for scale, metrics in (
        scale_metrics.items()
    ):
        print(
            f"scale={scale:.2f}: "
            f"pot={metrics['pot']:.3f} "
            f"cue_region="
            f"{metrics['cue_region']:.3f} "
            f"goal={metrics['goal']:.3f} "
            f"cue_dist="
            f"{metrics['mean_cue_distance']:.3f} "
            f"scratch="
            f"{metrics['scratch']:.3f} "
            f"clip="
            f"{metrics['clip_rate']:.3f}"
        )


def evaluate(args):
    device = resolve_device(
        args.device
    )

    base_agent = load_base_agent(
        args.base_checkpoint,
        device,
    )

    (
        residual_network,
        residual_checkpoint,
    ) = load_residual_network(
        args.residual_checkpoint,
        device,
    )

    residual_policy = (
        PositionResidualMicroPolicy(
            base_agent=base_agent,
            residual_network=(
                residual_network
            ),
            device=device,
        )
    )

    residual_scales = [
        float(
            value
        )
        for value in args.residual_scales
    ]

    print(
        f"device: {device}"
    )
    print(
        f"base checkpoint: "
        f"{args.base_checkpoint}"
    )
    print(
        f"residual checkpoint: "
        f"{args.residual_checkpoint}"
    )
    print(
        f"residual mode: "
        f"{residual_checkpoint.get('residual_mode', 'n/a')}"
    )
    print(
        f"episodes per seed: "
        f"{args.episodes}"
    )
    print(
        "seeds: "
        + ", ".join(
            str(seed)
            for seed in args.seeds
        )
    )
    print(
        "residual scales: "
        + ", ".join(
            f"{scale:.2f}"
            for scale in residual_scales
        )
    )
    print(
        "power / angle scale: "
        f"{args.power_scale:.3f} / "
        f"{args.angle_scale:.3f}"
    )

    base_by_seed = []
    scale_by_seed = {
        scale: []
        for scale in residual_scales
    }

    for seed in args.seeds:
        (
            base_metrics,
            scale_metrics,
        ) = evaluate_seed(
            base_agent=base_agent,
            residual_policy=(
                residual_policy
            ),
            seed=int(
                seed
            ),
            episodes=(
                args.episodes
            ),
            radius=args.radius,
            outcome_grid_size=(
                args.outcome_grid_size
            ),
            outcome_top_k=(
                args.outcome_top_k
            ),
            outcome_dedup_distance=(
                args.outcome_dedup_distance
            ),
            residual_scales=(
                residual_scales
            ),
            power_scale=(
                args.power_scale
            ),
            angle_scale=(
                args.angle_scale
            ),
        )

        base_by_seed.append(
            base_metrics
        )

        for scale in residual_scales:
            scale_by_seed[
                scale
            ].append(
                scale_metrics[
                    scale
                ]
            )

        print_seed_result(
            seed,
            base_metrics,
            scale_metrics,
        )

    base_aggregate = {}

    for key in METRIC_KEYS:
        values = np.asarray(
            [
                metrics[key]
                for metrics
                in base_by_seed
            ],
            dtype=np.float64,
        )
        base_aggregate[key] = {
            "mean": float(
                np.mean(
                    values
                )
            ),
            "std": float(
                np.std(
                    values
                )
            ),
        }

    print()
    print(
        "=== Aggregate across seeds ==="
    )
    print(
        "Base: "
        f"pot="
        f"{base_aggregate['pot']['mean']:.3f}"
        f"±{base_aggregate['pot']['std']:.3f} "
        f"goal="
        f"{base_aggregate['goal']['mean']:.3f}"
        f"±{base_aggregate['goal']['std']:.3f} "
        f"cue_dist="
        f"{base_aggregate['mean_cue_distance']['mean']:.3f}"
        f"±{base_aggregate['mean_cue_distance']['std']:.3f}"
    )

    best_scale = None
    best_goal = -np.inf

    for scale in residual_scales:
        metrics = aggregate(
            scale_by_seed[
                scale
            ]
        )

        mean_goal = metrics[
            "goal"
        ][
            "mean"
        ]

        if mean_goal > best_goal:
            best_goal = mean_goal
            best_scale = scale

        print(
            f"scale={scale:.2f}: "
            f"pot="
            f"{metrics['pot']['mean']:.3f}"
            f"±{metrics['pot']['std']:.3f} "
            f"cue_region="
            f"{metrics['cue_region']['mean']:.3f}"
            f"±{metrics['cue_region']['std']:.3f} "
            f"goal="
            f"{metrics['goal']['mean']:.3f}"
            f"±{metrics['goal']['std']:.3f} "
            f"cue_dist="
            f"{metrics['mean_cue_distance']['mean']:.3f}"
            f"±{metrics['mean_cue_distance']['std']:.3f} "
            f"scratch="
            f"{metrics['scratch']['mean']:.3f}"
            f"±{metrics['scratch']['std']:.3f} "
            f"clip="
            f"{metrics['clip_rate']['mean']:.3f}"
        )

    print()
    print(
        "best mean Position Goal scale: "
        f"{best_scale:.2f} "
        f"(goal={best_goal:.3f})"
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
        "--episodes",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            50137,
            50791,
            51473,
        ],
    )

    parser.add_argument(
        "--residual-scales",
        type=float,
        nargs="+",
        default=[
            0.25,
            0.50,
            0.75,
            1.00,
        ],
    )

    parser.add_argument(
        "--power-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--angle-scale",
        type=float,
        default=0.0,
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

    if args.episodes < 1:
        parser.error(
            "--episodes must be >= 1"
        )

    if len(args.seeds) < 1:
        parser.error(
            "--seeds requires at least "
            "one seed"
        )

    if len(
        args.residual_scales
    ) < 1:
        parser.error(
            "--residual-scales requires "
            "at least one value"
        )

    if any(
        scale < 0.0
        for scale
        in args.residual_scales
    ):
        parser.error(
            "residual scales must be >= 0"
        )

    return args


if __name__ == "__main__":
    evaluate(
        parse_args()
    )
