import argparse

import numpy as np
import torch

from scripts.eval_position_residual import (
    build_game_state,
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
    compose_residual_action,
)
from src.micro.position_residual_scale_gate import (
    AdaptivePositionResidualPolicy,
    PositionResidualScaleGate,
)


def load_scale_gate(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = (
        "gate",
        "scales",
        "input_dim",
        "hidden_dim",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Scale gate checkpoint is missing keys: "
            + ", ".join(missing)
        )

    scales = [
        float(value)
        for value in checkpoint[
            "scales"
        ]
    ]

    gate = PositionResidualScaleGate(
        input_dim=int(
            checkpoint[
                "input_dim"
            ]
        ),
        hidden_dim=int(
            checkpoint[
                "hidden_dim"
            ]
        ),
        num_scales=len(
            scales
        ),
    ).to(
        device
    )

    gate.load_state_dict(
        checkpoint[
            "gate"
        ]
    )
    gate.eval()

    threshold = float(
        checkpoint.get(
            "selection_threshold",
            0.0,
        )
    )

    return (
        gate,
        scales,
        threshold,
        checkpoint,
    )


def evaluate_seed(
    *,
    base_agent,
    residual_policy,
    adaptive_policy,
    seed,
    episodes,
    radius,
    outcome_grid_size,
    outcome_top_k,
    outcome_dedup_distance,
    fixed_scale,
    gate_scales,
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
    fixed_env = MicroBilliardEnv()
    adaptive_env = MicroBilliardEnv()

    oracle_envs = {
        scale: MicroBilliardEnv()
        for scale in gate_scales
    }

    rng = np.random.default_rng(
        seed
    )

    base_results = []
    fixed_results = []
    adaptive_results = []

    selected_scales = []
    confidences = []

    oracle_goal = 0

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
        macro_action = (
            sample.macro_action
        )
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

        residual_selection = (
            residual_policy.last_selection
        )

        if residual_selection is None:
            raise RuntimeError(
                "POSITION evaluation expected "
                "a residual selection."
            )

        fixed_action = (
            compose_residual_action(
                residual_selection.base_action,
                residual_selection.residual_action
                * float(
                    fixed_scale
                ),
            )
        )

        adaptive_action = (
            adaptive_policy.select_action(
                observation,
                goal,
                deterministic=True,
            )
        )

        adaptive_selection = (
            adaptive_policy.last_selection
        )

        if adaptive_selection is None:
            raise RuntimeError(
                "POSITION evaluation expected "
                "an adaptive scale selection."
            )

        selected_scales.append(
            adaptive_selection.residual_scale
        )
        confidences.append(
            adaptive_selection.confidence
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

        fixed_results.append(
            evaluate_action(
                fixed_env,
                game_state,
                macro_action,
                observation,
                goal,
                fixed_action,
            )
        )

        adaptive_results.append(
            evaluate_action(
                adaptive_env,
                game_state,
                macro_action,
                observation,
                goal,
                adaptive_action,
            )
        )

        oracle_success = False

        for scale in gate_scales:
            action = compose_residual_action(
                residual_selection.base_action,
                residual_selection.residual_action
                * float(scale),
            )

            result = evaluate_action(
                oracle_envs[
                    scale
                ],
                game_state,
                macro_action,
                observation,
                goal,
                action,
            )

            oracle_success = (
                oracle_success
                or bool(
                    result[
                        "goal"
                    ]
                )
            )

        oracle_goal += int(
            oracle_success
        )

    selection_counts = {
        scale: int(
            sum(
                np.isclose(
                    value,
                    scale,
                )
                for value
                in selected_scales
            )
        )
        for scale
        in gate_scales
    }

    return {
        "base": summarize(
            base_results
        ),
        "fixed": summarize(
            fixed_results
        ),
        "adaptive": summarize(
            adaptive_results
        ),
        "oracle_goal": float(
            oracle_goal
            / episodes
        ),
        "selection_counts": (
            selection_counts
        ),
        "mean_confidence": float(
            np.mean(
                confidences
            )
        ),
    }


def aggregate(
    rows,
    key,
):
    metric_names = (
        "pot",
        "cue_region",
        "goal",
        "scratch",
        "wrong_pocket",
        "mean_cue_distance",
        "mean_goal_reward",
    )

    output = {}

    for metric in metric_names:
        values = np.asarray(
            [
                row[
                    key
                ][
                    metric
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        output[
            metric
        ] = (
            float(
                np.mean(
                    values
                )
            ),
            float(
                np.std(
                    values
                )
            ),
        )

    return output


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

    (
        gate,
        scales,
        threshold,
        gate_checkpoint,
    ) = load_scale_gate(
        args.gate_checkpoint,
        device,
    )

    adaptive_policy = (
        AdaptivePositionResidualPolicy(
            base_agent=base_agent,
            residual_network=(
                residual_network
            ),
            gate=gate,
            scales=scales,
            selection_threshold=(
                threshold
            ),
            device=device,
        )
    )

    print(
        f"device: {device}"
    )
    print(
        f"residual mode: "
        f"{residual_checkpoint.get('residual_mode', 'n/a')}"
    )
    print(
        f"gate best epoch: "
        f"{gate_checkpoint.get('best_epoch', 'n/a')}"
    )
    print(
        f"gate threshold: "
        f"{threshold:.2f}"
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
        "candidate scales: "
        + ", ".join(
            f"{scale:.2f}"
            for scale in scales
        )
    )
    print(
        f"fixed comparison scale: "
        f"{args.fixed_scale:.2f}"
    )

    rows = []

    for seed in args.seeds:
        row = evaluate_seed(
            base_agent=base_agent,
            residual_policy=(
                residual_policy
            ),
            adaptive_policy=(
                adaptive_policy
            ),
            seed=int(
                seed
            ),
            episodes=args.episodes,
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
            fixed_scale=(
                args.fixed_scale
            ),
            gate_scales=scales,
        )

        rows.append(
            row
        )

        print()
        print(
            f"=== seed {seed} ==="
        )

        for label, key in (
            ("Base", "base"),
            (
                f"Fixed {args.fixed_scale:.2f}",
                "fixed",
            ),
            (
                "Adaptive Scale Gate",
                "adaptive",
            ),
        ):
            metrics = row[
                key
            ]
            print(
                f"{label}: "
                f"pot={metrics['pot']:.3f} "
                f"goal={metrics['goal']:.3f} "
                f"cue_dist="
                f"{metrics['mean_cue_distance']:.3f} "
                f"scratch="
                f"{metrics['scratch']:.3f}"
            )

        print(
            f"Oracle goal: "
            f"{row['oracle_goal']:.3f}"
        )
        print(
            "Selected scales: "
            + " ".join(
                (
                    f"{scale:.2f}:"
                    f"{row['selection_counts'][scale]}"
                )
                for scale in scales
            )
        )
        print(
            "Mean gate confidence: "
            f"{row['mean_confidence']:.3f}"
        )

    print()
    print(
        "=== Aggregate across seeds ==="
    )

    for label, key in (
        ("Base", "base"),
        (
            f"Fixed {args.fixed_scale:.2f}",
            "fixed",
        ),
        (
            "Adaptive Scale Gate",
            "adaptive",
        ),
    ):
        metrics = aggregate(
            rows,
            key,
        )

        print(
            f"{label}: "
            f"pot="
            f"{metrics['pot'][0]:.3f}"
            f"±{metrics['pot'][1]:.3f} "
            f"goal="
            f"{metrics['goal'][0]:.3f}"
            f"±{metrics['goal'][1]:.3f} "
            f"cue_dist="
            f"{metrics['mean_cue_distance'][0]:.3f}"
            f"±{metrics['mean_cue_distance'][1]:.3f} "
            f"scratch="
            f"{metrics['scratch'][0]:.3f}"
            f"±{metrics['scratch'][1]:.3f}"
        )

    oracle_values = np.asarray(
        [
            row[
                "oracle_goal"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    print(
        "Oracle goal: "
        f"{np.mean(oracle_values):.3f}"
        f"±{np.std(oracle_values):.3f}"
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
        "--gate-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fixed-scale",
        type=float,
        default=0.75,
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
            70137,
            70791,
            71473,
        ],
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

    return args


if __name__ == "__main__":
    evaluate(
        parse_args()
    )
