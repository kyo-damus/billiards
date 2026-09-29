import argparse
from collections import defaultdict

import numpy as np
import torch

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


def parse_lambdas(
    text,
):
    values = []

    for item in text.split(","):
        value = float(
            item.strip()
        )

        if not (
            0.0 <= value <= 1.0
        ):
            raise ValueError(
                "blend lambdas must be "
                "between 0 and 1."
            )

        values.append(
            value
        )

    if not values:
        raise ValueError(
            "At least one blend lambda "
            "is required."
        )

    return values


def init_stats():
    return {
        "episodes": 0,
        "position_pot": 0,
        "cue_region": 0,
        "position_goal": 0,
        "any_pocket": 0,
        "wrong_pocket": 0,
        "scratch": 0,
        "goal_reward": 0.0,
        "cue_distances": [],
    }


def update_stats(
    stats,
    base_reward,
    info,
    goal_info,
):
    stats["episodes"] += 1

    stats["position_pot"] += int(
        info.get(
            "success",
            False,
        )
    )

    stats["cue_region"] += int(
        goal_info.get(
            "cue_region_reached",
            False,
        )
    )

    stats["position_goal"] += int(
        goal_info.get(
            "goal_achieved",
            False,
        )
    )

    stats["any_pocket"] += int(
        info.get(
            "target_pocketed_anywhere",
            False,
        )
    )

    stats["wrong_pocket"] += int(
        info.get(
            "wrong_pocket",
            False,
        )
    )

    stats["scratch"] += int(
        info.get(
            "scratched",
            False,
        )
    )

    stats["goal_reward"] += float(
        goal_info.get(
            "reward",
            base_reward,
        )
    )

    cue_distance = goal_info.get(
        "cue_distance_after"
    )

    if cue_distance is not None:
        stats["cue_distances"].append(
            float(
                cue_distance
            )
        )


def print_stats(
    blend_lambda,
    stats,
):
    n = stats["episodes"]

    print()
    print(
        f"[blend lambda={blend_lambda:.2f}]"
    )
    print(
        f"episodes: {n}"
    )
    print(
        "position pot: "
        f"{stats['position_pot'] / n:.3f}"
    )
    print(
        "cue region: "
        f"{stats['cue_region'] / n:.3f}"
    )
    print(
        "position goal: "
        f"{stats['position_goal'] / n:.3f}"
    )
    print(
        "any pocket: "
        f"{stats['any_pocket'] / n:.3f}"
    )
    print(
        "wrong pocket: "
        f"{stats['wrong_pocket'] / n:.3f}"
    )
    print(
        "scratch: "
        f"{stats['scratch'] / n:.3f}"
    )

    if stats["cue_distances"]:
        print(
            "mean cue distance: "
            f"{np.mean(stats['cue_distances']):.3f} m"
        )
        print(
            "median cue distance: "
            f"{np.median(stats['cue_distances']):.3f} m"
        )


def main(
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

    lambdas = parse_lambdas(
        args.lambdas
    )

    print(
        f"device: {device}"
    )
    print(
        "base checkpoint:",
        args.base_checkpoint,
    )
    print(
        "BC checkpoint:",
        args.bc_checkpoint,
    )
    print(
        "blend lambdas:",
        ", ".join(
            f"{value:.2f}"
            for value in lambdas
        ),
    )

    base_agent = load_agent(
        args.base_checkpoint,
        device,
    )

    bc_agent = load_agent(
        args.bc_checkpoint,
        device,
    )

    sample_env = (
        MicroBilliardEnv()
    )

    eval_env = (
        MicroBilliardEnv()
    )

    sampler = (
        MixedTacticalGoalSampler(
            position_probability=1.0,
            position_mode="outcome",
            outcome_grid_size=(
                args.outcome_grid_size
            ),
            outcome_top_k=(
                args.outcome_top_k
            ),
            outcome_dedup_distance=(
                args.outcome_dedup_distance
            ),
        )
    )

    sampler.position_generator.target_radius = (
        args.radius
    )

    rng = np.random.default_rng(
        args.seed
    )

    stats_by_lambda = {
        blend_lambda: init_stats()
        for blend_lambda in lambdas
    }

    # Pairwise transition relative to lambda=0
    # if lambda 0 is included.
    base_successes = []
    success_by_lambda = {
        blend_lambda: []
        for blend_lambda in lambdas
    }

    for episode in range(
        args.episodes
    ):
        sample = sampler.sample(
            env=sample_env,
            rng=rng,
            seed=(
                args.seed
                + episode * 1000
            ),
        )

        observation = (
            sample.observation
        )

        goal = sample.goal
        macro_action = (
            sample.macro_action
        )

        state = build_game_state(
            sample_env
        )

        base_action = (
            base_agent.select_action(
                observation,
                goal,
                deterministic=True,
            )
        )

        bc_action = (
            bc_agent.select_action(
                observation,
                goal,
                deterministic=True,
            )
        )

        for blend_lambda in lambdas:
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

            eval_env.reset_to_game_state(
                state,
                macro_action,
            )

            (
                _,
                base_reward,
                _,
                _,
                info,
            ) = eval_env.step(
                action
            )

            next_observation = (
                eval_env
                .get_physical_observation()
            )

            (
                goal_reward,
                goal_info,
            ) = (
                calculate_goal_conditioned_reward(
                    base_reward=(
                        base_reward
                    ),
                    goal=goal,
                    before_observation=(
                        observation
                    ),
                    after_observation=(
                        next_observation
                    ),
                    shot_info=info,
                )
            )

            # Attach for optional logging helper.
            goal_info = dict(
                goal_info
            )
            goal_info["reward"] = (
                goal_reward
            )

            update_stats(
                stats_by_lambda[
                    blend_lambda
                ],
                base_reward,
                info,
                goal_info,
            )

            success_by_lambda[
                blend_lambda
            ].append(
                bool(
                    goal_info.get(
                        "goal_achieved",
                        False,
                    )
                )
            )

    print()
    print(
        "=== POSITION Actor Blend Evaluation ==="
    )
    print(
        f"episodes: {args.episodes}"
    )
    print(
        f"radius: {args.radius:.3f} m"
    )

    for blend_lambda in lambdas:
        print_stats(
            blend_lambda,
            stats_by_lambda[
                blend_lambda
            ],
        )

    if 0.0 in success_by_lambda:
        base_successes = (
            success_by_lambda[
                0.0
            ]
        )

        print()
        print(
            "=== Paired success changes vs lambda=0 ==="
        )

        for blend_lambda in lambdas:
            if blend_lambda == 0.0:
                continue

            current = (
                success_by_lambda[
                    blend_lambda
                ]
            )

            rescued = sum(
                (not base)
                and now
                for base, now
                in zip(
                    base_successes,
                    current,
                )
            )

            lost = sum(
                base
                and (not now)
                for base, now
                in zip(
                    base_successes,
                    current,
                )
            )

            both = sum(
                base
                and now
                for base, now
                in zip(
                    base_successes,
                    current,
                )
            )

            print(
                f"lambda={blend_lambda:.2f} "
                f"rescued={rescued:3d} "
                f"lost={lost:3d} "
                f"both={both:3d}"
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
        "--episodes",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--lambdas",
        type=str,
        default=(
            "0,0.25,0.5,0.75,1"
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
        "--seed",
        type=int,
        default=10000,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
