import argparse

import numpy as np
import torch

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

    agent.agent.critic.load_state_dict(
        checkpoint["critic"]
    )

    if "critic_target" in checkpoint:
        agent.agent.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

    return agent


def evaluate(args):

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"episodes: {args.episodes}")
    print(f"position radius: {args.radius:.3f} m")
    print(
        f"position mode: "
        f"{args.position_mode}"
    )

    if (
        args.position_mode
        == "outcome"
    ):
        print(
            "outcome coarse grid: "
            f"{args.outcome_grid_size} x "
            f"{args.outcome_grid_size}"
        )
        print(
            f"outcome top-k: "
            f"{args.outcome_top_k}"
        )
        print(
            "outcome dedup distance: "
            f"{args.outcome_dedup_distance:.3f} m"
        )

    print()

    agent = load_agent(
        args.checkpoint,
        device,
    )

    env = MicroBilliardEnv()

    # POSITION_ATTACKのみを生成
    sampler = MixedTacticalGoalSampler(
        position_probability=1.0,
        position_mode=(
            args.position_mode
        ),
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

    # 評価時は指定radiusに固定
    sampler.position_generator.target_radius = (
        args.radius
    )

    rng = np.random.default_rng(
        args.seed
    )

    episodes = 0

    position_pot = 0
    cue_region = 0
    position_goal = 0

    any_pocket = 0
    wrong_pocket = 0
    scratch = 0

    total_base_reward = 0.0
    total_goal_reward = 0.0

    cue_distances = []

    for episode in range(
        args.episodes
    ):

        sample = sampler.sample(
            env=env,
            rng=rng,
            seed=(
                args.seed
                + episode * 1000
            ),
        )

        observation = sample.observation
        tactical_goal = sample.goal

        action = agent.select_action(
            observation,
            tactical_goal,
            deterministic=True,
        )

        (
            _,
            base_reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_observation = (
            env.get_physical_observation()
        )

        (
            goal_reward,
            goal_info,
        ) = calculate_goal_conditioned_reward(
            base_reward=base_reward,
            goal=tactical_goal,
            before_observation=observation,
            after_observation=next_observation,
            shot_info=info,
        )

        pot_success = bool(
            info.get(
                "success",
                False,
            )
        )

        region_success = bool(
            goal_info.get(
                "cue_region_reached",
                False,
            )
        )

        full_success = bool(
            goal_info.get(
                "goal_achieved",
                False,
            )
        )

        position_pot += int(
            pot_success
        )

        cue_region += int(
            region_success
        )

        position_goal += int(
            full_success
        )

        any_pocket += int(
            info.get(
                "target_pocketed_anywhere",
                False,
            )
        )

        wrong_pocket += int(
            info.get(
                "wrong_pocket",
                False,
            )
        )

        scratch += int(
            info.get(
                "scratched",
                False,
            )
        )

        total_base_reward += float(
            base_reward
        )

        total_goal_reward += float(
            goal_reward
        )

        cue_distance_after = (
            goal_info.get(
                "cue_distance_after"
            )
        )

        if cue_distance_after is not None:
            cue_distances.append(
                float(
                    cue_distance_after
                )
            )

        episodes += 1

    if episodes == 0:
        raise RuntimeError(
            "No POSITION evaluation episodes."
        )

    print(
        "=== POSITION deterministic evaluation ==="
    )

    print(
        f"episodes: {episodes}"
    )

    print(
        "position pot: "
        f"{position_pot / episodes:.3f}"
    )

    print(
        "cue region: "
        f"{cue_region / episodes:.3f}"
    )

    print(
        "position goal: "
        f"{position_goal / episodes:.3f}"
    )

    print(
        "any pocket: "
        f"{any_pocket / episodes:.3f}"
    )

    print(
        "wrong pocket: "
        f"{wrong_pocket / episodes:.3f}"
    )

    print(
        "scratch: "
        f"{scratch / episodes:.3f}"
    )

    print(
        "avg base reward: "
        f"{total_base_reward / episodes:.3f}"
    )

    print(
        "avg goal reward: "
        f"{total_goal_reward / episodes:.3f}"
    )

    if cue_distances:
        print(
            "mean cue distance: "
            f"{np.mean(cue_distances):.3f} m"
        )

        print(
            "median cue distance: "
            f"{np.median(cue_distances):.3f} m"
        )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--position-mode",
        type=str,
        default="ideal",
        choices=[
            "ideal",
            "outcome",
        ],
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
    evaluate(
        parse_args()
    )
