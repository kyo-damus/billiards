import argparse

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
from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
    PositionResidualNetwork,
)


def resolve_device(requested):
    if requested == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return requested


def load_base_agent(
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


def load_residual_network(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = (
        "residual_network",
        "conditioned_state_dim",
        "action_dim",
        "hidden_dim",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Residual checkpoint is missing keys: "
            + ", ".join(missing)
        )

    network = PositionResidualNetwork(
        conditioned_state_dim=int(
            checkpoint[
                "conditioned_state_dim"
            ]
        ),
        action_dim=int(
            checkpoint[
                "action_dim"
            ]
        ),
        hidden_dim=int(
            checkpoint[
                "hidden_dim"
            ]
        ),
        zero_init_output=False,
    ).to(
        device
    )

    network.load_state_dict(
        checkpoint["residual_network"]
    )
    network.eval()

    return network, checkpoint


def build_game_state(env):
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


def evaluate_action(
    env,
    game_state,
    macro_action,
    observation,
    goal,
    action,
):
    env.reset_to_game_state(
        game_state,
        macro_action,
    )

    (
        _,
        base_reward,
        _,
        _,
        info,
    ) = env.step(
        action
    )

    next_observation = (
        env.get_physical_observation()
    )

    (
        goal_reward,
        goal_info,
    ) = calculate_goal_conditioned_reward(
        base_reward=base_reward,
        goal=goal,
        before_observation=observation,
        after_observation=next_observation,
        shot_info=info,
    )

    return {
        "pot": bool(
            info.get(
                "success",
                False,
            )
        ),
        "goal": bool(
            goal_info.get(
                "goal_achieved",
                False,
            )
        ),
        "cue_region": bool(
            goal_info.get(
                "cue_region_reached",
                False,
            )
        ),
        "scratch": bool(
            info.get(
                "scratched",
                False,
            )
        ),
        "wrong_pocket": bool(
            info.get(
                "wrong_pocket",
                False,
            )
        ),
        "cue_distance": (
            None
            if goal_info.get(
                "cue_distance_after"
            ) is None
            else float(
                goal_info[
                    "cue_distance_after"
                ]
            )
        ),
        "goal_reward": float(
            goal_reward
        ),
    }


def summarize(results):
    count = len(results)

    if count == 0:
        raise ValueError(
            "No evaluation results."
        )

    cue_distances = [
        item["cue_distance"]
        for item in results
        if item["cue_distance"] is not None
    ]

    return {
        "pot": float(
            np.mean(
                [
                    item["pot"]
                    for item in results
                ]
            )
        ),
        "goal": float(
            np.mean(
                [
                    item["goal"]
                    for item in results
                ]
            )
        ),
        "cue_region": float(
            np.mean(
                [
                    item["cue_region"]
                    for item in results
                ]
            )
        ),
        "scratch": float(
            np.mean(
                [
                    item["scratch"]
                    for item in results
                ]
            )
        ),
        "wrong_pocket": float(
            np.mean(
                [
                    item["wrong_pocket"]
                    for item in results
                ]
            )
        ),
        "mean_cue_distance": (
            float(
                np.mean(
                    cue_distances
                )
            )
            if cue_distances
            else float("nan")
        ),
        "mean_goal_reward": float(
            np.mean(
                [
                    item["goal_reward"]
                    for item in results
                ]
            )
        ),
    }


def print_summary(
    label,
    metrics,
):
    print(
        f"[{label}]"
    )
    print(
        f"position pot: "
        f"{metrics['pot']:.3f}"
    )
    print(
        f"cue region:   "
        f"{metrics['cue_region']:.3f}"
    )
    print(
        f"position goal:"
        f" {metrics['goal']:.3f}"
    )
    print(
        f"scratch:      "
        f"{metrics['scratch']:.3f}"
    )
    print(
        f"wrong pocket: "
        f"{metrics['wrong_pocket']:.3f}"
    )
    print(
        f"mean cue distance: "
        f"{metrics['mean_cue_distance']:.3f} m"
    )
    print(
        f"mean goal reward: "
        f"{metrics['mean_goal_reward']:.3f}"
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

    sampler = MixedTacticalGoalSampler(
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

    sampler.position_generator.target_radius = (
        args.radius
    )

    sample_env = MicroBilliardEnv()
    base_env = MicroBilliardEnv()
    residual_env = MicroBilliardEnv()

    rng = np.random.default_rng(
        args.seed
    )

    base_results = []
    residual_results = []

    residual_norms = []
    clip_count = 0

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

        residual_action = (
            residual_policy.select_action(
                observation,
                goal,
                deterministic=True,
            )
        )

        selection = (
            residual_policy.last_selection
        )

        if selection is None:
            raise RuntimeError(
                "POSITION evaluation expected "
                "a residual selection."
            )

        residual_norms.append(
            float(
                np.linalg.norm(
                    selection.residual_action
                )
            )
        )

        unclipped = (
            selection.base_action
            + selection.residual_action
        )

        if np.any(
            np.abs(
                unclipped
            )
            > 1.0
        ):
            clip_count += 1

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

        residual_results.append(
            evaluate_action(
                residual_env,
                game_state,
                macro_action,
                observation,
                goal,
                residual_action,
            )
        )

    base_metrics = summarize(
        base_results
    )
    residual_metrics = summarize(
        residual_results
    )

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
        f"residual best epoch: "
        f"{residual_checkpoint.get('best_epoch', 'n/a')}"
    )
    print(
        f"episodes: {args.episodes}"
    )
    print(
        f"position radius: "
        f"{args.radius:.3f} m"
    )
    print()

    print_summary(
        "Base Policy",
        base_metrics,
    )
    print()
    print_summary(
        "Position Residual",
        residual_metrics,
    )

    print()
    print(
        "=== Delta (Residual - Base) ==="
    )
    print(
        "position pot: "
        f"{residual_metrics['pot'] - base_metrics['pot']:+.3f}"
    )
    print(
        "cue region:   "
        f"{residual_metrics['cue_region'] - base_metrics['cue_region']:+.3f}"
    )
    print(
        "position goal:"
        f" {residual_metrics['goal'] - base_metrics['goal']:+.3f}"
    )
    print(
        "scratch:      "
        f"{residual_metrics['scratch'] - base_metrics['scratch']:+.3f}"
    )
    print(
        "mean cue distance: "
        f"{residual_metrics['mean_cue_distance'] - base_metrics['mean_cue_distance']:+.3f} m"
    )

    print()
    print(
        "mean predicted residual norm: "
        f"{np.mean(residual_norms):.4f}"
    )
    print(
        "action clip rate: "
        f"{clip_count / args.episodes:.3f}"
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
        default=50000,
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
