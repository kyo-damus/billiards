import argparse
from collections import Counter

import numpy as np
import torch

from src.common.types import GameState
from src.env.micro_billiard_env import MicroBilliardEnv
from src.goal.reward import calculate_goal_conditioned_reward
from src.micro.goal_conditioned_sac import GoalConditionedSACAgent
from src.micro.goal_sampler import MixedTacticalGoalSampler
from src.micro.position_blend_gate import (
    PositionBlendGate,
    build_position_blend_gate_features,
    choose_oracle_lambda_index,
)


def load_agent(path, device):
    agent = GoalConditionedSACAgent(
        action_dim=2,
        device=device,
    )
    checkpoint = torch.load(
        path,
        map_location=device,
    )
    agent.agent.actor.load_state_dict(
        checkpoint["actor"]
    )
    agent.agent.actor.eval()
    return agent


def load_gate(path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
    )
    lambdas = [
        float(value)
        for value in checkpoint["lambdas"]
    ]
    model = PositionBlendGate(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
        num_lambdas=len(lambdas),
    ).to(device)
    model.load_state_dict(
        checkpoint["gate"]
    )
    model.eval()

    gate_threshold = float(
        checkpoint.get(
            "selection_threshold",
            0.0,
        )
    )

    return (
        model,
        lambdas,
        gate_threshold,
    )


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
    state,
    observation,
    goal,
    macro_action,
    action,
):
    env.reset_to_game_state(
        state,
        macro_action,
    )
    (
        _,
        base_reward,
        _,
        _,
        info,
    ) = env.step(action)

    next_observation = (
        env.get_physical_observation()
    )

    (
        _,
        goal_info,
    ) = calculate_goal_conditioned_reward(
        base_reward=base_reward,
        goal=goal,
        before_observation=observation,
        after_observation=next_observation,
        shot_info=info,
    )

    return {
        "goal": bool(
            goal_info.get(
                "goal_achieved",
                False,
            )
        ),
        "pot": bool(
            info.get(
                "success",
                False,
            )
        ),
        "scratch": bool(
            info.get(
                "scratched",
                False,
            )
        ),
        "wrong": bool(
            info.get(
                "wrong_pocket",
                False,
            )
        ),
        "distance": float(
            goal_info.get(
                "cue_distance_after",
                float("nan"),
            )
        ),
    }


def summarize(results):
    distances = [
        item["distance"]
        for item in results
        if np.isfinite(
            item["distance"]
        )
    ]

    return {
        "n": len(results),
        "goal": float(
            np.mean(
                [
                    item["goal"]
                    for item in results
                ]
            )
        ),
        "pot": float(
            np.mean(
                [
                    item["pot"]
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
        "wrong": float(
            np.mean(
                [
                    item["wrong"]
                    for item in results
                ]
            )
        ),
        "mean_distance": float(
            np.mean(distances)
        ),
        "median_distance": float(
            np.median(distances)
        ),
    }


def print_summary(name, result):
    print()
    print(f"[{name}]")
    print(f"episodes: {result['n']}")
    print(
        "position goal: "
        f"{result['goal']:.3f}"
    )
    print(
        "position pot: "
        f"{result['pot']:.3f}"
    )
    print(
        "scratch: "
        f"{result['scratch']:.3f}"
    )
    print(
        "wrong pocket: "
        f"{result['wrong']:.3f}"
    )
    print(
        "mean cue distance: "
        f"{result['mean_distance']:.3f} m"
    )
    print(
        "median cue distance: "
        f"{result['median_distance']:.3f} m"
    )


def main(args):
    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    (
        gate,
        lambdas,
        gate_threshold,
    ) = load_gate(
        args.gate_checkpoint,
        device,
    )

    if 0.0 not in lambdas:
        raise ValueError(
            "Gate lambda set must include 0.0."
        )

    if args.fixed_lambda not in lambdas:
        raise ValueError(
            "--fixed-lambda must be in gate lambdas."
        )

    zero_index = lambdas.index(0.0)
    fixed_index = lambdas.index(
        args.fixed_lambda
    )

    base_agent = load_agent(
        args.base_checkpoint,
        device,
    )
    bc_agent = load_agent(
        args.bc_checkpoint,
        device,
    )

    sample_env = MicroBilliardEnv()
    eval_env = MicroBilliardEnv()

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

    rng = np.random.default_rng(
        args.seed
    )

    base_results = []
    fixed_results = []
    gate_results = []
    oracle_results = []

    gate_choices = Counter()
    gate_matches_oracle = 0
    success_capable = 0
    gate_success_when_capable = 0

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

        observation = sample.observation
        goal = sample.goal
        macro_action = sample.macro_action
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

        conditioned_state = (
            base_agent
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

        features_t = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            gate_logits = gate(
                features_t
            )

            gate_probabilities = (
                torch.sigmoid(
                    gate_logits
                )
            )

            gate_confidence, gate_choice = (
                torch.max(
                    gate_probabilities,
                    dim=1,
                )
            )

            gate_index = int(
                gate_choice.item()
            )

            if (
                float(
                    gate_confidence.item()
                )
                < gate_threshold
            ):
                gate_index = zero_index

        all_results = []

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
            ).astype(np.float32)

            all_results.append(
                evaluate_action(
                    eval_env,
                    state,
                    observation,
                    goal,
                    macro_action,
                    action,
                )
            )

        oracle_index = (
            choose_oracle_lambda_index(
                [
                    item["goal"]
                    for item in all_results
                ],
                [
                    item["pot"]
                    for item in all_results
                ],
                [
                    item["scratch"]
                    for item in all_results
                ],
                [
                    item["distance"]
                    for item in all_results
                ],
                zero_index=zero_index,
            )
        )

        if any(
            item["goal"]
            for item in all_results
        ):
            success_capable += 1
            if all_results[
                gate_index
            ]["goal"]:
                gate_success_when_capable += 1

        gate_matches_oracle += int(
            gate_index
            == oracle_index
        )

        gate_choices[
            lambdas[gate_index]
        ] += 1

        base_results.append(
            all_results[zero_index]
        )
        fixed_results.append(
            all_results[fixed_index]
        )
        gate_results.append(
            all_results[gate_index]
        )
        oracle_results.append(
            all_results[oracle_index]
        )

    print()
    print(
        "=== Adaptive POSITION Blend Gate Evaluation ==="
    )
    print(f"episodes: {args.episodes}")
    print(
        f"evaluation seed: {args.seed}"
    )

    print_summary(
        "Base lambda=0",
        summarize(base_results),
    )
    print_summary(
        f"Fixed lambda={args.fixed_lambda:.2f}",
        summarize(fixed_results),
    )
    print_summary(
        "Learned adaptive gate",
        summarize(gate_results),
    )
    print_summary(
        "Oracle selector upper bound",
        summarize(oracle_results),
    )

    print()
    print(
        "gate exact oracle-label accuracy: "
        f"{gate_matches_oracle / args.episodes:.3f}"
    )

    if success_capable > 0:
        print(
            "gate success on oracle-success-capable "
            "episodes: "
            f"{gate_success_when_capable / success_capable:.3f}"
        )

    print(
        "gate lambda selection counts:"
    )
    for blend_lambda in lambdas:
        print(
            f"  lambda={blend_lambda:.2f}: "
            f"{gate_choices[blend_lambda]}"
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
        "--gate-checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        default=0.40,
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

    return parser.parse_args()


if __name__ == "__main__":
    main(
        parse_args()
    )
