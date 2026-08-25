import argparse

import torch

from src.env.micro_billiard_env import MicroBilliardEnv
from src.micro.sac import SACAgent
from scripts.train_micro import sample_feasible_episode


def evaluate(args):
    env = MicroBilliardEnv()

    agent = SACAgent(
        state_dim=12,
        action_dim=2,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=agent.device,
    )

    agent.actor.load_state_dict(
        checkpoint["actor"]
    )

    agent.actor.eval()

    reward_sum = 0.0
    success_count = 0
    pocket_count = 0
    scratch_count = 0

    for episode in range(args.episodes):

        state = sample_feasible_episode(
            env=env,
            seed=args.seed + episode * 100,
        )

        action = agent.select_action(
            state,
            deterministic=True,
        )

        (
            _,
            reward,
            _,
            _,
            info,
        ) = env.step(action)

        reward_sum += reward

        success_count += int(
            info["success"]
        )

        pocket_count += int(
            info["target_pocketed_anywhere"]
        )

        scratch_count += int(
            info["scratched"]
        )

    n = args.episodes

    print()
    print("=== Micro SAC Evaluation ===")
    print(f"episodes:    {n}")
    print(f"avg reward:  {reward_sum / n:.3f}")
    print(f"success:     {success_count / n:.3f}")
    print(f"any pocket:  {pocket_count / n:.3f}")
    print(f"scratch:     {scratch_count / n:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/micro/micro_sac_final.pt",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())