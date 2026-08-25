import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.common.types import MacroAction, Strategy
from src.env.micro_billiard_env import MicroBilliardEnv
from src.micro.replay_buffer import ReplayBuffer
from src.micro.sac import SACAgent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(agent: SACAgent, path: Path, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
        },
        path,
    )

def sample_feasible_episode(
    env,
    seed,
    max_attempts=100,
):
    """
    Micro学習用に、直接ショット可能な
    (盤面, target ball, pocket) を生成する。
    """

    for attempt in range(max_attempts):

        target_ball = random.choice([0, 1])
        target_pocket = random.choice(range(6))

        macro_action = MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=target_ball,
            target_pocket=target_pocket,
        )

        env.set_macro_action(macro_action)

        state, _ = env.reset(
            seed=seed + attempt
        )

        if env.is_current_macro_feasible():
            return state

    raise RuntimeError(
        "Could not generate feasible micro task."
    )


def train(args):
    set_seed(args.seed)

    env = MicroBilliardEnv()

    agent = SACAgent(
        state_dim=12,
        action_dim=2,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )

    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size
    )

    print(f"Device: {agent.device}")
    print(f"Training steps: {args.steps}")

    reward_sum = 0.0
    success_count = 0
    scratch_count = 0
    possible_count = 0
    pocketed_anywhere_count = 0

    for step in range(1, args.steps + 1):

        state = sample_feasible_episode(
            env=env,
            seed=args.seed + step * 100,
        )

        # -----------------------------
        # Action selection
        # -----------------------------

        if step <= args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(
                state,
                deterministic=False,
            )

        # -----------------------------
        # One shot
        # -----------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        done = terminated or truncated

        replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        # -----------------------------
        # SAC update
        # -----------------------------

        metrics = None

        if (
            len(replay_buffer) >= args.batch_size
            and step > args.warmup_steps
        ):
            batch = replay_buffer.sample(
                batch_size=args.batch_size,
                device=agent.device,
            )

            metrics = agent.update(batch)

        # -----------------------------
        # Logging
        # -----------------------------

        reward_sum += reward
        success_count += int(info["success"])
        scratch_count += int(info["scratched"])
        possible_count += int(info["physically_possible"])
        pocketed_anywhere_count += int(info["target_pocketed_anywhere"])

        if step % args.log_interval == 0:

            avg_reward = (
                reward_sum / args.log_interval
            )

            success_rate = (
                success_count / args.log_interval
            )

            scratch_rate = (
                scratch_count / args.log_interval
            )

            possible_rate = (
                possible_count / args.log_interval
            )

            pocketed_anywhere_rate = (
                pocketed_anywhere_count / args.log_interval
            )

            message = (
                f"step={step:7d} "
                f"reward={avg_reward:8.3f} "
                f"success={success_rate:6.3f} "
                f"any_pocket={pocketed_anywhere_rate:6.3f} "
                f"scratch={scratch_rate:6.3f} "
                f"possible={possible_rate:6.3f}"
            )

            if metrics is not None:
                message += (
                    f" actor_loss={metrics['actor_loss']:8.3f}"
                    f" critic_loss={metrics['critic_loss']:8.3f}"
                    f" alpha={metrics['alpha']:7.4f}"
                )

            print(message)

            reward_sum = 0.0
            success_count = 0
            scratch_count = 0
            possible_count = 0
            pocketed_anywhere_count = 0

        # -----------------------------
        # Checkpoint
        # -----------------------------

        if step % args.save_interval == 0:
            save_checkpoint(
                agent,
                Path(args.output_dir)
                / f"micro_sac_step_{step}.pt",
                step,
            )

    save_checkpoint(
        agent,
        Path(args.output_dir)
        / "micro_sac_final.pt",
        args.steps,
    )

    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/micro",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())