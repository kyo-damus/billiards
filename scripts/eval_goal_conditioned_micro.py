import argparse
from collections import defaultdict

import numpy as np
import torch

from src.common.types import (
    MacroAction,
    Strategy,
)
from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.adapter import (
    macro_action_to_tactical_goal,
)
from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)


def unpack_reset_result(result):
    if (
        isinstance(result, tuple)
        and len(result) == 2
    ):
        return result[0]

    return result


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


def sample_feasible_case(
    env,
    target_ball,
    target_pocket,
    seed,
    max_attempts=200,
):
    macro_action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=target_ball,
        target_pocket=target_pocket,
    )

    for attempt in range(max_attempts):

        env.set_macro_action(
            macro_action
        )

        env.reset(
            seed=seed + attempt
        )

        if not env.is_current_macro_feasible():
            continue

        # Goalに依存しない9次元Physical Observation
        observation = (
            env.get_physical_observation()
        )

        tactical_goal = (
            macro_action_to_tactical_goal(
                macro_action
            )
        )

        return (
            observation,
            tactical_goal,
        )

    return None


def evaluate(args):

    device = (
        "cuda"
        if (
            args.device == "auto"
            and torch.cuda.is_available()
        )
        else (
            "cpu"
            if args.device == "auto"
            else args.device
        )
    )

    print(
        f"device: {device}"
    )

    agent = load_agent(
        args.checkpoint,
        device,
    )

    env = MicroBilliardEnv()

    stats = defaultdict(
        lambda: {
            "episodes": 0,
            "reward": 0.0,
            "success": 0,
            "any_pocket": 0,
            "wrong_pocket": 0,
            "scratch": 0,
        }
    )

    total_reward = 0.0
    total_success = 0
    total_any_pocket = 0
    total_scratch = 0
    total_episodes = 0
    total_wrong_pocket = 0

    seed_counter = args.seed

    # 2 balls x 6 pockets
    for target_ball in range(2):
        for target_pocket in range(6):

            key = (
                target_ball,
                target_pocket,
            )

            for _ in range(
                args.episodes_per_goal
            ):

                case = sample_feasible_case(
                    env=env,
                    target_ball=target_ball,
                    target_pocket=target_pocket,
                    seed=seed_counter,
                )

                seed_counter += 1000

                if case is None:
                    continue

                (
                    observation,
                    tactical_goal,
                ) = case

                action = agent.select_action(
                    observation,
                    tactical_goal,
                    deterministic=True,
                )

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                success = bool(
                    info.get(
                        "success",
                        False,
                    )
                )

                any_pocket = bool(
                    info.get(
                        "target_pocketed_anywhere",
                        False,
                    )
                )

                scratch = bool(
                    info.get(
                        "scratched",
                        False,
                    )
                )

                wrong_pocket = bool(
                    info.get(
                        "wrong_pocket",
                        False,
                    )
                )

                stats[key]["episodes"] += 1
                stats[key]["reward"] += float(
                    reward
                )
                stats[key]["success"] += int(
                    success
                )
                stats[key]["any_pocket"] += int(
                    any_pocket
                )
                stats[key]["scratch"] += int(
                    scratch
                )
                stats[key]["wrong_pocket"] += int(
                    wrong_pocket
                )

                total_reward += float(
                    reward
                )

                total_success += int(
                    success
                )

                total_any_pocket += int(
                    any_pocket
                )

                total_scratch += int(
                    scratch
                )

                total_wrong_pocket += int(
                    wrong_pocket
                )

                total_episodes += 1

    print()
    print("=== Per-goal evaluation ===")

    for target_ball in range(2):
        for target_pocket in range(6):

            key = (
                target_ball,
                target_pocket,
            )

            result = stats[key]

            n = result["episodes"]

            if n == 0:
                print(
                    f"ball={target_ball} "
                    f"pocket={target_pocket}: "
                    "no feasible cases"
                )
                continue

            print(
                f"ball={target_ball} "
                f"pocket={target_pocket} "
                f"n={n:4d} "
                f"reward="
                f"{result['reward'] / n:7.3f} "
                f"success="
                f"{result['success'] / n:6.3f} "
                f"any_pocket="
                f"{result['any_pocket'] / n:6.3f} "
                f"scratch="
                f"{result['scratch'] / n:6.3f}"
            )

    print()
    print("=== Overall ===")

    if total_episodes == 0:
        raise RuntimeError(
            "No feasible evaluation cases."
        )

    print(
        f"episodes: {total_episodes}"
    )

    print(
        f"avg reward: "
        f"{total_reward / total_episodes:.3f}"
    )

    print(
        f"success: "
        f"{total_success / total_episodes:.3f}"
    )

    print(
        f"any pocket: "
        f"{total_any_pocket / total_episodes:.3f}"
    )

    print(
        f"scratch: "
        f"{total_scratch / total_episodes:.3f}"
    )

    print(
        f"wrong_pocket="
        f"{result['wrong_pocket'] / n:6.3f} "
    )

    pocketed = result[
        "any_pocket"
    ]

    goal_precision = (
        result["success"] / pocketed
        if pocketed > 0
        else 0.0
    )

    print(
        f"goal_precision="
        f"{goal_precision:6.3f} "
    )

    print(
        f"wrong pocket: "
        f"{total_wrong_pocket / total_episodes:.3f}"
    )

    overall_goal_precision = (
        total_success / total_any_pocket
        if total_any_pocket > 0
        else 0.0
    )

    print(
        f"goal precision: "
        f"{overall_goal_precision:.3f}"
    )
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="goal_conditioned_micro.pt",
    )

    parser.add_argument(
        "--episodes-per-goal",
        type=int,
        default=100,
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
