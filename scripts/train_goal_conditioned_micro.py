import argparse
import random

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
from src.micro.goal_replay_buffer import (
    GoalConditionedReplayBuffer,
)
from src.goal.reward import (
    calculate_goal_conditioned_reward,
)
from src.goal.types import (
    TacticalGoalType,
)

from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)
from src.micro.goal_curriculum import (
    MixedGoalCurriculum,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_reset_result(result):
    """
    Gymnasium形式 (obs, info) と
    obsのみの両方に対応。
    """
    if (
        isinstance(result, tuple)
        and len(result) == 2
    ):
        return result[0]

    return result


# def sample_feasible_goal_episode(
#     env,
#     rng,
#     seed,
#     max_attempts=100,
# ):
#     """
#     現段階では DIRECT_ATTACK のみ。

#     ランダムな
#         target_ball
#         target_pocket
#     を選び、物理的に実行可能な盤面だけ採用する。
#     """

#     for attempt in range(max_attempts):

#         target_ball = int(
#             rng.integers(0, 2)
#         )

#         target_pocket = int(
#             rng.integers(0, 6)
#         )

#         macro_action = MacroAction(
#             strategy=Strategy.ATTACK,
#             target_ball=target_ball,
#             target_pocket=target_pocket,
#         )

#         env.set_macro_action(
#             macro_action
#         )

#         env.reset(
#             seed=seed + attempt
#         )

#         if not env.is_current_macro_feasible():
#             continue

#         observation = (
#             env.get_physical_observation()
#         )

#         if not env.is_current_macro_feasible():
#             continue

#         tactical_goal = (
#             macro_action_to_tactical_goal(
#                 macro_action
#             )
#         )

#         return (
#             observation,
#             tactical_goal,
#             macro_action,
#         )

#     raise RuntimeError(
#         "Could not sample a feasible "
#         "goal-conditioned episode."
#     )


def save_checkpoint(
    agent,
    path,
    step,
):
    sac = agent.agent

    checkpoint = {
        "step": step,
        "actor": sac.actor.state_dict(),
        "critic": sac.critic.state_dict(),
        "critic_target": sac.critic_target.state_dict(),
    }

    if hasattr(sac, "actor_optimizer"):
        checkpoint[
            "actor_optimizer"
        ] = sac.actor_optimizer.state_dict()

    if hasattr(sac, "critic_optimizer"):
        checkpoint[
            "critic_optimizer"
        ] = sac.critic_optimizer.state_dict()

    if hasattr(sac, "alpha_optimizer"):
        checkpoint[
            "alpha_optimizer"
        ] = sac.alpha_optimizer.state_dict()

    if hasattr(sac, "log_alpha"):
        checkpoint[
            "log_alpha"
        ] = sac.log_alpha.detach().cpu()

    torch.save(
        checkpoint,
        path,
    )

def train(args):

    set_seed(args.seed)

    rng = np.random.default_rng(
        args.seed
    )

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

    env = MicroBilliardEnv()

    # ---------------------------------
    # Goal sampler
    # ---------------------------------

    goal_sampler = MixedTacticalGoalSampler(
        position_probability=(
            args.position_probability_start
        )
    )

    # ---------------------------------
    # Curriculum
    # ---------------------------------

    curriculum = MixedGoalCurriculum(
        start_position_probability=(
            args.position_probability_start
        ),
        end_position_probability=(
            args.position_probability
        ),
        start_target_radius=(
            args.position_radius_start
        ),
        end_target_radius=(
            args.position_radius_end
        ),
        ramp_steps=(
            args.curriculum_steps
        ),
    )

    # ---------------------------------
    # Agent
    # ---------------------------------

    agent = GoalConditionedSACAgent(
        action_dim=2,
        learning_rate=args.learning_rate,
        device=device,
    )

    if args.init_checkpoint is not None:

        load_initial_checkpoint(
            agent=agent,
            checkpoint_path=(
                args.init_checkpoint
            ),
            device=device,
            mode=args.init_mode,
        )

    # ---------------------------------
    # Replay Buffer
    # ---------------------------------

    replay_buffer = (
        GoalConditionedReplayBuffer(
            capacity=args.buffer_size
        )
    )

    # ---------------------------------
    # Logging buffers
    # ---------------------------------

    reward_window = []
    success_window = []
    pocket_window = []
    scratch_window = []
    goal_achieved_window = []

    direct_count = 0
    direct_achieved = 0

    position_count = 0
    position_achieved = 0

    position_pot = 0
    position_region = 0

    latest_metrics = None

    for step in range(
        1,
        args.steps + 1,
    ):

        # ---------------------------------
        # Curriculum
        # ---------------------------------

        (
            current_position_probability,
            current_position_radius,
        ) = curriculum.values(
            step
        )

        goal_sampler.position_probability = (
            current_position_probability
        )

        goal_sampler.position_generator.target_radius = (
            current_position_radius
        )

        # ---------------------------------
        # Goal sampling
        # ---------------------------------

        sample = goal_sampler.sample(
            env=env,
            rng=rng,
            seed=args.seed + step * 100,
        )

        observation = (
            sample.observation
        )

        tactical_goal = (
            sample.goal
        )

        macro_action = (
            sample.macro_action
        )


        # ---------------------------------
        # Action
        # ---------------------------------

        if step <= args.warmup:
            action = env.action_space.sample()

        else:
            action = agent.select_action(
                observation,
                tactical_goal,
                deterministic=False,
            )

        # ---------------------------------
        # Environment step
        # ---------------------------------

        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_observation = (
            env.get_physical_observation()
        )

        base_reward = float(
            reward
        )

        (
            reward,
            goal_reward_info,
        ) = calculate_goal_conditioned_reward(
            base_reward=base_reward,
            goal=tactical_goal,
            before_observation=observation,
            after_observation=next_observation,
            shot_info=info,
        )

        goal_achieved = bool(
            goal_reward_info[
                "goal_achieved"
            ]
        )

        goal_achieved_window.append(
            float(goal_achieved)
        )

        if (
            tactical_goal.goal_type
            == TacticalGoalType.DIRECT_ATTACK
        ):
            direct_count += 1

            direct_achieved += int(
                goal_achieved
            )

        elif (
            tactical_goal.goal_type
            == TacticalGoalType.POSITION_ATTACK
        ):
            position_count += 1

            position_achieved += int(
                goal_achieved
            )

            position_pot += int(
                info.get(
                    "success",
                    False,
                )
            )

            position_region += int(
                goal_reward_info.get(
                    "cue_region_reached",
                    False,
                )
            )

        done = bool(
            terminated or truncated
        )

        # ---------------------------------
        # Replay Buffer
        # ---------------------------------

        replay_buffer.push(
            observation=observation,
            goal=tactical_goal,
            action=action,
            reward=float(reward),
            next_observation=next_observation,
            done=done,
        )

        # ---------------------------------
        # SAC update
        # ---------------------------------

        if (
            step > args.warmup
            and len(replay_buffer)
            >= args.batch_size
        ):
            batch = replay_buffer.sample(
                batch_size=args.batch_size,
                device=agent.device,
            )

            latest_metrics = (
                agent.update(batch)
            )

        # ---------------------------------
        # Logging
        # ---------------------------------

        reward_window.append(
            float(reward)
        )

        success_window.append(
            float(
                info.get(
                    "success",
                    False,
                )
            )
        )

        pocket_window.append(
            float(
                info.get(
                    "target_pocketed_anywhere",
                    False,
                )
            )
        )

        scratch_window.append(
            float(
                info.get(
                    "scratched",
                    False,
                )
            )
        )

        if step % args.log_interval == 0:

            avg_reward = float(
                np.mean(reward_window)
            )

            success_rate = float(
                np.mean(success_window)
            )

            pocket_rate = float(
                np.mean(pocket_window)
            )

            scratch_rate = float(
                np.mean(scratch_window)
            )

            goal_rate = float(
                np.mean(
                    goal_achieved_window
                )
            )

            message = (
                f"step={step:6d} "
                f"reward={avg_reward:7.3f} "
                f"success={success_rate:6.3f} "
                f"goal={goal_rate:6.3f} "
                f"any_pocket={pocket_rate:6.3f} "
                f"scratch={scratch_rate:6.3f}"
            )

            direct_rate = (
                direct_achieved / direct_count
                if direct_count > 0
                else 0.0
            )

            position_rate = (
                position_achieved / position_count
                if position_count > 0
                else 0.0
            )

            position_pot_rate = (
                position_pot / position_count
                if position_count > 0
                else 0.0
            )

            position_region_rate = (
                position_region / position_count
                if position_count > 0
                else 0.0
            )

            message += (
                f" direct_goal="
                f"{direct_rate:6.3f}"
                f" position_goal="
                f"{position_rate:6.3f}"
            )

            message += (
                f" pos_pot="
                f"{position_pot_rate:6.3f}"
                f" pos_region="
                f"{position_region_rate:6.3f}"
            )

            message += (
                f" pos_p="
                f"{current_position_probability:.2f}"
                f" pos_r="
                f"{current_position_radius:.3f}"
            )

            if latest_metrics is not None:
                message += (
                    f" actor_loss="
                    f"{latest_metrics['actor_loss']:.4f}"
                    f" critic_loss="
                    f"{latest_metrics['critic_loss']:.4f}"
                    f" alpha="
                    f"{latest_metrics['alpha']:.4f}"
                )

            print(message)

            reward_window.clear()
            success_window.clear()
            pocket_window.clear()
            scratch_window.clear()
            goal_achieved_window.clear()

            direct_count = 0
            direct_achieved = 0

            position_count = 0
            position_achieved = 0

            position_pot = 0
            position_region = 0

        # ---------------------------------
        # Checkpoint
        # ---------------------------------

        if (
            args.save_interval > 0
            and step % args.save_interval == 0
        ):
            save_checkpoint(
                agent,
                args.checkpoint,
                step,
            )

    save_checkpoint(
        agent,
        args.checkpoint,
        args.steps,
    )

    print()
    print(
        "Goal-conditioned SAC "
        "training completed."
    )

    print(
        f"checkpoint: "
        f"{args.checkpoint}"
    )

def load_initial_checkpoint(
    agent,
    checkpoint_path,
    device,
    mode="actor",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    sac = agent.agent

    # =================================
    # Actor
    # =================================

    sac.actor.load_state_dict(
        checkpoint["actor"]
    )

    # =================================
    # Critic
    # =================================

    if mode in (
        "actor_critic",
        "full",
    ):
        sac.critic.load_state_dict(
            checkpoint["critic"]
        )

        if "critic_target" in checkpoint:
            sac.critic_target.load_state_dict(
                checkpoint[
                    "critic_target"
                ]
            )
        else:
            sac.critic_target.load_state_dict(
                checkpoint["critic"]
            )

    # =================================
    # Entropy temperature
    # =================================

    if mode == "full":

        if "log_alpha" in checkpoint:

            saved_log_alpha = (
                checkpoint["log_alpha"]
                .to(
                    device=sac.log_alpha.device,
                    dtype=sac.log_alpha.dtype,
                )
            )

            with torch.no_grad():
                sac.log_alpha.copy_(
                    saved_log_alpha
                )

    if mode not in (
        "actor",
        "actor_critic",
        "full",
    ):
        raise ValueError(
            f"Unknown init mode: {mode}"
        )

    print(
        "initialized from:",
        checkpoint_path,
    )

    print(
        "initialization mode:",
        mode,
    )

    alpha_value = (
        sac.alpha.detach()
        .cpu()
        .item()
    )

    print(
        f"initial alpha: "
        f"{alpha_value:.4f}"
    )

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "goal_conditioned_micro.pt"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
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

    parser.add_argument(
        "--position-probability",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--init-mode",
        type=str,
        default="actor",
        choices=[
            "actor",
            "actor_critic",
            "full",
        ],
    )

    parser.add_argument(
        "--position-probability-start",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--position-radius-start",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--position-radius-end",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(
        parse_args()
    )