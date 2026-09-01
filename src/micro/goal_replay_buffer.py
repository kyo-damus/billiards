import numpy as np

from src.goal.conditioner import (
    encode_micro_goal,
)

from src.micro.replay_buffer import (
    ReplayBuffer,
)


class GoalConditionedReplayBuffer:
    """
    (observation, goal) を結合して
    通常のSAC ReplayBufferへ保存する。

    現在のMicroは1-shotなので、
    transition中は同一goalを使用する。
    """

    def __init__(
        self,
        capacity: int = 100_000,
    ):
        self.buffer = ReplayBuffer(
            capacity=capacity
        )

    def _build_state(
        self,
        observation,
        goal,
    ):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        goal_vector = encode_micro_goal(
            observation,
            goal,
        )

        return np.concatenate(
            [
                observation,
                goal_vector,
            ]
        ).astype(np.float32)

    def push(
        self,
        observation,
        goal,
        action,
        reward,
        next_observation,
        done,
    ):
        state = self._build_state(
            observation,
            goal,
        )

        next_goal_vector = encode_micro_goal(
            next_observation,
            goal,
        )

        next_state = np.concatenate(
            [
                next_observation,
                next_goal_vector,
            ]
        ).astype(np.float32)

        self.buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def sample(
        self,
        batch_size,
        device,
    ):
        return self.buffer.sample(
            batch_size=batch_size,
            device=device,
        )

    def __len__(self):
        return len(self.buffer)
