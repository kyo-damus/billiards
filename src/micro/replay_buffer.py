from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size, device):
        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            np.stack(actions),
            dtype=torch.float32,
            device=device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        next_states = torch.as_tensor(
            np.stack(next_states),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):
        return len(self.buffer)