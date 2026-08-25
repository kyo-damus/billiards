import torch
import torch.nn as nn


class TwinQCritic(nn.Module):
    """SAC用のTwin Q Network."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat(
            [state, action],
            dim=-1,
        )

        return (
            self.q1(state_action),
            self.q2(state_action),
        )