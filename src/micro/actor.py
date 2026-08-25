import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianActor(nn.Module):
    """SAC用の確率的Actor."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)

        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)

        std = log_std.exp()
        distribution = Normal(mu, std)

        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = distribution.log_prob(pre_tanh)

        # tanhによる確率密度変換の補正
        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True,
        )

        return action, log_prob

    def deterministic(self, state):
        """評価時などに平均行動を使用する."""
        mu, _ = self.forward(state)

        return torch.tanh(mu)