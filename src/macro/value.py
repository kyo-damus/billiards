import torch.nn as nn

from src.macro.encoding import (
    MACRO_STATE_DIM,
)


class MacroValueNetwork(nn.Module):
    """
    GameStateの長期価値 V(s) を推定する。
    """

    def __init__(
        self,
        state_dim: int = MACRO_STATE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)
        