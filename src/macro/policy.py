import torch.nn as nn

from src.macro.encoding import (
    MACRO_STATE_DIM,
    NUM_MACRO_ACTIONS,
)


class MacroPolicyNetwork(nn.Module):
    """
    GameStateから12個のMacroActionに対するlogitを出力する。
    """

    def __init__(
        self,
        state_dim: int = MACRO_STATE_DIM,
        hidden_dim: int = 128,
        num_actions: int = NUM_MACRO_ACTIONS,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state):
        return self.network(state)
        