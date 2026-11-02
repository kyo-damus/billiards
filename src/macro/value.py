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
        


def load_macro_state_value_network(checkpoint_path, device="cpu"):
    """Load explicitly tagged state V(s) from completed mixed games."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    required = (
        "state_value_network", "state_dim", "hidden_dim",
        "target_kind", "turn_aware", "reward_kind",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            "State V checkpoint is missing keys: " + ", ".join(missing)
        )
    if (
        int(checkpoint["state_dim"]) != MACRO_STATE_DIM
        or str(checkpoint["target_kind"]) != "selected_action_mc_return"
        or not bool(checkpoint["turn_aware"])
        or str(checkpoint["reward_kind"]) != "macro_local_reward"
    ):
        raise ValueError(
            "Expected 12D turn-aware State V from Macro local MC returns."
        )
    network = MacroValueNetwork(
        state_dim=MACRO_STATE_DIM,
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    network.load_state_dict(checkpoint["state_value_network"])
    network.eval()
    return network, checkpoint
