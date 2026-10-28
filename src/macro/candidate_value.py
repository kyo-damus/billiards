import torch
import torch.nn as nn

from src.macro.encoding import (
    MACRO_STATE_DIM,
)

from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
)


class MacroCandidateValueNetwork(
    nn.Module
):
    """
    GameStateとMacroCandidateの組から
    候補依存の価値 Q(s, c) を推定する。

    state-onlyのV(s)と異なり、

        value(state, candidate)

    とすることで、DIRECT / POSITIONなど
    異なる戦術候補を同じ盤面上で比較する。
    """

    def __init__(
        self,
        state_dim=MACRO_STATE_DIM,
        candidate_dim=MACRO_CANDIDATE_DIM,
        hidden_dim=128,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.candidate_dim = candidate_dim

        input_dim = (
            state_dim
            + candidate_dim
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        state,
        candidates,
    ):
        """
        state:
            (state_dim,)
            または
            (N, state_dim)

        candidates:
            (N, candidate_dim)

        return:
            (N,)
            各候補のQ(s, c)
        """

        if candidates.dim() == 1:
            candidates = (
                candidates.unsqueeze(0)
            )

        num_candidates = (
            candidates.shape[0]
        )

        if state.dim() == 1:
            state = (
                state
                .unsqueeze(0)
                .expand(
                    num_candidates,
                    -1,
                )
            )

        if (
            state.shape[0]
            != num_candidates
        ):
            raise ValueError(
                "state and candidate "
                "batch sizes differ"
            )

        x = torch.cat(
            [
                state,
                candidates,
            ],
            dim=-1,
        )

        values = self.network(x)

        return values.squeeze(-1)



def load_macro_candidate_value_network(
    checkpoint_path,
    device="cpu",
):
    """
    train_macro_candidate_value.py が保存したcheckpointから
    候補依存Qネットワークを復元する。
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = (
        "candidate_value_network",
        "state_dim",
        "candidate_dim",
        "hidden_dim",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Candidate value checkpoint "
            "is missing keys: "
            + ", ".join(missing)
        )

    state_dim = int(
        checkpoint["state_dim"]
    )
    candidate_dim = int(
        checkpoint["candidate_dim"]
    )

    if state_dim != MACRO_STATE_DIM:
        raise ValueError(
            "Candidate value checkpoint "
            f"state_dim={state_dim}, "
            f"expected {MACRO_STATE_DIM}."
        )

    if candidate_dim != MACRO_CANDIDATE_DIM:
        raise ValueError(
            "Candidate value checkpoint "
            f"candidate_dim={candidate_dim}, "
            f"expected {MACRO_CANDIDATE_DIM}."
        )

    network = MacroCandidateValueNetwork(
        state_dim=state_dim,
        candidate_dim=candidate_dim,
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
    ).to(
        device
    )

    network.load_state_dict(
        checkpoint[
            "candidate_value_network"
        ]
    )

    network.eval()

    return (
        network,
        checkpoint,
    )
