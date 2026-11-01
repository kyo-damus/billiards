import torch
import torch.nn as nn

from src.macro.encoding import (
    MACRO_STATE_DIM,
)

from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
)


class MacroCandidatePolicyNetwork(
    nn.Module
):
    """
    GameStateとMacroCandidateの組を受け取り、
    その候補のlogitを返す。

    固定12クラス分類ではなく、

        score(state, candidate)

    とすることで可変個の候補を扱う。
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

        logits = self.network(x)

        return logits.squeeze(-1)
        


def load_macro_candidate_policy_network(
    checkpoint_path,
    device="cpu",
):
    """
    train_macro_candidate_policy.py が保存したcheckpointから
    Candidate Policyを復元する。
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required = (
        "candidate_policy_network",
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
            "Candidate policy checkpoint "
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
            "Candidate policy checkpoint "
            f"state_dim={state_dim}, "
            f"expected {MACRO_STATE_DIM}."
        )

    if candidate_dim != MACRO_CANDIDATE_DIM:
        raise ValueError(
            "Candidate policy checkpoint "
            f"candidate_dim={candidate_dim}, "
            f"expected {MACRO_CANDIDATE_DIM}."
        )

    network = MacroCandidatePolicyNetwork(
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
            "candidate_policy_network"
        ]
    )
    network.eval()

    return (
        network,
        checkpoint,
    )
