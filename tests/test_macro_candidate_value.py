import torch

from src.macro.candidate_value import (
    MacroCandidateValueNetwork,
)
from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
)
from src.macro.encoding import (
    MACRO_STATE_DIM,
)


def test_candidate_value_variable_count():

    value = (
        MacroCandidateValueNetwork()
    )

    state = torch.zeros(
        MACRO_STATE_DIM,
        dtype=torch.float32,
    )

    candidate = torch.zeros(
        MACRO_CANDIDATE_DIM,
        dtype=torch.float32,
    )

    candidates_3 = (
        candidate
        .unsqueeze(0)
        .repeat(
            3,
            1,
        )
    )

    values_3 = value(
        state,
        candidates_3,
    )

    assert values_3.shape == (3,)

    candidates_7 = (
        candidate
        .unsqueeze(0)
        .repeat(
            7,
            1,
        )
    )

    values_7 = value(
        state,
        candidates_7,
    )

    assert values_7.shape == (7,)


def test_candidate_value_batched_state():

    value = (
        MacroCandidateValueNetwork()
    )

    states = torch.zeros(
        4,
        MACRO_STATE_DIM,
        dtype=torch.float32,
    )

    candidates = torch.zeros(
        4,
        MACRO_CANDIDATE_DIM,
        dtype=torch.float32,
    )

    output = value(
        states,
        candidates,
    )

    assert output.shape == (4,)



def test_candidate_value_checkpoint_roundtrip(
    tmp_path,
):
    from src.macro.candidate_value import (
        load_macro_candidate_value_network,
    )

    original = (
        MacroCandidateValueNetwork(
            hidden_dim=32,
        )
    )

    checkpoint_path = (
        tmp_path
        / "candidate_value.pt"
    )

    torch.save(
        {
            "candidate_value_network": (
                original.state_dict()
            ),
            "state_dim": (
                MACRO_STATE_DIM
            ),
            "candidate_dim": (
                MACRO_CANDIDATE_DIM
            ),
            "hidden_dim": 32,
        },
        checkpoint_path,
    )

    restored, checkpoint = (
        load_macro_candidate_value_network(
            checkpoint_path,
            device="cpu",
        )
    )

    state = torch.zeros(
        MACRO_STATE_DIM,
        dtype=torch.float32,
    )
    candidates = torch.zeros(
        3,
        MACRO_CANDIDATE_DIM,
        dtype=torch.float32,
    )

    with torch.no_grad():
        expected = original(
            state,
            candidates,
        )
        actual = restored(
            state,
            candidates,
        )

    assert torch.allclose(
        expected,
        actual,
    )
    assert (
        checkpoint["hidden_dim"]
        == 32
    )
