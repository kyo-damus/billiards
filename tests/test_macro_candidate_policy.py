import numpy as np
import torch

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_DIM,
    encode_macro_candidate,
)

from src.macro.candidate_policy import (
    MacroCandidatePolicyNetwork,
)

from src.macro.encoding import (
    MACRO_STATE_DIM,
)


def make_direct_candidate():

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=1,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


def make_position_candidate():

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=1,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,

        cue_target_position=(
            0.7,
            1.2,
        ),

        cue_target_radius=0.1,

        next_target_ball=1,
        next_target_pocket=5,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


def test_candidate_encoding_shape():

    encoded = encode_macro_candidate(
        make_direct_candidate()
    )

    assert encoded.shape == (
        MACRO_CANDIDATE_DIM,
    )

    assert (
        MACRO_CANDIDATE_DIM
        == 25
    )


def test_position_and_direct_differ():

    direct = encode_macro_candidate(
        make_direct_candidate()
    )

    position = encode_macro_candidate(
        make_position_candidate()
    )

    assert not np.allclose(
        direct,
        position,
    )


def test_candidate_policy_variable_count():

    policy = (
        MacroCandidatePolicyNetwork()
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

    logits_3 = policy(
        state,
        candidates_3,
    )

    assert logits_3.shape == (3,)

    candidates_7 = (
        candidate
        .unsqueeze(0)
        .repeat(
            7,
            1,
        )
    )

    logits_7 = policy(
        state,
        candidates_7,
    )

    assert logits_7.shape == (7,)
    


def test_candidate_policy_checkpoint_roundtrip(
    tmp_path,
):
    from src.macro.candidate_policy import (
        load_macro_candidate_policy_network,
    )

    original = (
        MacroCandidatePolicyNetwork(
            hidden_dim=32,
        )
    )

    checkpoint_path = (
        tmp_path
        / "candidate_policy.pt"
    )

    torch.save(
        {
            "candidate_policy_network": (
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
        load_macro_candidate_policy_network(
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



def test_mixed_candidate_policy_v2_checkpoint_requires_explicit_opt_in(
    tmp_path,
):
    import pytest
    from src.macro.candidate_policy import (
        load_macro_candidate_policy_network,
    )

    original = MacroCandidatePolicyNetwork(
        candidate_dim=31,
        hidden_dim=16,
    )
    path = tmp_path / "mixed_candidate_policy.pt"
    torch.save(
        {
            "candidate_policy_network": original.state_dict(),
            "state_dim": MACRO_STATE_DIM,
            "candidate_dim": 31,
            "candidate_encoding_version": 2,
            "turn_aware": True,
            "hidden_dim": 16,
        },
        path,
    )

    # Existing Attack-only code may only load 25-D by default.
    with pytest.raises(ValueError, match="expected 25"):
        load_macro_candidate_policy_network(path)

    restored, metadata = load_macro_candidate_policy_network(
        path, expected_candidate_dim=31
    )
    assert metadata["candidate_encoding_version"] == 2
    state = torch.zeros(MACRO_STATE_DIM)
    candidates = torch.zeros(3, 31)
    with torch.no_grad():
        torch.testing.assert_close(
            original(state, candidates),
            restored(state, candidates),
        )


def test_mixed_policy_loader_rejects_missing_v2_metadata(tmp_path):
    import pytest
    from src.macro.candidate_policy import (
        load_macro_candidate_policy_network,
    )

    network = MacroCandidatePolicyNetwork(
        candidate_dim=31, hidden_dim=16
    )
    path = tmp_path / "invalid_mixed_policy.pt"
    torch.save(
        {
            "candidate_policy_network": network.state_dict(),
            "state_dim": MACRO_STATE_DIM,
            "candidate_dim": 31,
            "hidden_dim": 16,
        },
        path,
    )
    with pytest.raises(ValueError, match="31-D v2"):
        load_macro_candidate_policy_network(
            path, expected_candidate_dim=31
        )
