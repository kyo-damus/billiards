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
    