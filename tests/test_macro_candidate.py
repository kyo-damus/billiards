import numpy as np

from src.common.types import (
    GameState,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.candidate_generator import (
    MacroCandidateGenerator,
    ExpandedMacroCandidateGenerator,
)



def make_state():

    return GameState(
        ball_positions=np.array(
            [
                [0.80, 1.118],
                [0.40, 1.118],
                [0.80, 0.40],
            ],
            dtype=np.float32,
        ),

        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_macro_candidate_generation():

    generator = (
        MacroCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    assert len(candidates) > 0

    assert all(
        isinstance(
            candidate,
            MacroCandidate,
        )
        for candidate in candidates
    )


def test_direct_candidate_has_goal():

    generator = (
        MacroCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    candidate = candidates[0]

    assert (
        candidate.goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
    )

    assert (
        candidate.goal.target_ball
        == candidate.action.target_ball
    )

    assert (
        candidate.goal.target_pocket
        == candidate.action.target_pocket
    )

def test_expanded_generator_has_multiple_goal_types():

    generator = (
        ExpandedMacroCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    goal_types = {
        candidate.goal.goal_type
        for candidate in candidates
    }

    assert (
        TacticalGoalType.DIRECT_ATTACK
        in goal_types
    )

    assert (
        TacticalGoalType.POSITION_ATTACK
        in goal_types
    )