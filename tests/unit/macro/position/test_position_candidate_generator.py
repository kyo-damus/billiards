import numpy as np

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
)

from src.common.types import (
    GameState,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)


def make_state():

    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.80, 1.118],

                # ball 0
                [0.40, 1.118],

                # ball 1
                [0.80, 0.80],
            ],
            dtype=np.float32,
        ),

        score=np.zeros(
            2,
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_position_candidates_are_generated():

    generator = (
        PositionAttackCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    assert len(candidates) > 0

    assert all(
        candidate.goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
        for candidate in candidates
    )


def test_position_goal_has_next_shot():

    generator = (
        PositionAttackCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    goal = candidates[0].goal

    assert (
        goal.next_target_ball
        is not None
    )

    assert (
        goal.next_target_pocket
        is not None
    )

    assert (
        goal.cue_target_position
        is not None
    )

    assert (
        goal.cue_target_radius
        is not None
    )


def test_cue_target_is_inside_table():

    generator = (
        PositionAttackCandidateGenerator()
    )

    candidates = generator.generate(
        make_state()
    )

    for candidate in candidates:

        x, y = (
            candidate.goal
            .cue_target_position
        )

        assert 0.0 < x < TABLE_WIDTH
        assert 0.0 < y < TABLE_LENGTH
        