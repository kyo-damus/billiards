import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
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


class SingleActionGenerator:

    def generate(
        self,
        state,
    ):
        return [
            MacroAction(
                strategy=Strategy.ATTACK,
                target_ball=0,
                target_pocket=1,
            )
        ]


def test_invalid_parameters_are_rejected():

    try:
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=1
        )
    except ValueError:
        pass
    else:
        assert False

    try:
        OutcomeGuidedPositionCandidateGenerator(
            top_k=0
        )
    except ValueError:
        pass
    else:
        assert False


def test_outcome_guided_generator_runs():

    generator = (
        OutcomeGuidedPositionCandidateGenerator(
            action_generator=(
                SingleActionGenerator()
            ),
            grid_size=5,
            top_k=8,
        )
    )

    witnesses = (
        generator.generate_with_witnesses(
            make_state()
        )
    )

    assert isinstance(
        witnesses,
        list,
    )

    assert len(witnesses) <= 8

    for item in witnesses:
        assert (
            item.candidate.action
            .target_ball
            == 0
        )

        assert (
            item.candidate.action
            .target_pocket
            == 1
        )

        assert (
            item.candidate.goal
            .cue_target_position
            is not None
        )

        assert (
            item.candidate.goal
            .next_target_ball
            == 1
        )

        assert (
            item.candidate.goal
            .next_target_pocket
            is not None
        )

        assert len(
            item.witness_action
        ) == 2

        assert np.isfinite(
            item.next_shot_distance
        )


def test_generate_matches_witness_candidates():

    generator = (
        OutcomeGuidedPositionCandidateGenerator(
            action_generator=(
                SingleActionGenerator()
            ),
            grid_size=5,
            top_k=4,
        )
    )

    candidates = generator.generate(
        make_state()
    )

    witnesses = (
        generator.generate_with_witnesses(
            make_state()
        )
    )

    assert candidates == [
        item.candidate
        for item in witnesses
    ]
