import numpy as np

from src.common.types import GameState
from src.game.simple_rules import SimplifiedGameRules


def make_state(
    pockets=None,
    score=None,
    player=0,
):
    if pockets is None:
        pockets = [-1, -1, -1]

    if score is None:
        score = [0.0, 0.0]

    return GameState(
        ball_positions=np.array(
            [
                [0.5, 0.3],
                [0.5, 0.8],
                [0.8, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.array(
            score,
            dtype=np.float32,
        ),
        current_player=player,
        ball_pocket_indices=np.array(
            pockets,
            dtype=np.int32,
        ),
    )


def test_pocket_increases_score():
    rules = SimplifiedGameRules()

    previous = make_state()

    physical = make_state(
        pockets=[
            -1,
             0,
            -1,
        ]
    )

    result = rules.apply(
        previous_state=previous,
        physical_state=physical,
        shot_info={
            "scratched": False,
        },
    )

    assert result.reward == 1.0

    np.testing.assert_array_equal(
        result.next_state.score,
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    assert result.turn_changed is False
    assert result.next_state.current_player == 0


def test_miss_changes_turn():
    rules = SimplifiedGameRules()

    previous = make_state()
    physical = make_state()

    result = rules.apply(
        previous_state=previous,
        physical_state=physical,
        shot_info={
            "scratched": False,
        },
    )

    assert result.reward == 0.0
    assert result.turn_changed is True

    assert (
        result.next_state.current_player
        == 1
    )


def test_all_object_balls_pocketed_ends_game():
    rules = SimplifiedGameRules()

    previous = make_state(
        pockets=[
            -1,
             0,
            -1,
        ],
        score=[
            1.0,
            0.0,
        ],
    )

    physical = make_state(
        pockets=[
            -1,
             0,
             3,
        ],
        score=[
            1.0,
            0.0,
        ],
    )

    result = rules.apply(
        previous_state=previous,
        physical_state=physical,
        shot_info={
            "scratched": False,
        },
    )

    assert result.reward == 1.0
    assert result.terminated is True

    np.testing.assert_array_equal(
        result.next_state.score,
        np.array(
            [2.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_scratch_ends_simplified_game():
    rules = SimplifiedGameRules()

    previous = make_state()

    physical = make_state(
        pockets=[
             0,
            -1,
            -1,
        ]
    )

    result = rules.apply(
        previous_state=previous,
        physical_state=physical,
        shot_info={
            "scratched": True,
        },
    )

    assert result.reward == -1.0
    assert result.terminated is True
    assert result.turn_changed is True
    