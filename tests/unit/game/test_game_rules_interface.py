import numpy as np

from src.common.types import GameState
from src.game.rules import GameTransitionResult


def test_game_transition_result():

    state = GameState(
        ball_positions=np.array(
            [
                [0.5, 0.3],
                [0.5, 0.8],
                [0.9, 1.5],
            ],
            dtype=np.float32,
        ),
        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        current_player=0,
    )

    result = GameTransitionResult(
        next_state=state,
        reward=1.0,
        terminated=False,
        turn_changed=False,
    )

    assert result.next_state is state
    assert result.reward == 1.0
    assert result.terminated is False
    assert result.turn_changed is False
    