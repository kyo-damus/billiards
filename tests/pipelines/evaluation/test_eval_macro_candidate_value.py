import numpy as np
import torch

from scripts.eval_macro_candidate_value import (
    predict_candidates,
)
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
from src.macro.candidate_value import (
    MacroCandidateValueNetwork,
)


def make_state():
    return GameState(
        ball_positions=np.array(
            [
                [0.5, 0.5],
                [0.3, 1.0],
                [0.8, 1.5],
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


def make_candidate(
    pocket,
):
    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=pocket,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=pocket,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


def test_predict_candidates_shape():

    network = (
        MacroCandidateValueNetwork()
    )

    candidates = [
        make_candidate(0),
        make_candidate(1),
        make_candidate(2),
    ]

    prediction = predict_candidates(
        network,
        make_state(),
        candidates,
        device="cpu",
    )

    assert prediction.shape == (3,)
