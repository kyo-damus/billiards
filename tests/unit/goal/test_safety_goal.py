import numpy as np

from src.common.types import (
    GameState,
)
from src.goal.evaluator import (
    TacticalGoalEvaluator,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


def test_safety_goal_achieved_inside_region():

    state = GameState(
        ball_positions=np.array(
            [
                [0.20, 1.10],
                [0.50, 1.00],
                [0.90, 1.40],
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

    goal = TacticalGoal(
        goal_type=TacticalGoalType.SAFETY,
        cue_target_position=(
            0.20,
            1.10,
        ),
        cue_target_radius=0.12,
    )

    evaluator = TacticalGoalEvaluator()

    assert evaluator.is_achieved(
        goal,
        state,
    )


def test_safety_goal_fails_after_scratch():

    state = GameState(
        ball_positions=np.array(
            [
                [0.20, 1.10],
                [0.50, 1.00],
                [0.90, 1.40],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=np.array(
            [0, -1, -1],
            dtype=np.int32,
        ),
    )

    goal = TacticalGoal(
        goal_type=TacticalGoalType.SAFETY,
        cue_target_position=(
            0.20,
            1.10,
        ),
        cue_target_radius=0.12,
    )

    evaluator = TacticalGoalEvaluator()

    assert not evaluator.is_achieved(
        goal,
        state,
    )
