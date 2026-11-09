import numpy as np
import pytest

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
from src.macro.position_feasibility import (
    PositionFeasibilityResult,
    PositionGoalFeasibilityChecker,
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


def make_candidate():

    return MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=1,
        ),
        goal=TacticalGoal(
            goal_type=(
                TacticalGoalType.POSITION_ATTACK
            ),
            target_ball=0,
            target_pocket=1,
            cue_target_position=(
                0.50,
                1.10,
            ),
            cue_target_radius=0.10,
            next_target_ball=1,
            next_target_pocket=5,
        ),
    )


def test_check_actions_runs_with_custom_action_set():

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=5,
        )
    )

    actions = np.array(
        [
            [-0.75, -0.25],
            [-0.25, 0.25],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )

    result = checker.check_actions(
        make_state(),
        make_candidate(),
        actions,
    )

    assert isinstance(
        result,
        PositionFeasibilityResult,
    )

    assert (
        result.tested_actions
        <= len(actions)
    )

    assert (
        result.tested_actions
        > 0
    )


def test_check_actions_rejects_invalid_shape():

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=5,
        )
    )

    with pytest.raises(
        ValueError
    ):
        checker.check_actions(
            make_state(),
            make_candidate(),
            np.zeros(
                3,
                dtype=np.float32,
            ),
        )


def test_check_actions_rejects_out_of_range_action():

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=5,
        )
    )

    with pytest.raises(
        ValueError
    ):
        checker.check_actions(
            make_state(),
            make_candidate(),
            np.array(
                [
                    [1.1, 0.0],
                ],
                dtype=np.float32,
            ),
        )
