import numpy as np

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


def make_candidate():

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=1,
    )

    # radiusを非常に広くし、
    # potできればPOSITION条件も
    # 満たしやすくしたテスト用Goal
    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
        cue_target_position=(
            0.50,
            1.10,
        ),
        cue_target_radius=2.0,
        next_target_ball=1,
        next_target_pocket=5,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
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


def test_position_feasibility_checker_runs():

    checker = (
        PositionGoalFeasibilityChecker(
            grid_size=5,
        )
    )

    result = checker.check(
        make_state(),
        make_candidate(),
    )

    assert isinstance(
        result,
        PositionFeasibilityResult,
    )

    assert (
        result.tested_actions
        > 0
    )

    assert isinstance(
        result.reachable,
        bool,
    )

    assert isinstance(
        result.pot_found,
        bool,
    )
    