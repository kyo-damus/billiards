import numpy as np

from scripts.train_goal_conditioned_micro import (
    _init_goal_sampler_worker,
    _sample_goal_worker,
)
from src.goal.types import (
    TacticalGoalType,
)


def test_parallel_goal_worker_direct_mode_runs():

    _init_goal_sampler_worker(
        position_mode="ideal",
        outcome_grid_size=3,
        outcome_top_k=4,
        outcome_dedup_distance=0.10,
    )

    (
        state,
        goal,
        macro_action,
    ) = _sample_goal_worker(
        (
            123,
            0.0,
            0.10,
        )
    )

    assert (
        state.ball_positions.shape
        == (3, 2)
    )

    assert (
        goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
    )

    assert (
        macro_action.target_ball
        == goal.target_ball
    )

    assert (
        macro_action.target_pocket
        == goal.target_pocket
    )


def test_parallel_goal_worker_outcome_mode_runs():

    _init_goal_sampler_worker(
        position_mode="outcome",
        outcome_grid_size=3,
        outcome_top_k=4,
        outcome_dedup_distance=0.10,
    )

    (
        state,
        goal,
        macro_action,
    ) = _sample_goal_worker(
        (
            321,
            1.0,
            0.10,
        )
    )

    assert (
        state.ball_positions.shape
        == (3, 2)
    )

    assert (
        goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
    )

    assert (
        goal.cue_target_position
        is not None
    )

    assert np.isfinite(
        np.asarray(
            goal.cue_target_position
        )
    ).all()

    assert (
        macro_action.target_ball
        == goal.target_ball
    )

    assert (
        macro_action.target_pocket
        == goal.target_pocket
    )
