import numpy as np

from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)
from src.goal.types import (
    TacticalGoalType,
)
from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)


def test_outcome_position_sampler_runs():

    rng = np.random.default_rng(
        0
    )

    env = MicroBilliardEnv()

    sampler = (
        MixedTacticalGoalSampler(
            position_probability=1.0,
            position_mode="outcome",
            outcome_grid_size=3,
            outcome_top_k=4,
            outcome_dedup_distance=0.10,
        )
    )

    sample = sampler.sample(
        env=env,
        rng=rng,
        seed=100,
    )

    assert (
        sample.goal.goal_type
        == TacticalGoalType.POSITION_ATTACK
    )

    assert (
        sample.goal.cue_target_position
        is not None
    )

    assert (
        sample.goal.next_target_ball
        is not None
    )

    assert (
        sample.goal.next_target_pocket
        is not None
    )


def test_invalid_position_mode_is_rejected():

    try:
        MixedTacticalGoalSampler(
            position_mode="unknown"
        )
    except ValueError:
        return

    assert False, (
        "ValueError was not raised"
    )
