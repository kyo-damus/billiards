import numpy as np

from src.env.micro_billiard_env import (
    MicroBilliardEnv,
)

from src.goal.types import (
    TacticalGoalType,
)

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)

from src.micro.goal_sampler import (
    MixedTacticalGoalSampler,
)


def test_direct_only_sampler():

    rng = np.random.default_rng(0)

    env = MicroBilliardEnv()

    sampler = MixedTacticalGoalSampler(
        position_probability=0.0,
    )

    sample = sampler.sample(
        env=env,
        rng=rng,
        seed=0,
    )

    assert (
        sample.goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
    )

    assert sample.observation.shape == (
        PHYSICAL_OBSERVATION_DIM,
    )


def test_position_only_sampler():

    rng = np.random.default_rng(0)

    env = MicroBilliardEnv()

    sampler = MixedTacticalGoalSampler(
        position_probability=1.0,
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
        sample.goal.cue_target_radius
        is not None
    )


def test_sample_action_matches_goal():

    rng = np.random.default_rng(1)

    env = MicroBilliardEnv()

    sampler = MixedTacticalGoalSampler(
        position_probability=0.5,
    )

    sample = sampler.sample(
        env=env,
        rng=rng,
        seed=200,
    )

    assert (
        sample.macro_action.target_ball
        == sample.goal.target_ball
    )

    assert (
        sample.macro_action.target_pocket
        == sample.goal.target_pocket
    )


def test_goal_type_sampling_is_not_candidate_count_based():

    rng = np.random.default_rng(123)

    sampler = MixedTacticalGoalSampler(
        position_probability=0.5,
    )

    position_count = 0

    num_samples = 1000

    for _ in range(num_samples):

        goal_type = (
            sampler.sample_goal_type(
                rng
            )
        )

        if (
            goal_type
            == TacticalGoalType.POSITION_ATTACK
        ):
            position_count += 1

    ratio = (
        position_count
        / num_samples
    )

    # 固定seedなので安定するが、
    # 少し余裕を持たせる
    assert 0.40 < ratio < 0.60
    