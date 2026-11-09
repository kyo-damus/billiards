import numpy as np

from src.micro.goal_curriculum import (
    MixedGoalCurriculum,
)


def test_curriculum_start():

    curriculum = MixedGoalCurriculum(
        start_position_probability=0.2,
        end_position_probability=0.5,
        start_target_radius=0.2,
        end_target_radius=0.1,
        ramp_steps=10000,
    )

    probability, radius = (
        curriculum.values(0)
    )

    assert np.isclose(
        probability,
        0.2,
    )

    assert np.isclose(
        radius,
        0.2,
    )


def test_curriculum_middle():

    curriculum = MixedGoalCurriculum(
        ramp_steps=10000,
    )

    probability, radius = (
        curriculum.values(5000)
    )

    assert np.isclose(
        probability,
        0.35,
    )

    assert np.isclose(
        radius,
        0.15,
    )


def test_curriculum_end():

    curriculum = MixedGoalCurriculum(
        ramp_steps=10000,
    )

    probability, radius = (
        curriculum.values(20000)
    )

    assert np.isclose(
        probability,
        0.5,
    )

    assert np.isclose(
        radius,
        0.1,
    )
    