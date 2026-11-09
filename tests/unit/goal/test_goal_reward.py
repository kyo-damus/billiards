import numpy as np

from src.goal.reward import (
    POSITION_SUCCESS_BONUS,
    calculate_goal_conditioned_reward,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.micro.goal_observation import (
    build_physical_observation,
)


def make_observation(
    cue_position,
):

    positions = np.array(
        [
            cue_position,
            [0.40, 1.118],
            [0.80, 0.80],
        ],
        dtype=np.float32,
    )

    pockets = np.array(
        [-1, -1, -1],
        dtype=np.int32,
    )

    return build_physical_observation(
        positions,
        pockets,
    )


def make_position_goal():

    return TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),

        target_ball=0,
        target_pocket=1,

        cue_target_position=(
            0.70,
            1.00,
        ),

        cue_target_radius=0.10,

        next_target_ball=1,
        next_target_pocket=5,
    )


def test_direct_attack_keeps_base_reward():

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
    )

    observation = make_observation(
        (0.8, 1.1)
    )

    reward, info = (
        calculate_goal_conditioned_reward(
            base_reward=3.0,
            goal=goal,
            before_observation=observation,
            after_observation=observation,
            shot_info={
                "success": True,
                "scratched": False,
            },
        )
    )

    assert reward == 3.0
    assert info["goal_achieved"]


def test_position_closer_gives_positive_shaping():

    before = make_observation(
        (0.90, 1.00)
    )

    after = make_observation(
        (0.75, 1.00)
    )

    reward, info = (
        calculate_goal_conditioned_reward(
            base_reward=0.0,
            goal=make_position_goal(),
            before_observation=before,
            after_observation=after,
            shot_info={
                "success": False,
                "scratched": False,
            },
        )
    )

    assert (
        info["position_progress"]
        > 0.0
    )

    assert reward > 0.0


def test_position_farther_gives_negative_shaping():

    before = make_observation(
        (0.75, 1.00)
    )

    after = make_observation(
        (1.00, 1.00)
    )

    reward, info = (
        calculate_goal_conditioned_reward(
            base_reward=0.0,
            goal=make_position_goal(),
            before_observation=before,
            after_observation=after,
            shot_info={
                "success": False,
                "scratched": False,
            },
        )
    )

    assert (
        info["position_progress"]
        < 0.0
    )

    assert reward < 0.0


def test_position_success_gets_bonus():

    before = make_observation(
        (0.90, 1.00)
    )

    after = make_observation(
        (0.70, 1.00)
    )

    reward, info = (
        calculate_goal_conditioned_reward(
            base_reward=10.0,
            goal=make_position_goal(),
            before_observation=before,
            after_observation=after,
            shot_info={
                "success": True,
                "scratched": False,
            },
        )
    )

    assert (
        info["cue_region_reached"]
    )

    assert (
        info["goal_achieved"]
    )

    assert (
        info["position_bonus"]
        == POSITION_SUCCESS_BONUS
    )

    assert reward > 10.0


def test_scratch_disables_position_bonus():

    before = make_observation(
        (0.90, 1.00)
    )

    after = make_observation(
        (0.70, 1.00)
    )

    reward, info = (
        calculate_goal_conditioned_reward(
            base_reward=-10.0,
            goal=make_position_goal(),
            before_observation=before,
            after_observation=after,
            shot_info={
                "success": True,
                "scratched": True,
            },
        )
    )

    assert not info[
        "goal_achieved"
    ]

    assert (
        info["position_bonus"]
        == 0.0
    )

    assert (
        info["position_progress"]
        == 0.0
    )

    assert reward == -10.0
    