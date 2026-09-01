import numpy as np

from src.goal.conditioner import (
    encode_micro_goal,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.micro.goal_observation import (
    build_physical_observation,
)


def test_same_state_different_goal():

    positions = np.array(
        [
            [0.8, 1.118],
            [0.4, 1.118],
            [0.8, 0.4],
        ],
        dtype=np.float32,
    )

    pockets = np.array(
        [-1, -1, -1],
        dtype=np.int32,
    )

    observation = (
        build_physical_observation(
            positions,
            pockets,
        )
    )

    goal_a = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
    )

    goal_b = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=1,
        target_pocket=5,
    )

    encoded_a = encode_micro_goal(
        observation,
        goal_a,
    )

    encoded_b = encode_micro_goal(
        observation,
        goal_b,
    )

    # 物理状態は同じ
    assert observation.shape == (9,)

    # Goalを変えると条件表現が変わる
    assert not np.allclose(
        encoded_a,
        encoded_b,
    )

def test_goal_identity_is_explicitly_encoded():

    positions = np.array(
        [
            [0.8, 1.118],
            [0.4, 1.118],
            [0.8, 0.4],
        ],
        dtype=np.float32,
    )

    pockets = np.array(
        [-1, -1, -1],
        dtype=np.int32,
    )

    observation = build_physical_observation(
        positions,
        pockets,
    )

    goal = TacticalGoal(
        goal_type=TacticalGoalType.DIRECT_ATTACK,
        target_ball=1,
        target_pocket=5,
    )

    encoded = encode_micro_goal(
        observation,
        goal,
    )

    assert encoded.shape == (25,)

    # ball 1 one-hot
    assert encoded[17] == 0.0
    assert encoded[18] == 1.0

    # pocket 5 one-hot
    assert encoded[19 + 5] == 1.0

    assert np.sum(
        encoded[19:25]
    ) == 1.0