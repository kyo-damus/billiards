import numpy as np

from scripts.train_outcome_witness_bc import (
    _conditioned_state,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)


def test_conditioned_state_shape():

    observation = np.zeros(
        PHYSICAL_OBSERVATION_DIM,
        dtype=np.float32,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
        cue_target_position=(
            0.50,
            1.00,
        ),
        cue_target_radius=0.10,
        next_target_ball=1,
        next_target_pocket=5,
    )

    state = _conditioned_state(
        observation,
        goal,
    )

    assert state.shape == (
        34,
    )

    assert state.dtype == (
        np.float32
    )
