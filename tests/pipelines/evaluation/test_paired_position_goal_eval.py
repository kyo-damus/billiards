import numpy as np

from src.common.types import (
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
from scripts.eval_paired_position_goal_generation import (
    candidate_key,
)


def make_candidate(
    cue_target,
):

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=1,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
        cue_target_position=(
            cue_target
        ),
        cue_target_radius=0.10,
        next_target_ball=1,
        next_target_pocket=5,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


def test_candidate_key_ignores_cue_target():

    a = make_candidate(
        (0.20, 0.30)
    )

    b = make_candidate(
        (0.90, 1.20)
    )

    assert (
        candidate_key(a)
        == candidate_key(b)
    )
