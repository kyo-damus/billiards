import numpy as np

from src.common.types import GameState
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.local_evaluation import (
    MacroLocalEvaluator,
)


def make_state(
    cue=(0.80, 1.00),
    target_pocket=2,
    cue_pocket=-1,
):
    return GameState(
        ball_positions=np.array(
            [
                cue,
                [0.40, 1.10],
                [1.00, 1.50],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=np.array(
            [
                cue_pocket,
                target_pocket,
                -1,
            ],
            dtype=np.int32,
        ),
    )


def make_direct_goal():
    return TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=2,
    )


def make_position_goal():
    return TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=2,
        cue_target_position=(
            0.80,
            1.00,
        ),
        cue_target_radius=0.10,
        next_target_ball=1,
        next_target_pocket=3,
    )


def test_direct_success_score_is_pot_reward():
    evaluator = MacroLocalEvaluator()

    result = evaluator.evaluate(
        goal=make_direct_goal(),
        state=make_state(),
        shot_info={
            "success": True,
            "scratched": False,
            "wrong_pocket": False,
        },
    )

    assert result.target_potted is True
    assert result.scratch is False
    assert result.position_quality is None
    assert result.next_shot_feasible is True
    assert np.isclose(
        result.score,
        1.25,
    )


def test_wrong_pocket_is_penalized():
    evaluator = MacroLocalEvaluator()

    state = make_state(
        target_pocket=1,
    )

    result = evaluator.evaluate(
        goal=make_direct_goal(),
        state=state,
        shot_info={
            "success": False,
            "scratched": False,
            "wrong_pocket": True,
        },
    )

    assert result.target_potted is False
    assert result.wrong_pocket is True
    assert np.isclose(
        result.score,
        -0.5,
    )


def test_position_center_gets_full_quality_bonus():
    evaluator = MacroLocalEvaluator()

    result = evaluator.evaluate(
        goal=make_position_goal(),
        state=make_state(
            cue=(0.80, 1.00),
        ),
        shot_info={
            "success": True,
            "scratched": False,
            "wrong_pocket": False,
        },
    )

    assert (
        result.position_goal_satisfied
        is True
    )
    assert np.isclose(
        result.cue_distance,
        0.0,
    )
    assert np.isclose(
        result.position_quality,
        1.0,
    )
    assert result.next_shot_feasible is True
    assert np.isclose(
        result.score,
        1.75,
    )


def test_failed_pot_does_not_receive_position_bonus():
    evaluator = MacroLocalEvaluator()

    result = evaluator.evaluate(
        goal=make_position_goal(),
        state=make_state(
            cue=(0.80, 1.00),
            target_pocket=-1,
        ),
        shot_info={
            "success": False,
            "scratched": False,
            "wrong_pocket": False,
        },
    )

    assert (
        result.position_goal_satisfied
        is False
    )
    assert np.isclose(
        result.position_quality,
        1.0,
    )
    assert result.next_shot_feasible is False
    assert np.isclose(
        result.score,
        0.0,
    )


def test_position_miss_does_not_receive_position_bonus():

    evaluator = MacroLocalEvaluator()

    result = evaluator.evaluate(
        goal=make_position_goal(),
        state=make_state(
            cue=(0.80, 1.28),
        ),
        shot_info={
            "success": True,
            "scratched": False,
            "wrong_pocket": False,
        },
    )

    assert (
        result.position_goal_satisfied
        is False
    )
    assert (
        result.position_quality
        is not None
    )
    assert (
        result.position_quality
        > 0.0
    )
    assert result.next_shot_feasible is True

    # Position Goal未達の微小qualityは
    # scoreへ加算しない。
    assert np.isclose(
        result.score,
        1.25,
    )


def test_scratch_disables_position_quality_bonus():
    evaluator = MacroLocalEvaluator()

    result = evaluator.evaluate(
        goal=make_position_goal(),
        state=make_state(
            cue_pocket=0,
        ),
        shot_info={
            "success": True,
            "scratched": True,
            "wrong_pocket": False,
        },
    )

    assert result.scratch is True
    assert result.position_quality == 0.0
    assert result.next_shot_feasible is False
    assert (
        result.position_goal_satisfied
        is False
    )
    assert np.isclose(
        result.score,
        -1.0,
    )
