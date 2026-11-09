import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.goal.adapter import (
    macro_action_to_tactical_goal,
)

from src.goal.encoder import (
    TACTICAL_GOAL_DIM,
    encode_tactical_goal,
)

from src.goal.evaluator import (
    TacticalGoalEvaluator,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)


def test_macro_action_to_goal():

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=3,
    )

    goal = macro_action_to_tactical_goal(
        action
    )

    assert (
        goal.goal_type
        == TacticalGoalType.DIRECT_ATTACK
    )

    assert goal.target_ball == 0
    assert goal.target_pocket == 3


def test_goal_encoding_shape():

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=1,
        target_pocket=5,
    )

    encoded = encode_tactical_goal(goal)

    assert encoded.shape == (
        TACTICAL_GOAL_DIM,
    )

    assert encoded.dtype == np.float32


def test_position_goal_encoding():

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=1,
        cue_target_position=(
            0.5,
            1.0,
        ),
        cue_target_radius=0.1,
    )

    encoded = encode_tactical_goal(goal)

    assert encoded.shape == (
        TACTICAL_GOAL_DIM,
    )

    # cue region flag
    assert encoded[11] == 1.0


def test_direct_attack_goal_achieved():

    state = GameState(
        ball_positions=np.array(
            [
                [0.5, 0.3],
                [0.04, 0.04],
                [0.8, 1.5],
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
                -1,
                 0,
                -1,
            ],
            dtype=np.int32,
        ),
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=0,
    )

    evaluator = TacticalGoalEvaluator()

    assert evaluator.is_achieved(
        goal,
        state,
    )
    
def test_position_attack_goal_achieved():

    state = GameState(
        ball_positions=np.array(
            [
                # cue stopped near desired position
                [0.70, 1.00],

                # ball 0 pocketed
                [0.03, 1.10],

                # ball 1
                [0.80, 0.50],
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
                -1,
                 1,
                -1,
            ],
            dtype=np.int32,
        ),
    )

    goal = TacticalGoal(
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

    evaluator = TacticalGoalEvaluator()

    assert evaluator.is_achieved(
        goal,
        state,
    )