import numpy as np
import torch

from src.common.types import GameState

from src.game.simple_rules import (
    SimplifiedGameRules,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.candidate_generator import (
    ExpandedMacroCandidateGenerator,
)

from src.macro.candidate_policy import (
    MacroCandidatePolicyNetwork,
)

from src.macro.mcts import MCTS

from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)

from src.macro.value import (
    MacroValueNetwork,
)

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)


class RecordingMicroAgent:
    """
    integration test用。

    TacticalGoalがMicroまで届いたことを記録し、
    固定の連続行動を返す。
    """

    def __init__(self):
        self.received_goals = []

    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        assert observation.shape == (
            PHYSICAL_OBSERVATION_DIM,
        )

        assert isinstance(
            goal,
            TacticalGoal,
        )

        self.received_goals.append(
            goal
        )

        # 弱めの決定論的ショット
        return np.array(
            [-1.0, 0.0],
            dtype=np.float32,
        )


def make_state():

    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.80, 1.118],

                # object ball 0
                [0.40, 1.118],

                # object ball 1
                [0.80, 0.80],
            ],
            dtype=np.float32,
        ),

        score=np.zeros(
            2,
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def test_candidate_mcts_runs_with_real_physics():

    np.random.seed(0)
    torch.manual_seed(0)

    micro_agent = RecordingMicroAgent()

    transition = PhysicsMacroTransitionModel(
        micro_agent=micro_agent,
        deterministic=True,
        game_rules=SimplifiedGameRules(),
    )

    generator = (
        ExpandedMacroCandidateGenerator()
    )

    policy = (
        MacroCandidatePolicyNetwork()
    )

    value = MacroValueNetwork()

    mcts = MCTS(
        policy_network=policy,
        value_network=value,
        action_generator=generator,
        transition_model=transition,

        # integration確認なので少数
        simulations=2,

        device="cpu",
        candidate_mode=True,
    )

    root = mcts.search(
        make_state()
    )

    # --------------------------------
    # Candidate生成
    # --------------------------------

    assert len(root.children) > 0

    assert all(
        isinstance(
            candidate,
            MacroCandidate,
        )
        for candidate in root.children
    )

    goal_types = {
        candidate.goal.goal_type
        for candidate in root.children
    }

    assert (
        TacticalGoalType.DIRECT_ATTACK
        in goal_types
    )

    assert (
        TacticalGoalType.POSITION_ATTACK
        in goal_types
    )

    # --------------------------------
    # MCTS → PhysicsTransition
    # --------------------------------

    transitioned_children = [
        child
        for child
        in root.children.values()
        if child.state is not None
    ]

    assert (
        len(transitioned_children)
        > 0
    )

    child = transitioned_children[0]

    assert isinstance(
        child.state,
        GameState,
    )

    assert (
        child.state.ball_positions.shape
        == (3, 2)
    )

    assert (
        child.state.ball_pocket_indices.shape
        == (3,)
    )

    # --------------------------------
    # TacticalGoal → Micro
    # --------------------------------

    assert (
        len(
            micro_agent.received_goals
        )
        > 0
    )

    assert all(
        isinstance(
            goal,
            TacticalGoal,
        )
        for goal
        in micro_agent.received_goals
    )
    