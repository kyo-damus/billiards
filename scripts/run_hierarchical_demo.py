import numpy as np
import torch

from src.common.types import GameState

from src.controller.hierarchical_agent import (
    HierarchicalAgent,
)

from src.game.simple_rules import (
    SimplifiedGameRules,
)

from src.macro.action_generator import (
    MacroActionGenerator,
)

from src.macro.mcts import MCTS

from src.macro.physics_transition import (
    PhysicsMacroTransitionModel,
)

from src.macro.policy import (
    MacroPolicyNetwork,
)

from src.macro.value import (
    MacroValueNetwork,
)

# from src.micro.sac import SACAgent

from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)


def make_initial_state():
    return GameState(
        ball_positions=np.array(
            [
                # cue
                [0.80, 1.118],

                # object ball 0
                [0.40, 1.118],

                # object ball 1
                [0.80, 0.40],
            ],
            dtype=np.float32,
        ),

        score=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),

        current_player=0,

        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # =========================================
    # Macro
    # =========================================

    action_generator = MacroActionGenerator()

    policy = MacroPolicyNetwork()
    value = MacroValueNetwork()

    # =========================================
    # Micro
    # =========================================

    micro_agent = GoalConditionedSACAgent(
        action_dim=2,
        device=device,
    )

    # =========================================
    # Physics + Game Rules
    # =========================================

    transition = PhysicsMacroTransitionModel(
        micro_agent=micro_agent,
        deterministic=True,
        game_rules=SimplifiedGameRules(),
    )

    # =========================================
    # Macro MCTS
    # =========================================

    mcts = MCTS(
        policy_network=policy,
        value_network=value,
        action_generator=action_generator,
        transition_model=transition,

        # デモなのでごく少数
        simulations=3,

        device="cpu",
    )

    # =========================================
    # Hierarchical Agent
    # =========================================

    agent = HierarchicalAgent(
        mcts=mcts,
        transition_model=transition,
    )

    state = make_initial_state()

    # =========================================
    # Before
    # =========================================

    print()
    print("=== Initial Game State ===")

    print(
        "ball positions:"
    )
    print(state.ball_positions)

    print(
        "pocket states:",
        state.ball_pocket_indices,
    )

    print(
        "score:",
        state.score,
    )

    print(
        "current player:",
        state.current_player,
    )

    feasible_actions = (
        action_generator.generate(state)
    )

    print()
    print(
        "feasible MacroActions:",
        len(feasible_actions),
    )

    for action in feasible_actions:
        print(
            " ",
            action.strategy.value,
            f"ball={action.target_ball}",
            f"pocket={action.target_pocket}",
        )

    # =========================================
    # One hierarchical decision
    # =========================================

    result = agent.step(state)

    # =========================================
    # Result
    # =========================================

    print()
    print("=== Selected MacroAction ===")

    print(
        "strategy:",
        result.macro_action.strategy.value,
    )

    print(
        "target ball:",
        result.macro_action.target_ball,
    )

    print(
        "target pocket:",
        result.macro_action.target_pocket,
    )

    print()
    print("=== Next Game State ===")

    print(
        "ball positions:"
    )
    print(
        result.next_state.ball_positions
    )

    print(
        "pocket states:",
        result.next_state.ball_pocket_indices,
    )

    print(
        "score:",
        result.next_state.score,
    )

    print(
        "current player:",
        result.next_state.current_player,
    )

    print(
        "reward:",
        result.reward,
    )

    print(
        "game terminated:",
        result.terminated,
    )

    print()
    print(
        "Hierarchical pipeline completed."
    )


if __name__ == "__main__":
    main()
    