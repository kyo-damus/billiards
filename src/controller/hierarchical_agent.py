from dataclasses import dataclass

from src.common.types import (
    GameState,
    MacroAction,
)

from src.macro.transition import (
    MacroTransitionResult,
)


@dataclass
class HierarchicalStepResult:
    """
    階層型エージェントが1回意思決定した結果。
    """

    macro_action: MacroAction

    next_state: GameState
    reward: float
    terminated: bool


class HierarchicalAgent:
    """
    Macro MCTS と Micro制御を統合する上位エージェント。

    処理:

        GameState
            ↓
        Macro MCTS
            ↓
        MacroAction
            ↓
        PhysicsMacroTransitionModel
            ↓
        Micro SAC
            ↓
        FastFiz
            ↓
        next GameState
    """

    def __init__(
        self,
        mcts,
        transition_model,
    ):
        self.mcts = mcts
        self.transition_model = transition_model

    # ============================================================
    # Planning
    # ============================================================

    def select_macro_action(
        self,
        state: GameState,
    ) -> MacroAction:
        """
        MCTSでMacroActionを決定する。
        """

        return self.mcts.select_action(state)

    # ============================================================
    # Execution
    # ============================================================

    def execute_macro_action(
        self,
        state: GameState,
        action: MacroAction,
    ) -> MacroTransitionResult:
        """
        選択されたMacroActionを
        Micro + FastFizで実際に実行する。
        """

        return self.transition_model.step(
            state,
            action,
        )

    # ============================================================
    # Full hierarchical step
    # ============================================================

    def step(
        self,
        state: GameState,
    ) -> HierarchicalStepResult:
        """
        Macro意思決定から物理実行までを1回行う。
        """

        macro_action = self.select_macro_action(
            state
        )

        transition_result = (
            self.execute_macro_action(
                state,
                macro_action,
            )
        )

        return HierarchicalStepResult(
            macro_action=macro_action,
            next_state=transition_result.next_state,
            reward=transition_result.reward,
            terminated=transition_result.terminated,
        )
        