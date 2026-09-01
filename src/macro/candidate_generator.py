from src.goal.adapter import (
    macro_action_to_tactical_goal,
)

from src.macro.action_generator import (
    MacroActionGenerator,
)

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)


class MacroCandidateGenerator:
    """
    MacroActionGeneratorが生成した
    物理的に実行可能なActionを、
    TacticalGoal付きCandidateへ変換する。

    現段階ではDIRECT_ATTACKのみ。
    """

    def __init__(
        self,
        action_generator=None,
    ):
        self.action_generator = (
            action_generator
            if action_generator is not None
            else MacroActionGenerator()
        )

    def generate(
        self,
        state,
    ):
        actions = (
            self.action_generator.generate(
                state
            )
        )

        candidates = []

        for action in actions:

            goal = (
                macro_action_to_tactical_goal(
                    action
                )
            )

            candidates.append(
                MacroCandidate(
                    action=action,
                    goal=goal,
                )
            )

        return candidates

class ExpandedMacroCandidateGenerator:
    """
    複数種類のMacro候補をまとめて生成する。

    現在:
        DIRECT_ATTACK
        POSITION_ATTACK

    将来:
        SAFETY
        etc.
    """

    def __init__(
        self,
        direct_generator=None,
        position_generator=None,
    ):
        self.direct_generator = (
            direct_generator
            if direct_generator is not None
            else MacroCandidateGenerator()
        )

        self.position_generator = (
            position_generator
            if position_generator is not None
            else PositionAttackCandidateGenerator()
        )

    def generate(
        self,
        state,
    ):
        candidates = []

        candidates.extend(
            self.direct_generator.generate(
                state
            )
        )

        candidates.extend(
            self.position_generator.generate(
                state
            )
        )

        return candidates