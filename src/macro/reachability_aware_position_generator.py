from dataclasses import dataclass

from src.macro.candidate import (
    MacroCandidate,
)
from src.macro.position_candidate_generator import (
    PositionAttackCandidateGenerator,
)
from src.macro.position_feasibility import (
    PositionFeasibilityResult,
    PositionGoalFeasibilityChecker,
)


@dataclass(frozen=True)
class RankedPositionCandidate:
    """
    POSITION_ATTACK候補と、
    現在のMicro action spaceに対する
    近似的な実現可能性評価の組。
    """

    candidate: MacroCandidate
    feasibility: PositionFeasibilityResult


class ReachabilityAwarePositionCandidateGenerator:
    """
    従来のPOSITION_ATTACK候補を、
    現在のMicro action spaceでの実現可能性を考慮して
    ranking / pruningする。

    重要:
        feasibilityは有限個のaction gridによる近似評価であり、
        reachable=Falseは数学的な到達不能を意味しない。

    優先順位:
        1. cue target regionまで到達できた候補
        2. pot可能でbest cue distanceが小さい候補
        3. coarse gridでpotが見つからない候補

    generate()では、pot可能候補が1つ以上ある場合は
    coarse gridでpotが見つからなかった候補を除外する。

    ただし全候補でpotが見つからなかった場合は、
    Macro候補が完全に消えるのを避けるため、
    fallback_to_unfiltered=Trueなら元候補へfallbackする。
    """

    def __init__(
        self,
        base_generator=None,
        feasibility_checker=None,
        top_k=8,
        fallback_to_unfiltered=True,
    ):
        self.base_generator = (
            base_generator
            if base_generator is not None
            else PositionAttackCandidateGenerator()
        )

        self.feasibility_checker = (
            feasibility_checker
            if feasibility_checker is not None
            else PositionGoalFeasibilityChecker(
                grid_size=5,
            )
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        self.top_k = int(top_k)
        self.fallback_to_unfiltered = bool(
            fallback_to_unfiltered
        )

    def rank(
        self,
        state,
    ):
        """
        全POSITION候補を実現可能性で評価し、
        良い候補から順に返す。
        """

        candidates = (
            self.base_generator.generate(
                state
            )
        )

        ranked = []

        for candidate in candidates:
            feasibility = (
                self.feasibility_checker.check(
                    state,
                    candidate,
                )
            )

            ranked.append(
                RankedPositionCandidate(
                    candidate=candidate,
                    feasibility=feasibility,
                )
            )

        ranked.sort(
            key=self._sort_key
        )

        return ranked

    def select_ranked(
        self,
        ranked,
    ):
        """
        rank()済みの候補から、
        Macroへ渡す候補を選択する。

        Returns
        -------
        list[RankedPositionCandidate]
        """

        if not ranked:
            return []

        pot_feasible = [
            item
            for item in ranked
            if item.feasibility.pot_found
        ]

        if pot_feasible:
            pool = pot_feasible

        elif self.fallback_to_unfiltered:
            pool = list(ranked)

        else:
            return []

        return pool[
            : self.top_k
        ]

    def generate(
        self,
        state,
    ):
        """
        ExpandedMacroCandidateGeneratorへ
        そのまま渡せるMacroCandidate列を返す。
        """

        ranked = self.rank(
            state
        )

        selected = self.select_ranked(
            ranked
        )

        return [
            item.candidate
            for item in selected
        ]

    @staticmethod
    def _sort_key(
        item,
    ):
        feasibility = (
            item.feasibility
        )

        # reachableを最優先し、
        # 次にpot_found、
        # 最後にcue targetまでの距離で並べる。
        return (
            0
            if feasibility.reachable
            else 1,

            0
            if feasibility.pot_found
            else 1,

            float(
                feasibility.best_cue_distance
            ),
        )
