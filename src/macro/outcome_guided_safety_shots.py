"""Outcome-guided physical safety shot proposals (pre-Macro v0).

These proposals contain explicit physical ShotAction parameters. They are
not MacroCandidate and are not passed into the Attack-trained Q/Policy.
"""

from dataclasses import dataclass

from src.common.types import ShotAction

from src.micro.defense_shot import (
    DefenseShotExecutor,
    DefenseShotResult,
    plan_defense_shot,
)
from src.macro.opponent_threat import (
    OpponentDirectThreat,
    OpponentDirectThreatEvaluator,
)


def is_eligible_safety_shot(
    result: DefenseShotResult,
    targeted_ball: int,
) -> bool:
    """Minimal legal-contact and turn-change screen for Defense v0."""
    return bool(
        result.first_contact_verified
        and result.first_contact_ball == targeted_ball
        and not result.scratched
        and not result.terminated
        and result.turn_changed
        and result.newly_pocketed_object_count == 0
    )


@dataclass(frozen=True)
class EvaluatedSafetyShot:
    """A physically executed shot and its model-based opponent threat."""

    targeted_ball: int
    shot: ShotAction
    result: DefenseShotResult
    opponent_threat: OpponentDirectThreat
    threat_cost: float


class OutcomeGuidedSafetyShotGenerator:
    """
    Enumerate a small physical shot grid, discard misses/fouls/pots,
    and score the resulting opponent-turn state using DIRECT FastFiz
    rollouts.

    The threat metric is deterministic conditional on today's Micro.
    This is a model-based, potentially optimistic search, NOT a learned
    Safety Micro policy nor an evaluation of game-winning probability.
    """

    def __init__(
        self,
        defense_executor: DefenseShotExecutor,
        opponent_threat_evaluator: OpponentDirectThreatEvaluator,
        powers=(0.65, 1.10),
        angle_offsets_deg=(-12.0, 0.0, 12.0),
        tip_offsets_mm=((0.0, 0.0), (0.0, 5.0)),
        top_k=4,
        geometric_weight=0.02,
    ):
        self.defense_executor = defense_executor
        self.opponent_threat_evaluator = (
            opponent_threat_evaluator
        )
        self.powers = tuple(float(v) for v in powers)
        self.angle_offsets_deg = tuple(
            float(v) for v in angle_offsets_deg
        )
        self.tip_offsets_mm = tuple(
            (float(x), float(y))
            for x, y in tip_offsets_mm
        )
        self.top_k = int(top_k)
        self.geometric_weight = float(geometric_weight)

        if (
            not self.powers
            or not self.angle_offsets_deg
            or not self.tip_offsets_mm
            or self.top_k < 1
            or self.geometric_weight < 0
        ):
            raise ValueError(
                "Shot grids must be nonempty; top_k positive; "
                "geometric_weight nonnegative."
            )

        self.last_stats = {}

    def generate(self, state) -> list[EvaluatedSafetyShot]:
        if state.current_player not in (0, 1):
            raise ValueError(
                "current_player must be 0 or 1."
            )

        stats = {
            "attempted": 0,
            "invalid_physics": 0,
            "wrong_or_no_first_contact": 0,
            "scratch": 0,
            "pocketed": 0,
            "other_ineligible": 0,
            "eligible": 0,
        }
        self.last_stats = stats

        if state.is_pocketed(0):
            return []

        candidates = []

        for ball in (0, 1):
            if state.is_pocketed(ball + 1):
                continue

            for power in self.powers:
                for angle_offset in self.angle_offsets_deg:
                    for tip_x, tip_y in self.tip_offsets_mm:
                        shot = plan_defense_shot(
                            state,
                            target_ball=ball,
                            power=power,
                            angle_offset_deg=angle_offset,
                            tip_x_mm=tip_x,
                            tip_y_mm=tip_y,
                        )
                        stats["attempted"] += 1

                        try:
                            result = self.defense_executor.execute(
                                state,
                                target_ball=ball,
                                shot=shot,
                            )
                        except RuntimeError:
                            stats["invalid_physics"] += 1
                            continue

                        if (
                            not result.first_contact_verified
                            or result.first_contact_ball != ball
                        ):
                            stats["wrong_or_no_first_contact"] += 1
                            continue
                        if result.scratched:
                            stats["scratch"] += 1
                            continue
                        if result.newly_pocketed_object_count:
                            stats["pocketed"] += 1
                            continue
                        if not is_eligible_safety_shot(result, ball):
                            stats["other_ineligible"] += 1
                            continue

                        stats["eligible"] += 1
                        threat = self.opponent_threat_evaluator.evaluate(
                            result.next_state,
                            defending_player=state.current_player,
                        )

                        cost = (
                            float(threat.successful_pot_count)
                            + self.geometric_weight
                            * float(threat.geometric_candidate_count)
                        )

                        candidates.append(
                            EvaluatedSafetyShot(
                                targeted_ball=ball,
                                shot=shot,
                                result=result,
                                opponent_threat=threat,
                                threat_cost=cost,
                            )
                        )

        # Primary criterion: opponent's executable DIRECT successes.
        # Geometric opportunities break near-ties; center-tip/low
        # power are deterministic tie breakers, not safety guarantees.
        candidates.sort(
            key=lambda entry: (
                entry.opponent_threat.successful_pot_count,
                entry.opponent_threat.geometric_candidate_count,
                abs(entry.shot.tip_x_mm)
                + abs(entry.shot.tip_y_mm),
                entry.shot.power,
            )
        )

        return candidates[:self.top_k]
