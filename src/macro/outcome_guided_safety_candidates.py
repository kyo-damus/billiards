"""Adapter from FastFiz-selected physical Safety shots to MacroCandidate.

This creates a *candidate representation*. It does NOT add these
candidates to the existing Attack-v0 MCTS or pretrained 25-D networks.
"""

from src.common.types import MacroAction, Strategy
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate, PhysicalShotSpec


class OutcomeGuidedSafetyMacroCandidateGenerator:
    def __init__(
        self,
        shot_generator,
        cue_target_radius=0.12,
    ):
        self.shot_generator = shot_generator
        self.cue_target_radius = float(cue_target_radius)

        if self.cue_target_radius <= 0.0:
            raise ValueError(
                "cue_target_radius must be positive."
            )

    def generate(self, state) -> list[MacroCandidate]:
        shot_proposals = self.shot_generator.generate(state)
        candidates = []

        for proposal in shot_proposals:
            ball = int(proposal.targeted_ball)

            # We use the FastFiz-predicted resting cue position
            # as a provisional TacticalGoal, not as ground truth
            # about an unknown future or a learned Micro outcome.
            cue = proposal.result.next_state.ball_positions[0]

            candidate = MacroCandidate(
                action=MacroAction(
                    strategy=Strategy.DEFENSE,
                    target_ball=ball,
                    target_pocket=None,
                ),
                goal=TacticalGoal(
                    goal_type=TacticalGoalType.SAFETY,
                    target_ball=ball,
                    cue_target_position=(
                        float(cue[0]),
                        float(cue[1]),
                    ),
                    cue_target_radius=self.cue_target_radius,
                ),
                physical_shot=PhysicalShotSpec(
                    angle=float(proposal.shot.angle),
                    power=float(proposal.shot.power),
                    tip_x_mm=float(proposal.shot.tip_x_mm),
                    tip_y_mm=float(proposal.shot.tip_y_mm),
                ),
            )
            candidates.append(candidate)

        return candidates
