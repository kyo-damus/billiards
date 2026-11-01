"""Execute explicit SAFETY MacroCandidates without changing ATTACK Micro.

A Defense candidate must supply a physical shot; no learned Micro SAC
policy is run for this branch. Opponent quality belongs in turn-aware
lookahead, not as an extra immediate reward (to avoid double counting).
"""

from math import isfinite, hypot

from src.common.types import (
    GameState,
    ShotAction,
    Strategy,
)
from src.goal.types import TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.macro.defense_local_evaluation import (
    DefenseLocalEvaluation,
)
from src.macro.outcome_guided_safety_shots import (
    is_eligible_safety_shot,
)
from src.macro.transition import MacroTransitionResult
from src.micro.defense_shot import DefenseShotExecutor


class DefenseMacroTransitionModel:
    def __init__(
        self,
        executor: DefenseShotExecutor | None = None,
        wrong_contact_penalty: float = 1.0,
    ):
        if not (
            isfinite(wrong_contact_penalty)
            and wrong_contact_penalty >= 0.0
        ):
            raise ValueError(
                "wrong_contact_penalty must be finite and >= 0."
            )

        self.executor = (
            executor if executor is not None
            else DefenseShotExecutor()
        )
        self.wrong_contact_penalty = float(
            wrong_contact_penalty
        )
        self.last_evaluation = None

    def step(
        self,
        state: GameState,
        candidate: MacroCandidate,
    ) -> MacroTransitionResult:
        if not isinstance(candidate, MacroCandidate):
            raise TypeError(
                "Defense requires a MacroCandidate with physical_shot."
            )

        action = candidate.action
        goal = candidate.goal
        spec = candidate.physical_shot

        if (
            action.strategy != Strategy.DEFENSE
            or goal.goal_type != TacticalGoalType.SAFETY
        ):
            raise ValueError(
                "Defense transition requires DEFENSE/SAFETY."
            )

        if (
            action.target_ball not in (0, 1)
            or action.target_pocket is not None
            or spec is None
        ):
            raise ValueError(
                "Executable Defense requires a target_ball, "
                "no target_pocket, and physical_shot."
            )

        if (
            goal.target_ball is not None
            and goal.target_ball != action.target_ball
        ):
            raise ValueError(
                "Safety goal and action target_ball disagree."
            )

        if (
            goal.cue_target_position is None
            or goal.cue_target_radius is None
            or not isfinite(goal.cue_target_radius)
            or goal.cue_target_radius <= 0.0
        ):
            raise ValueError(
                "SAFETY needs a cue target position "
                "and positive radius."
            )

        target_x, target_y = goal.cue_target_position
        if not (
            isfinite(target_x)
            and isfinite(target_y)
        ):
            raise ValueError(
                "Cue target position must be finite."
            )

        shot = ShotAction(
            angle=spec.angle,
            power=spec.power,
            tip_x_mm=spec.tip_x_mm,
            tip_y_mm=spec.tip_y_mm,
        )

        physical = self.executor.execute(
            state,
            target_ball=action.target_ball,
            shot=shot,
        )

        cue_x, cue_y = (
            physical.next_state.ball_positions[0]
        )
        distance = hypot(
            float(cue_x) - target_x,
            float(cue_y) - target_y,
        )
        cue_reached = (
            distance <= goal.cue_target_radius
        )

        safety_eligible = is_eligible_safety_shot(
            physical,
            action.target_ball,
        )

        # The game reward accounts for actual pocket/scratch events.
        # Add a v0 penalty for a missed/wrong first ball contact,
        # except on scratches (already penalized by game rules).
        reward = float(
            physical.game_reward
        )

        if (
            not physical.first_contact_verified
            and not physical.scratched
        ):
            reward -= self.wrong_contact_penalty

        evaluation = DefenseLocalEvaluation(
            first_contact_ball=(
                physical.first_contact_ball
            ),
            intended_first_contact=(
                physical.first_contact_verified
            ),
            scratch=physical.scratched,
            turn_changed=physical.turn_changed,
            object_ball_pocketed=(
                physical.newly_pocketed_object_count > 0
            ),
            safety_eligible=safety_eligible,
            cue_distance=distance,
            cue_region_reached=cue_reached,
            goal_achieved=(
                safety_eligible and cue_reached
            ),
            score=reward,
        )
        self.last_evaluation = evaluation

        return MacroTransitionResult(
            next_state=physical.next_state,
            reward=reward,
            terminated=physical.terminated,
            evaluation=evaluation,
        )


class MixedStrategyMacroTransitionModel:
    """Dispatch by strategy without modifying the existing ATTACK path."""

    def __init__(
        self,
        attack_transition_model,
        defense_transition_model,
    ):
        self.attack_transition_model = (
            attack_transition_model
        )
        self.defense_transition_model = (
            defense_transition_model
        )
        self.last_evaluation = None

    def step(
        self,
        state: GameState,
        choice,
    ) -> MacroTransitionResult:
        action = (
            choice.action
            if isinstance(choice, MacroCandidate)
            else choice
        )

        if action.strategy == Strategy.ATTACK:
            result = self.attack_transition_model.step(
                state, choice
            )
        elif action.strategy == Strategy.DEFENSE:
            result = self.defense_transition_model.step(
                state, choice
            )
        else:
            raise ValueError(
                "Unknown Macro strategy."
            )

        self.last_evaluation = result.evaluation
        return result
