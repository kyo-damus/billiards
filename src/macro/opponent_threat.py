from dataclasses import dataclass

from src.goal.types import TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.macro.candidate_generator import MacroCandidateGenerator


@dataclass(frozen=True)
class OpponentDirectThreat:
    """
    Threat from the opponent's perspective after the turn changes.

    These are deterministic FastFiz outcomes for the *current*
    Micro policy, not probabilities or official billiards rules.

    geometric_candidate_count:
        DIRECT shots that pass the geometry feasibility filter.
    successful_pot_count:
        Among those, shots that the Micro policy actually pockets
        without scratching in FastFiz.
    best_success_reward:
        Maximum local reward of a successful DIRECT shot, or 0
        if no candidate succeeds.
    """

    geometric_candidate_count: int
    successful_pot_count: int
    best_success_reward: float
    best_candidate: MacroCandidate | None

    @property
    def has_successful_direct(self) -> bool:
        return self.successful_pot_count > 0


class OpponentDirectThreatEvaluator:
    """
    One-shot opponent DIRECT threat, evaluated with FastFiz.

    This is a diagnostic building block; it does not change
    MacroLocalEvaluator reward, Q targets, or MCTS.

    Use on a nonterminal post-shot GameState where the turn
    belongs to the opponent of defending_player.

    The transition model must return MacroLocalEvaluation so
    scratch and actual pocket success can be distinguished.
    """

    def __init__(
        self,
        transition_model,
        candidate_generator=None,
    ):
        self.transition_model = transition_model
        self.candidate_generator = (
            candidate_generator
            if candidate_generator is not None
            else MacroCandidateGenerator()
        )

    def evaluate(
        self,
        state,
        defending_player,
    ) -> OpponentDirectThreat:
        if defending_player not in (0, 1):
            raise ValueError(
                "defending_player must be 0 or 1."
            )

        if state.current_player == defending_player:
            raise ValueError(
                "Threat requires an opponent-to-move state; "
                "the defending player still has the turn."
            )

        if state.is_pocketed(0):
            raise ValueError(
                "Cannot evaluate opponent threat after "
                "a scratch; this game is terminal."
            )

        candidates = list(
            self.candidate_generator.generate(
                state
            )
        )

        best_candidate = None
        best_reward = None
        successes = 0

        for candidate in candidates:
            if (
                not isinstance(candidate, MacroCandidate)
                or candidate.goal.goal_type
                != TacticalGoalType.DIRECT_ATTACK
            ):
                raise TypeError(
                    "Opponent threat requires DIRECT "
                    "MacroCandidate objects."
                )

            result = self.transition_model.step(
                state,
                candidate,
            )

            evaluation = result.evaluation
            if evaluation is None:
                raise RuntimeError(
                    "Opponent threat needs a "
                    "transition with local evaluation."
                )

            succeeded = bool(
                evaluation.target_potted
                and not evaluation.scratch
            )

            if not succeeded:
                continue

            successes += 1

            score = float(
                evaluation.score
            )

            if (
                best_reward is None
                or score > best_reward
            ):
                best_reward = score
                best_candidate = candidate

        return OpponentDirectThreat(
            geometric_candidate_count=len(candidates),
            successful_pot_count=successes,
            best_success_reward=(
                float(best_reward)
                if best_reward is not None
                else 0.0
            ),
            best_candidate=best_candidate,
        )
