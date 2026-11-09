"""Build Attack/Safety components for mixed Macro runs.

Moved from scripts without changing physics or candidate semantics.
"""

from src.pipelines.attack_components import make_components
from src.macro.defense_transition import (
    DefenseMacroTransitionModel,
    MixedStrategyMacroTransitionModel,
)
from src.macro.opponent_threat import OpponentDirectThreatEvaluator
from src.macro.outcome_guided_safety_candidates import (
    OutcomeGuidedSafetyMacroCandidateGenerator,
)
from src.macro.outcome_guided_safety_shots import (
    OutcomeGuidedSafetyShotGenerator,
)
from src.micro.defense_shot import DefenseShotExecutor


def make_mixed_components(args, device):
    attack, attack_generator, direct_generator = make_components(
        args, device
    )

    defense_executor = DefenseShotExecutor()
    defense_transition = DefenseMacroTransitionModel(
        executor=defense_executor,
    )
    transition = MixedStrategyMacroTransitionModel(
        attack_transition_model=attack,
        defense_transition_model=defense_transition,
    )
    threat = OpponentDirectThreatEvaluator(
        transition_model=attack,
        candidate_generator=direct_generator,
    )
    shot_generator = OutcomeGuidedSafetyShotGenerator(
        defense_executor=defense_executor,
        opponent_threat_evaluator=threat,
        powers=args.safety_powers,
        angle_offsets_deg=args.safety_angles,
        tip_offsets_mm=[
            (0.0, float(tip_y))
            for tip_y in args.safety_tip_y_mm
        ],
        top_k=args.safety_top_k,
    )
    candidate_adapter = OutcomeGuidedSafetyMacroCandidateGenerator(
        shot_generator=shot_generator,
    )

    return (
        transition,
        attack_generator,
        direct_generator,
        shot_generator,
        candidate_adapter,
    )
