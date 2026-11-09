"""Legacy imports for the Attack/Safety mixed component factory.

The component construction logic is in src.pipelines.mixed_components.
"""

from src.pipelines.mixed_components import (
    DefenseMacroTransitionModel,
    DefenseShotExecutor,
    MixedStrategyMacroTransitionModel,
    OpponentDirectThreatEvaluator,
    OutcomeGuidedSafetyMacroCandidateGenerator,
    OutcomeGuidedSafetyShotGenerator,
    make_components,
    make_mixed_components,
)
