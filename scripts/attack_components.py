"""Legacy imports for the Attack component factory.

Implementations live in src.pipelines.attack_components. Keep this path
for existing command-line scripts and downstream experiment imports.
"""

from src.pipelines.attack_components import (
    ExpandedMacroCandidateGenerator,
    FixedScalePositionResidualPolicy,
    MacroCandidateGenerator,
    MacroLocalEvaluator,
    OutcomeGuidedPositionCandidateGenerator,
    PhysicsMacroTransitionModel,
    PositionResidualMicroPolicy,
    SimplifiedGameRules,
    build_attack_components_with_metadata,
    load_base_agent,
    load_residual_network,
    make_components,
)
