"""Shared Attack component construction for both CLI and mixed pipelines.

The original parameters, return shape, and checkpoint loading behavior
are retained. CLI modules re-export these functions for compatibility.
"""

from src.pipelines.checkpoint_loading import (
    load_base_agent,
    load_residual_network,
)
from src.game.simple_rules import SimplifiedGameRules
from src.macro.candidate_generator import (
    ExpandedMacroCandidateGenerator,
    MacroCandidateGenerator,
)
from src.macro.local_evaluation import MacroLocalEvaluator
from src.macro.outcome_guided_position_generator import (
    OutcomeGuidedPositionCandidateGenerator,
)
from src.macro.physics_transition import PhysicsMacroTransitionModel
from src.micro.fixed_scale_position_residual_policy import (
    FixedScalePositionResidualPolicy,
)
from src.micro.position_residual_policy import PositionResidualMicroPolicy


def build_attack_components_with_metadata(
    args,
    device,
):
    base_agent = load_base_agent(
        args.base_checkpoint,
        device,
    )

    (
        residual_network,
        residual_checkpoint,
    ) = load_residual_network(
        args.residual_checkpoint,
        device,
    )

    residual_policy = (
        PositionResidualMicroPolicy(
            base_agent=base_agent,
            residual_network=(
                residual_network
            ),
            device=device,
        )
    )

    micro_policy = (
        FixedScalePositionResidualPolicy(
            residual_policy=(
                residual_policy
            ),
            residual_scale=(
                args.residual_scale
            ),
            power_scale=1.0,
            angle_scale=0.0,
        )
    )

    transition = (
        PhysicsMacroTransitionModel(
            micro_agent=micro_policy,
            deterministic=True,
            game_rules=(
                SimplifiedGameRules()
            ),
            local_evaluator=(
                MacroLocalEvaluator()
            ),
        )
    )

    direct_generator = (
        MacroCandidateGenerator()
    )

    position_generator = (
        OutcomeGuidedPositionCandidateGenerator(
            grid_size=(
                args.outcome_grid_size
            ),
            target_radius=(
                args.radius
            ),
            top_k=(
                args.outcome_top_k
            ),
            dedup_distance=(
                args.outcome_dedup_distance
            ),
        )
    )

    root_generator = (
        ExpandedMacroCandidateGenerator(
            direct_generator=(
                direct_generator
            ),
            position_generator=(
                position_generator
            ),
        )
    )

    return (
        (
            transition,
            root_generator,
            direct_generator,
        ),
        residual_checkpoint,
    )


def make_components(args, device):
    """Backward-compatible public factory: preserve the original 3-tuple."""
    components, _ = build_attack_components_with_metadata(args, device)
    return components
