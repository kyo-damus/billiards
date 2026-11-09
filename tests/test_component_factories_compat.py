"""Contracts and import compatibility for moved Attack/Mixed factories."""

from types import SimpleNamespace
from unittest.mock import Mock

from scripts import attack_components as legacy_attack
from scripts import mixed_components as legacy_mixed
from src.pipelines import attack_components as attack
from src.pipelines import mixed_components as mixed


def test_factories_reexport_exact_original_functions():
    assert legacy_attack.make_components is attack.make_components
    assert (
        legacy_attack.build_attack_components_with_metadata
        is attack.build_attack_components_with_metadata
    )
    assert legacy_attack.load_base_agent is attack.load_base_agent
    assert legacy_attack.load_residual_network is attack.load_residual_network
    assert legacy_mixed.make_components is attack.make_components
    assert legacy_mixed.make_mixed_components is mixed.make_mixed_components


def test_attack_factory_keeps_three_tuple_and_checkpoint_metadata(monkeypatch):
    args = SimpleNamespace(
        base_checkpoint="base.pt",
        residual_checkpoint="residual.pt",
        residual_scale=0.75,
        outcome_grid_size=3,
        radius=0.10,
        outcome_top_k=2,
        outcome_dedup_distance=0.11,
    )
    base, residual, metadata = object(), object(), {"version": "keep"}
    load_base = Mock(return_value=base)
    load_residual = Mock(return_value=(residual, metadata))
    monkeypatch.setattr(attack, "load_base_agent", load_base)
    monkeypatch.setattr(attack, "load_residual_network", load_residual)

    constructors = {}
    for name in (
        "PositionResidualMicroPolicy",
        "FixedScalePositionResidualPolicy",
        "PhysicsMacroTransitionModel",
        "SimplifiedGameRules",
        "MacroLocalEvaluator",
        "MacroCandidateGenerator",
        "OutcomeGuidedPositionCandidateGenerator",
        "ExpandedMacroCandidateGenerator",
    ):
        constructors[name] = Mock(return_value=object())
        monkeypatch.setattr(attack, name, constructors[name])

    components, checkpoint_metadata = attack.build_attack_components_with_metadata(
        args, "cpu"
    )
    assert checkpoint_metadata is metadata
    assert components == (
        constructors["PhysicsMacroTransitionModel"].return_value,
        constructors["ExpandedMacroCandidateGenerator"].return_value,
        constructors["MacroCandidateGenerator"].return_value,
    )
    load_base.assert_called_once_with("base.pt", "cpu")
    load_residual.assert_called_once_with("residual.pt", "cpu")
    assert constructors["PositionResidualMicroPolicy"].call_args.kwargs == {
        "base_agent": base,
        "residual_network": residual,
        "device": "cpu",
    }
    assert constructors["FixedScalePositionResidualPolicy"].call_args.kwargs == {
        "residual_policy": constructors["PositionResidualMicroPolicy"].return_value,
        "residual_scale": 0.75,
        "power_scale": 1.0,
        "angle_scale": 0.0,
    }
    assert constructors["PhysicsMacroTransitionModel"].call_args.kwargs == {
        "micro_agent": constructors["FixedScalePositionResidualPolicy"].return_value,
        "deterministic": True,
        "game_rules": constructors["SimplifiedGameRules"].return_value,
        "local_evaluator": constructors["MacroLocalEvaluator"].return_value,
    }
    assert constructors["OutcomeGuidedPositionCandidateGenerator"].call_args.kwargs == {
        "grid_size": 3,
        "target_radius": 0.10,
        "top_k": 2,
        "dedup_distance": 0.11,
    }
    assert constructors["ExpandedMacroCandidateGenerator"].call_args.kwargs == {
        "direct_generator": constructors["MacroCandidateGenerator"].return_value,
        "position_generator": constructors[
            "OutcomeGuidedPositionCandidateGenerator"
        ].return_value,
    }
    assert attack.make_components(args, "cpu") == components


def test_mixed_factory_keeps_five_tuple_and_safety_parameters(monkeypatch):
    args = SimpleNamespace(
        safety_powers=[0.65, 1.1],
        safety_angles=[-12.0, 0.0, 12.0],
        safety_tip_y_mm=[0.0, 5.0],
        safety_top_k=4,
    )
    attack_transition, attack_generator, direct_generator = (
        object(), object(), object()
    )
    make_attack = Mock(return_value=(
        attack_transition, attack_generator, direct_generator
    ))
    monkeypatch.setattr(mixed, "make_components", make_attack)

    constructors = {}
    for name in (
        "DefenseShotExecutor",
        "DefenseMacroTransitionModel",
        "MixedStrategyMacroTransitionModel",
        "OpponentDirectThreatEvaluator",
        "OutcomeGuidedSafetyShotGenerator",
        "OutcomeGuidedSafetyMacroCandidateGenerator",
    ):
        constructors[name] = Mock(return_value=object())
        monkeypatch.setattr(mixed, name, constructors[name])

    built = mixed.make_mixed_components(args, "cpu")
    assert built == (
        constructors["MixedStrategyMacroTransitionModel"].return_value,
        attack_generator,
        direct_generator,
        constructors["OutcomeGuidedSafetyShotGenerator"].return_value,
        constructors["OutcomeGuidedSafetyMacroCandidateGenerator"].return_value,
    )
    make_attack.assert_called_once_with(args, "cpu")
    assert constructors["MixedStrategyMacroTransitionModel"].call_args.kwargs == {
        "attack_transition_model": attack_transition,
        "defense_transition_model": constructors[
            "DefenseMacroTransitionModel"
        ].return_value,
    }
    assert constructors["OpponentDirectThreatEvaluator"].call_args.kwargs == {
        "transition_model": attack_transition,
        "candidate_generator": direct_generator,
    }
    assert constructors["OutcomeGuidedSafetyShotGenerator"].call_args.kwargs == {
        "defense_executor": constructors["DefenseShotExecutor"].return_value,
        "opponent_threat_evaluator": constructors[
            "OpponentDirectThreatEvaluator"
        ].return_value,
        "powers": [0.65, 1.1],
        "angle_offsets_deg": [-12.0, 0.0, 12.0],
        "tip_offsets_mm": [(0.0, 0.0), (0.0, 5.0)],
        "top_k": 4,
    }
    assert constructors[
        "OutcomeGuidedSafetyMacroCandidateGenerator"
    ].call_args.kwargs == {
        "shot_generator": constructors[
            "OutcomeGuidedSafetyShotGenerator"
        ].return_value,
    }
