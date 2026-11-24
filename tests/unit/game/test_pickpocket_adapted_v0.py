"""CPU-only test contracts for PickPocket-style v0 nine-ball adaptation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.game.agents.baselines.pickpocket.agent import (
    PhysicalErrorModel, PickPocketAdaptedAgent,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal
from src.game.nine_ball_session import create_nine_ball_rack


def test_error_model_zero_noise_preserves_physical_shot():
    from src.micro.shot_control import PhysicalShot5D

    shot = PhysicalShot5D(1.8, 32.0, 10.0, 2.5, 0.0)
    model = PhysicalErrorModel(0.0, 0.0, 0.0, 0.0, 0.0)
    got = model.perturb(shot, np.random.default_rng(3))
    assert got == shot
    with pytest.raises(ValueError, match="nonnegative"):
        PhysicalErrorModel(azimuth_deg=-0.01)


def test_error_model_never_exits_tip_disk():
    from src.micro.shot_control import PhysicalShot5D

    shot = PhysicalShot5D(1.8, 32.0, 69.0, 10.0, 6.0)
    model = PhysicalErrorModel(azimuth_deg=3, speed_m_s=9, elevation_deg=9,
                               tip_x_mm=20, tip_y_mm=20)
    rng = np.random.default_rng(23)
    for _ in range(100):
        got = model.perturb(shot, rng)
        assert got.speed_m_s > 0
        assert 0 <= got.elevation_deg <= 69
        assert got.tip_x_mm ** 2 + got.tip_y_mm ** 2 <= 11.99 ** 2 + 1e-5


def test_attack_candidate_generator_keeps_nine_ball_lowest_and_multiple_pockets(monkeypatch):
    import src.game.agents.baselines.pickpocket.agent as module

    monkeypatch.setattr(module, "direct_pocket_candidates", lambda state, target: (0, 1))
    monkeypatch.setattr(module, "reference_angle_deg", lambda state, goal: 13.0)
    agent = object.__new__(PickPocketAdaptedAgent)
    agent.max_candidates = 4
    state = replace(create_nine_ball_rack(breaker=0, seed=3), break_shot=False)
    candidates = agent._attack_candidates(state)
    assert len(candidates) == 4
    assert {g.target_ball for g, _ in candidates} == {1}
    assert {g.target_pocket for g, _ in candidates} == {0, 1}
    assert all(g.kind == "DIRECT_ATTACK" for g, _ in candidates)


def test_sampling_scores_exact_pot_and_does_not_invent_outcome(monkeypatch):
    import src.game.agents.baselines.pickpocket.agent as module

    state = replace(create_nine_ball_rack(breaker=0, seed=3), break_shot=False)
    goal = NineBallGoal("DIRECT_ATTACK", 1, 0)
    shot = module.PhysicalShot5D(1.6, 40.0, 10.0, 0.0, 0.0)
    pot_indices = state.ball_pocket_indices.copy()
    pot_indices[state.array_index_for_ball(1)] = 0

    def fake_sim(*args, **kwargs):
        return (
            SimpleNamespace(pocket_indices=pot_indices),
            SimpleNamespace(foul=False, illegal_break=False, rack_winner=None,
                            turn_changed=False, next_state=state),
        )

    monkeypatch.setattr(module, "simulate_nine_ball_shot", fake_sim)
    agent = object.__new__(PickPocketAdaptedAgent)
    agent.samples = 3
    agent.error_model = PhysicalErrorModel(0, 0, 0, 0, 0)
    agent.simulator = object()
    agent.rules = object()
    result = agent._score_candidate(
        state, module.NineBallContext(), goal, shot,
        rng=np.random.default_rng(10),
    )
    assert result.accepted == 3
    assert result.invalid == 0
    assert result.pot_rate == 1.0


def test_accept_table_requires_no_simulation_or_micro(monkeypatch):
    state = replace(create_nine_ball_rack(breaker=0, seed=3), break_shot=False)
    agent = object.__new__(PickPocketAdaptedAgent)
    monkeypatch.setattr(agent, "_select", lambda *args: pytest.fail("no sim for table choice"))
    result = agent.choose_match_shot(
        state, module_context("illegal_break"), None,
    )
    assert result.macro_goal.kind == "ACCEPT_TABLE"
    assert result.shot is None


def module_context(choice):
    from src.game.nine_ball_rules import NineBallContext
    return NineBallContext(pending_choice=choice)
