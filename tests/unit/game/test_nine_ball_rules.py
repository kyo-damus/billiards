"""Core 9-ball rules + rack/match control using deterministic fake physics.

The rules tests require no native FastFiz build or GPU.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.game.match import MatchConfig
from src.game.nine_ball_rules import (
    BALL_RADIUS_M, FOOT_SPOT, NineBallContext, NineBallRules,
)
from src.game.nine_ball_session import (
    NineBallMatchSession, NineBallRackSession, create_nine_ball_rack,
)


def event(kind, ball1, ball2=None, time=0.1):
    return SimpleNamespace(
        kind=kind, ball1=ball1, ball2=ball2, time_s=time,
    )


def physical(
    previous, *, contact=1, pocketed=(), scratch=False, rail=True,
    early_rail=False, precondition=0, placement=0,
):
    pockets = previous.ball_pocket_indices.copy()
    for number in pocketed:
        pockets[number] = 1
    if scratch:
        pockets[0] = 0
    events = []
    if early_rail:
        events.append(event("RAIL_COLLISION", 0, time=0.05))
    if contact is not None:
        events.append(event("BALL_COLLISION", 0, contact, time=0.1))
    if rail:
        events.append(event(
            "RAIL_COLLISION", contact if contact is not None else 1,
            time=0.2,
        ))
    return SimpleNamespace(
        positions=previous.ball_positions.copy(),
        pocket_indices=pockets,
        first_contact_ball=contact,
        events=tuple(events),
        precondition_code=precondition,
        placement_code=placement,
    )


@pytest.mark.parametrize("seed", [0, 50, 123])
def test_diamond_rack_has_one_at_apex_nine_at_foot_spot(seed):
    state = create_nine_ball_rack(breaker=0, seed=seed)
    assert state.ball_count == 10
    assert state.break_shot
    np.testing.assert_allclose(state.ball_positions[9], FOOT_SPOT, atol=1e-4)
    assert state.ball_positions[1, 1] < state.ball_positions[9, 1]
    for i in range(10):
        for j in range(i + 1, 10):
            distance = np.linalg.norm(
                state.ball_positions[i] - state.ball_positions[j]
            )
            assert distance >= 2 * BALL_RADIUS_M - 1e-6


def test_legal_contact_and_rail_without_pot_changes_turn():
    state = replace(create_nine_ball_rack(breaker=0, seed=2), break_shot=False)
    result = NineBallRules().apply(
        state, physical(state), NineBallContext()
    )
    assert not result.foul
    assert result.turn_changed
    assert result.next_state.current_player == 1
    assert not result.rack_finished


def test_pocketed_lowest_keeps_turn():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state, physical(state, pocketed=(1,), rail=False),
        NineBallContext(),
    )
    assert not result.foul
    assert not result.turn_changed
    assert result.newly_pocketed == (1,)


def test_legal_nine_ball_combo_wins_rack():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state, physical(state, pocketed=(9,), rail=False),
        NineBallContext(),
    )
    assert not result.foul
    assert result.rack_winner == 0
    assert result.rack_finished
    assert not result.nine_respotted


def test_wrong_first_with_nine_pocketed_is_foul_and_respotted():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state,
        physical(state, contact=2, pocketed=(9,), rail=False),
        NineBallContext(),
    )
    assert result.foul
    assert result.rack_winner is None
    assert result.nine_respotted
    assert result.next_state.ball_pocket_indices[9] == -1
    assert result.next_state.ball_in_hand
    assert result.next_state.current_player == 1
    assert "wrong_ball_first" in result.foul_reasons


def test_scratch_is_foul_and_next_player_has_ball_in_hand():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state, physical(state, scratch=True), NineBallContext(),
    )
    assert result.foul
    assert result.next_state.ball_in_hand
    assert result.turn_changed


def test_rail_before_first_contact_does_not_count():
    state = replace(create_nine_ball_rack(breaker=0, seed=2), break_shot=False)
    result = NineBallRules().apply(
        state,
        physical(state, rail=False, early_rail=True),
        NineBallContext(),
    )
    assert "no_rail_after_contact" in result.foul_reasons


def test_missing_first_object_contact_is_foul():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state, physical(state, contact=None, rail=False),
        NineBallContext(),
    )
    assert "no_object_contact" in result.foul_reasons


def test_break_without_pot_requires_four_distinct_rail_contacts():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = physical(state)
    result.events += tuple(
        event("RAIL_COLLISION", n, time=0.3)
        for n in (2, 3, 4)
    )
    legal = NineBallRules().apply(state, result, NineBallContext())
    assert not legal.foul
    assert legal.next_context.push_out_available
    result.events = result.events[:-1]
    illegal = NineBallRules().apply(state, result, NineBallContext())
    assert "insufficient_break_rails" in illegal.foul_reasons


def test_three_consecutive_fouls_awards_rack_to_opponent():
    state = create_nine_ball_rack(breaker=0, seed=2)
    result = NineBallRules().apply(
        state, physical(state, contact=2),
        NineBallContext(consecutive_fouls=(2, 1)),
    )
    assert result.rack_winner == 1
    assert result.foul
    assert result.next_context.consecutive_fouls == (3, 1)


def test_push_out_cannot_be_misclassified_as_normal_shot():
    state = create_nine_ball_rack(breaker=0, seed=2)
    with pytest.raises(NotImplementedError, match="Push-out"):
        NineBallRules().apply(
            state, physical(state), NineBallContext(),
            declared_push_out=True,
        )


class FakeSimulator:
    def __init__(self):
        self.state = None
        self.outputs = []

    def restore(self, state):
        self.state = state

    def step(self, *, target_ball, shot):
        return self.outputs.pop(0)


def test_session_executes_physics_and_finishes_rack():
    state = create_nine_ball_rack(breaker=1, seed=8)
    sim = FakeSimulator()
    sim.outputs = [physical(state, pocketed=(9,), rail=False)]
    session = NineBallRackSession(sim, state)
    result = session.step(target_ball=1, shot=object())
    assert result.rack_winner == 1
    assert session.winner == 1
    with pytest.raises(RuntimeError, match="Rack already ended"):
        session.step(target_ball=1, shot=object())


def test_session_foul_requires_cue_placement():
    state = create_nine_ball_rack(breaker=1, seed=8)
    sim = FakeSimulator()
    sim.outputs = [physical(state, contact=2)]
    session = NineBallRackSession(sim, state)
    result = session.step(target_ball=1, shot=object())
    assert result.foul
    assert session.state.ball_in_hand
    with pytest.raises(ValueError, match="placement required"):
        session.step(target_ball=1, shot=object())
    with pytest.raises(ValueError, match="overlaps"):
        session.place_cue_ball(*state.ball_positions[1])
    session.place_cue_ball(0.6, 0.35)
    assert not session.state.ball_in_hand
    assert session.state.current_player == 0


def test_match_session_records_racks_and_alternates_breaker():
    class WinningBreakSimulator(FakeSimulator):
        def step(self, *, target_ball, shot):
            return physical(self.state, pocketed=(9,), rail=False)

    match = NineBallMatchSession(
        simulator_factory=lambda config: WinningBreakSimulator(),
        match_config=MatchConfig(race_to=2, rule_id="nine_ball"),
        seed=200,
    )
    assert match.state.current_player == 0
    assert match.step(target_ball=1, shot=object()).rack_winner == 0
    assert match.state.current_player == 1
    assert match.step(target_ball=1, shot=object()).rack_winner == 1
    assert match.state.current_player == 0
    assert match.step(target_ball=1, shot=object()).rack_winner == 0
    assert match.match.snapshot().rack_wins == (2, 1)
    assert match.match.snapshot().finished
    assert len(match.rack_history) == 3
    with pytest.raises(RuntimeError, match="Match already ended"):
        match.step(target_ball=1, shot=object())
