"""Race-to match manager; rack winner is supplied by a separate rules engine."""

import pytest

from src.game.match import MatchConfig, RaceToMatch


def test_alternate_break_and_race_to_three():
    m = RaceToMatch(MatchConfig(race_to=3, rule_id="nine_ball"))
    assert m.snapshot().next_breaker == 0
    s1 = m.finish_rack(1)
    assert s1.rack_wins == (0, 1)
    assert s1.next_breaker == 1
    s2 = m.finish_rack(0)
    assert s2.rack_wins == (1, 1)
    assert s2.next_breaker == 0
    m.finish_rack(1)
    end = m.finish_rack(1)
    assert end.rack_wins == (1, 3)
    assert end.racks_completed == 4
    assert end.finished
    assert end.match_winner == 1
    assert end.next_breaker is None
    with pytest.raises(RuntimeError, match="already finished"):
        m.finish_rack(0)


@pytest.mark.parametrize(
    "policy,expected_breaker",
    [("winner", 1), ("loser", 0), ("alternate", 1)],
)
def test_break_policy_is_independent_of_rack_winner(policy, expected_breaker):
    m = RaceToMatch(
        MatchConfig(race_to=2, rule_id="practice", break_policy=policy)
    )
    assert m.finish_rack(1).next_breaker == expected_breaker


@pytest.mark.parametrize(
    "kwargs", [
        {"race_to": 0, "rule_id": "x"},
        {"race_to": 2, "rule_id": ""},
        {"race_to": 2, "rule_id": "x", "first_breaker": 2},
        {"race_to": 2, "rule_id": "x", "break_policy": "invalid"},
    ],
)
def test_invalid_match_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        MatchConfig(**kwargs)


def test_bad_winner_is_rejected_without_changing_match():
    m = RaceToMatch(MatchConfig(race_to=2, rule_id="x"))
    with pytest.raises(ValueError):
        m.finish_rack(5)
    assert m.snapshot().racks_completed == 0
