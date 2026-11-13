"""Rule-neutral multiple-rack race-to-N match bookkeeping.

The rules engine reports who won each rack. This module does NOT decide
fouls, the winner of a shot, or official nine-ball legality.
"""

from dataclasses import dataclass
from math import comb
from typing import Literal, Optional


BreakPolicy = Literal["alternate", "winner", "loser"]


@dataclass(frozen=True)
class MatchConfig:
    race_to: int
    rule_id: str
    first_breaker: int = 0
    break_policy: BreakPolicy = "alternate"

    def __post_init__(self):
        if self.race_to < 1:
            raise ValueError("race_to must be >= 1.")
        if not self.rule_id or not self.rule_id.strip():
            raise ValueError("rule_id must be nonempty.")
        if self.first_breaker not in (0, 1):
            raise ValueError("first_breaker must be 0 or 1.")
        if self.break_policy not in ("alternate", "winner", "loser"):
            raise ValueError("break_policy must be alternate, winner, or loser.")


@dataclass(frozen=True)
class MatchSnapshot:
    rack_wins: tuple[int, int]
    racks_completed: int
    next_breaker: Optional[int]
    match_winner: Optional[int]

    @property
    def finished(self) -> bool:
        return self.match_winner is not None


@dataclass(frozen=True)
class MatchPosition:
    """Public pre-shot match state, independent of rack layouts and seeds."""

    race_to: int
    rack_wins: tuple[int, int]
    breaker: int
    break_policy: BreakPolicy = "alternate"

    @property
    def winner(self):
        return next((p for p, wins in enumerate(self.rack_wins) if wins >= self.race_to), None)

    def after_rack(self, winner: int):
        if self.winner is not None:
            raise RuntimeError("Match is already finished.")
        if winner not in (0, 1):
            raise ValueError("Rack winner must be player 0 or 1.")
        wins = list(self.rack_wins)
        wins[winner] += 1
        breaker = (
            1 - self.breaker if self.break_policy == "alternate"
            else winner if self.break_policy == "winner" else 1 - winner
        )
        return MatchPosition(self.race_to, tuple(wins), breaker, self.break_policy)

    def fair_match_value(self, player: int) -> float:
        """Signed match win probability when all remaining racks are fair."""
        winner = self.winner
        if winner is not None:
            return 1.0 if winner == player else -1.0
        own = self.race_to - self.rack_wins[player]
        other = self.race_to - self.rack_wins[1 - player]
        trials = own + other - 1
        probability = sum(comb(trials, i) for i in range(own, trials + 1)) / (2 ** trials)
        return 2.0 * probability - 1.0


class RaceToMatch:
    """Track a race-to-N with an explicit break order and no tie racks."""

    def __init__(self, config: MatchConfig):
        self.config = config
        self.position = MatchPosition(config.race_to, (0, 0), config.first_breaker, config.break_policy)

    def snapshot(self) -> MatchSnapshot:
        return MatchSnapshot(
            rack_wins=self.position.rack_wins,
            racks_completed=sum(self.position.rack_wins),
            next_breaker=None if self.position.winner is not None else self.position.breaker,
            match_winner=self.position.winner,
        )

    def finish_rack(self, winner: int) -> MatchSnapshot:
        """Submit one completed rack; no winner is invented from points."""
        self.position = self.position.after_rack(winner)
        return self.snapshot()
