"""Rule-neutral multiple-rack race-to-N match bookkeeping.

The rules engine reports who won each rack. This module does NOT decide
fouls, the winner of a shot, or official nine-ball legality.
"""

from dataclasses import dataclass
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


class RaceToMatch:
    """Track a race-to-N with an explicit break order and no tie racks."""

    def __init__(self, config: MatchConfig):
        self.config = config
        self._wins = [0, 0]
        self._racks_completed = 0
        self._breaker = config.first_breaker
        self._winner = None

    def snapshot(self) -> MatchSnapshot:
        return MatchSnapshot(
            rack_wins=tuple(self._wins),
            racks_completed=self._racks_completed,
            next_breaker=None if self._winner is not None else self._breaker,
            match_winner=self._winner,
        )

    def finish_rack(self, winner: int) -> MatchSnapshot:
        """Submit one completed rack; no winner is invented from points."""
        if self._winner is not None:
            raise RuntimeError("Match is already finished.")
        if winner not in (0, 1):
            raise ValueError("Rack winner must be player 0 or 1.")

        self._wins[winner] += 1
        self._racks_completed += 1
        if self._wins[winner] >= self.config.race_to:
            self._winner = winner
        elif self.config.break_policy == "alternate":
            self._breaker = (
                self.config.first_breaker ^ (self._racks_completed % 2)
            )
        elif self.config.break_policy == "winner":
            self._breaker = winner
        else:
            self._breaker = 1 - winner
        return self.snapshot()
