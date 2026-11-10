"""Ball-count-independent rack data contract for future competitive environments.

No physics or rules are implemented here. Keep the existing three-ball
GameState/12-D Macro encoders unchanged until versioned replacements exist.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TableGameConfig:
    """Rule identifier and numbered object-ball inventory for one rack."""

    rule_id: str
    object_ball_numbers: tuple[int, ...]

    def __post_init__(self):
        numbers = tuple(int(number) for number in self.object_ball_numbers)
        if not self.rule_id or not self.rule_id.strip():
            raise ValueError("rule_id must be nonempty.")
        if not numbers or any(number <= 0 for number in numbers):
            raise ValueError("Object ball numbers must be positive and nonempty.")
        if len(set(numbers)) != len(numbers):
            raise ValueError("Object ball numbers must be unique.")
        object.__setattr__(self, "object_ball_numbers", numbers)

    @property
    def ball_numbers(self) -> tuple[int, ...]:
        return (0, *self.object_ball_numbers)

    @classmethod
    def two_object_training(cls):
        return cls("simplified_two_object", (1, 2))

    @classmethod
    def nine_ball_inventory(cls):
        """Numbered inventory only; NOT a nine-ball rules implementation."""
        return cls("nine_ball", tuple(range(1, 10)))


@dataclass(frozen=True)
class RackState:
    """Full rack state with explicit ball identity and rule-independent flags.

    Row zero is the cue ball (number 0). Further rows follow
    config.object_ball_numbers. Pocket indices use -1 for on-table and 0..5
    for pocketed balls. A scratched cue ball is permitted as a physical
    transition result; applying placement/foul rules is the rule engine's job.
    """

    config: TableGameConfig
    ball_positions: np.ndarray
    ball_pocket_indices: np.ndarray
    current_player: int
    score: np.ndarray
    ball_in_hand: bool = False
    break_shot: bool = False
    turn_index: int = 0

    def __post_init__(self):
        positions = np.asarray(self.ball_positions, dtype=np.float32).copy()
        pockets = np.asarray(self.ball_pocket_indices, dtype=np.int32).copy()
        score = np.asarray(self.score, dtype=np.float32).copy()
        n_balls = len(self.config.ball_numbers)
        if positions.shape != (n_balls, 2):
            raise ValueError(
                f"ball_positions must have shape ({n_balls}, 2), got {positions.shape}"
            )
        if pockets.shape != (n_balls,):
            raise ValueError(
                f"ball_pocket_indices must have shape ({n_balls},), got {pockets.shape}"
            )
        if score.shape != (2,):
            raise ValueError("score must have shape (2,).")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(score)):
            raise ValueError("Positions and score must be finite.")
        if np.any(pockets < -1) or np.any(pockets > 5):
            raise ValueError("Pocket indices must be -1 or in 0..5.")
        if self.current_player not in (0, 1):
            raise ValueError("current_player must be 0 or 1.")
        if self.turn_index < 0:
            raise ValueError("turn_index must be >= 0.")
        positions.setflags(write=False)
        pockets.setflags(write=False)
        score.setflags(write=False)
        object.__setattr__(self, "ball_positions", positions)
        object.__setattr__(self, "ball_pocket_indices", pockets)
        object.__setattr__(self, "score", score)

    @property
    def ball_count(self) -> int:
        return len(self.config.ball_numbers)

    def array_index_for_ball(self, ball_number: int) -> int:
        try:
            return self.config.ball_numbers.index(int(ball_number))
        except ValueError as exc:
            raise ValueError(f"Unknown ball number: {ball_number}") from exc

    def on_table_object_balls(self) -> tuple[int, ...]:
        return tuple(
            number for index, number in enumerate(self.config.ball_numbers)
            if index > 0 and self.ball_pocket_indices[index] == -1
        )

    def is_pocketed(self, ball_number: int) -> bool:
        return bool(
            self.ball_pocket_indices[self.array_index_for_ball(ball_number)] >= 0
        )

    @classmethod
    def from_legacy(cls, game_state, *, config: Optional[TableGameConfig] = None):
        """Explicit adapter for old (3, 2) GameState. No new inference support."""
        if config is None:
            config = TableGameConfig.two_object_training()
        if config.ball_numbers != (0, 1, 2):
            raise ValueError("Legacy GameState can only be adapted as 3 balls.")
        return cls(
            config=config,
            ball_positions=game_state.ball_positions,
            ball_pocket_indices=game_state.ball_pocket_indices,
            current_player=game_state.current_player,
            score=game_state.score,
        )
