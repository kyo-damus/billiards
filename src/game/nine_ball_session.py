"""One-rack and race-to-N state machines driven by a physics backend.

The simulator can be FastFiz or a deterministic test double. Shot decisions
come from the caller. This does not implement all WPA referee conditions.
"""

from dataclasses import replace
from math import sqrt

import numpy as np

from src.game.nine_ball_rules import (
    BALL_RADIUS_M, FOOT_SPOT, TABLE_LENGTH_M, TABLE_WIDTH_M,
    NineBallContext, NineBallRules,
)
from src.game.rack_state import RackState, TableGameConfig


def create_nine_ball_rack(*, breaker: int, seed: int) -> RackState:
    """Diamond rack with 1 at apex and 9 at foot spot, 2..8 shuffled."""
    config = TableGameConfig.nine_ball_inventory()
    if breaker not in (0, 1):
        raise ValueError("breaker must be 0 or 1")
    rng = np.random.default_rng(seed)
    other_numbers = list(rng.permutation(np.arange(2, 9)))
    diameter = 2 * BALL_RADIUS_M + 0.00015
    row_distance = diameter * sqrt(3) / 2
    x0, y0 = FOOT_SPOT
    positions = np.zeros((10, 2), dtype=np.float32)
    positions[0] = [x0, TABLE_LENGTH_M * 0.25 - 0.10]
    row_sizes = (1, 2, 3, 2, 1)
    for row, count in enumerate(row_sizes):
        y = y0 + (row - 2) * row_distance
        for col in range(count):
            x = x0 + (col - (count - 1) * 0.5) * diameter
            number = (
                1 if row == 0 else
                9 if row == 2 and col == 1 else
                other_numbers.pop(0)
            )
            positions[number] = [x, y]
    assert not other_numbers
    return RackState(
        config=config,
        ball_positions=positions,
        ball_pocket_indices=np.full(10, -1, dtype=np.int32),
        current_player=breaker,
        score=np.zeros(2, dtype=np.float32),
        break_shot=True,
    )


class NineBallRackSession:
    """Physics + rules for one rack; the caller supplies the shot."""

    def __init__(self, simulator, initial_state: RackState, *, rules=None):
        if initial_state.config.rule_id != "nine_ball":
            raise ValueError("NineBallRackSession requires a nine_ball RackState")
        self.simulator = simulator
        self.rules = rules or NineBallRules()
        self.state = initial_state
        self.context = NineBallContext()
        self.winner = None
        self.simulator.restore(initial_state)

    def place_cue_ball(self, x: float, y: float) -> RackState:
        """After a foul, place the cue ball with physical placement validation."""
        if self.winner is not None:
            raise RuntimeError("Rack already ended")
        if not self.state.ball_in_hand:
            raise ValueError("Cue placement is only allowed with ball in hand")
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Cue placement must be finite")
        if not (BALL_RADIUS_M <= x <= TABLE_WIDTH_M - BALL_RADIUS_M and
                BALL_RADIUS_M <= y <= TABLE_LENGTH_M - BALL_RADIUS_M):
            raise ValueError("Cue placement outside playing surface")
        positions = self.state.ball_positions.copy()
        pockets = self.state.ball_pocket_indices.copy()
        positions[0] = [x, y]
        pockets[0] = -1
        candidate = replace(
            self.state,
            ball_positions=positions,
            ball_pocket_indices=pockets,
            ball_in_hand=False,
        )
        self.simulator.restore(candidate)
        if hasattr(self.simulator, "native"):
            valid = self.simulator.native.get_ball_placement_code() == 0
        else:
            valid = all(
                pockets[i] >= 0 or np.linalg.norm(positions[0] - positions[i])
                >= 2 * BALL_RADIUS_M + 1e-6
                for i in range(1, len(pockets))
            )
        if not valid:
            self.simulator.restore(self.state)
            raise ValueError("Cue placement overlaps a ball or violates table constraints")
        self.state = candidate
        return candidate

    def step(self, *, target_ball: int, shot, declared_push_out: bool = False):
        if self.winner is not None:
            raise RuntimeError("Rack already ended")
        if self.state.ball_in_hand:
            raise ValueError("Cue-ball placement required before shooting")
        if declared_push_out:
            raise NotImplementedError("Push-out requires option/choice support")
        physical = self.simulator.step(target_ball=target_ball, shot=shot)
        result = self.rules.apply(self.state, physical, self.context)
        self.state = result.next_state
        self.context = result.next_context
        self.winner = result.rack_winner
        # A respotted 9-ball is a rule result, not a physical event.
        if self.winner is None:
            self.simulator.restore(self.state)
        return result


class NineBallMatchSession:
    """Several independent racks played until race-to-N winner exists."""

    def __init__(self, *, simulator_factory, match_config, seed: int):
        from src.game.match import RaceToMatch
        if match_config.rule_id != "nine_ball":
            raise ValueError("Match must use nine_ball rule_id")
        self.simulator_factory = simulator_factory
        self.match = RaceToMatch(match_config)
        self.seed = seed
        self.rack_history = []
        self.rack = None
        self._start_next_rack()

    def _start_next_rack(self):
        snapshot = self.match.snapshot()
        if snapshot.finished:
            raise RuntimeError("Match already ended")
        new_rack = create_nine_ball_rack(
            breaker=snapshot.next_breaker,
            seed=self.seed + snapshot.racks_completed,
        )
        self.rack = NineBallRackSession(
            simulator=self.simulator_factory(new_rack.config),
            initial_state=new_rack,
        )

    @property
    def state(self) -> RackState:
        return self.rack.state

    def place_cue_ball(self, x: float, y: float):
        return self.rack.place_cue_ball(x, y)

    def step(self, *, target_ball: int, shot, declared_push_out: bool = False):
        if self.match.snapshot().finished:
            raise RuntimeError("Match already ended")
        result = self.rack.step(
            target_ball=target_ball, shot=shot,
            declared_push_out=declared_push_out,
        )
        if result.rack_winner is not None:
            self.rack_history.append({
                "rack": self.match.snapshot().racks_completed,
                "winner": result.rack_winner,
                "turns": result.next_state.turn_index,
            })
            snapshot = self.match.finish_rack(result.rack_winner)
            if not snapshot.finished:
                self._start_next_rack()
        return result
