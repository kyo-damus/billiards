"""Physical elevated DIRECT probe remains isolated from SAC/MCTS."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.common.types import GameState, MacroAction, Strategy
from src.env.simulator_state import SimulatorSnapshot
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate
from src.macro.elevated_direct_threat import ElevatedDirectThreatProbe


def state():
    return GameState(
        ball_positions=np.array(
            [[0.4, 0.5], [0.5, 1.0], [0.8, 1.7]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=1,
        ball_pocket_indices=np.array([-1, -1, -1], dtype=np.int32),
    )


def direct():
    return MacroCandidate(
        action=MacroAction(
            strategy=Strategy.ATTACK,
            target_ball=0,
            target_pocket=3,
        ),
        goal=TacticalGoal(
            goal_type=TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=3,
        ),
    )


class FixedGenerator:
    def __init__(self, candidates):
        self.candidates = candidates

    def generate(self, current):
        return list(self.candidates)


class FakeSim:
    def __init__(self):
        self.calls = 0

    def restore(self, snapshot):
        self.original = snapshot
        self.current = snapshot

    def step(self, target_ball, power, angle, tip_x_mm, tip_y_mm,
             cue_elevation_deg):
        self.calls += 1
        pocket_indices = self.original.pocket_indices.copy()
        if cue_elevation_deg == 35.0:
            pocket_indices[target_ball + 1] = 3
        self.current = SimulatorSnapshot(
            positions=self.original.positions.copy(),
            pocket_indices=pocket_indices,
        )
        return SimpleNamespace(
            precondition_code=0,
            first_contact_ball=target_ball,
        )

    def snapshot(self):
        return self.current


def test_elevation_probe_counts_target_pocket_routes():
    sim = FakeSim()
    probe = ElevatedDirectThreatProbe(
        direct_generator=FixedGenerator([direct()]),
        elevations=[10.0, 35.0],
        angle_offsets=[0.0],
        powers=[1.0],
        tip_y_mm=[0.0],
        simulator=sim,
    )
    result = probe.evaluate(state())
    assert result.candidate_count == 1
    assert result.attempted == 2
    assert result.physically_valid == 2
    assert result.successful_routes == 1
    assert result.successful_routes_by_elevation == (0, 1)
    assert sim.calls == 2


def test_elevation_probe_deduplicates_identical_direct_routes():
    probe = ElevatedDirectThreatProbe(
        direct_generator=FixedGenerator([direct(), direct()]),
        elevations=[35],
        angle_offsets=[0],
        powers=[1.0],
        tip_y_mm=[0],
        simulator=FakeSim(),
    )
    result = probe.evaluate(state())
    assert result.candidate_count == 1
    assert result.attempted == 1
    assert result.successful_routes == 1


def test_elevation_probe_rejects_invalid_grids():
    with pytest.raises(ValueError, match="grid"):
        ElevatedDirectThreatProbe(
            direct_generator=FixedGenerator([]),
            powers=[],
            simulator=FakeSim(),
        )
    with pytest.raises(ValueError, match="grid"):
        ElevatedDirectThreatProbe(
            direct_generator=FixedGenerator([]),
            elevations=[75],
            simulator=FakeSim(),
        )


def test_elevation_probe_does_not_count_wrong_first_contact():
    class WrongFirst(FakeSim):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            return SimpleNamespace(
                precondition_code=0,
                first_contact_ball=1,
            )

    result = ElevatedDirectThreatProbe(
        direct_generator=FixedGenerator([direct()]),
        elevations=[35.0],
        angle_offsets=[0.0],
        powers=[1.0],
        tip_y_mm=[0.0],
        simulator=WrongFirst(),
    ).evaluate(state())
    assert result.physically_valid == 1
    assert result.successful_routes == 0
