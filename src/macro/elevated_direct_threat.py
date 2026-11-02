"""Physical opponent DIRECT-pot probe with explicit cue elevation.

This is a coarse, deterministic FastFiz grid diagnostic, NOT the
learned 10-degree SAC policy, a full game rule validator, or a
substitute for the trained 31D mixed Candidate Q.

A successful *route* must physically pot the specified target ball
in the specified pocket, avoid cue scratch, and contact the intended
ball first. Results are within the enumerated parameter grid only.
"""

from dataclasses import dataclass
from math import atan2, degrees, isfinite

import numpy as np

from src.common.table import POCKET_POSITIONS
from src.env.fastfiz_simulator import FastFizSimulator
from src.env.shot_geometry import ghost_ball_position
from src.env.simulator_state import SimulatorSnapshot
from src.goal.types import TacticalGoalType


@dataclass(frozen=True)
class ElevatedDirectThreatResult:
    candidate_count: int
    attempted: int
    physically_valid: int
    successful_routes: int
    successful_routes_by_elevation: tuple[int, ...]


class ElevatedDirectThreatProbe:
    """A reusable, explicitly bounded opponent physical-shot sweep."""

    def __init__(
        self,
        direct_generator,
        elevations=(10.0, 20.0, 35.0),
        angle_offsets=(-5.0, 0.0, 5.0),
        powers=(0.65, 1.0, 1.5),
        tip_y_mm=(0.0, 5.0),
        simulator=None,
    ):
        self.direct_generator = direct_generator
        self.elevations = tuple(float(v) for v in elevations)
        self.angle_offsets = tuple(float(v) for v in angle_offsets)
        self.powers = tuple(float(v) for v in powers)
        self.tip_y_mm = tuple(float(v) for v in tip_y_mm)

        if (
            not self.elevations
            or not all(isfinite(v) and 0 <= v <= 70 for v in self.elevations)
            or not self.angle_offsets
            or not all(isfinite(v) and abs(v) <= 35 for v in self.angle_offsets)
            or not self.powers
            or not all(isfinite(v) and 0.5 <= v <= 3.0 for v in self.powers)
            or not self.tip_y_mm
            or not all(isfinite(v) and abs(v) <= 12 for v in self.tip_y_mm)
        ):
            raise ValueError("Invalid elevation/direct shot probe grid.")

        self.sim = simulator if simulator is not None else FastFizSimulator()

    def evaluate(self, state) -> ElevatedDirectThreatResult:
        if state.is_pocketed(0):
            raise ValueError("Opponent cue ball must be on table.")

        positions = np.asarray(state.ball_positions, dtype=np.float32)
        pockets = np.asarray(state.ball_pocket_indices, dtype=np.int32)
        if positions.shape != (3, 2) or pockets.shape != (3,):
            raise ValueError("Unexpected GameState geometry.")

        direct_candidates = list(self.direct_generator.generate(state))
        routes = []
        seen_routes = set()
        for c in direct_candidates:
            if c.goal.goal_type != TacticalGoalType.DIRECT_ATTACK:
                raise ValueError("Only DIRECT_ATTACK candidates are supported.")
            ball = c.action.target_ball
            pocket = c.action.target_pocket
            if ball not in (0, 1) or pocket not in POCKET_POSITIONS:
                raise ValueError("Invalid DIRECT target/pocket.")
            if pockets[ball + 1] >= 0:
                raise ValueError("DIRECT target is already pocketed.")
            route = (ball, pocket)
            if route in seen_routes:
                continue
            seen_routes.add(route)
            ghost = ghost_ball_position(
                positions[ball + 1],
                POCKET_POSITIONS[pocket],
            )
            shot_dir = ghost - positions[0]
            if np.linalg.norm(shot_dir) < 1e-8:
                continue
            angle = degrees(atan2(float(shot_dir[1]), float(shot_dir[0])))
            routes.append((route, angle))

        initial = SimulatorSnapshot(
            positions=positions.copy(),
            pocket_indices=pockets.copy(),
        )
        attempts = 0
        valid = 0
        success_by_elevation = []
        successful_all = set()

        for elevation in self.elevations:
            successful_this = set()
            for ((ball, pocket), ideal_angle) in routes:
                for offset in self.angle_offsets:
                    for power in self.powers:
                        for tip in self.tip_y_mm:
                            self.sim.restore(initial)
                            result = self.sim.step(
                                target_ball=ball,
                                power=power,
                                angle=ideal_angle + offset,
                                tip_x_mm=0.0,
                                tip_y_mm=tip,
                                cue_elevation_deg=elevation,
                            )
                            attempts += 1
                            if result.precondition_code != 0:
                                continue
                            valid += 1
                            if result.first_contact_ball != ball:
                                continue
                            snapshot = self.sim.snapshot()
                            if (
                                snapshot.pocket_indices[0] < 0
                                and snapshot.pocket_indices[ball + 1] == pocket
                            ):
                                successful_this.add((ball, pocket))
            successful_all.update(successful_this)
            success_by_elevation.append(len(successful_this))

        return ElevatedDirectThreatResult(
            candidate_count=len(routes),
            attempted=attempts,
            physically_valid=valid,
            successful_routes=len(successful_all),
            successful_routes_by_elevation=tuple(success_by_elevation),
        )
