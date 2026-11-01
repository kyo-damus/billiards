"""Minimal, independent FastFiz execution path for Defense shots.

This module deliberately does not modify the ATTACK Micro SAC,
MacroCandidate encoding, or existing reward function.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.common.types import GameState, ShotAction
from src.env.fastfiz_simulator import FastFizSimulator
from src.env.simulator_state import SimulatorSnapshot
from src.game.simple_rules import SimplifiedGameRules


@dataclass(frozen=True)
class DefenseShotResult:
    """Physical/game transition from one planned defensive shot."""

    shot: ShotAction
    targeted_ball: int
    physical_state: GameState
    next_state: GameState
    game_reward: float
    terminated: bool
    turn_changed: bool
    scratched: bool
    object_ball_moved: bool
    newly_pocketed_object_count: int

    # First cue/object contact from FastFiz (-1=no collision).
    # A hit on the intended target is only a v0 contact check,
    # not proof that all official billiard rules are satisfied.
    first_contact_ball: int = -1
    first_contact_verified: bool = False


def plan_defense_shot(
    state: GameState,
    target_ball: int,
    power: float,
    angle_offset_deg: float = 0.0,
    tip_x_mm: float = 0.0,
    tip_y_mm: float = 0.0,
) -> ShotAction:
    """
    Construct a physical shot aimed at an object-ball center.

    Unlike ATTACK, this does not use a ghost-ball pocket line.
    Angle offset is in DEGREES and the returned ShotAction uses
    an absolute table angle and physical power.

    This is a shot proposal, not a guarantee of ball contact.
    """
    if target_ball not in (0, 1):
        raise ValueError(
            "target_ball must be 0 or 1."
        )

    if state.is_pocketed(0) or state.is_pocketed(
        target_ball + 1
    ):
        raise ValueError(
            "Cue and targeted object ball must be on the table."
        )

    if not (
        math.isfinite(power)
        and 0.5 <= power <= 3.0
    ):
        raise ValueError(
            "power must be finite and in [0.5, 3.0]."
        )

    if not (
        math.isfinite(angle_offset_deg)
        and abs(angle_offset_deg) <= 35.0
    ):
        raise ValueError(
            "angle_offset_deg must be finite and in [-35, 35]."
        )

    tip = np.asarray(
        [tip_x_mm, tip_y_mm],
        dtype=np.float64,
    )

    if (
        not np.all(np.isfinite(tip))
        or float(np.dot(tip, tip)) > 144.0
    ):
        raise ValueError(
            "Tip offsets must be within 12 mm radius."
        )

    positions = np.asarray(
        state.ball_positions,
        dtype=np.float32,
    )

    if positions.shape != (3, 2):
        raise ValueError(
            "ball_positions must have shape (3, 2)."
        )

    cue = positions[0]
    target = positions[
        target_ball + 1
    ]
    direction = target - cue

    if float(
        np.linalg.norm(direction)
    ) < 1e-6:
        raise ValueError(
            "Cue and target positions must be distinct."
        )

    base_angle = math.degrees(
        math.atan2(
            float(direction[1]),
            float(direction[0]),
        )
    )

    return ShotAction(
        angle=float(
            base_angle + angle_offset_deg
        ),
        power=float(power),
        tip_x_mm=float(tip_x_mm),
        tip_y_mm=float(tip_y_mm),
    )


class DefenseShotExecutor:
    """
    Execute one explicit physical shot in FastFiz, then apply
    the same simplified game rules as ATTACK.

    Reports the first cue/object collision, but does not
    implement official foul rules, ATTACK reward shaping,
    or learned Micro behavior.
    """

    def __init__(
        self,
        simulator=None,
        game_rules=None,
        movement_tolerance=1e-3,
    ):
        self.sim = (
            simulator
            if simulator is not None
            else FastFizSimulator()
        )
        self.game_rules = (
            game_rules
            if game_rules is not None
            else SimplifiedGameRules()
        )
        self.movement_tolerance = float(
            movement_tolerance
        )

        if self.movement_tolerance < 0.0:
            raise ValueError(
                "movement_tolerance must be >= 0."
            )

    def execute(
        self,
        state: GameState,
        target_ball: int,
        shot: ShotAction,
    ) -> DefenseShotResult:
        if target_ball not in (0, 1):
            raise ValueError(
                "target_ball must be 0 or 1."
            )

        if (
            state.is_pocketed(0)
            or state.is_pocketed(
                target_ball + 1
            )
        ):
            raise ValueError(
                "Cannot shoot with pocketed cue "
                "or targeted ball."
            )

        if (
            not math.isfinite(shot.power)
            or not 0.5 <= shot.power <= 3.0
            or not math.isfinite(shot.angle)
        ):
            raise ValueError(
                "Invalid physical shot power/angle."
            )

        if (
            not np.isclose(shot.spin_x, 0.0)
            or not np.isclose(shot.spin_y, 0.0)
        ):
            raise ValueError(
                "Defense v0 does not support spin."
            )

        tip = np.asarray(
            [shot.tip_x_mm, shot.tip_y_mm],
            dtype=np.float64,
        )

        if (
            not np.all(np.isfinite(tip))
            or float(np.dot(tip, tip)) > 144.0
        ):
            raise ValueError(
                "Tip offsets must be within 12 mm radius."
            )

        positions = np.asarray(
            state.ball_positions,
            dtype=np.float32,
        )

        if positions.shape != (3, 2):
            raise ValueError(
                "ball_positions must have shape (3, 2)."
            )

        self.sim.restore(
            SimulatorSnapshot(
                positions=positions.copy(),
                pocket_indices=np.asarray(
                    state.ball_pocket_indices,
                    dtype=np.int32,
                ).copy(),
            )
        )

        step = self.sim.step(
            target_ball=int(target_ball),
            power=float(shot.power),
            angle=float(shot.angle),
            tip_x_mm=float(shot.tip_x_mm),
            tip_y_mm=float(shot.tip_y_mm),
        )

        # The current C++ wrapper returns -1 for an invalid
        # physical shot. Reject it instead of mistaking it
        # for a legitimate miss/turn change.
        if step.legacy_reward < 0.0:
            raise RuntimeError(
                "FastFiz rejected this physical shot."
            )

        snapshot = self.sim.snapshot()

        physical_state = GameState(
            ball_positions=snapshot.positions.copy(),
            score=state.score.copy(),
            current_player=state.current_player,
            ball_pocket_indices=(
                snapshot.pocket_indices.copy()
            ),
        )

        scratched = bool(
            physical_state.is_pocketed(0)
        )

        result = self.game_rules.apply(
            previous_state=state,
            physical_state=physical_state,
            shot_info={
                "scratched": scratched,
            },
        )

        was_on_table = (
            np.asarray(
                state.ball_pocket_indices[1:],
            ) < 0
        )
        now_pocketed = (
            np.asarray(
                snapshot.pocket_indices[1:],
            ) >= 0
        )

        newly_pocketed_count = int(
            np.count_nonzero(
                was_on_table & now_pocketed
            )
        )

        movements = np.linalg.norm(
            snapshot.positions[1:]
            - positions[1:],
            axis=1,
        )

        object_ball_moved = bool(
            np.any(
                movements[
                    was_on_table
                ] > self.movement_tolerance
            )
        )

        return DefenseShotResult(
            shot=shot,
            targeted_ball=target_ball,
            physical_state=physical_state,
            next_state=result.next_state,
            game_reward=float(result.reward),
            terminated=bool(result.terminated),
            turn_changed=bool(result.turn_changed),
            scratched=scratched,
            object_ball_moved=object_ball_moved,
            newly_pocketed_object_count=(
                newly_pocketed_count
            ),
            first_contact_ball=(
                step.first_contact_ball
            ),
            first_contact_verified=(
                step.first_contact_ball == target_ball
            ),
        )
