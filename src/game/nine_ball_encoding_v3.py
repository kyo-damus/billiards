"""Version-3 multi-ball state and Macro/Micro goal representation.

Fixed 16-ball slots refer to actual ball numbers 0..15; they do NOT
relabel object ball 9 as an array index. This does not load the old
12-D/31-D Macro or 2-D Micro networks.
"""

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from typing import Optional

import numpy as np

from src.common.table import POCKET_POSITIONS, TABLE_LENGTH, TABLE_WIDTH
from src.env.shot_geometry import (
    BALL_DIAMETER, ghost_ball_position, point_to_segment_distance,
)
from src.micro.shot_control import PhysicalShot5D

ENCODING_VERSION = 3
MAX_BALL_NUMBER = 15
MACRO_STATE_V3_DIM = 72
MACRO_GOAL_V3_DIM = 36
MICRO_CONDITIONED_V3_DIM = MACRO_STATE_V3_DIM + MACRO_GOAL_V3_DIM
GOAL_KINDS = ("BREAK", "DIRECT_ATTACK", "POSITION_ATTACK", "SAFETY")


@dataclass(frozen=True)
class NineBallGoal:
    kind: str
    target_ball: int
    target_pocket: Optional[int] = None
    cue_stop_xy: Optional[tuple[float, float]] = None
    cue_stop_radius: Optional[float] = None
    explicit_shot: Optional[PhysicalShot5D] = None

    def __post_init__(self):
        if self.kind not in GOAL_KINDS:
            raise ValueError(f"Unknown goal kind: {self.kind}")
        if not (1 <= self.target_ball <= MAX_BALL_NUMBER):
            raise ValueError("target_ball must be a physical ball ID 1..15")
        if self.target_pocket is not None and self.target_pocket not in range(6):
            raise ValueError("target_pocket must be 0..5")
        if (self.cue_stop_xy is None) != (self.cue_stop_radius is None):
            raise ValueError("cue stop position and radius must be supplied together")
        if self.cue_stop_xy is not None:
            if (
                len(self.cue_stop_xy) != 2
                or not np.all(np.isfinite(self.cue_stop_xy))
                or not np.isfinite(self.cue_stop_radius)
                or self.cue_stop_radius <= 0
            ):
                raise ValueError("Invalid cue stop region")


def encode_rack_v3(state, context=None) -> np.ndarray:
    """16 x (present, on-table, x, y) + 8 rule/turn features."""
    out = np.zeros(MACRO_STATE_V3_DIM, dtype=np.float32)
    for index, ball_number in enumerate(state.config.ball_numbers):
        if not 0 <= ball_number <= MAX_BALL_NUMBER:
            raise ValueError("v3 encoder supports numbered balls 0..15")
        k = 4 * ball_number
        active = state.ball_pocket_indices[index] == -1
        out[k] = 1.0
        out[k + 1] = float(active)
        if active:
            out[k + 2] = state.ball_positions[index, 0] / TABLE_WIDTH
            out[k + 3] = state.ball_positions[index, 1] / TABLE_LENGTH

    out[64 + state.current_player] = 1.0
    out[66] = float(state.ball_in_hand)
    out[67] = float(state.break_shot)
    out[68] = len(state.on_table_object_balls()) / 15.0
    if context is not None:
        out[69] = min(context.consecutive_fouls[0], 3) / 3.0
        out[70] = min(context.consecutive_fouls[1], 3) / 3.0
        out[71] = float(context.push_out_available)
    return out


def encode_goal_v3(goal: NineBallGoal) -> np.ndarray:
    """4 goal kind + 16 real-ball IDs + 6 pockets + 3 stop + 1 shot flag + 6 shot."""
    v = np.zeros(MACRO_GOAL_V3_DIM, dtype=np.float32)
    v[GOAL_KINDS.index(goal.kind)] = 1.0
    v[4 + goal.target_ball] = 1.0
    if goal.target_pocket is not None:
        v[20 + goal.target_pocket] = 1.0
    if goal.cue_stop_xy is not None:
        v[26:29] = (
            goal.cue_stop_xy[0] / TABLE_WIDTH,
            goal.cue_stop_xy[1] / TABLE_LENGTH,
            goal.cue_stop_radius / np.hypot(TABLE_WIDTH, TABLE_LENGTH),
        )
    if goal.explicit_shot is not None:
        shot = goal.explicit_shot
        v[29] = 1.0
        v[30:36] = (
            shot.speed_m_s / 3.0,
            sin(radians(shot.azimuth_deg)),
            cos(radians(shot.azimuth_deg)),
            shot.elevation_deg / 70.0,
            shot.tip_x_mm / 12.0,
            shot.tip_y_mm / 12.0,
        )
    return v


def conditioned_observation_v3(state, goal, context=None) -> np.ndarray:
    return np.concatenate(
        (encode_rack_v3(state, context), encode_goal_v3(goal))
    ).astype(np.float32)


def reference_angle_deg(state, goal: NineBallGoal) -> float:
    cue = state.ball_positions[state.array_index_for_ball(0)]
    target = state.ball_positions[state.array_index_for_ball(goal.target_ball)]
    destination = target
    if goal.target_pocket is not None:
        destination = ghost_ball_position(
            np.asarray(target, dtype=np.float64),
            np.asarray(POCKET_POSITIONS[goal.target_pocket], dtype=np.float64),
        )
    diff = destination - cue
    if np.linalg.norm(diff) < 1e-7:
        raise ValueError("Cannot aim from cue directly at the reference point")
    return degrees(atan2(float(diff[1]), float(diff[0])))


def direct_pocket_candidates(state, target):
    """Yield pockets with clear cue-to-ghost and object-to-pocket paths."""
    cue = np.asarray(state.ball_positions[0], dtype=np.float64)
    ball = np.asarray(
        state.ball_positions[state.array_index_for_ball(target)], dtype=np.float64
    )
    other = [
        np.asarray(state.ball_positions[i], dtype=np.float64)
        for i, number in enumerate(state.config.ball_numbers)
        if number not in (0, target) and state.ball_pocket_indices[i] < 0
    ]
    for pocket_id, pocket in POCKET_POSITIONS.items():
        pocket = np.asarray(pocket, dtype=np.float64)
        ghost = ghost_ball_position(ball, pocket)
        if not (
            BALL_DIAMETER / 2 < ghost[0] < TABLE_WIDTH - BALL_DIAMETER / 2
            and BALL_DIAMETER / 2 < ghost[1] < TABLE_LENGTH - BALL_DIAMETER / 2
        ):
            continue
        if any(
            point_to_segment_distance(obj, cue, ghost) < BALL_DIAMETER + 0.004
            or point_to_segment_distance(obj, ball, pocket) < BALL_DIAMETER + 0.004
            for obj in other
        ):
            continue
        yield pocket_id


def generate_nine_ball_goals(
    state, *, learned_safety=False, learned_break=False, position_goal=None,
) -> tuple[NineBallGoal, ...]:
    """Unweighted candidate set; no equal-frequency strategy balancing.

    Learned BREAK and SAFETY goals condition the same 5-D Micro as attacks.
    A position_goal callback may replace or omit each legacy stop region.
    """
    if state.ball_in_hand:
        raise ValueError("Cue placement must precede shot candidate generation")
    on_table = state.on_table_object_balls()
    if not on_table:
        raise ValueError("No object balls left")
    target = min(on_table)
    cue = np.asarray(state.ball_positions[0], dtype=np.float64)
    ball = np.asarray(
        state.ball_positions[state.array_index_for_ball(target)], dtype=np.float64
    )
    heading = degrees(atan2(ball[1] - cue[1], ball[0] - cue[0]))
    if state.break_shot:
        shot = None if learned_break else PhysicalShot5D(2.5, heading, 10.0, 0.0, 0.0)
        return (NineBallGoal("BREAK", target, explicit_shot=shot),)

    candidates = []
    next_balls = sorted(n for n in on_table if n > target)
    for pocket_id in direct_pocket_candidates(state, target):
        direct = NineBallGoal("DIRECT_ATTACK", target, pocket_id)
        candidates.append(direct)
        if not next_balls:
            continue
        if position_goal is not None:
            position = position_goal(state, direct, next_balls[0])
            if position is not None:
                candidates.append(position)
        else:
            next_pos = state.ball_positions[
                state.array_index_for_ball(next_balls[0])
            ]
            # A simple next-ball region, not an optimized stopping point.
            stop = (
                float(np.clip(next_pos[0], 0.1, TABLE_WIDTH - 0.1)),
                float(np.clip(next_pos[1] - 0.25, 0.1, TABLE_LENGTH - 0.1)),
            )
            candidates.append(NineBallGoal(
                "POSITION_ATTACK", target, pocket_id, stop, 0.20,
            ))
    # Preserve a lowest-ball fallback even without a visible direct pot.
    if not candidates:
        candidates.append(NineBallGoal("DIRECT_ATTACK", target))
    safety_shot = None if learned_safety else PhysicalShot5D(0.8, heading, 10.0, 0.0, 0.0)
    candidates.append(NineBallGoal("SAFETY", target, explicit_shot=safety_shot))
    return tuple(candidates)
