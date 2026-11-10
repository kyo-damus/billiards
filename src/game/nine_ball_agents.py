"""Replaceable 9-ball action provider; current default is geometry, NOT RL.

Both the game driver and the data schema accept future Macro+Micro policies
with the same choose_shot/placement interface. No network weights are loaded.
"""

from dataclasses import dataclass
from math import atan2, degrees, isfinite

import numpy as np

from src.common.table import POCKET_POSITIONS, TABLE_LENGTH, TABLE_WIDTH
from src.env.shot_geometry import (
    BALL_DIAMETER, ghost_ball_position, point_to_segment_distance,
)
from src.micro.shot_control import PhysicalShot5D


@dataclass(frozen=True)
class ShotPlan:
    target_ball: int
    shot: PhysicalShot5D
    goal_label: str
    target_pocket: int | None = None


class GeometryNineBallAgent:
    """Deterministic direct-shot/lowest-ball fallback for pipeline verification.

    This is a deliberately weak, nonlearned agent. A logged win against
    itself is NOT evidence that hierarchical RL works.
    """

    def __init__(self, *, break_speed: float = 2.5, shot_speed: float = 1.6):
        self.break_speed = float(break_speed)
        self.shot_speed = float(shot_speed)
        if (
            not isfinite(self.break_speed) or self.break_speed <= 0
            or not isfinite(self.shot_speed) or self.shot_speed <= 0
        ):
            raise ValueError("Physical shot speeds must be finite and positive.")

    def cue_placement_candidates(self, state):
        """Grid positions outside object-ball diameters, ordered for attack."""
        if not state.ball_in_hand:
            raise ValueError("Cue placement requested without ball in hand.")
        target = min(state.on_table_object_balls())
        ball = state.ball_positions[state.array_index_for_ball(target)]
        choices = []
        # Robust fallback grid. Session still validates actual native placement.
        for y in np.linspace(0.12, TABLE_LENGTH - 0.12, 13):
            for x in np.linspace(0.12, TABLE_WIDTH - 0.12, 7):
                xy = np.asarray((x, y), dtype=np.float64)
                if all(
                    np.linalg.norm(xy - state.ball_positions[i])
                    >= BALL_DIAMETER + 0.006
                    for i in range(1, state.ball_count)
                    if state.ball_pocket_indices[i] < 0
                ):
                    # Favor an approach from the head of the table.
                    score = float(np.linalg.norm(xy - ball)) + 0.2 * abs(x - TABLE_WIDTH / 2)
                    choices.append((score, float(x), float(y)))
        choices.sort()
        return tuple((x, y) for _, x, y in choices)

    def choose_shot(self, state) -> ShotPlan:
        if state.ball_in_hand:
            raise ValueError("Cue ball must be placed before selecting shot.")
        available = state.on_table_object_balls()
        if not available:
            raise ValueError("No remaining object balls.")
        target = min(available)
        cue = np.asarray(state.ball_positions[0], dtype=np.float64)
        ball = np.asarray(
            state.ball_positions[state.array_index_for_ball(target)],
            dtype=np.float64,
        )
        if state.break_shot:
            angle = degrees(atan2(ball[1] - cue[1], ball[0] - cue[0]))
            return ShotPlan(
                target, PhysicalShot5D(self.break_speed, angle, 10.0, 0.0, 0.0),
                goal_label="BREAK_HEURISTIC",
            )

        other = [
            np.asarray(state.ball_positions[i], dtype=np.float64)
            for i in range(1, state.ball_count)
            if state.ball_pocket_indices[i] < 0
            and i != state.array_index_for_ball(target)
        ]
        proposals = []
        for pocket_id, pocket in POCKET_POSITIONS.items():
            ghost = ghost_ball_position(ball, np.asarray(pocket))
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
            direction = ghost - cue
            if np.linalg.norm(direction) < 1e-8:
                continue
            dist = np.linalg.norm(direction) + np.linalg.norm(pocket - ball)
            angle = degrees(atan2(direction[1], direction[0]))
            proposals.append((float(dist), pocket_id, angle))
        if proposals:
            _, pocket_id, angle = min(proposals)
            return ShotPlan(
                target, PhysicalShot5D(self.shot_speed, angle, 10.0, 0.0, 0.0),
                goal_label="DIRECT_GEOMETRY",
                target_pocket=int(pocket_id),
            )
        # No clear direct shot: still aim at lowest ball, not a genuine SAFETY.
        angle = degrees(atan2(ball[1] - cue[1], ball[0] - cue[0]))
        return ShotPlan(
            target, PhysicalShot5D(self.shot_speed, angle, 10.0, 0.0, 0.0),
            goal_label="LOWEST_BALL_FALLBACK",
        )
