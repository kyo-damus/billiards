"""Generate cue-stop goals from legal nine-ball shot outcomes."""

from functools import partial
from itertools import product

import numpy as np

from src.common.table import POCKET_POSITIONS
from src.env.shot_geometry import ghost_ball_position
from src.game.nine_ball_encoding_v3 import (
    NineBallGoal, direct_pocket_candidates, generate_nine_ball_goals,
    reference_angle_deg,
)
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, simulate_nine_ball_shot,
)
from src.micro.shot_control import Shot5DDecoder


POSITION_RADIUS_M = 0.10
_DECODER = Shot5DDecoder()
# Nine proposals per current pocket, inside the Micro action space.
# Vary speed and follow/draw around the ghost-ball aiming direction.
_PROBES = tuple(
    (speed, 0.0, -0.8, 0.0, tip_y)
    for speed, tip_y in product((-0.6, 0.0, 0.6), (0.0, -0.4, 0.4))
)


def continues_to_next_ball(next_ball, outcome):
    """The shooter keeps the table and the intended next ball stays live."""
    return (
        next_ball is not None and not outcome.foul and not outcome.turn_changed
        and outcome.rack_winner is None
        and not outcome.next_state.is_pocketed(next_ball)
    )


def _approach_distance(cue, ball, pocket):
    direction = ghost_ball_position(ball, pocket) - cue
    distance = float(np.linalg.norm(direction))
    # A clear segment from the wrong side of the ghost cannot pot the ball.
    if distance < 1e-7 or np.dot(direction, pocket - ball) <= 0.0:
        return np.inf
    return distance


def _next_shot_distance(state, ball_number):
    cue = state.ball_positions[0].astype(np.float64)
    ball = state.ball_positions[state.array_index_for_ball(ball_number)].astype(np.float64)
    return min((
        _approach_distance(cue, ball, POCKET_POSITIONS[pocket].astype(np.float64))
        for pocket in direct_pocket_candidates(state, ball_number)
    ), default=np.inf)


def _witnessed_position(state, direct, next_ball, *, context, simulator, rules):
    target_index = state.array_index_for_ball(direct.target_ball)
    cue = state.ball_positions[0].astype(np.float64)
    ball = state.ball_positions[target_index].astype(np.float64)
    pocket = POCKET_POSITIONS[direct.target_pocket].astype(np.float64)
    if not np.isfinite(_approach_distance(cue, ball, pocket)):
        return None
    heading = reference_angle_deg(state, direct)
    best_distance = np.inf
    stop = None
    for action in _PROBES:
        shot = _DECODER.decode(action, heading)
        try:
            physical, outcome = simulate_nine_ball_shot(
                simulator, rules, state, context, target_ball=direct.target_ball, shot=shot,
            )
        except InvalidPhysicalTransition:
            continue
        if (
            physical.pocket_indices[target_index] != direct.target_pocket
            or not continues_to_next_ball(next_ball, outcome)
        ):
            continue
        distance = _next_shot_distance(outcome.next_state, next_ball)
        if distance < best_distance:
            best_distance = distance
            stop = tuple(float(x) for x in outcome.next_state.ball_positions[0])
    if stop is None:
        return None
    return NineBallGoal(
        "POSITION_ATTACK", direct.target_ball, direct.target_pocket, stop, POSITION_RADIUS_M,
    )


def generate_outcome_position_goals(state, context, simulator, rules, *, learned_safety=False):
    """Offer at most one physically witnessed cue-stop goal per current pocket."""
    position_goal = partial(
        _witnessed_position, context=context, simulator=simulator, rules=rules,
    )
    return generate_nine_ball_goals(
        state, learned_safety=learned_safety, position_goal=position_goal,
    )
