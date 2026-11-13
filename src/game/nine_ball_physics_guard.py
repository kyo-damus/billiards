"""Strict diagnostics for physically impossible 9-ball transitions.

A native FastFiz failure is NOT a billiards foul, opponent win, or valid
MCTS observation. Never assign an invented reward to an unexecuted shot.
"""

import numpy as np

from src.game.rack_state import RackState


class InvalidPhysicalTransition(RuntimeError):
    """The engine could not produce a trustworthy physical shot or next rack."""


class NativeShotRejected(InvalidPhysicalTransition):
    """FastFiz rejected execution and returned native diagnostic flags."""

    def __init__(self, precondition, placement):
        self.precondition = precondition
        self.placement = placement
        super().__init__(
            f"native precondition={precondition}, placement={placement}; "
            "no physical shot was executed"
        )


def validate_physical_step(physical):
    """Reject native nonexecution rather than passing it into game rules.

    Lightweight test doubles may not expose native diagnostic attributes.
    Real PhysicalStep always provides both codes.
    """
    precondition = int(getattr(physical, "precondition_code", 0))
    placement = int(getattr(physical, "placement_code", 0))
    if precondition or placement:
        raise NativeShotRejected(precondition, placement)


def validate_rack_layout(state, *, tolerance_m=1e-6):
    """Reject substantial physical overlaps after the rules and any respot.

    Below 1 micrometer, float32 quantization can produce slight penetration.
    We DO NOT change ball coordinates here. If native execution rejects even
    a tiny overlap, that candidate is separately rejected as invalid.
    """
    if not isinstance(state, RackState):
        return  # MCTS unit-test doubles do not hold physical coordinates
    if not 0 <= tolerance_m <= 1e-4:
        raise ValueError("Invalid collision tolerance")
    active = np.flatnonzero(state.ball_pocket_indices < 0)
    diameter = 0.05715
    for offset, i in enumerate(active):
        for j in active[offset + 1:]:
            separation = float(np.linalg.norm(
                state.ball_positions[int(i)].astype(np.float64)
                - state.ball_positions[int(j)].astype(np.float64)
            ))
            overlap = diameter - separation
            if overlap > tolerance_m:
                ball_a = state.config.ball_numbers[int(i)]
                ball_b = state.config.ball_numbers[int(j)]
                raise InvalidPhysicalTransition(
                    f"overlapping balls {ball_a},{ball_b}: "
                    f"penetration={overlap:.9f} m; "
                    "discard simulated transition rather than score as foul"
                )


def simulate_nine_ball_shot(simulator, rules, state, context, *, target_ball, shot):
    """Simulate from an independent rack snapshot and validate the successor."""
    validate_rack_layout(state)
    simulator.restore(state)
    try:
        physical = simulator.step(target_ball=target_ball, shot=shot)
    except RuntimeError as exc:
        if "Bad Shot Exception" not in str(exc):
            raise
        raise InvalidPhysicalTransition(f"FastFiz rejected shot: {exc}") from exc
    validate_physical_step(physical)
    outcome = rules.apply(state, physical, context)
    validate_rack_layout(outcome.next_state)
    return physical, outcome
