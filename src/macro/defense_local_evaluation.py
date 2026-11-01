"""Per-shot diagnostics for an executable Defense/SAFETY candidate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DefenseLocalEvaluation:
    """
    Immediate Defense v0 diagnostics.

    score is from the *acting player's* perspective. It contains
    the simplified game reward and an optional missed/wrong first
    contact penalty. It intentionally does NOT include opponent
    DIRECT lookahead: turn-aware MCTS or Q targets handle that.
    """

    first_contact_ball: int
    intended_first_contact: bool
    scratch: bool
    turn_changed: bool
    object_ball_pocketed: bool
    safety_eligible: bool
    cue_distance: float | None
    cue_region_reached: bool
    goal_achieved: bool
    score: float
