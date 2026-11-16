"""Event-based core nine-ball rules for the variable-ball physical simulator.

Optional WPA break and push-out rules use native head-string crossing data.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from src.game.rack_state import RackState

BALL_RADIUS_M = 0.028575
TABLE_WIDTH_M = 1.116
TABLE_LENGTH_M = 2.236
FOOT_SPOT = (TABLE_WIDTH_M / 2, TABLE_LENGTH_M * 0.75)


@dataclass(frozen=True)
class NineBallContext:
    """Rack-local enforcement state; three-foul warning is automatic."""

    consecutive_fouls: tuple[int, int] = (0, 0)
    push_out_available: bool = False
    pending_choice: str | None = None

    def __post_init__(self):
        if len(self.consecutive_fouls) != 2 or any(
            n < 0 for n in self.consecutive_fouls
        ):
            raise ValueError("Expected two nonnegative foul counts.")
        if self.pending_choice not in (None, "illegal_break", "push_out"):
            raise ValueError("Unknown post-break choice")
        if self.pending_choice and self.push_out_available:
            raise ValueError("Resolve the post-break choice before declaring a push-out")


@dataclass(frozen=True)
class NineBallTransition:
    next_state: RackState
    next_context: NineBallContext
    foul: bool
    foul_reasons: tuple[str, ...]
    turn_changed: bool
    rack_winner: Optional[int]
    newly_pocketed: tuple[int, ...]
    nine_respotted: bool
    illegal_break: bool = False

    @property
    def rack_finished(self) -> bool:
        return self.rack_winner is not None


def _spot_nine(positions, pockets, config):
    """Spot 9 on the long string, avoiding balls already on the table.

    Prefer foot spot, then towards foot rail, then towards head rail.
    """
    idx = config.ball_numbers.index(9)
    x, y = FOOT_SPOT
    spacing = 2 * BALL_RADIUS_M + 1e-4
    locations = [(x, y)]
    for direction in (1, -1):
        for step in range(1, int(TABLE_LENGTH_M / spacing) + 1):
            new_y = y + direction * step * spacing
            if BALL_RADIUS_M <= new_y <= TABLE_LENGTH_M - BALL_RADIUS_M:
                locations.append((x, new_y))
    for trial in locations:
        if all(
            pockets[i] >= 0 or i == idx or
            np.linalg.norm(positions[i] - np.asarray(trial)) >= spacing
            for i in range(len(pockets))
        ):
            positions[idx] = trial
            pockets[idx] = -1
            return
    raise RuntimeError("No legal long-string location for respotted 9-ball")


class NineBallRules:
    """Physics-to-rules transition, independent of Macro/Micro models."""

    def __init__(self, *, wpa_break_rules=False):
        self.wpa_break_rules = wpa_break_rules

    def validate_shot(self, state, context, declared_push_out):
        if context.pending_choice is not None:
            raise ValueError("Resolve the pending shot choice before shooting")
        if declared_push_out and not self.wpa_break_rules:
            raise NotImplementedError("Push-out requires WPA break rules")
        if declared_push_out and (not context.push_out_available or state.break_shot or state.ball_in_hand):
            raise ValueError("Push-out is unavailable")

    def resolve_choice(self, state, context, *, play: bool):
        """The chooser is the current player before an accept/pass decision."""
        if not self.wpa_break_rules or context.pending_choice is None:
            raise ValueError("No pending shot choice")
        next_state = state if play else replace(state, current_player=1 - state.current_player)
        next_context = replace(
            context, pending_choice=None,
            push_out_available=context.pending_choice == "illegal_break" and not play,
        )
        return next_state, next_context

    def apply(
        self,
        previous: RackState,
        physical,
        context: NineBallContext,
        *,
        declared_push_out: bool = False,
    ) -> NineBallTransition:
        if previous.config.rule_id != "nine_ball" or tuple(
            previous.config.object_ball_numbers
        ) != tuple(range(1, 10)):
            raise ValueError("NineBallRules requires a complete 1..9 inventory")
        if previous.ball_in_hand:
            raise ValueError("Place cue ball legally before executing this shot")
        if declared_push_out or context.pending_choice is not None:
            self.validate_shot(previous, context, declared_push_out)

        before = previous.ball_pocket_indices
        pockets = np.asarray(physical.pocket_indices, dtype=np.int32).copy()
        positions = np.asarray(physical.positions, dtype=np.float32).copy()
        if pockets.shape != before.shape or positions.shape != previous.ball_positions.shape:
            raise ValueError("Physical result ball inventory differs from rack")
        if np.any((before >= 0) & (pockets != before)):
            raise ValueError("Already pocketed balls cannot change pocket without spotting")
        if previous.is_pocketed(9):
            raise ValueError("9-ball is already pocketed; rack should have ended or 9 spotted")

        prev_on_table = previous.on_table_object_balls()
        if not prev_on_table:
            raise ValueError("No object balls remain on table")
        expected_first = min(prev_on_table)
        newly = tuple(
            number for number in prev_on_table
            if pockets[previous.array_index_for_ball(number)] >= 0
        )

        reasons = []
        if int(physical.precondition_code) != 0:
            reasons.append("invalid_shot_parameters")
        if int(physical.placement_code) != 0:
            reasons.append("invalid_ball_placement")
        if pockets[0] >= 0:
            reasons.append("cue_scratch")
        if self.wpa_break_rules and previous.break_shot and previous.ball_positions[0, 1] >= TABLE_LENGTH_M / 4:
            reasons.append("cue_not_behind_head_string")

        hit = physical.first_contact_ball
        if not declared_push_out:
            if hit is None:
                reasons.append("no_object_contact")
            elif int(hit) != expected_first:
                reasons.append("wrong_ball_first")

        # Time-ordered native events: rail contact only counts at or after
        # the first cue/object-ball collision, unless an object was pocketed.
        events = tuple(physical.events)
        cue_hit_times = [
            e.time_s for e in events
            if e.kind == "BALL_COLLISION"
            and (e.ball1 == 0 or e.ball2 == 0)
            and (e.ball2 if e.ball1 == 0 else e.ball1) == hit
        ] if hit is not None else []
        contact_time = min(cue_hit_times) if cue_hit_times else None
        rail_after_hit = contact_time is not None and any(
            e.kind == "RAIL_COLLISION" and e.time_s >= contact_time
            for e in events
        )
        if not declared_push_out and hit is not None and not newly and not rail_after_hit:
            reasons.append("no_rail_after_contact")

        if previous.break_shot and not newly:
            distinct_object_rails = {
                int(e.ball1) for e in events
                if e.kind == "RAIL_COLLISION"
                and e.ball1 in prev_on_table
            }
            if len(distinct_object_rails) < 4:
                reasons.append("insufficient_break_rails")

        foul = bool(reasons)
        illegal_break = False
        if self.wpa_break_rules and previous.break_shot and not foul:
            crossed = getattr(physical, "head_string_crossings", None)
            if crossed is None:
                raise RuntimeError("WPA break rules require a FastFiz build with head-string crossing data")
            qualifying = set(newly).union(set(crossed).intersection(prev_on_table))
            illegal_break = len(qualifying) < 3
        shooter = previous.current_player
        other = 1 - shooter
        fouls = list(context.consecutive_fouls)
        if foul:
            fouls[shooter] += 1
        else:
            fouls[shooter] = 0

        winner = None
        if foul and fouls[shooter] >= 3:
            winner = other
        elif not foul and not illegal_break and not declared_push_out and 9 in newly:
            winner = shooter

        nine_respotted = 9 in newly and (foul or declared_push_out or illegal_break)
        if nine_respotted:
            _spot_nine(positions, pockets, previous.config)
        choice = None
        if illegal_break:
            choice = "illegal_break"
        elif declared_push_out and not foul:
            choice = "push_out"
        turn_changed = foul or choice is not None or (not newly and winner is None)
        if winner is not None:
            turn_changed = False
        next_player = other if turn_changed else shooter
        next_state = RackState(
            config=previous.config,
            ball_positions=positions,
            ball_pocket_indices=pockets,
            current_player=next_player,
            score=previous.score,
            ball_in_hand=foul and winner is None,
            break_shot=False,
            turn_index=previous.turn_index + 1,
        )
        next_context = NineBallContext(
            consecutive_fouls=tuple(fouls),
            push_out_available=previous.break_shot and not foul and choice is None and winner is None,
            pending_choice=choice,
        )
        return NineBallTransition(
            next_state=next_state,
            next_context=next_context,
            foul=foul,
            foul_reasons=tuple(reasons),
            turn_changed=turn_changed,
            rack_winner=winner,
            newly_pocketed=newly,
            nine_respotted=nine_respotted,
            illegal_break=illegal_break,
        )
