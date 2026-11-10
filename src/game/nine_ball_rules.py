"""Event-based core nine-ball rules for the variable-ball physical simulator.

This does NOT claim full WPA tournament compliance: the three-ball-break
head-string crossing regulation, push-out option, jumped-ball recognition,
and referee-only infractions need additional interfaces.
"""

from dataclasses import dataclass
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

    def __post_init__(self):
        if len(self.consecutive_fouls) != 2 or any(
            n < 0 for n in self.consecutive_fouls
        ):
            raise ValueError("Expected two nonnegative foul counts.")


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
    """Physics-to-rules transition, independent of Macro/Micro models.

    On a break with no pocket, four distinct object balls must reach a rail.
    The WPA three-ball head-string crossing regulation is NOT inferred from
    final positions, and push-out remains explicitly unsupported.
    """

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
        if declared_push_out:
            raise NotImplementedError(
                "Push-out and opponent choice are not implemented; "
                "do not silently treat a push-out as an ordinary shot"
            )

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

        hit = physical.first_contact_ball
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
        if hit is not None and not newly and not rail_after_hit:
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
        elif not foul and 9 in newly:
            winner = shooter

        nine_respotted = 9 in newly and foul
        if nine_respotted:
            _spot_nine(positions, pockets, previous.config)
        turn_changed = foul or (not newly and winner is None)
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
            push_out_available=(previous.break_shot and not foul and winner is None),
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
        )
