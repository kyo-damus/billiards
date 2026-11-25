"""Experimental, read-only geometry ranking of ball-in-hand placement options.

This does NOT change the live nine-ball / MCTS placement policy.
Use it only for controlled comparisons of alternative legal cue locations.
"""

from dataclasses import replace

import numpy as np

from src.common.table import POCKET_POSITIONS
from src.env.shot_geometry import ghost_ball_position
from src.game.nine_ball_encoding_v3 import direct_pocket_candidates
from src.game.nine_ball_rules import BALL_RADIUS_M, TABLE_WIDTH_M, TABLE_LENGTH_M


def geometry_placement_ranking(state, candidates):
    """Rank legal-looking cue positions by visible target pockets, not proximity alone.

    Native FastFiz legality is separately checked before any candidate is used.
    Explicitly prefer more unobstructed pockets, then the shortest available
    stroke + object-to-pocket route. Preserve the original index as a stable
    tie-breaker. Return all positions so no legal option is silently lost.
    """
    if not state.ball_in_hand:
        raise ValueError("Geometry placement requires ball_in_hand=True")
    available = state.on_table_object_balls()
    if not available:
        raise ValueError("No legal object ball to target")
    target = min(available)
    index = state.array_index_for_ball(target)
    ball = state.ball_positions[index].astype(np.float64)
    ranked = []
    for i, (x, y) in enumerate(candidates):
        p = np.asarray((x, y), dtype=np.float64)
        if (not np.all(np.isfinite(p))
                or not (BALL_RADIUS_M <= p[0] <= TABLE_WIDTH_M - BALL_RADIUS_M)
                or not (BALL_RADIUS_M <= p[1] <= TABLE_LENGTH_M - BALL_RADIUS_M)):
            continue
        if any(
            np.linalg.norm(
                p - state.ball_positions[j].astype(np.float64)
            ) < 2 * BALL_RADIUS_M + 0.006
            for j in range(1, len(state.config.ball_numbers))
            if state.ball_pocket_indices[j] < 0
        ):
            continue
        positions = state.ball_positions.copy()
        pockets = state.ball_pocket_indices.copy()
        positions[0] = p
        pockets[0] = -1
        proposed = replace(
            state, ball_positions=positions, ball_pocket_indices=pockets,
            ball_in_hand=False,
        )
        options = list(direct_pocket_candidates(proposed, target))
        min_path = min(
            (
                float(np.linalg.norm(
                    ghost_ball_position(
                        ball, np.asarray(POCKET_POSITIONS[pocket], dtype=np.float64)
                    ) - p
                ) + np.linalg.norm(
                    ball - np.asarray(POCKET_POSITIONS[pocket], dtype=np.float64)
                ))
                for pocket in options
            ), default=float("inf"),
        )
        ranked.append({
            "xy": [float(x), float(y)],
            "original_index": i,
            "clear_pocket_ids": options,
            "clear_pocket_count": len(options),
            "best_clear_path_length_m": min_path if np.isfinite(min_path) else None,
        })
    return sorted(
        ranked,
        key=lambda item: (
            -item["clear_pocket_count"],
            item["best_clear_path_length_m"]
            if item["best_clear_path_length_m"] is not None else float("inf"),
            item["original_index"],
        ),
    )


def sample_placements(original_ranked, geometric_ranked, *, max_placements=17):
    """Baseline first, plus best geometry and evenly dispersed other options."""
    if max_placements < 2:
        raise ValueError("max_placements must be >= 2")
    original = {tuple(x["xy"]): x for x in geometric_ranked}
    in_original_order = sorted(original.values(), key=lambda x: x["original_index"])
    if not in_original_order:
        raise ValueError("No legal-looking cue placements")
    baseline = in_original_order[0]
    chosen = {tuple(baseline["xy"]): baseline}
    geometry_budget = max(1, (max_placements - 1) // 2)
    for item in geometric_ranked[:geometry_budget]:
        if len(chosen) >= max_placements:
            break
        chosen[tuple(item["xy"])] = item
    # Cover farther/broader options, not only high geometric scores.
    remaining = [x for x in in_original_order if tuple(x["xy"]) not in chosen]
    if remaining:
        for index in np.linspace(0, len(remaining) - 1, num=min(len(remaining), max_placements - len(chosen)), dtype=int):
            item = remaining[int(index)]
            chosen[tuple(item["xy"])] = item
    # Fill any empty positions deterministically.
    for item in geometric_ranked:
        if len(chosen) == max_placements:
            break
        chosen[tuple(item["xy"])] = item
    return list(chosen.values())[:max_placements]
