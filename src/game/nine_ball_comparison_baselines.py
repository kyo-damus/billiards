"""Training-free nine-ball comparison agents sharing the proposal's Micro actor.

These are *adapted baselines*, not reproductions of named papers.
PotOnlyGreedyAgent intentionally excludes position, safety and strategic
push-out choice; physical execution and nine-ball rules are unchanged.
"""
from dataclasses import replace

import numpy as np

from src.common.table import POCKET_POSITIONS
from src.env.shot_geometry import ghost_ball_position
from src.game.nine_ball_encoding_v3 import NineBallGoal, direct_pocket_candidates
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER


class PotOnlyGreedyAgent(NineBallMicroSACAgent):
    """Nearest geometrically clear pocket; no learned Macro or tree search.

    The same trained goal-conditioned Micro produces the physical shot.
    If no pocket is clear, still try the lowest-numbered ball (no SAFETY).
    WPA table-choice states are handled by always accepting the table.
    """

    def __init__(self, micro_checkpoint, *, device="cpu"):
        super().__init__(
            micro_checkpoint,
            macro_policy_checkpoint=None,
            deterministic=True,
            learned_safety=False,
            learned_break=True,
            outcome_position=False,
            match_aware=True,
            device=device,
        )
        self.break_decoder = WPA_NINE_BALL_BREAK_DECODER

    def _choose_goal(self, state, context=None, match_position=None):
        if state.ball_in_hand:
            raise ValueError("Place cue ball before choosing a goal")
        balls = state.on_table_object_balls()
        if not balls:
            raise ValueError("No remaining object balls")
        target = min(balls)
        if context is not None and context.pending_choice is not None:
            return NineBallGoal("ACCEPT_TABLE", target)
        if state.break_shot:
            return NineBallGoal("BREAK", target)

        pockets = tuple(direct_pocket_candidates(state, target))
        if not pockets:
            return NineBallGoal("DIRECT_ATTACK", target)

        cue = np.asarray(
            state.ball_positions[state.array_index_for_ball(0)], dtype=np.float64
        )
        ball = np.asarray(
            state.ball_positions[state.array_index_for_ball(target)], dtype=np.float64
        )

        def shot_distance(pocket_id):
            pocket = np.asarray(POCKET_POSITIONS[pocket_id], dtype=np.float64)
            ghost = ghost_ball_position(ball, pocket)
            return float(np.linalg.norm(ghost - cue) + np.linalg.norm(pocket - ball))

        chosen = min(pockets, key=lambda pocket_id: (shot_distance(pocket_id), pocket_id))
        return NineBallGoal("DIRECT_ATTACK", target, int(chosen))

    def choose_match_shot(self, state, context, match_position):
        result = super().choose_match_shot(state, context, match_position)
        return replace(
            result,
            goal_label=f"POT_ONLY_GREEDY_{result.macro_goal.kind}",
        )
