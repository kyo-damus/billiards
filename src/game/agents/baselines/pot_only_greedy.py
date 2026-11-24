"""Training-free adapted nine-ball comparisons sharing the proposal's 5D Micro.

These are NOT faithful replications of any named published method.
"""
from dataclasses import replace
from functools import partial

import numpy as np

from src.common.table import POCKET_POSITIONS
from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.env.shot_geometry import ghost_ball_position
from src.game.nine_ball_encoding_v3 import NineBallGoal, direct_pocket_candidates
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_physics_guard import (
    InvalidPhysicalTransition, simulate_nine_ball_shot, simulate_with_cue_clearance,
)
from src.game.nine_ball_rules import NineBallRules
from src.micro.shot_control import WPA_NINE_BALL_BREAK_DECODER


class PotOnlyGreedyAgent(NineBallMicroSACAgent):
    """Nearest clear pocket, no strategic defense, positioning or tree search.

    If there is no clear pocket, aim at the lowest-numbered ball.  The shared
    trained Micro controls each shot. WPA table choices always accept the table.
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
        self._physics_simulator = None
        self._wpa_rules = NineBallRules(wpa_break_rules=True)

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

        cue = np.asarray(state.ball_positions[state.array_index_for_ball(0)], dtype=np.float64)
        ball = np.asarray(state.ball_positions[state.array_index_for_ball(target)], dtype=np.float64)

        def path_length(pocket_id):
            pocket = np.asarray(POCKET_POSITIONS[pocket_id], dtype=np.float64)
            ghost = ghost_ball_position(ball, pocket)
            return float(np.linalg.norm(ghost - cue) + np.linalg.norm(pocket - ball))

        chosen = min(pockets, key=lambda pocket_id: (path_length(pocket_id), pocket_id))
        return NineBallGoal("DIRECT_ATTACK", target, int(chosen))

    def choose_match_shot(self, state, context, match_position):
        result = super().choose_match_shot(state, context, match_position)
        if result.shot is not None:
            if self._physics_simulator is None:
                self._physics_simulator = MultiBallFastFizSimulator(state.config)
            simulate = partial(
                simulate_nine_ball_shot, self._physics_simulator, self._wpa_rules,
                state, context, target_ball=result.target_ball,
                declared_push_out=False,
            )
            try:
                shot, _, _, _ = simulate_with_cue_clearance(
                    lambda attempt: simulate(shot=attempt), result.shot,
                )
                result = replace(result, shot=shot)
            except InvalidPhysicalTransition:
                # Do not invent a physical transition or an opponent foul.
                # The live collector flags the unexecutable shot separately.
                pass
        return replace(result, goal_label=f"POT_ONLY_GREEDY_{result.macro_goal.kind}")
