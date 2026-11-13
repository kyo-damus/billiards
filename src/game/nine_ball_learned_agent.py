"""5D Micro SAC inference behind the existing nine-ball self-play interface.

Goal selection is still a deterministic geometric placeholder unless a
separately trained version-3 Macro Candidate Policy is explicitly provided.
Never load old 12D/31D Macro or 2D Micro checkpoints into this adapter.
"""

from pathlib import Path

import numpy as np
import torch

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.nine_ball_agents import GeometryNineBallAgent, ShotPlan
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_STATE_V3_DIM,
    MICRO_CONDITIONED_V3_DIM, conditioned_observation_v3,
    MATCH_MACRO_STATE_DIM, encode_goal_v3, encode_macro_state, generate_nine_ball_goals,
    reference_angle_deg,
)
from src.game.nine_ball_position import generate_outcome_position_goals
from src.game.nine_ball_rules import NineBallContext, NineBallRules
from src.game.nine_ball_policy_v3 import load_macro_policy
from src.micro.actor import GaussianActor
from src.micro.shot_control import NINE_BALL_BREAK_DECODER, Shot5DDecoder


MICRO_KIND = "nine_ball_goal_conditioned_micro_sac_5d"


def _load_versioned(checkpoint, *, model_kind, state_dim, action_dim=None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        int(ckpt.get("format_version", -1)) != ENCODING_VERSION
        or ckpt.get("model_kind") != model_kind
        or int(ckpt.get("state_dim", -1)) != state_dim
        or (action_dim is not None and int(ckpt.get("action_dim", -1)) != action_dim)
    ):
        raise ValueError(
            f"Incompatible checkpoint {checkpoint}: expected version-3 "
            f"{model_kind} with state_dim={state_dim} "
            f"and action_dim={action_dim}."
        )
    return ckpt


class NineBallMicroSACAgent(GeometryNineBallAgent):
    """Micro learned, Macro geometric unless an explicitly tagged v3 Policy exists."""

    def __init__(
        self,
        micro_checkpoint,
        *,
        macro_policy_checkpoint=None,
        device="cpu",
        deterministic=True,
        learned_safety=False,
        learned_break=False,
        outcome_position=False,
        match_aware=False,
    ):
        super().__init__()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for nine-ball inference")
        micro = _load_versioned(
            micro_checkpoint,
            model_kind=MICRO_KIND,
            state_dim=MICRO_CONDITIONED_V3_DIM,
            action_dim=5,
        )
        self.actor = GaussianActor(
            state_dim=MICRO_CONDITIONED_V3_DIM,
            action_dim=5,
            hidden_dim=int(micro["hidden_dim"]),
        ).to(self.device)
        self.actor.load_state_dict(micro["actor"], strict=True)
        self.actor.eval()
        self.deterministic = deterministic
        self.learned_safety = learned_safety
        self.learned_break = learned_break
        self.outcome_position = outcome_position
        self.match_aware = match_aware
        self.position_simulator = None
        self.decoder = Shot5DDecoder()
        self.macro_policy = None
        if macro_policy_checkpoint is not None:
            self.macro_policy, _ = load_macro_policy(
                macro_policy_checkpoint, device=self.device,
                state_dim=MATCH_MACRO_STATE_DIM if match_aware else MACRO_STATE_V3_DIM,
            )

    def _choose_goal(self, state, context=None, match_position=None):
        if self.outcome_position and not state.break_shot:
            if self.position_simulator is None:
                self.position_simulator = MultiBallFastFizSimulator(state.config)
            goals = generate_outcome_position_goals(
                state, context or NineBallContext(), self.position_simulator, NineBallRules(),
                learned_safety=self.learned_safety,
            )
        else:
            goals = generate_nine_ball_goals(
                state, learned_safety=self.learned_safety, learned_break=self.learned_break,
            )
        if len(goals) == 1:
            return goals[0]
        if self.macro_policy is not None:
            s = torch.from_numpy(encode_macro_state(state, context, match_position)).to(self.device)
            c = torch.from_numpy(np.stack([
                encode_goal_v3(goal) for goal in goals
            ])).to(self.device)
            with torch.no_grad():
                scores = self.macro_policy(s, c).cpu().numpy()
            return goals[int(np.argmax(scores))]
        # Fallback Macro policy is geometry (NOT learned or MCTS).
        attack = [goal for goal in goals if goal.kind == "DIRECT_ATTACK"]
        if attack:
            return min(
                attack,
                key=lambda goal: abs(
                    reference_angle_deg(state, goal)
                    - reference_angle_deg(
                        state,
                        type(goal)("DIRECT_ATTACK", goal.target_ball),
                    )
                ),
            )
        return goals[-1]

    def plan_goal(self, state, goal, context=None) -> ShotPlan:
        """Decode a supplied Macro goal, without selecting or changing it.

        Used for counterfactual one-shot Macro supervision: every goal must
        be evaluated from the same original pre-shot state.
        """
        if state.ball_in_hand:
            raise ValueError("Place cue ball before Micro shot generation")
        if goal.explicit_shot is not None:
            return ShotPlan(
                target_ball=goal.target_ball,
                shot=goal.explicit_shot,
                goal_label=f"{goal.kind}_EXPLICIT",
                target_pocket=goal.target_pocket,
                macro_goal=goal,
            )
        observation = conditioned_observation_v3(state, goal, context)
        state_tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if self.deterministic:
                action = self.actor.deterministic(state_tensor)
            else:
                action, _ = self.actor.sample(state_tensor)
        normalized = action.squeeze(0).cpu().numpy()
        decoder = NINE_BALL_BREAK_DECODER if goal.kind == "BREAK" else self.decoder
        shot = decoder.decode(normalized, reference_angle_deg(state, goal))
        return ShotPlan(
            target_ball=goal.target_ball,
            shot=shot,
            goal_label=goal.kind + "_MICRO_SAC5D_" + (
                "MACRO_POLICY" if self.macro_policy is not None
                else "GEOMETRY_MACRO"
            ),
            target_pocket=goal.target_pocket,
            macro_goal=goal,
        )

    def choose_shot(self, state, context=None) -> ShotPlan:
        return self.choose_match_shot(state, context, None)

    def choose_match_shot(self, state, context, match_position) -> ShotPlan:
        if state.ball_in_hand:
            raise ValueError("Place cue ball before Micro shot generation")
        if self.match_aware and match_position is None:
            raise ValueError("Match-aware Macro requires a match position")
        goal = self._choose_goal(state, context, match_position if self.match_aware else None)
        return self.plan_goal(state, goal, context)
