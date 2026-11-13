"""5D Micro SAC inference behind the existing nine-ball self-play interface.

Goal selection is still a deterministic geometric placeholder unless a
separately trained version-3 Macro Candidate Policy is explicitly provided.
Never load old 12D/31D Macro or 2D Micro checkpoints into this adapter.
"""

from pathlib import Path

import numpy as np
import torch

from src.game.nine_ball_agents import GeometryNineBallAgent, ShotPlan
from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM,
    MICRO_CONDITIONED_V3_DIM, conditioned_observation_v3,
    encode_goal_v3, encode_rack_v3, generate_nine_ball_goals,
    reference_angle_deg,
)
from src.macro.candidate_policy import MacroCandidatePolicyNetwork
from src.micro.actor import GaussianActor
from src.micro.shot_control import Shot5DDecoder


MICRO_KIND = "nine_ball_goal_conditioned_micro_sac_5d"
MACRO_KIND = "nine_ball_macro_candidate_policy_v3"


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
        self.decoder = Shot5DDecoder()
        self.macro_policy = None
        if macro_policy_checkpoint is not None:
            macro = _load_versioned(
                macro_policy_checkpoint,
                model_kind=MACRO_KIND,
                state_dim=MACRO_STATE_V3_DIM,
            )
            if int(macro.get("candidate_dim", -1)) != MACRO_GOAL_V3_DIM:
                raise ValueError("Macro v3 candidate dimension mismatch")
            self.macro_policy = MacroCandidatePolicyNetwork(
                state_dim=MACRO_STATE_V3_DIM,
                candidate_dim=MACRO_GOAL_V3_DIM,
                hidden_dim=int(macro["hidden_dim"]),
            ).to(self.device)
            self.macro_policy.load_state_dict(
                macro["candidate_policy_network"], strict=True
            )
            self.macro_policy.eval()

    def _choose_goal(self, state, context=None):
        goals = generate_nine_ball_goals(state, learned_safety=self.learned_safety)
        if len(goals) == 1:
            return goals[0]
        if self.macro_policy is not None:
            s = torch.from_numpy(encode_rack_v3(state, context)).to(self.device)
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
        shot = self.decoder.decode(normalized, reference_angle_deg(state, goal))
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
        if state.ball_in_hand:
            raise ValueError("Place cue ball before Micro shot generation")
        goal = self._choose_goal(state, context)
        return self.plan_goal(state, goal, context)
