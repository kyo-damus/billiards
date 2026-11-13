"""Load rack and match versions of the nine-ball Macro Candidate Policy."""

import torch

from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM,
)
from src.macro.candidate_policy import MacroCandidatePolicyNetwork


def load_macro_policy(path, *, device, state_dim):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    old_dim = checkpoint.get("state_dim")
    hidden_dim = checkpoint.get("hidden_dim")
    if (
        checkpoint.get("format_version") != ENCODING_VERSION
        or checkpoint.get("model_kind") != "nine_ball_macro_candidate_policy_v3"
        or checkpoint.get("candidate_dim") != MACRO_GOAL_V3_DIM
        or old_dim not in (MACRO_STATE_V3_DIM, state_dim)
        or state_dim not in (MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM)
        or not isinstance(hidden_dim, int) or hidden_dim < 1
    ):
        raise ValueError("Incompatible nine-ball Macro Policy state or candidate dimensions")
    model = MacroCandidatePolicyNetwork(state_dim, MACRO_GOAL_V3_DIM, hidden_dim)
    weights = checkpoint["candidate_policy_network"]
    if old_dim != state_dim:
        # Preserve rack and goal columns; start with zero weights for match inputs.
        old = MacroCandidatePolicyNetwork(old_dim, MACRO_GOAL_V3_DIM, hidden_dim)
        old.load_state_dict(weights, strict=True)
        first = old.network[0].weight.detach()
        extra = first.new_zeros((hidden_dim, state_dim - old_dim))
        weights = dict(weights)
        weights["network.0.weight"] = torch.cat((first[:, :old_dim], extra, first[:, old_dim:]), dim=1)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), hidden_dim
