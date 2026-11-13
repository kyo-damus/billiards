"""Strictly tagged 9-ball Macro Q/V loaders for future search integration.

These networks regress behavior-policy *selected-action terminal returns*.
Do not treat Q predictions on unseen/unexecuted goal candidates as an oracle.
"""

import torch

from src.game.nine_ball_encoding_v3 import (
    ENCODING_VERSION, MACRO_GOAL_V3_DIM, MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM,
)
from src.macro.candidate_value import MacroCandidateValueNetwork
from src.macro.value import MacroValueNetwork

TARGET_KIND = "selected_action_terminal_match_return"
REWARD_KIND = "terminal_match_win_loss"


def _read(path, expected_kind, *, needs_candidate, match_aware=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    correct = (
        ckpt.get("format_version") == ENCODING_VERSION
        and ckpt.get("model_kind") == expected_kind
        and ckpt.get("state_dim") in (
            (MACRO_STATE_V3_DIM, MATCH_MACRO_STATE_DIM) if match_aware else (MACRO_STATE_V3_DIM,)
        )
        and ckpt.get("target_kind") == TARGET_KIND
        and ckpt.get("reward_kind") == REWARD_KIND
        and ckpt.get("turn_aware") is True
        and isinstance(ckpt.get("hidden_dim"), int)
        and ckpt["hidden_dim"] > 0
    )
    if needs_candidate:
        correct = correct and ckpt.get("candidate_dim") == MACRO_GOAL_V3_DIM
    if not correct:
        raise ValueError(f"Incompatible 9-ball Macro Q/V version or target contract: {path}")
    return ckpt


def load_macro_q_v3(checkpoint_path, *, device="cpu", match_aware=False):
    c = _read(
        checkpoint_path, "nine_ball_macro_selected_action_q_v3",
        needs_candidate=True, match_aware=match_aware,
    )
    model = MacroCandidateValueNetwork(
        state_dim=c["state_dim"],
        candidate_dim=MACRO_GOAL_V3_DIM,
        hidden_dim=c["hidden_dim"],
    ).to(device)
    model.load_state_dict(c["candidate_value_network"], strict=True)
    model.eval()
    return model, c


def load_state_v_v3(checkpoint_path, *, device="cpu", match_aware=False):
    c = _read(
        checkpoint_path, "nine_ball_macro_state_v_v3",
        needs_candidate=False, match_aware=match_aware,
    )
    model = MacroValueNetwork(
        state_dim=c["state_dim"],
        hidden_dim=c["hidden_dim"],
    ).to(device)
    model.load_state_dict(c["state_value_network"], strict=True)
    model.eval()
    return model, c
