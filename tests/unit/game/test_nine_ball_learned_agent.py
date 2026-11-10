"""New 5D learned Micro inference adapter and optional Macro policy contract."""

import torch
import pytest

from src.game.nine_ball_encoding_v3 import (
    MACRO_STATE_V3_DIM, MACRO_GOAL_V3_DIM, MICRO_CONDITIONED_V3_DIM,
)
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_session import create_nine_ball_rack
from src.macro.candidate_policy import MacroCandidatePolicyNetwork
from src.micro.actor import GaussianActor


def save_micro(tmp_path, *, version=3, action_dim=5):
    path = tmp_path / "micro.pt"
    actor = GaussianActor(
        state_dim=MICRO_CONDITIONED_V3_DIM,
        action_dim=5,
        hidden_dim=16,
    )
    torch.save({
        "format_version": version,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": MICRO_CONDITIONED_V3_DIM,
        "action_dim": action_dim,
        "hidden_dim": 16,
        "actor": actor.state_dict(),
    }, path)
    return path


def test_pretrained_micro_format_loads_and_shoots_all_five_parameters(tmp_path):
    path = save_micro(tmp_path)
    agent = NineBallMicroSACAgent(path, device="cpu")
    rack = create_nine_ball_rack(breaker=0, seed=1)
    break_shot = agent.choose_shot(rack)
    assert break_shot.goal_label == "BREAK_EXPLICIT"
    from dataclasses import replace
    regular = replace(rack, break_shot=False)
    shot = agent.choose_shot(regular)
    assert shot.target_ball == 1
    assert "MICRO_SAC5D" in shot.goal_label
    physical = shot.shot.to_fastfiz_kwargs()
    assert set(physical) == {
        "power", "angle", "cue_elevation_deg", "tip_x_mm", "tip_y_mm"
    }


def test_old_micro_format_is_rejected(tmp_path):
    path = save_micro(tmp_path, version=2)
    with pytest.raises(ValueError, match="Incompatible"):
        NineBallMicroSACAgent(path)


def test_macro_policy_requires_distinct_versioned_checkpoint(tmp_path):
    path = save_micro(tmp_path)
    macro_path = tmp_path / "macro.pt"
    model = MacroCandidatePolicyNetwork(
        state_dim=MACRO_STATE_V3_DIM,
        candidate_dim=MACRO_GOAL_V3_DIM,
        hidden_dim=16,
    )
    torch.save({
        "format_version": 3,
        "model_kind": "nine_ball_macro_candidate_policy_v3",
        "state_dim": MACRO_STATE_V3_DIM,
        "candidate_dim": MACRO_GOAL_V3_DIM,
        "hidden_dim": 16,
        "candidate_policy_network": model.state_dict(),
    }, macro_path)
    agent = NineBallMicroSACAgent(path, macro_policy_checkpoint=macro_path)
    from dataclasses import replace
    rack = replace(create_nine_ball_rack(breaker=0, seed=1), break_shot=False)
    shot = agent.choose_shot(rack)
    assert shot.goal_label.endswith("MACRO_POLICY")
    # Loader should reject the legacy Macro policy dimensions even if there
    # happens to be a torch model with compatible parameter names.
    ckp = torch.load(macro_path, weights_only=True)
    ckp["candidate_dim"] = 31
    torch.save(ckp, macro_path)
    with pytest.raises(ValueError, match="candidate dimension"):
        NineBallMicroSACAgent(path, macro_policy_checkpoint=macro_path)
