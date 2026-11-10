"""Scoped 9-ball Macro v3 bootstrap tests; no FastFiz native build required."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.build_nine_ball_macro_oracle import (
    evaluate_goal, observed_samples, one_step_score,
)
from scripts.train_nine_ball_macro_policy_v3 import (
    load_dataset, parse_args, split_by_match, train,
)
from src.game.nine_ball_encoding_v3 import NineBallGoal
from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_session import create_nine_ball_rack
from src.micro.actor import GaussianActor


def test_one_step_score_respects_fouls_and_correct_pocket():
    state = create_nine_ball_rack(breaker=0, seed=4)
    pockets = np.full(10, -1, dtype=np.int32)
    pockets[1] = 3
    physical = SimpleNamespace(
        pocket_indices=pockets,
        positions=state.ball_positions,
    )
    correct = NineBallGoal("DIRECT_ATTACK", target_ball=1, target_pocket=3)
    wrong = NineBallGoal("DIRECT_ATTACK", target_ball=1, target_pocket=4)
    safety = NineBallGoal("SAFETY", target_ball=1)
    legal = SimpleNamespace(
        foul=False, rack_winner=None, newly_pocketed=(1,),
        turn_changed=False,
    )
    foul = SimpleNamespace(
        foul=True, rack_winner=None, newly_pocketed=(1,),
        turn_changed=True,
    )
    assert one_step_score(correct, state, physical, legal) == 2.0
    assert one_step_score(wrong, state, physical, legal) == 0.4
    assert one_step_score(safety, state, physical, legal) == 0.0
    assert one_step_score(correct, state, physical, foul) == -2.0


def test_one_step_oracle_restores_original_board_per_candidate():
    state = create_nine_ball_rack(breaker=0, seed=4)
    goal = NineBallGoal("DIRECT_ATTACK", target_ball=1, target_pocket=0)
    calls = []

    class Agent:
        def plan_goal(self, board, request, context):
            assert board is state
            assert request is goal
            return SimpleNamespace(target_ball=1, shot="shot")

    class Simulator:
        def restore(self, board):
            assert board is state
            calls.append("restore")

        def step(self, *, target_ball, shot):
            calls.append("step")
            return SimpleNamespace(
                positions=state.ball_positions,
                pocket_indices=state.ball_pocket_indices,
            )

    class Rules:
        def apply(self, board, physical, context):
            return SimpleNamespace(
                foul=False, rack_winner=None, newly_pocketed=(),
                turn_changed=True,
            )

    for _ in range(2):
        result = evaluate_goal(
            state, None, goal, agent=Agent(),
            simulator=Simulator(), rules=Rules(),
        )
        assert result["score"] == 0.0
    assert calls == ["restore", "step", "restore", "step"]


def test_v3_macro_policy_checkpoint_and_match_holdout(tmp_path):
    states = np.zeros((4, 72), dtype=np.float32)
    candidates = np.zeros((8, 36), dtype=np.float32)
    for i in range(4):
        states[i, i] = 1.0
        candidates[2 * i, 0] = 1.0
        candidates[2 * i + 1, 1] = 1.0
    path = tmp_path / "oracle.npz"
    np.savez_compressed(
        path,
        states=states,
        candidates=candidates,
        teacher_scores=np.array([2, 0, 0, 2, 2, 0, 0, 2], dtype=np.float32),
        offsets=np.array([0, 2, 4, 6, 8]),
        match_seeds=np.array([100, 100, 200, 200]),
    )
    arrays = load_dataset(path)
    training, validation, match_seeds = split_by_match(
        arrays[-1], validation_fraction=0.5, seed=4,
    )
    assert len(training) == len(validation) == 2
    assert set(arrays[-1][training]).isdisjoint(set(arrays[-1][validation]))
    assert len(match_seeds) == 1
    ckpt = tmp_path / "macro.pt"
    report = train(parse_args([
        "--dataset", str(path), "--output", str(ckpt),
        "--epochs", "2", "--hidden-dim", "16",
        "--validation-fraction", "0.5",
    ]))
    assert ckpt.is_file() and report["validation_boards"] == 2
    saved = torch.load(ckpt, weights_only=True)
    assert saved["format_version"] == 3
    assert saved["model_kind"] == "nine_ball_macro_candidate_policy_v3"
    assert saved["state_dim"] == 72
    assert saved["candidate_dim"] == 36
    assert saved["teacher_kind"] == "fixed_micro_one_step_fastfiz"

    micro_path = tmp_path / "micro.pt"
    actor = GaussianActor(state_dim=108, action_dim=5, hidden_dim=16)
    torch.save({
        "format_version": 3,
        "model_kind": "nine_ball_goal_conditioned_micro_sac_5d",
        "state_dim": 108, "action_dim": 5, "hidden_dim": 16,
        "actor": actor.state_dict(),
    }, micro_path)
    agent = NineBallMicroSACAgent(
        micro_path, macro_policy_checkpoint=ckpt, device="cpu",
    )
    state = create_nine_ball_rack(breaker=0, seed=3)
    from dataclasses import replace
    shot = agent.choose_shot(replace(state, break_shot=False))
    assert shot.goal_label.endswith(("MACRO_POLICY", "_EXPLICIT"))


def test_reject_single_match_dataset(tmp_path):
    with pytest.raises(ValueError, match="distinct source matches"):
        split_by_match(
            np.array([7, 7, 7]), validation_fraction=0.5, seed=1,
        )
