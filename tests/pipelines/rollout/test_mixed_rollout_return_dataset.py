"""Executed-only Monte Carlo Q label conversion, separate from oracle Q."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_mixed_rollout_return_dataset import (
    build_return_dataset,
    load_completed_rollout,
    signed_executed_returns,
)
from scripts.train_macro_candidate_value import load_dataset
from src.common.types import GameState, MacroAction, Strategy
from src.controller.mixed_hierarchical_rollout import (
    HierarchicalTurnRecord,
)
from src.controller.mixed_rollout_dataset import save_mixed_rollout_npz
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate


def make_state(player=0, score=(0.0, 0.0), pockets=(-1, -1, -1)):
    return GameState(
        ball_positions=np.asarray(
            [[0.5, 0.4], [0.6, 0.9], [0.9, 1.5]], dtype=np.float32
        ),
        score=np.asarray(score, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.asarray(pockets, dtype=np.int32),
    )


def selected_candidate():
    return MacroCandidate(
        action=MacroAction(Strategy.ATTACK, 0, 0),
        goal=TacticalGoal(
            TacticalGoalType.DIRECT_ATTACK, target_ball=0, target_pocket=0
        ),
    )


def create_rollout(path, *, completed=True, switch=False):
    before = make_state(player=0)
    between = make_state(
        player=1 if switch else 0,
        score=(0.0, 0.0),
    )
    after = make_state(
        player=1 if switch else 0,
        score=(0.0, 1.0) if switch else (1.0, 0.0),
        pockets=(-1, 0, 5),
    )
    first = HierarchicalTurnRecord(
        turn_index=0, player_before=0, choice=selected_candidate(),
        reward=-0.5, player_after=between.current_player,
        terminated=False, score_delta=(0.0, 0.0),
        newly_pocketed_balls=(), state_before=before,
        state_after=between,
    )
    second = HierarchicalTurnRecord(
        turn_index=1, player_before=between.current_player,
        choice=selected_candidate(), reward=1.0,
        player_after=after.current_player,
        terminated=completed,
        score_delta=(
            (0.0, 1.0) if switch else (1.0, 0.0)
        ), newly_pocketed_balls=(0, 1),
        state_before=between, state_after=after,
    )
    save_mixed_rollout_npz(
        SimpleNamespace(
            records=[first, second],
            terminated=completed,
            stop_reason="game_terminated" if completed else "max_turns",
        ),
        path,
    )
    return path


def test_signed_returns_track_player_perspective():
    np.testing.assert_allclose(
        signed_executed_returns([-0.5, 1.0], [0, 0], 0.99),
        [0.49, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        signed_executed_returns([-0.5, 1.0], [0, 1], 0.99),
        [-1.49, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        signed_executed_returns([1.0, 2.0, 3.0], [0, 1, 0], 1),
        [2.0, -1.0, 3.0],
    )


def test_converted_dataset_loads_with_existing_q_trainer(tmp_path):
    first = create_rollout(tmp_path / "first.npz")
    second = create_rollout(
        tmp_path / "second.npz", switch=True,
    )
    output, n_episodes, n_rows = build_return_dataset(
        [first, second],
        tmp_path / "nested" / "returns.npz",
        gamma=0.99,
    )
    assert (n_episodes, n_rows) == (2, 4)
    states, candidates, targets, ids = load_dataset(output)
    assert states.shape == (4, 12)
    assert candidates.shape == (4, 31)
    assert ids.tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(
        targets, [0.49, 1.0, -1.49, 1.0], atol=1e-6
    )
    with np.load(output, allow_pickle=False) as dataset:
        assert dataset["episode_id"].tolist() == [0, 0, 1, 1]
        assert str(dataset["target_kind"]) == "selected_action_mc_return"
        assert bool(dataset["turn_aware"])
        assert int(dataset["candidate_encoding_version"]) == 2


def test_rejects_incomplete_executed_rollout(tmp_path):
    path = create_rollout(tmp_path / "partial.npz", completed=False)
    with pytest.raises(ValueError, match="incomplete trajectory"):
        load_completed_rollout(path, gamma=0.99)
    assert not (tmp_path / "out.npz").exists()


def test_rejects_missing_or_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="At least one rollout"):
        build_return_dataset([], tmp_path / "none.npz")
    with pytest.raises(ValueError, match="gamma"):
        signed_executed_returns([1.0], [0], -0.1)
    bad = tmp_path / "bad.npz"
    np.savez(bad, x=np.zeros(1))
    with pytest.raises(ValueError, match="missing fields"):
        load_completed_rollout(bad, gamma=0.99)
