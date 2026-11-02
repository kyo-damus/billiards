"""No-FastFiz unit tests for collecting independent mixed rollout files."""

import numpy as np
import pytest

from scripts.collect_mixed_rollouts import collect_rollouts
from src.common.types import GameState, MacroAction, Strategy
from src.controller.hierarchical_agent import HierarchicalStepResult
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate
from scripts.train_macro_candidate_value import load_dataset


def state_factory(seed):
    # Even seed terminates in one action; odd seed is truncated.
    cue_x = 0.5 if seed % 2 == 0 else 0.8
    return GameState(
        ball_positions=np.asarray(
            [[cue_x, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=0,
        ball_pocket_indices=np.full(3, -1, dtype=np.int32),
    )


class StubAgent:
    def __init__(self):
        self.calls = 0

    def step(self, state):
        self.calls += 1
        completes = state.ball_positions[0, 0] < 0.6
        next_state = GameState(
            ball_positions=state.ball_positions.copy(),
            score=np.asarray(
                [2.0, 0.0] if completes else [0.0, 0.0],
                dtype=np.float32,
            ),
            current_player=0 if completes else 1,
            ball_pocket_indices=np.asarray(
                [-1, 0, 3] if completes else [-1, -1, -1],
                dtype=np.int32,
            ),
        )
        choice = MacroCandidate(
            action=MacroAction(Strategy.ATTACK, 0, 0),
            goal=TacticalGoal(
                TacticalGoalType.DIRECT_ATTACK,
                target_ball=0, target_pocket=0,
            ),
        )
        return HierarchicalStepResult(
            macro_action=choice,
            next_state=next_state,
            reward=1.0 if completes else 0.0,
            terminated=bool(completes),
        )


def test_collect_saves_completed_and_partial_separately(tmp_path):
    agent = StubAgent()
    folder = tmp_path / "rollouts"
    returns = tmp_path / "completed_returns.npz"
    result = collect_rollouts(
        agent,
        state_factory,
        games=2,
        start_seed=94000,
        max_turns=1,
        output_dir=folder,
        return_dataset=returns,
    )
    assert (result.saved_games, result.completed_games) == (2, 1)
    assert (result.unfinished_games, result.total_executed_turns) == (1, 2)
    assert result.completed_executed_turns == 1
    assert result.completed_paths == (folder / "rollout_seed_94000.npz",)
    assert result.return_dataset == returns
    assert agent.calls == 2
    assert (folder / "rollout_seed_94001.npz").exists()

    with np.load(folder / "rollout_seed_94001.npz", allow_pickle=False) as data:
        assert str(data["stop_reason"]) == "max_turns"
        assert not bool(data["rollout_terminated"])

    state, candidates, targets, ids = load_dataset(returns)
    assert state.shape == (1, 12)
    assert candidates.shape == (1, 31)
    assert targets.tolist() == [1.0]
    assert ids.tolist() == [0]
    with np.load(returns, allow_pickle=False) as data:
        assert data["episode_id"].tolist() == [0]
        assert str(data["target_kind"]) == "selected_action_mc_return"
        assert str(data["source_files"][0]).endswith(
            "rollout_seed_94000.npz"
        )


def test_all_incomplete_still_saved_but_no_return_dataset(tmp_path):
    result = collect_rollouts(
        StubAgent(),
        state_factory,
        games=1,
        start_seed=94001,
        max_turns=1,
        output_dir=tmp_path / "logs",
        return_dataset=tmp_path / "none.npz",
    )
    assert result.saved_games == 1
    assert result.completed_games == 0
    assert result.return_dataset is None
    assert (tmp_path / "logs" / "rollout_seed_94001.npz").exists()
    assert not (tmp_path / "none.npz").exists()


def test_refuse_overwrite_preflight(tmp_path):
    agent = StubAgent()
    out = tmp_path / "logs"
    collect_rollouts(
        agent,
        state_factory,
        games=1,
        start_seed=94000,
        max_turns=1,
        output_dir=out,
    )
    assert agent.calls == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_rollouts(
            agent,
            state_factory,
            games=2,
            start_seed=94000,
            max_turns=1,
            output_dir=out,
        )
    assert agent.calls == 1
    assert not (out / "rollout_seed_94001.npz").exists()


def test_collector_rejects_bad_arguments(tmp_path):
    agent = StubAgent()
    for games, max_turns, gamma in (
        (0, 1, 0.99),
        (1, 0, 0.99),
        (1, 1, -0.1),
    ):
        with pytest.raises(ValueError):
            collect_rollouts(
                agent,
                state_factory,
                games=games,
                start_seed=0,
                max_turns=max_turns,
                output_dir=tmp_path / "bad",
                gamma=gamma,
            )
    assert agent.calls == 0
