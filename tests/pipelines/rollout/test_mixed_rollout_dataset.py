"""Typed 31-D rollout experience export, separate from Q targets."""

from types import SimpleNamespace

import numpy as np

from src.common.types import GameState, MacroAction, Strategy
from src.controller.mixed_hierarchical_rollout import (
    HierarchicalTurnRecord,
    RootCandidateRecord,
)
from src.controller.mixed_rollout_dataset import save_mixed_rollout_npz
from src.goal.types import TacticalGoal, TacticalGoalType
from src.macro.candidate import MacroCandidate, PhysicalShotSpec


def state(player=0, score=(0, 0), pockets=(-1, -1, -1)):
    return GameState(
        ball_positions=np.array(
            [[0.5, 0.3], [0.5, 0.8], [0.9, 1.5]],
            dtype=np.float32,
        ),
        score=np.array(score, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.array(pockets, dtype=np.int32),
    )


def attack():
    return MacroCandidate(
        action=MacroAction(Strategy.ATTACK, 0, 0),
        goal=TacticalGoal(
            TacticalGoalType.DIRECT_ATTACK,
            target_ball=0,
            target_pocket=0,
        ),
    )


def safety():
    return MacroCandidate(
        action=MacroAction(Strategy.DEFENSE, 0, None),
        goal=TacticalGoal(
            TacticalGoalType.SAFETY,
            target_ball=0,
            cue_target_position=(0.6, 0.4),
            cue_target_radius=0.1,
        ),
        physical_shot=PhysicalShotSpec(
            angle=90.0, power=0.65, tip_y_mm=5.0
        ),
    )


def test_npz_saves_executed_states_and_full_candidate_root(tmp_path):
    initial = state()
    after = state(
        player=0, score=(1, 0), pockets=(-1, 5, -1)
    )
    chosen = attack()
    alt = safety()
    record = HierarchicalTurnRecord(
        turn_index=0,
        player_before=0,
        choice=chosen,
        reward=-0.5,
        player_after=0,
        terminated=False,
        score_delta=(1.0, 0.0),
        newly_pocketed_balls=(0,),
        state_before=initial,
        state_after=after,
        root_candidates=(
            RootCandidateRecord(chosen, 4, 0.7, 1.1, True),
            RootCandidateRecord(alt, 2, 0.3, None, False),
        ),
    )
    rollout = SimpleNamespace(
        records=[record],
        stop_reason="max_turns",
        terminated=False,
    )
    output = tmp_path / "smoke" / "rollout.npz"
    assert save_mixed_rollout_npz(rollout, output, seed=94000) == output

    with np.load(output, allow_pickle=False) as data:
        assert int(data["format_version"]) == 1
        assert int(data["candidate_encoding_version"]) == 2
        assert str(data["data_kind"]) == "executed_mixed_rollout"
        assert data["state"].shape == (1, 12)
        assert data["next_state"].shape == (1, 12)
        assert data["candidate"].shape == (1, 31)
        assert data["root_candidate"].shape == (2, 31)
        assert data["root_turn_index"].tolist() == [0, 0]
        assert data["root_visit_count"].tolist() == [4, 2]
        assert data["root_selected"].tolist() == [True, False]
        assert np.isnan(data["root_predicted_q"][1])
        assert data["macro_local_reward"].tolist() == [-0.5]
        assert data["score_delta"].tolist() == [[1.0, 0.0]]
        assert data["state_pocket_indices"].tolist() == [[-1, -1, -1]]
        assert data["next_state_pocket_indices"].tolist() == [[-1, 5, -1]]
        assert data["player_before"].tolist() == [0]
        assert str(data["stop_reason"]) == "max_turns"
        assert not bool(data["rollout_terminated"])


def test_empty_rollout_produces_typed_empty_matrices(tmp_path):
    rollout = SimpleNamespace(
        records=[],
        stop_reason="no_feasible_macro_candidates",
        terminated=False,
    )
    output = tmp_path / "empty.npz"
    save_mixed_rollout_npz(rollout, output)
    with np.load(output, allow_pickle=False) as data:
        assert data["state"].shape == (0, 12)
        assert data["candidate"].shape == (0, 31)
        assert data["root_candidate"].shape == (0, 31)
        assert data["root_turn_index"].shape == (0,)
        assert str(data["stop_reason"]) == "no_feasible_macro_candidates"


def test_npz_preserves_defense_shot_encoding_and_turn_switch(tmp_path):
    initial = state()
    chosen = safety()
    after = state(player=1)
    record = HierarchicalTurnRecord(
        turn_index=0,
        player_before=0,
        choice=chosen,
        reward=0.0,
        player_after=1,
        terminated=False,
        score_delta=(0.0, 0.0),
        newly_pocketed_balls=(),
        state_before=initial,
        state_after=after,
    )
    output = tmp_path / "defense.npz"
    save_mixed_rollout_npz(
        SimpleNamespace(
            records=[record], stop_reason="max_turns", terminated=False
        ),
        output,
    )
    with np.load(output, allow_pickle=False) as data:
        assert data["candidate"].shape == (1, 31)
        assert data["candidate"][0, 1] == 1
        assert data["candidate"][0, 25] == 1
        assert data["player_after"].tolist() == [1]
