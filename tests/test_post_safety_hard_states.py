from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval_post_safety_hard_states import (
    board_fingerprint,
    select_hard_states_from_safety,
)
from src.common.types import GameState, ShotAction


def make_state(player=0, cue_x=0.50, cue_pocketed=False):
    return GameState(
        ball_positions=np.asarray(
            [
                [cue_x, 0.30],
                [0.50, 0.80],
                [0.90, 1.50],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(2, dtype=np.float32),
        current_player=player,
        ball_pocket_indices=np.asarray(
            [0 if cue_pocketed else -1, -1, -1],
            dtype=np.int32,
        ),
    )


def make_proposal(state, successes, ball=0, angle=90.0):
    return SimpleNamespace(
        targeted_ball=ball,
        result=SimpleNamespace(next_state=state),
        opponent_threat=SimpleNamespace(
            successful_pot_count=successes,
        ),
        shot=ShotAction(
            angle=angle,
            power=0.65,
            tip_y_mm=5.0,
        ),
    )


def test_selects_only_hard_opponent_states_and_deduplicates():
    root = make_state(player=0)
    hard = make_state(player=1, cue_x=0.60)
    near_duplicate = make_state(player=1, cue_x=0.601)
    easy = make_state(player=1, cue_x=0.70)
    unchanged_turn = make_state(player=0, cue_x=0.80)
    scratched = make_state(
        player=1, cue_x=0.90, cue_pocketed=True,
    )

    proposals = [
        make_proposal(hard, 0, ball=1),
        make_proposal(near_duplicate, 0),
        make_proposal(easy, 3),
        make_proposal(unchanged_turn, 0),
        make_proposal(scratched, 0),
    ]

    seen = set()
    selected = select_hard_states_from_safety(
        root, proposals, max_successful_direct=0, seen=seen,
    )

    assert len(selected) == 1
    assert selected[0].source_target_ball == 1
    assert selected[0].source_direct_successes == 0
    assert np.isclose(selected[0].source_tip_y_mm, 5.0)
    assert len(seen) == 1

    # Repeating the same proposal list does not duplicate boards.
    assert select_hard_states_from_safety(
        root, proposals, max_successful_direct=0, seen=seen
    ) == []


def test_threshold_controls_hard_state_filter():
    root = make_state(player=0)
    proposal = make_proposal(
        make_state(player=1, cue_x=0.7), 1
    )
    assert not select_hard_states_from_safety(
        root, [proposal], max_successful_direct=0,
    )
    assert len(select_hard_states_from_safety(
        root, [proposal], max_successful_direct=1,
    )) == 1


def test_state_fingerprint_respects_player_and_pockets():
    a = make_state(player=0)
    b = make_state(player=1)
    c = make_state(player=0, cue_pocketed=True)
    assert board_fingerprint(a) != board_fingerprint(b)
    assert board_fingerprint(a) != board_fingerprint(c)


def test_bad_threshold_or_bin_size_is_rejected():
    with pytest.raises(ValueError, match="max_successful_direct"):
        select_hard_states_from_safety(
            make_state(), [], max_successful_direct=-1
        )
    with pytest.raises(ValueError, match="bin_size"):
        board_fingerprint(make_state(), bin_size=0.0)
