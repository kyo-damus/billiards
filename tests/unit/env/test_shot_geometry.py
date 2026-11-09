import numpy as np

from src.env.shot_geometry import (
    ghost_ball_position,
    is_direct_shot_feasible,
)


def test_ghost_ball_is_behind_target():
    target = np.array(
        [0.5, 1.0],
        dtype=np.float32,
    )

    pocket = np.array(
        [0.5, 2.0],
        dtype=np.float32,
    )

    ghost = ghost_ball_position(
        target,
        pocket,
    )

    assert ghost[1] < target[1]


def test_clear_direct_shot_is_feasible():
    cue = np.array([0.5, 0.3])
    target = np.array([0.5, 0.8])
    other = np.array([0.9, 1.2])
    pocket = np.array([0.5, 2.0])

    assert is_direct_shot_feasible(
        cue_pos=cue,
        target_pos=target,
        other_pos=other,
        pocket_pos=pocket,
        table_width=1.116,
        table_length=2.236,
    )


def test_blocked_shot_is_not_feasible():
    cue = np.array([0.5, 0.3])
    target = np.array([0.5, 0.8])

    # 手球と対象球の間を塞ぐ
    other = np.array([0.5, 0.55])

    pocket = np.array([0.5, 2.0])

    assert not is_direct_shot_feasible(
        cue_pos=cue,
        target_pos=target,
        other_pos=other,
        pocket_pos=pocket,
        table_width=1.116,
        table_length=2.236,
    )