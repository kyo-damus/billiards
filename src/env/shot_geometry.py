import numpy as np


BALL_RADIUS = 0.028575
BALL_DIAMETER = 2.0 * BALL_RADIUS


def ghost_ball_position(
    target_pos: np.ndarray,
    pocket_pos: np.ndarray,
) -> np.ndarray:
    """
    対象球をポケット方向へ送り出すために、
    衝突時に手球中心が存在すべき位置（ghost ball）。
    """

    direction = pocket_pos - target_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-8:
        raise ValueError("target and pocket are too close")

    unit = direction / distance

    return target_pos - BALL_DIAMETER * unit


def point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start

    length_sq = np.dot(segment, segment)

    if length_sq < 1e-12:
        return float(np.linalg.norm(point - start))

    t = np.dot(
        point - start,
        segment,
    ) / length_sq

    t = np.clip(t, 0.0, 1.0)

    closest = start + t * segment

    return float(
        np.linalg.norm(point - closest)
    )


def angle_difference_deg(
    angle_a: float,
    angle_b: float,
) -> float:
    """2角度の最小差をdegreeで返す."""

    return (
        (angle_a - angle_b + 180.0)
        % 360.0
        - 180.0
    )


def is_direct_shot_feasible(
    cue_pos: np.ndarray,
    target_pos: np.ndarray,
    other_pos: np.ndarray | None,
    pocket_pos: np.ndarray,
    table_width: float,
    table_length: float,
    max_angle_offset: float = 45.0,
) -> bool:
    """
    現在のMicro actionで直接ポケットを狙えるかを
    幾何的に簡易判定する。
    """

    ghost = ghost_ball_position(
        target_pos,
        pocket_pos,
    )

    # ghost ball位置がテーブル内部にある
    margin = BALL_RADIUS

    if not (
        margin < ghost[0] < table_width - margin
        and margin < ghost[1] < table_length - margin
    ):
        return False

    # ---------------------------------
    # 現action表現 ±45° で届くか
    # ---------------------------------

    center_angle = np.degrees(
        np.arctan2(
            target_pos[1] - cue_pos[1],
            target_pos[0] - cue_pos[0],
        )
    )

    ghost_angle = np.degrees(
        np.arctan2(
            ghost[1] - cue_pos[1],
            ghost[0] - cue_pos[0],
        )
    )

    required_offset = abs(
        angle_difference_deg(
            ghost_angle,
            center_angle,
        )
    )

    if required_offset > max_angle_offset:
        return False

    # ---------------------------------
    # 手球 -> ghost ball の進路
    # ---------------------------------

    clearance = BALL_DIAMETER + 0.005

    if other_pos is not None:

        cue_path_clearance = (
            point_to_segment_distance(
                other_pos,
                cue_pos,
                ghost,
            )
        )

        if cue_path_clearance < clearance:
            return False

        object_path_clearance = (
            point_to_segment_distance(
                other_pos,
                target_pos,
                pocket_pos,
            )
        )

        if object_path_clearance < clearance:
            return False

    return True
