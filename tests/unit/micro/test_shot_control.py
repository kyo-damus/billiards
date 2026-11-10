"""5-DOF physical action decoding tests; independent of SAC/FastFiz."""

import numpy as np
import pytest

from src.micro.shot_control import PhysicalShot5D, Shot5DDecoder


def test_center_action_maps_to_middle_speed_and_elevation():
    shot = Shot5DDecoder().decode(np.zeros(5), reference_angle_deg=90.0)
    assert shot.speed_m_s == pytest.approx(1.75)
    assert shot.azimuth_deg == pytest.approx(90)
    assert shot.elevation_deg == pytest.approx(15)
    assert shot.tip_x_mm == 0
    assert shot.tip_y_mm == 0
    kwargs = shot.to_fastfiz_kwargs()
    assert kwargs["angle"] == 90
    assert kwargs["cue_elevation_deg"] == 15


def test_action_bounds_and_tip_disk_projection():
    shot = Shot5DDecoder().decode(
        np.array([1, -1, 1, 1, 1]), reference_angle_deg=30.0
    )
    assert shot.speed_m_s == pytest.approx(3.0)
    assert shot.azimuth_deg == pytest.approx(20.0)
    assert shot.elevation_deg == pytest.approx(30.0)
    assert np.hypot(shot.tip_x_mm, shot.tip_y_mm) == pytest.approx(12.0)


@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros(4), np.ones(6),
                                   [0, 0, 0, 0, 2], [0, 0, float("nan"), 0, 0]])
def test_invalid_5d_action_rejected(bad):
    with pytest.raises(ValueError):
        Shot5DDecoder().decode(bad, reference_angle_deg=0)


def test_invalid_elevation_and_tip_rejected_by_physical_contract():
    with pytest.raises(ValueError, match="elevation"):
        PhysicalShot5D(1, 90, 75, 0, 0)
    with pytest.raises(ValueError, match="tip"):
        PhysicalShot5D(1, 90, 10, 12, 12)


def test_decoder_requires_sensible_ranges():
    with pytest.raises(ValueError, match="speed"):
        Shot5DDecoder(min_speed=2, max_speed=1)
    with pytest.raises(ValueError, match="elevation"):
        Shot5DDecoder(max_elevation_deg=75)
