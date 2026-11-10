"""Native tests: build the opt-in FastFiz extension before running these."""

import numpy as np
import pytest

from src.env.multi_ball_fastfiz import MultiBallFastFizSimulator
from src.game.rack_state import RackState, TableGameConfig
from src.micro.shot_control import PhysicalShot5D


def make_rack(config):
    n = len(config.ball_numbers)
    positions = [[0.55, 0.35]] + [
        [0.20 + 0.23 * (i % 3), 1.14 + 0.19 * (i // 3)]
        for i in range(n - 1)
    ]
    return RackState(
        config=config,
        ball_positions=np.asarray(positions, dtype=np.float32),
        ball_pocket_indices=np.full(n, -1, dtype=np.int32),
        current_player=0,
        score=np.zeros(2, dtype=np.float32),
    )


@pytest.mark.parametrize("config", [
    TableGameConfig.two_object_training(),
    TableGameConfig.nine_ball_inventory(),
])
def test_native_multiball_roundtrip_and_shot(config):
    sim = MultiBallFastFizSimulator(config)
    rack = make_rack(config)
    sim.restore(rack)
    xy, pocket = sim.snapshot()
    np.testing.assert_allclose(xy, rack.ball_positions, atol=1e-6)
    np.testing.assert_array_equal(pocket, rack.ball_pocket_indices)
    result = sim.step(
        target_ball=config.object_ball_numbers[0],
        shot=PhysicalShot5D(1.0, 90.0, 10.0, 0.0, 0.0),
    )
    assert result.positions.shape == (len(config.ball_numbers), 2)
    assert result.pocket_indices.shape == (len(config.ball_numbers),)
    assert result.first_contact_ball is None or (
        result.first_contact_ball in config.object_ball_numbers
    )
    assert isinstance(result.events, tuple)
    sim.restore(rack)
    xy2, pocket2 = sim.snapshot()
    np.testing.assert_allclose(xy2, rack.ball_positions, atol=1e-6)
    np.testing.assert_array_equal(pocket2, rack.ball_pocket_indices)


def test_native_rejects_target_ball_not_in_inventory():
    config = TableGameConfig("subset", (2, 5, 9))
    sim = MultiBallFastFizSimulator(config)
    sim.restore(make_rack(config))
    with pytest.raises(ValueError, match="Target"):
        sim.step(
            target_ball=1,
            shot=PhysicalShot5D(1.0, 90.0, 10.0, 0, 0),
        )
