import numpy as np

from src.env.fastfiz_simulator import FastFizSimulator
from src.env.simulator_state import SimulatorSnapshot


def test_get_and_set_state():
    sim = FastFizSimulator()

    state = sim.reset()

    assert state.shape == (6,)

    sim.set_state(state)

    restored_state = sim.get_state()

    np.testing.assert_allclose(state, restored_state)


def test_step():
    sim = FastFizSimulator()

    sim.reset()

    result = sim.step(
        target_ball=0,
        power=0.5,
        angle=90.0,
    )

    assert result.state.shape == (6,)

def test_ball_is_not_pocketed_after_reset():
    sim = FastFizSimulator()

    sim.reset()

    assert sim.get_pocket_index(-1) == -1
    assert sim.get_pocket_index(0) == -1
    assert sim.get_pocket_index(1) == -1

def test_snapshot_restore():
    sim = FastFizSimulator()

    sim.reset()

    snapshot = sim.snapshot()

    changed_state = np.array(
        [
            0.3, 0.3,
            0.4, 0.8,
            0.8, 1.5,
        ],
        dtype=np.float32,
    )

    sim.set_state(changed_state)

    sim.restore(snapshot)

    restored = sim.snapshot()

    np.testing.assert_allclose(
        restored.positions,
        snapshot.positions,
    )

    np.testing.assert_array_equal(
        restored.pocket_indices,
        snapshot.pocket_indices,
    )

def test_restore_pocketed_ball():
    sim = FastFizSimulator()

    snapshot = SimulatorSnapshot(
        positions=np.array(
            [
                [0.5, 0.5],
                [0.04, 0.04],
                [0.8, 1.5],
            ],
            dtype=np.float32,
        ),

        pocket_indices=np.array(
            [
                -1,  # cue
                 0,  # obj0 = SW pocket
                -1,  # obj1
            ],
            dtype=np.int32,
        ),
    )

    sim.restore(snapshot)

    assert sim.get_pocket_index(-1) == -1
    assert sim.get_pocket_index(0) == 0
    assert sim.get_pocket_index(1) == -1
