import numpy as np

from src.env.fastfiz_simulator import FastFizSimulator


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