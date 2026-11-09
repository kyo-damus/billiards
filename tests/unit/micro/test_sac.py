import numpy as np

from src.micro.replay_buffer import ReplayBuffer
from src.micro.sac import SACAgent


STATE_DIM = 12
ACTION_DIM = 2


def test_select_action():

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device="cpu",
    )

    state = np.zeros(
        STATE_DIM,
        dtype=np.float32,
    )

    action = agent.select_action(state)

    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_sac_update():

    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device="cpu",
    )

    buffer = ReplayBuffer(capacity=100)

    for _ in range(20):

        state = np.random.randn(
            STATE_DIM
        ).astype(np.float32)

        action = np.random.uniform(
            -1.0,
            1.0,
            size=ACTION_DIM,
        ).astype(np.float32)

        next_state = np.random.randn(
            STATE_DIM
        ).astype(np.float32)

        buffer.push(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    batch = buffer.sample(
        batch_size=8,
        device=agent.device,
    )

    metrics = agent.update(batch)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "alpha" in metrics