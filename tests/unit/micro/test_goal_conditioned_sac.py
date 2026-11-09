import numpy as np

from src.goal.encoder import (
    TACTICAL_GOAL_DIM,
)

from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)

from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)

from src.micro.goal_replay_buffer import (
    GoalConditionedReplayBuffer,
)

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
    build_physical_observation,
)

OBS_DIM = PHYSICAL_OBSERVATION_DIM
ACTION_DIM = 2

def make_goal():
    return TacticalGoal(
        goal_type=(
            TacticalGoalType.DIRECT_ATTACK
        ),
        target_ball=0,
        target_pocket=3,
    )


def test_conditioned_state_shape():
    agent = GoalConditionedSACAgent(
        observation_dim=OBS_DIM,
        device="cpu",
    )

    observation = make_observation()

    state = agent.build_conditioned_state(
        observation,
        make_goal(),
    )

    assert state.shape == (
        agent.conditioned_state_dim,
    )

def test_goal_conditioned_select_action():
    agent = GoalConditionedSACAgent(
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device="cpu",
    )

    observation = make_observation()

    action = agent.select_action(
        observation,
        make_goal(),
        deterministic=True,
    )

    assert action.shape == (
        ACTION_DIM,
    )

    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

def test_goal_replay_buffer():
    buffer = GoalConditionedReplayBuffer(
        capacity=100,
    )

    observation = make_observation()

    next_observation = make_observation(
        cue=(0.75, 1.10),
        ball0=(0.35, 1.118),
        ball1=(0.80, 0.40),
    )

    buffer.push(
        observation=observation,
        goal=make_goal(),
        action=np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        ),
        reward=1.0,
        next_observation=next_observation,
        done=True,
    )

    assert len(buffer) == 1


def test_goal_conditioned_sac_update():
    agent = GoalConditionedSACAgent(
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        device="cpu",
    )

    buffer = GoalConditionedReplayBuffer(
        capacity=100,
    )

    rng = np.random.default_rng(0)

    for i in range(20):

        cue = (
            float(rng.uniform(0.6, 0.9)),
            float(rng.uniform(0.8, 1.4)),
        )

        ball0 = (
            float(rng.uniform(0.25, 0.5)),
            float(rng.uniform(0.8, 1.4)),
        )

        ball1 = (
            float(rng.uniform(0.6, 0.9)),
            float(rng.uniform(0.3, 0.7)),
        )

        observation = make_observation(
            cue=cue,
            ball0=ball0,
            ball1=ball1,
        )

        next_observation = make_observation(
            cue=(
                cue[0] - 0.01,
                cue[1],
            ),
            ball0=ball0,
            ball1=ball1,
        )

        action = rng.uniform(
            -1.0,
            1.0,
            size=ACTION_DIM,
        ).astype(np.float32)

        buffer.push(
            observation=observation,
            goal=make_goal(),
            action=action,
            reward=float(i % 2),
            next_observation=next_observation,
            done=True,
        )

    batch = buffer.sample(
        batch_size=8,
        device=agent.device,
    )

    metrics = agent.update(batch)

    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "alpha" in metrics
    
def make_observation(
    cue=(0.80, 1.118),
    ball0=(0.40, 1.118),
    ball1=(0.80, 0.40),
):
    positions = np.array(
        [
            cue,
            ball0,
            ball1,
        ],
        dtype=np.float32,
    )

    pocket_indices = np.array(
        [-1, -1, -1],
        dtype=np.int32,
    )

    return build_physical_observation(
        positions,
        pocket_indices,
    )