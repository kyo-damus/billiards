import numpy as np

from src.goal.encoder import (
    TACTICAL_GOAL_DIM,
    encode_tactical_goal,
)

from src.goal.conditioner import (
    MICRO_GOAL_DIM,
    encode_micro_goal,
)

from src.micro.goal_observation import (
    PHYSICAL_OBSERVATION_DIM,
)

from src.micro.sac import SACAgent


DEFAULT_MICRO_OBS_DIM = 9


class GoalConditionedSACAgent:
    """
    TacticalGoalを条件入力として使用するSAC。

    π(a | observation, goal)
    Q(observation, goal, a)

    内部のSACアルゴリズム自体は既存SACAgentを再利用する。
    """

    def __init__(
        self,
        observation_dim=PHYSICAL_OBSERVATION_DIM,
        goal_dim=MICRO_GOAL_DIM,
        action_dim: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device=None,
    ):
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim

        self.conditioned_state_dim = (
            observation_dim + goal_dim
        )

        self.agent = SACAgent(
            state_dim=self.conditioned_state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            device=device,
        )

    @property
    def device(self):
        return self.agent.device

    @property
    def actor(self):
        return self.agent.actor

    @property
    def critic(self):
        return self.agent.critic

    @property
    def alpha(self):
        return self.agent.alpha

    def build_conditioned_state(
        self,
        observation,
        goal,
    ) -> np.ndarray:
        """
        observation + encoded TacticalGoal
        をNN入力へ変換する。
        """

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (
            self.observation_dim,
        ):
            raise ValueError(
                "observation must have shape "
                f"({self.observation_dim},), "
                f"got {observation.shape}"
            )

        goal_vector = encode_micro_goal(
            observation,
            goal,
        )

        if goal_vector.shape != (
            self.goal_dim,
        ):
            raise ValueError(
                "goal vector dimension mismatch"
            )

        return np.concatenate(
            [
                observation,
                goal_vector,
            ]
        ).astype(np.float32)

    def select_action(
        self,
        observation,
        goal,
        deterministic=False,
    ):
        conditioned_state = (
            self.build_conditioned_state(
                observation,
                goal,
            )
        )

        return self.agent.select_action(
            conditioned_state,
            deterministic=deterministic,
        )

    def update(self, batch):
        """
        batchは既存SACAgentと同じ形式まで
        GoalConditionedReplayBuffer側で変換する。
        """

        return self.agent.update(batch)
        