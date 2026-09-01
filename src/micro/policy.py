from typing import Protocol

import numpy as np

from src.goal.types import TacticalGoal


class GoalConditionedMicroPolicy(Protocol):

    def select_action(
        self,
        observation,
        goal: TacticalGoal,
        deterministic: bool = True,
    ) -> np.ndarray:
        ...
        