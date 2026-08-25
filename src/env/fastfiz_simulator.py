from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


# CMakeで生成された billiard_env_cpp を読み込む
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = PROJECT_ROOT / "build"

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import billiard_env_cpp


@dataclass(frozen=True)
class SimulatorStepResult:
    """FastFizによる1ショットの物理計算結果."""

    state: np.ndarray

    # 現C++実装との互換用。
    # RLのreward/terminationには将来的に使用しない。
    legacy_reward: float
    legacy_done: bool


class FastFizSimulator:
    """
    Python側からFastFizを扱うためのラッパー。

    RL固有の処理（reward, observation変換など）は
    このクラスには持たせない。
    """

    def __init__(self):
        self._sim = billiard_env_cpp.BilliardSimulator()

    def reset(self) -> np.ndarray:
        self._sim.reset()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.asarray(
            self._sim.get_state(),
            dtype=np.float32,
        )

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)

        if state.shape != (6,):
            raise ValueError(
                f"state must have shape (6,), got {state.shape}"
            )

        self._sim.set_state(state.tolist())

    def step(
        self,
        target_ball: int,
        power: float,
        angle: float,
    ) -> SimulatorStepResult:

        result = self._sim.step(
            int(target_ball),
            float(power),
            float(angle),
        )

        if len(result) != 8:
            raise RuntimeError(
                f"Unexpected simulator output length: {len(result)}"
            )

        return SimulatorStepResult(
            state=np.asarray(result[:6], dtype=np.float32),
            legacy_reward=float(result[6]),
            legacy_done=bool(result[7]),
        )