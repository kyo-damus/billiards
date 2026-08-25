from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from src.env.simulator_state import SimulatorSnapshot


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

    def get_pocket_index(self, ball_id: int) -> int:
        """
        球が入ったポケット番号を返す。

        -1: ポケットされていない
        0: SW
        1: W
        2: NW
        3: NE
        4: E
        5: SE

        ball_id:
            -1: cue ball
            0: object ball 0
            1: object ball 1
        """
        return int(
            self._sim.get_pocket_index(int(ball_id))
        )

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32)

        if state.shape != (6,):
            raise ValueError(
                f"state must have shape (6,), got {state.shape}"
            )

        self._sim.set_state(state.tolist())

    def snapshot(self) -> SimulatorSnapshot:
        positions = self.get_state().reshape(3, 2)

        pocket_indices = np.array(
            [
                self.get_pocket_index(-1),
                self.get_pocket_index(0),
                self.get_pocket_index(1),
            ],
            dtype=np.int32,
        )

        return SimulatorSnapshot(
            positions=positions.copy(),
            pocket_indices=pocket_indices,
        )


    def restore(
        self,
        snapshot: SimulatorSnapshot,
    ) -> None:

        self._sim.set_full_state(
            snapshot.positions.reshape(-1).tolist(),
            snapshot.pocket_indices.tolist(),
        )

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