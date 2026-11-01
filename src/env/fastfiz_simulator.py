from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from src.env.simulator_state import SimulatorSnapshot


# FastFiz TableState precondition/placement bitmask values.
FASTFIZ_PRECONDITION_FLAGS = {
    1: "BAD_A_VAL",
    2: "BAD_B_VAL",
    4: "BAD_THETA_VAL",
    8: "BAD_PHI_VAL",
    16: "BAD_V_VAL",
    32: "BAD_X_VAL",
    64: "BAD_Y_VAL",
    128: "CUE_STICK_COLLISION",
    256: "BALL_OVERLAP",
}


def describe_fastfiz_precondition(code: int) -> str:
    code = int(code)
    if code == 0:
        return "OK_PRECONDITION"
    flags = [
        name
        for bit, name in FASTFIZ_PRECONDITION_FLAGS.items()
        if code & bit
    ]
    unrecognized = code & ~sum(FASTFIZ_PRECONDITION_FLAGS)
    if unrecognized:
        flags.append(f"UNKNOWN_BITS({unrecognized})")
    return "|".join(flags) if flags else f"UNKNOWN({code})"


class FastFizPreconditionError(RuntimeError):
    """FastFiz rejected a shot before execution; original codes retained."""

    def __init__(self, precondition_code, placement_code):
        self.precondition_code = int(precondition_code)
        self.placement_code = int(placement_code)
        super().__init__(
            "FastFiz rejected this physical shot: "
            f"precondition={self.precondition_code} "
            f"({describe_fastfiz_precondition(self.precondition_code)}); "
            f"ball_placement={self.placement_code} "
            f"({describe_fastfiz_precondition(self.placement_code)})"
        )


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

    # -1=no cue/object collision, 0/1=first contacted object ball.
    first_contact_ball: int = -1

    # FastFiz bitmasks before this shot (0 = OK).
    precondition_code: int = 0
    placement_code: int = 0


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
        tip_x_mm: float = 0.0,
        tip_y_mm: float = 0.0,
    ) -> SimulatorStepResult:

        offsets = np.asarray(
            [tip_x_mm, tip_y_mm],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(offsets))
            or float(np.dot(offsets, offsets)) > 144.0
        ):
            raise ValueError(
                "Tip offset must be within a 12 mm radius."
            )

        # Fail loudly if the native extension was not rebuilt after
        # enabling precondition diagnostics; never fabricate code 0.
        if not (
            hasattr(self._sim, "get_last_precondition_code")
            and hasattr(self._sim, "get_ball_placement_code")
        ):
            raise RuntimeError(
                "billiard_env_cpp is outdated. Rebuild it with "
                "'cmake --build build -j4' before using FastFiz diagnostics."
            )

        placement_code_before = int(
            self._sim.get_ball_placement_code()
        )
        result = self._sim.step(
            int(target_ball),
            float(power),
            float(angle),
            float(tip_x_mm),
            float(tip_y_mm),
        )

        if len(result) != 8:
            raise RuntimeError(
                f"Unexpected simulator output length: {len(result)}"
            )

        return SimulatorStepResult(
            state=np.asarray(result[:6], dtype=np.float32),
            legacy_reward=float(result[6]),
            legacy_done=bool(result[7]),
            first_contact_ball=int(
                self._sim.get_last_first_contact_ball()
            ),
            precondition_code=int(
                self._sim.get_last_precondition_code()
            ),
            placement_code=placement_code_before,
        )