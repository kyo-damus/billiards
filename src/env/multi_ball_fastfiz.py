"""Optional multi-ball native FastFiz API; no game rules or learned policy."""

from dataclasses import dataclass

import numpy as np

from src.env.fastfiz_simulator import billiard_env_cpp
from src.game.rack_state import RackState, TableGameConfig
from src.micro.shot_control import PhysicalShot5D


FASTFIZ_EVENTS = {
    0: "NO_EVENT", 1: "STATE_CHANGE", 2: "BALL_COLLISION",
    3: "RAIL_COLLISION", 4: "POCKETED", 5: "CUE_STRIKE",
    6: "MISCUE", 7: "UNKNOWN_EVENT",
}


@dataclass(frozen=True)
class PhysicalEvent:
    time_s: float
    kind: str
    ball1: int
    ball2: int | None


@dataclass(frozen=True)
class PhysicalStep:
    positions: np.ndarray
    pocket_indices: np.ndarray
    first_contact_ball: int | None
    events: tuple[PhysicalEvent, ...]
    precondition_code: int
    placement_code: int

    @property
    def valid_shot(self):
        return self.precondition_code == 0


class MultiBallFastFizSimulator:
    """Ball IDs: 0=cue, objects=their actual numbers, 1 through 15."""

    def __init__(self, config: TableGameConfig):
        if not hasattr(billiard_env_cpp, "BilliardSimulatorMulti"):
            raise RuntimeError(
                "Updated native BilliardSimulatorMulti is unavailable. "
                "Compile cpp_simulator.cpp in a separate CMake build directory "
                "and set BILLIARDS_FASTFIZ_BUILD_DIR before starting Python."
            )
        if any(number > 15 for number in config.ball_numbers):
            raise ValueError("FastFiz supports numbered object balls 1..15.")
        self.config = config
        self.native = billiard_env_cpp.BilliardSimulatorMulti(list(config.ball_numbers))

    def restore(self, state: RackState):
        if state.config != self.config:
            raise ValueError("Rack configuration does not match simulator.")
        self.native.set_full_state(
            state.ball_positions.reshape(-1).tolist(),
            state.ball_pocket_indices.tolist(),
        )

    def snapshot(self):
        n = len(self.config.ball_numbers)
        position = np.asarray(self.native.get_state(), dtype=np.float32).reshape(n, 2)
        pockets = np.asarray([
            self.native.get_pocket_index(int(ball_number))
            for ball_number in self.config.ball_numbers
        ], dtype=np.int32)
        return position.copy(), pockets.copy()

    def step(self, *, target_ball: int, shot: PhysicalShot5D) -> PhysicalStep:
        if target_ball not in self.config.object_ball_numbers:
            raise ValueError("Target must be a configured object ball.")
        if not isinstance(shot, PhysicalShot5D):
            raise TypeError("shot must be a PhysicalShot5D.")
        placement = int(self.native.get_ball_placement_code())
        raw = np.asarray(
            self.native.step(int(target_ball), **shot.to_fastfiz_kwargs()),
            dtype=np.float32,
        )
        n = len(self.config.ball_numbers)
        if raw.shape != (2 * n + 2,):
            raise RuntimeError(f"Native multiball output shape mismatch: {raw.shape}")
        pos, pockets = self.snapshot()
        first = int(self.native.get_last_first_contact_ball())
        events = tuple(
            PhysicalEvent(
                time_s=float(t),
                kind=FASTFIZ_EVENTS.get(int(typ), f"UNKNOWN_{typ}"),
                ball1=int(a),
                ball2=None if int(b) == 16 else int(b),
            )
            for t, typ, a, b in self.native.get_last_events()
        )
        return PhysicalStep(
            positions=pos,
            pocket_indices=pockets,
            first_contact_ball=None if first < 0 else first,
            events=events,
            precondition_code=int(self.native.get_last_precondition_code()),
            placement_code=placement,
        )
