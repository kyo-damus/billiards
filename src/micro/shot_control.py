"""Five physically independent FastFiz shot controls, not a trained 5-D SAC.

Decoded controls: speed, horizontal angle, cue elevation, left/right cue-tip
offset and up/down cue-tip offset. The reference angle may come from Macro.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PhysicalShot5D:
    speed_m_s: float
    azimuth_deg: float
    elevation_deg: float
    tip_x_mm: float
    tip_y_mm: float

    def __post_init__(self):
        values = (
            self.speed_m_s, self.azimuth_deg, self.elevation_deg,
            self.tip_x_mm, self.tip_y_mm,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Shot parameters must be finite.")
        if not 0.0 <= self.elevation_deg <= 70.0:
            raise ValueError("FastFiz cue elevation must be in [0,70] degrees.")
        if self.speed_m_s <= 0.0:
            raise ValueError("Shot speed must be positive.")
        if self.tip_x_mm ** 2 + self.tip_y_mm ** 2 > 144.0 + 1e-6:
            raise ValueError("Cue-tip offsets must be inside the 12 mm disk.")

    def to_fastfiz_kwargs(self):
        """Pass separately from the diagnostic target_ball argument."""
        return {
            "power": self.speed_m_s,
            "angle": self.azimuth_deg,
            "cue_elevation_deg": self.elevation_deg,
            "tip_x_mm": self.tip_x_mm,
            "tip_y_mm": self.tip_y_mm,
        }


@dataclass(frozen=True)
class Shot5DDecoder:
    min_speed: float = 0.5
    max_speed: float = 3.0
    max_angle_offset_deg: float = 10.0
    min_elevation_deg: float = 0.0
    max_elevation_deg: float = 30.0
    tip_radius_mm: float = 12.0

    def __post_init__(self):
        v = (
            self.min_speed, self.max_speed,
            self.max_angle_offset_deg,
            self.min_elevation_deg, self.max_elevation_deg,
            self.tip_radius_mm,
        )
        if not all(math.isfinite(x) for x in v):
            raise ValueError("Decoder bounds must be finite.")
        if self.min_speed <= 0 or self.min_speed >= self.max_speed:
            raise ValueError("Invalid speed bounds.")
        if self.max_angle_offset_deg <= 0:
            raise ValueError("max_angle_offset_deg must be positive.")
        if not (0 <= self.min_elevation_deg < self.max_elevation_deg <= 70):
            raise ValueError("Invalid elevation bounds.")
        if not 0 < self.tip_radius_mm <= 12:
            raise ValueError("tip_radius_mm must be in (0,12].")

    def decode(self, normalized_action, reference_angle_deg: float) -> PhysicalShot5D:
        """Map normalized [-1,1]^5 into a physically valid FastFiz shot.

        Tip x/y are projected onto a disk to avoid physically invalid
        diagonal corner offsets. This decoder is versioned separately
        from the existing 2-D Micro checkpoint and does not load it.
        """
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (5,):
            raise ValueError(f"Expected normalized 5-D action, got {action.shape}")
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1.0):
            raise ValueError("5-D action must be finite and in [-1,1].")
        if not math.isfinite(reference_angle_deg):
            raise ValueError("Reference angle must be finite.")
        speed = self.min_speed + (
            (float(action[0]) + 1.0) * 0.5
            * (self.max_speed - self.min_speed)
        )
        angle = reference_angle_deg + (
            float(action[1]) * self.max_angle_offset_deg
        )
        elevation = self.min_elevation_deg + (
            (float(action[2]) + 1.0) * 0.5
            * (self.max_elevation_deg - self.min_elevation_deg)
        )
        tip = np.asarray(action[3:5], dtype=np.float64).copy()
        norm = float(np.linalg.norm(tip))
        if norm > 1.0:
            tip /= norm
        # FastFiz's pybind11 interface casts doubles to 32-bit floats.
        # At exactly 12 mm, float32 rounding can make a*a+b*b > 144.0f,
        # which the native simulator rejects with "Invalid shot parameters".
        # Stay just inside the disk for *all* normalized actor outputs,
        # including diagonal corner actions during random warmup.
        native_safe_radius = min(self.tip_radius_mm, 11.99)
        return PhysicalShot5D(
            speed_m_s=speed,
            azimuth_deg=angle,
            elevation_deg=elevation,
            tip_x_mm=float(tip[0]) * native_safe_radius,
            tip_y_mm=float(tip[1]) * native_safe_radius,
        )


# Keep exploration around the apex-ball line while allowing all five controls.
NINE_BALL_BREAK_DECODER = Shot5DDecoder(
    min_speed=2.0, max_angle_offset_deg=2.0, max_elevation_deg=15.0, tip_radius_mm=6.0,
)
