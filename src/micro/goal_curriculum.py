from dataclasses import dataclass


@dataclass
class MixedGoalCurriculum:
    """
    DIRECT_ATTACKからPOSITION_ATTACKへ
    段階的に学習対象を広げる。

    同時にcue target regionも
    徐々に狭くする。
    """

    start_position_probability: float = 0.20
    end_position_probability: float = 0.50

    start_target_radius: float = 0.20
    end_target_radius: float = 0.10

    ramp_steps: int = 10_000

    def values(
        self,
        step: int,
    ) -> tuple[float, float]:

        if self.ramp_steps <= 0:
            ratio = 1.0

        else:
            ratio = min(
                max(
                    step / self.ramp_steps,
                    0.0,
                ),
                1.0,
            )

        position_probability = (
            self.start_position_probability
            + ratio
            * (
                self.end_position_probability
                - self.start_position_probability
            )
        )

        target_radius = (
            self.start_target_radius
            + ratio
            * (
                self.end_target_radius
                - self.start_target_radius
            )
        )

        return (
            float(position_probability),
            float(target_radius),
        )
        