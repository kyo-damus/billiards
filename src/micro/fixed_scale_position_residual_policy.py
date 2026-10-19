import numpy as np

from src.micro.position_residual_policy import (
    PositionResidualMicroPolicy,
    compose_residual_action,
)


class FixedScalePositionResidualPolicy:
    """
    Wrap a trained PositionResidualMicroPolicy and apply
    a fixed scale to the learned residual at inference.

    The recommended current setting is power-only residual
    with residual_scale=0.75 and angle_scale=0.0.
    """

    def __init__(
        self,
        residual_policy,
        residual_scale=0.75,
        power_scale=1.0,
        angle_scale=0.0,
    ):
        if not isinstance(
            residual_policy,
            PositionResidualMicroPolicy,
        ):
            raise TypeError(
                "residual_policy must be a "
                "PositionResidualMicroPolicy."
            )

        self.residual_policy = residual_policy
        self.base_agent = (
            residual_policy.base_agent
        )

        self.residual_scale = float(
            residual_scale
        )
        self.power_scale = float(
            power_scale
        )
        self.angle_scale = float(
            angle_scale
        )

        if self.residual_scale < 0.0:
            raise ValueError(
                "residual_scale must be >= 0."
            )
        if self.power_scale < 0.0:
            raise ValueError(
                "power_scale must be >= 0."
            )
        if self.angle_scale < 0.0:
            raise ValueError(
                "angle_scale must be >= 0."
            )

        self.last_selection = None
        self.direct_call_count = 0
        self.position_call_count = 0

    def select_action(
        self,
        observation,
        goal,
        deterministic=True,
    ):
        action = (
            self.residual_policy.select_action(
                observation,
                goal,
                deterministic=deterministic,
            )
        )

        selection = (
            self.residual_policy.last_selection
        )

        if selection is None:
            self.direct_call_count += 1
            self.last_selection = None
            return np.asarray(
                action,
                dtype=np.float32,
            )

        self.position_call_count += 1

        applied_residual = np.asarray(
            selection.residual_action,
            dtype=np.float32,
        ).copy()

        applied_residual[0] *= (
            self.power_scale
        )
        applied_residual[1] *= (
            self.angle_scale
        )
        applied_residual *= (
            self.residual_scale
        )

        final_action = (
            compose_residual_action(
                selection.base_action,
                applied_residual,
            )
        )

        self.last_selection = selection

        return final_action
