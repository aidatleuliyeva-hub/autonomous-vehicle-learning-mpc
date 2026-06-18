from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.models.vehicle_parameters import VehicleParameters


Array = NDArray[np.float64]


class PolicyPredictor(Protocol):
    def predict(self, features: Array) -> Array:
        ...


class ELMPolicyController:
    """Fast ELM approximation of the MPC control policy."""

    def __init__(
        self,
        predictor: PolicyPredictor,
        parameters: VehicleParameters,
        sample_time: float = 0.05,
    ) -> None:
        self.predictor = predictor
        self.parameters = parameters
        self.sample_time = sample_time

        self.last_raw_prediction = 0.0
        self.last_command = 0.0

    def control(
        self,
        state: Array,
        previous_steering: float,
        curvature_preview: Array,
        current_curvature: float,
    ) -> float:
        state = np.asarray(state, dtype=float)
        curvature_preview = np.asarray(
            curvature_preview,
            dtype=float,
        )

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")

        if curvature_preview.ndim != 1:
            raise ValueError(
                "curvature_preview must be one-dimensional"
            )

        preview_5 = curvature_preview[
            min(5, len(curvature_preview) - 1)
        ]
        preview_10 = curvature_preview[
            min(10, len(curvature_preview) - 1)
        ]
        preview_20 = curvature_preview[
            min(20, len(curvature_preview) - 1)
        ]

        features = np.array(
            [
                state[0],
                state[1],
                state[2],
                state[3],
                previous_steering,
                current_curvature,
                preview_5,
                preview_10,
                preview_20,
                self.parameters.longitudinal_speed,
            ],
            dtype=float,
        ).reshape(1, -1)

        raw_prediction = float(
            self.predictor.predict(features)[0]
        )

        rate_step_limit = (
            self.parameters.max_steering_rate
            * self.sample_time
        )

        command = float(
            np.clip(
                raw_prediction,
                previous_steering - rate_step_limit,
                previous_steering + rate_step_limit,
            )
        )

        command = float(
            np.clip(
                command,
                -self.parameters.max_steering_angle,
                self.parameters.max_steering_angle,
            )
        )

        self.last_raw_prediction = raw_prediction
        self.last_command = command

        return command