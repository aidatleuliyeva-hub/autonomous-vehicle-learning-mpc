from math import radians
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.controllers.mpc_controller import MPCController


Array = NDArray[np.float64]


class ErrorPredictor(Protocol):
    def predict(self, features: Array) -> Array:
        ...


class LearningEnhancedMPC:
    """MPC with safe ELM predictive-error compensation."""

    def __init__(
        self,
        mpc: MPCController,
        error_predictor: ErrorPredictor,
        lateral_gain: float = 0.5,
        heading_gain: float = 0.2,
        correction_limit: float = radians(0.25),
        filter_coefficient: float = 0.15,
        ood_soft_limit: float = 3.0,
        ood_hard_limit: float = 5.0,
    ) -> None:
        self.mpc = mpc
        self.error_predictor = error_predictor

        self.lateral_gain = lateral_gain
        self.heading_gain = heading_gain
        self.correction_limit = correction_limit
        self.filter_coefficient = filter_coefficient

        self.ood_soft_limit = ood_soft_limit
        self.ood_hard_limit = ood_hard_limit

        self.filtered_prediction = np.zeros(2)

        self.last_nominal_steering = 0.0
        self.last_compensation = 0.0
        self.last_prediction = np.zeros(2)
        self.last_ood_weight = 1.0

    def reset(self) -> None:
        self.filtered_prediction = np.zeros(2)

        self.last_nominal_steering = 0.0
        self.last_compensation = 0.0
        self.last_prediction = np.zeros(2)
        self.last_ood_weight = 1.0

    def _clip_prediction(
        self,
        prediction: Array,
    ) -> Array:
        output_scaler = getattr(
            self.error_predictor,
            "output_scaler",
            None,
        )

        if output_scaler is None:
            return prediction

        mean = np.asarray(
            output_scaler.mean_,
            dtype=float,
        )

        scale = np.asarray(
            output_scaler.scale_,
            dtype=float,
        )

        return np.clip(
            prediction,
            mean - 4.0 * scale,
            mean + 4.0 * scale,
        )

    def _calculate_ood_weight(
        self,
        features: Array,
    ) -> float:
        input_scaler = getattr(
            self.error_predictor,
            "input_scaler",
            None,
        )

        if input_scaler is None:
            return 1.0

        standardized_features = (
            input_scaler.transform(features)
        )

        maximum_distance = float(
            np.max(
                np.abs(standardized_features)
            )
        )

        if maximum_distance <= self.ood_soft_limit:
            return 1.0

        if maximum_distance >= self.ood_hard_limit:
            return 0.0

        return (
            self.ood_hard_limit - maximum_distance
        ) / (
            self.ood_hard_limit
            - self.ood_soft_limit
        )

    def control(
        self,
        state: Array,
        previous_steering: float,
        curvature_preview: Array,
        current_curvature: float,
    ) -> float:
        state = np.asarray(state, dtype=float)

        nominal_steering = self.mpc.control(
            state,
            previous_steering=previous_steering,
            curvature_preview=curvature_preview,
        )

        steering_rate = (
            nominal_steering - previous_steering
        ) / self.mpc.sample_time

        features = np.array(
            [
                state[0],
                state[1],
                state[2],
                state[3],
                nominal_steering,
                steering_rate,
                current_curvature,
                self.mpc.model.p.longitudinal_speed,
            ],
            dtype=float,
        ).reshape(1, -1)

        prediction = np.asarray(
            self.error_predictor.predict(features)[0],
            dtype=float,
        )

        prediction = self._clip_prediction(
            prediction
        )

        ood_weight = self._calculate_ood_weight(
            features
        )

        self.filtered_prediction = (
            self.filter_coefficient * prediction
            + (1.0 - self.filter_coefficient)
            * self.filtered_prediction
        )

        compensation = -ood_weight * (
            self.lateral_gain
            * self.filtered_prediction[0]
            + self.heading_gain
            * self.filtered_prediction[1]
        )

        compensation = float(
            np.clip(
                compensation,
                -self.correction_limit,
                self.correction_limit,
            )
        )

        enhanced_steering = (
            nominal_steering + compensation
        )

        steering_limit = (
            self.mpc.model.p.max_steering_angle
        )

        rate_step_limit = (
            self.mpc.model.p.max_steering_rate
            * self.mpc.sample_time
        )

        enhanced_steering = float(
            np.clip(
                enhanced_steering,
                previous_steering - rate_step_limit,
                previous_steering + rate_step_limit,
            )
        )

        enhanced_steering = float(
            np.clip(
                enhanced_steering,
                -steering_limit,
                steering_limit,
            )
        )

        self.last_nominal_steering = nominal_steering

        self.last_compensation = (
            enhanced_steering
            - nominal_steering
        )

        self.last_prediction = prediction.copy()
        self.last_ood_weight = ood_weight

        return enhanced_steering