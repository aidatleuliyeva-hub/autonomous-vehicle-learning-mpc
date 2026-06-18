import numpy as np

from src.controllers.learning_enhanced_mpc import (
    LearningEnhancedMPC,
)
from src.controllers.mpc_controller import MPCController


class ConstantErrorPredictor:
    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return np.tile(
            self.prediction,
            (len(features), 1),
        )


def test_learning_mpc_respects_constraints() -> None:
    mpc = MPCController()

    predictor = ConstantErrorPredictor(
        np.array([0.1, 0.05])
    )

    controller = LearningEnhancedMPC(
        mpc,
        predictor,
    )

    previous_steering = 0.0

    steering = controller.control(
        state=np.array([0.0, 0.0, 2.0, 0.2]),
        previous_steering=previous_steering,
        curvature_preview=np.zeros(
            mpc.prediction_horizon
        ),
        current_curvature=0.0,
    )

    rate_limit = (
        mpc.model.p.max_steering_rate
        * mpc.sample_time
    )

    assert (
        abs(steering - previous_steering)
        <= rate_limit + 1e-5
    )

    assert (
        abs(steering)
        <= mpc.model.p.max_steering_angle
    )


def test_positive_error_creates_negative_compensation() -> None:
    mpc = MPCController()

    predictor = ConstantErrorPredictor(
        np.array([0.01, 0.005])
    )

    controller = LearningEnhancedMPC(
        mpc,
        predictor,
        filter_coefficient=1.0,
    )

    controller.control(
        state=np.zeros(4),
        previous_steering=0.0,
        curvature_preview=np.zeros(
            mpc.prediction_horizon
        ),
        current_curvature=0.0,
    )

    assert controller.last_compensation < 0.0