import numpy as np

from src.controllers.elm_policy_controller import (
    ELMPolicyController,
)
from src.models.vehicle_parameters import VehicleParameters


class ConstantPolicy:
    def __init__(self, command: float) -> None:
        self.command = command

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(features), self.command)


def test_policy_controller_respects_rate_limit() -> None:
    parameters = VehicleParameters()

    controller = ELMPolicyController(
        predictor=ConstantPolicy(1.0),
        parameters=parameters,
    )

    command = controller.control(
        state=np.zeros(4),
        previous_steering=0.0,
        curvature_preview=np.zeros(30),
        current_curvature=0.0,
    )

    allowed_change = (
        parameters.max_steering_rate
        * controller.sample_time
    )

    assert abs(command) <= allowed_change + 1e-12


def test_policy_controller_respects_angle_limit() -> None:
    parameters = VehicleParameters()

    controller = ELMPolicyController(
        predictor=ConstantPolicy(10.0),
        parameters=parameters,
        sample_time=1.0,
    )

    command = controller.control(
        state=np.zeros(4),
        previous_steering=0.0,
        curvature_preview=np.zeros(30),
        current_curvature=0.0,
    )

    assert (
        abs(command)
        <= parameters.max_steering_angle
    )