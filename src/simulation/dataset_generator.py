from dataclasses import dataclass
from math import radians

import numpy as np
import pandas as pd

from src.controllers.mpc_controller import MPCController
from src.models.linear_bicycle import LinearBicycleModel
from src.models.nonlinear_bicycle import (
    NonlinearBicycleModel,
)
from src.models.vehicle_parameters import VehicleParameters
from src.simulation.path_tracking import (
    calculate_tracking_error,
    create_control_state,
    nearest_reference_index,
)
from src.simulation.reference_paths import ReferencePath


FEATURE_COLUMNS = [
    "lateral_velocity",
    "yaw_rate",
    "lateral_error",
    "heading_error",
    "steering",
    "steering_rate",
    "curvature",
    "velocity",
]

TARGET_COLUMNS = [
    "lateral_prediction_error",
    "heading_prediction_error",
]


@dataclass(frozen=True)
class DatasetScenario:
    name: str
    path: ReferencePath
    nominal_parameters: VehicleParameters
    real_parameters: VehicleParameters

    steering_bias: float = 0.0
    measurement_noise_std: tuple[
        float,
        float,
        float,
        float,
    ] = (
        0.02,
        0.005,
        0.005,
        radians(0.05),
    )

    random_seed: int = 42


def generate_scenario_data(
    scenario: DatasetScenario,
    sample_time: float = 0.05,
    prediction_horizon: int = 30,
    maximum_duration: float = 18.0,
) -> pd.DataFrame:
    linear_model = LinearBicycleModel(
        scenario.nominal_parameters
    )

    controller = MPCController(
        model=linear_model,
        sample_time=sample_time,
        prediction_horizon=prediction_horizon,
    )

    vehicle = NonlinearBicycleModel(
        scenario.real_parameters
    )

    random_generator = np.random.default_rng(
        scenario.random_seed
    )

    vehicle_state = np.zeros(5)
    previous_steering = 0.0
    reference_index = 0

    rows: list[dict[str, float | str | int]] = []

    path_spacing = (
        scenario.path.x[1] - scenario.path.x[0]
    )

    travelled_per_step = (
        scenario.nominal_parameters.longitudinal_speed
        * sample_time
    )

    maximum_steps = int(
        maximum_duration / sample_time
    )

    noise_scale = np.asarray(
        scenario.measurement_noise_std,
        dtype=float,
    )

    for step in range(maximum_steps):
        reference_index = nearest_reference_index(
            scenario.path,
            vehicle_state[0],
            vehicle_state[1],
            start_index=reference_index,
        )

        tracking_error = calculate_tracking_error(
            vehicle_state,
            scenario.path,
            reference_index,
        )

        true_control_state = create_control_state(
            vehicle_state,
            tracking_error,
        )

        measured_control_state = (
            true_control_state
            + random_generator.normal(
                loc=0.0,
                scale=noise_scale,
            )
        )

        preview_offsets = np.arange(
            prediction_horizon
        ) * travelled_per_step

        preview_indices = (
            reference_index
            + np.rint(
                preview_offsets / path_spacing
            ).astype(int)
        )

        preview_indices = np.clip(
            preview_indices,
            0,
            len(scenario.path.curvature) - 1,
        )

        curvature_preview = (
            scenario.path.curvature[preview_indices]
        )

        steering = controller.control(
            measured_control_state,
            previous_steering=previous_steering,
            curvature_preview=curvature_preview,
        )

        steering_rate = (
            steering - previous_steering
        ) / sample_time

        current_curvature = float(
            scenario.path.curvature[reference_index]
        )

        nominal_next_state = (
            controller.ad @ measured_control_state
            + controller.bd.flatten() * steering
            + controller.ed.flatten()
            * current_curvature
        )

        effective_steering = (
            steering + scenario.steering_bias
        )

        next_vehicle_state = vehicle.step(
            vehicle_state,
            effective_steering,
            sample_time,
        )

        next_reference_index = nearest_reference_index(
            scenario.path,
            next_vehicle_state[0],
            next_vehicle_state[1],
            start_index=reference_index,
        )

        next_tracking_error = calculate_tracking_error(
            next_vehicle_state,
            scenario.path,
            next_reference_index,
        )

        actual_next_state = create_control_state(
            next_vehicle_state,
            next_tracking_error,
        )

        prediction_error = (
            actual_next_state - nominal_next_state
        )

        rows.append(
            {
                "scenario": scenario.name,
                "step": step,
                "lateral_velocity":
                    measured_control_state[0],
                "yaw_rate":
                    measured_control_state[1],
                "lateral_error":
                    measured_control_state[2],
                "heading_error":
                    measured_control_state[3],
                "steering": steering,
                "steering_rate": steering_rate,
                "curvature": current_curvature,
                "velocity":
                    scenario.nominal_parameters
                    .longitudinal_speed,
                "lateral_prediction_error":
                    prediction_error[2],
                "heading_prediction_error":
                    prediction_error[3],
            }
        )

        vehicle_state = next_vehicle_state
        reference_index = next_reference_index
        previous_steering = steering

        if vehicle_state[0] >= scenario.path.x[-1]:
            break

    return pd.DataFrame(rows)