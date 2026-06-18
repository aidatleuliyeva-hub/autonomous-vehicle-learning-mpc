import json
from dataclasses import dataclass
from math import radians
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.controllers.learning_enhanced_mpc import (
    LearningEnhancedMPC,
)
from src.controllers.mpc_controller import MPCController
from src.learning.extreme_learning_machine import (
    ExtremeLearningMachineRegressor,
)
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
from src.simulation.reference_paths import (
    ReferencePath,
    sinusoidal_path,
)


@dataclass
class SimulationResult:
    vehicle_states: np.ndarray
    control_states: np.ndarray
    steering: np.ndarray
    compensation: np.ndarray
    computation_times: np.ndarray
    sample_time: float

    @property
    def time(self) -> np.ndarray:
        return (
            np.arange(len(self.vehicle_states))
            * self.sample_time
        )


def run_simulation(
    path: ReferencePath,
    nominal_parameters: VehicleParameters,
    real_parameters: VehicleParameters,
    error_predictor: (
        ExtremeLearningMachineRegressor | None
    ),
    sample_time: float = 0.05,
    maximum_duration: float = 18.0,
    compensator_parameters: (
        dict[str, float] | None
    ) = None,
) -> SimulationResult:
    linear_model = LinearBicycleModel(
        nominal_parameters
    )

    mpc = MPCController(
        model=linear_model,
        sample_time=sample_time,
        prediction_horizon=30,
    )

    learning_controller = None

    if error_predictor is not None:
        parameters = compensator_parameters or {}

        learning_controller = LearningEnhancedMPC(
            mpc=mpc,
            error_predictor=error_predictor,
            **parameters,
        )

    vehicle = NonlinearBicycleModel(
        real_parameters
    )

    vehicle_state = np.array(
        [
            path.x[0],
            path.y[0],
            path.heading[0],
            0.0,
            0.0,
        ],
        dtype=float,
    )
    previous_steering = 0.0
    reference_index = 0

    steering_bias = radians(0.45)

    vehicle_states = []
    control_states = []
    steering_commands = []
    compensations = []
    computation_times = []

    path_spacing = path.x[1] - path.x[0]

    travelled_per_step = (
        nominal_parameters.longitudinal_speed
        * sample_time
    )

    maximum_steps = int(
        maximum_duration / sample_time
    )

    for _ in range(maximum_steps):
        reference_index = nearest_reference_index(
            path,
            vehicle_state[0],
            vehicle_state[1],
            start_index=reference_index,
        )

        tracking_error = calculate_tracking_error(
            vehicle_state,
            path,
            reference_index,
        )

        control_state = create_control_state(
            vehicle_state,
            tracking_error,
        )

        preview_offsets = np.arange(
            mpc.prediction_horizon
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
            len(path.curvature) - 1,
        )

        curvature_preview = (
            path.curvature[preview_indices]
        )

        current_curvature = float(
            path.curvature[reference_index]
        )

        start_time = perf_counter()

        if learning_controller is None:
            steering = mpc.control(
                control_state,
                previous_steering=previous_steering,
                curvature_preview=curvature_preview,
            )
            compensation = 0.0
        else:
            steering = learning_controller.control(
                state=control_state,
                previous_steering=previous_steering,
                curvature_preview=curvature_preview,
                current_curvature=current_curvature,
            )

            compensation = (
                learning_controller.last_compensation
            )

        computation_time = (
            perf_counter() - start_time
        )

        vehicle_states.append(vehicle_state.copy())
        control_states.append(control_state.copy())
        steering_commands.append(steering)
        compensations.append(compensation)
        computation_times.append(computation_time)

        effective_steering = (
            steering + steering_bias
        )

        vehicle_state = vehicle.step(
            vehicle_state,
            effective_steering,
            sample_time,
        )

        previous_steering = steering

        if vehicle_state[0] >= path.x[-1]:
            break

    return SimulationResult(
        vehicle_states=np.asarray(vehicle_states),
        control_states=np.asarray(control_states),
        steering=np.asarray(steering_commands),
        compensation=np.asarray(compensations),
        computation_times=np.asarray(
            computation_times
        ),
        sample_time=sample_time,
    )


def calculate_metrics(
    result: SimulationResult,
) -> dict[str, float]:
    lateral_error = result.control_states[:, 2]
    heading_error = result.control_states[:, 3]

    return {
        "lateral_rmse": float(
            np.sqrt(np.mean(lateral_error**2))
        ),
        "maximum_lateral_error": float(
            np.max(np.abs(lateral_error))
        ),
        "heading_rmse": float(
            np.sqrt(np.mean(heading_error**2))
        ),
        "control_effort": float(
            np.mean(result.steering**2)
        ),
        "average_computation_ms": float(
            np.mean(result.computation_times)
            * 1000.0
        ),
    }


def main() -> None:
    sample_time = 0.05

    # This trajectory was not used for ELM training.
    test_path = sinusoidal_path(
        length=220.0,
        spacing=0.1,
        amplitude=2.8,
        wavelength=92.0,
    )

    nominal_parameters = VehicleParameters(
        mass=1500.0,
        yaw_inertia=2250.0,
        front_cornering_stiffness=80_000.0,
        rear_cornering_stiffness=80_000.0,
        longitudinal_speed=14.5,
        friction_coefficient=0.9,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    real_parameters = VehicleParameters(
        mass=1660.0,
        yaw_inertia=2480.0,
        front_cornering_stiffness=68_000.0,
        rear_cornering_stiffness=72_000.0,
        longitudinal_speed=14.5,
        friction_coefficient=0.82,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    elm = ExtremeLearningMachineRegressor.load(
        "data/models/elm_predictive_error.joblib"
    )

    with Path(
        "data/models/compensator_config.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        compensator_parameters = json.load(file)

    print(
        "Using fixed compensator parameters: "
        f"{compensator_parameters}"
    )

    nominal_result = run_simulation(
        path=test_path,
        nominal_parameters=nominal_parameters,
        real_parameters=real_parameters,
        error_predictor=None,
        sample_time=sample_time,
    )

    learning_result = run_simulation(
        path=test_path,
        nominal_parameters=nominal_parameters,
        real_parameters=real_parameters,
        error_predictor=elm,
        sample_time=sample_time,
        compensator_parameters=compensator_parameters,
    )

    nominal_metrics = calculate_metrics(
        nominal_result
    )

    learning_metrics = calculate_metrics(
        learning_result
    )

    print("Nominal MPC metrics:")

    for name, value in nominal_metrics.items():
        print(f"  {name}: {value:.8f}")

    print("\nLearning-enhanced MPC metrics:")

    for name, value in learning_metrics.items():
        print(f"  {name}: {value:.8f}")

    improvement = (
        1.0
        - learning_metrics["lateral_rmse"]
        / nominal_metrics["lateral_rmse"]
    ) * 100.0

    print(
        "\nLateral RMSE improvement: "
        f"{improvement:.2f}%"
    )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(11, 13),
        constrained_layout=True,
    )

    axes[0].plot(
        test_path.x,
        test_path.y,
        "k--",
        label="Reference path",
    )
    axes[0].plot(
        nominal_result.vehicle_states[:, 0],
        nominal_result.vehicle_states[:, 1],
        label="Nominal MPC",
    )
    axes[0].plot(
        learning_result.vehicle_states[:, 0],
        learning_result.vehicle_states[:, 1],
        label="MPC + ELM",
    )
    axes[0].set_ylabel("Lateral position Y [m]")
    axes[0].set_xlabel("Longitudinal position X [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        nominal_result.time,
        nominal_result.control_states[:, 2],
        label="Nominal MPC",
    )
    axes[1].plot(
        learning_result.time,
        learning_result.control_states[:, 2],
        label="MPC + ELM",
    )
    axes[1].set_ylabel("Lateral error [m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        nominal_result.time,
        np.rad2deg(nominal_result.steering),
        label="Nominal MPC",
    )
    axes[2].plot(
        learning_result.time,
        np.rad2deg(learning_result.steering),
        label="MPC + ELM",
    )
    axes[2].set_ylabel("Steering [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        learning_result.time,
        np.rad2deg(
            learning_result.compensation
        ),
    )
    axes[3].set_ylabel("ELM compensation [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    figure.suptitle(
        "Nominal MPC versus learning-enhanced MPC"
    )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure.savefig(
        output_directory
        / "nominal_vs_learning_mpc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()