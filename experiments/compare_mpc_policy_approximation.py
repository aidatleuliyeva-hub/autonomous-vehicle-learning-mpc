from dataclasses import dataclass
from math import radians
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.controllers.elm_policy_controller import (
    ELMPolicyController,
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
class Result:
    vehicle_states: np.ndarray
    control_states: np.ndarray
    steering: np.ndarray
    computation_times: np.ndarray
    sample_time: float

    @property
    def time(self) -> np.ndarray:
        return (
            np.arange(len(self.vehicle_states))
            * self.sample_time
        )


def simulate(
    path: ReferencePath,
    nominal: VehicleParameters,
    real: VehicleParameters,
    policy_model: (
        ExtremeLearningMachineRegressor | None
    ),
    sample_time: float = 0.05,
) -> Result:
    horizon = 30

    mpc = MPCController(
        model=LinearBicycleModel(nominal),
        sample_time=sample_time,
        prediction_horizon=horizon,
    )

    policy_controller = None

    if policy_model is not None:
        policy_controller = ELMPolicyController(
            predictor=policy_model,
            parameters=nominal,
            sample_time=sample_time,
        )

    vehicle = NonlinearBicycleModel(real)

    vehicle_state = np.array(
        [
            path.x[0],
            path.y[0],
            path.heading[0],
            0.0,
            0.0,
        ]
    )

    previous_steering = 0.0
    reference_index = 0
    steering_bias = radians(0.4)

    states = []
    control_states = []
    steering_commands = []
    computation_times = []

    path_spacing = path.x[1] - path.x[0]

    travelled_per_step = (
        nominal.longitudinal_speed * sample_time
    )

    for _ in range(int(18.0 / sample_time)):
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

        preview_offsets = (
            np.arange(horizon)
            * travelled_per_step
        )

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

        start = perf_counter()

        if policy_controller is None:
            steering = mpc.control(
                control_state,
                previous_steering,
                curvature_preview,
            )
        else:
            steering = policy_controller.control(
                control_state,
                previous_steering,
                curvature_preview,
                current_curvature,
            )

        computation_times.append(
            perf_counter() - start
        )

        states.append(vehicle_state.copy())
        control_states.append(control_state.copy())
        steering_commands.append(steering)

        vehicle_state = vehicle.step(
            vehicle_state,
            steering + steering_bias,
            sample_time,
        )

        previous_steering = steering

        if vehicle_state[0] >= path.x[-1]:
            break

    return Result(
        vehicle_states=np.asarray(states),
        control_states=np.asarray(control_states),
        steering=np.asarray(steering_commands),
        computation_times=np.asarray(
            computation_times
        ),
        sample_time=sample_time,
    )


def metrics(result: Result) -> dict[str, float]:
    lateral_error = result.control_states[:, 2]

    return {
        "lateral_rmse": float(
            np.sqrt(np.mean(lateral_error**2))
        ),
        "maximum_error": float(
            np.max(np.abs(lateral_error))
        ),
        "control_effort": float(
            np.mean(result.steering**2)
        ),
        "computation_ms": float(
            np.mean(result.computation_times)
            * 1000.0
        ),
    }


def main() -> None:
    path = sinusoidal_path(
        length=210.0,
        spacing=0.1,
        amplitude=2.6,
        wavelength=88.0,
    )

    nominal = VehicleParameters(
        longitudinal_speed=14.2,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    real = VehicleParameters(
        mass=1640.0,
        yaw_inertia=2470.0,
        front_cornering_stiffness=67_000.0,
        rear_cornering_stiffness=73_000.0,
        longitudinal_speed=14.2,
        friction_coefficient=0.82,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    policy_model = (
        ExtremeLearningMachineRegressor.load(
            "data/models/elm_mpc_policy.joblib"
        )
    )

    mpc_result = simulate(
        path,
        nominal,
        real,
        policy_model=None,
    )

    elm_result = simulate(
        path,
        nominal,
        real,
        policy_model=policy_model,
    )

    mpc_metrics = metrics(mpc_result)
    elm_metrics = metrics(elm_result)

    print("Exact MPC:")

    for name, value in mpc_metrics.items():
        print(f"  {name}: {value:.8f}")

    print("\nELM policy:")

    for name, value in elm_metrics.items():
        print(f"  {name}: {value:.8f}")

    degradation = (
        elm_metrics["lateral_rmse"]
        / mpc_metrics["lateral_rmse"]
        - 1.0
    ) * 100.0

    speedup = (
        mpc_metrics["computation_ms"]
        / elm_metrics["computation_ms"]
    )

    print(
        "\nTracking RMSE difference: "
        f"{degradation:.2f}%"
    )

    print(f"Online speedup: {speedup:.1f}x")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        constrained_layout=True,
    )

    axes[0].plot(
        path.x,
        path.y,
        "k--",
        label="Reference",
    )
    axes[0].plot(
        mpc_result.vehicle_states[:, 0],
        mpc_result.vehicle_states[:, 1],
        label="Exact MPC",
    )
    axes[0].plot(
        elm_result.vehicle_states[:, 0],
        elm_result.vehicle_states[:, 1],
        label="ELM policy",
    )
    axes[0].legend()
    axes[0].set_ylabel("Y [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        mpc_result.time,
        mpc_result.control_states[:, 2],
        label="Exact MPC",
    )
    axes[1].plot(
        elm_result.time,
        elm_result.control_states[:, 2],
        label="ELM policy",
    )
    axes[1].set_ylabel("Lateral error [m]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        mpc_result.time,
        np.rad2deg(mpc_result.steering),
        label="Exact MPC",
    )
    axes[2].plot(
        elm_result.time,
        np.rad2deg(elm_result.steering),
        label="ELM policy",
    )
    axes[2].set_ylabel("Steering [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        "Exact MPC versus ELM policy approximation"
    )

    Path("results").mkdir(exist_ok=True)

    figure.savefig(
        "results/mpc_policy_closed_loop.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()