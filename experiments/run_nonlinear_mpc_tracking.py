from math import radians
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from src.simulation.reference_paths import (
    double_lane_change_path,
)


def main() -> None:
    sample_time = 0.05
    maximum_duration = 15.0

    reference_path = double_lane_change_path(
        length=180.0,
        lane_width=3.5,
        first_change=45.0,
        second_change=125.0,
        transition_length=12.0,
    )

    nominal_parameters = VehicleParameters(
        mass=1500.0,
        yaw_inertia=2250.0,
        front_cornering_stiffness=80_000.0,
        rear_cornering_stiffness=80_000.0,
        longitudinal_speed=15.0,
        friction_coefficient=0.9,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    linear_model = LinearBicycleModel(
        nominal_parameters
    )

    controller = MPCController(
        model=linear_model,
        sample_time=sample_time,
        prediction_horizon=30,
    )

    real_parameters = VehicleParameters(
        mass=1575.0,
        yaw_inertia=2350.0,
        front_cornering_stiffness=76_000.0,
        rear_cornering_stiffness=78_000.0,
        longitudinal_speed=15.0,
        friction_coefficient=0.88,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )

    vehicle = NonlinearBicycleModel(real_parameters)

    vehicle_state = np.zeros(5)
    previous_steering = 0.0
    reference_index = 0

    steering_bias = radians(0.2)

    vehicle_states = []
    control_states = []
    steering_commands = []

    path_spacing = (
        reference_path.x[1] - reference_path.x[0]
    )

    maximum_steps = int(
        maximum_duration / sample_time
    )

    for _ in range(maximum_steps):
        reference_index = nearest_reference_index(
            reference_path,
            vehicle_state[0],
            vehicle_state[1],
            start_index=reference_index,
        )

        tracking_error = calculate_tracking_error(
            vehicle_state,
            reference_path,
            reference_index,
        )

        control_state = create_control_state(
            vehicle_state,
            tracking_error,
        )

        travelled_per_step = (
            nominal_parameters.longitudinal_speed
            * sample_time
        )

        preview_offsets = np.arange(
            controller.prediction_horizon
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
            len(reference_path.curvature) - 1,
        )

        curvature_preview = (
            reference_path.curvature[preview_indices]
        )

        steering_command = controller.control(
            control_state,
            previous_steering,
            curvature_preview,
        )

        effective_steering = (
            steering_command + steering_bias
        )

        vehicle_states.append(vehicle_state.copy())
        control_states.append(control_state.copy())
        steering_commands.append(steering_command)

        vehicle_state = vehicle.step(
            vehicle_state,
            effective_steering,
            sample_time,
        )

        previous_steering = steering_command

        if vehicle_state[0] >= reference_path.x[-1]:
            break

    vehicle_states = np.asarray(vehicle_states)
    control_states = np.asarray(control_states)
    steering_commands = np.asarray(steering_commands)

    time = (
        np.arange(len(vehicle_states)) * sample_time
    )

    lateral_errors = control_states[:, 2]
    heading_errors = control_states[:, 3]

    lateral_rmse = float(
        np.sqrt(np.mean(lateral_errors**2))
    )

    maximum_lateral_error = float(
        np.max(np.abs(lateral_errors))
    )

    print(
        f"Lateral-error RMSE: {lateral_rmse:.4f} m"
    )
    print(
        "Maximum lateral error: "
        f"{maximum_lateral_error:.4f} m"
    )
    print(
        "Maximum steering command: "
        f"{np.rad2deg(np.max(np.abs(steering_commands))):.2f} deg"
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        constrained_layout=True,
    )

    axes[0].plot(
        reference_path.x,
        reference_path.y,
        "--",
        label="Reference path",
    )
    axes[0].plot(
        vehicle_states[:, 0],
        vehicle_states[:, 1],
        label="Nominal MPC",
    )
    axes[0].set_xlabel("Longitudinal position X [m]")
    axes[0].set_ylabel("Lateral position Y [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        lateral_errors,
        label="Lateral error [m]",
    )
    axes[1].plot(
        time,
        np.rad2deg(heading_errors),
        label="Heading error [deg]",
    )
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Tracking errors")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time,
        np.rad2deg(steering_commands),
    )
    axes[2].axhline(
        12.0,
        color="red",
        linestyle="--",
    )
    axes[2].axhline(
        -12.0,
        color="red",
        linestyle="--",
    )
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Steering [deg]")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        "Curvature-aware MPC with model mismatch"
    )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure.savefig(
        output_directory
        / "nonlinear_mpc_tracking.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()