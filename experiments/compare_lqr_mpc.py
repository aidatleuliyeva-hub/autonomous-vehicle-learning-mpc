from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers.lqr_controller import LQRController
from src.controllers.mpc_controller import MPCController


def main() -> None:
    sample_time = 0.05
    duration = 8.0

    lqr = LQRController(sample_time=sample_time)

    mpc = MPCController(
        sample_time=sample_time,
        prediction_horizon=20,
    )

    time = np.arange(
        0.0,
        duration + sample_time,
        sample_time,
    )

    initial_state = np.array(
        [0.0, 0.0, 1.5, np.deg2rad(5.0)]
    )

    lqr_state = initial_state.copy()
    mpc_state = initial_state.copy()

    previous_mpc_steering = 0.0

    lqr_states = []
    mpc_states = []

    lqr_inputs = []
    mpc_inputs = []

    for _ in time:
        lqr_states.append(lqr_state.copy())
        mpc_states.append(mpc_state.copy())

        lqr_steering = lqr.control(lqr_state)

        mpc_steering = mpc.control(
            mpc_state,
            previous_mpc_steering,
        )

        lqr_inputs.append(lqr_steering)
        mpc_inputs.append(mpc_steering)

        lqr_state = (
            lqr.ad @ lqr_state
            + lqr.bd.flatten() * lqr_steering
        )

        mpc_state = (
            mpc.ad @ mpc_state
            + mpc.bd.flatten() * mpc_steering
        )

        previous_mpc_steering = mpc_steering

    lqr_states = np.asarray(lqr_states)
    mpc_states = np.asarray(mpc_states)

    lqr_inputs = np.asarray(lqr_inputs)
    mpc_inputs = np.asarray(mpc_inputs)

    lqr_rmse = np.sqrt(
        np.mean(lqr_states[:, 2] ** 2)
    )

    mpc_rmse = np.sqrt(
        np.mean(mpc_states[:, 2] ** 2)
    )

    print(f"LQR lateral-error RMSE: {lqr_rmse:.4f} m")
    print(f"MPC lateral-error RMSE: {mpc_rmse:.4f} m")

    print(
        "Maximum MPC steering rate: "
        f"{np.max(np.abs(np.diff(mpc_inputs))) / sample_time:.4f} rad/s"
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        time,
        lqr_states[:, 2],
        label="LQR",
    )
    axes[0].plot(
        time,
        mpc_states[:, 2],
        label="Constrained MPC",
    )
    axes[0].set_ylabel("Lateral error [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        np.rad2deg(lqr_states[:, 3]),
        label="LQR",
    )
    axes[1].plot(
        time,
        np.rad2deg(mpc_states[:, 3]),
        label="Constrained MPC",
    )
    axes[1].set_ylabel("Heading error [deg]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time,
        np.rad2deg(lqr_inputs),
        label="LQR",
    )
    axes[2].plot(
        time,
        np.rad2deg(mpc_inputs),
        label="Constrained MPC",
    )

    steering_limit = np.rad2deg(
        mpc.model.p.max_steering_angle
    )

    axes[2].axhline(
        steering_limit,
        color="red",
        linestyle="--",
        label="Steering limits",
    )
    axes[2].axhline(
        -steering_limit,
        color="red",
        linestyle="--",
    )

    axes[2].set_ylabel("Steering [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        "LQR and constrained MPC comparison"
    )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure.savefig(
        output_directory / "lqr_mpc_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()