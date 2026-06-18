from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers.lqr_controller import LQRController


def main() -> None:
    controller = LQRController(sample_time=0.05)

    duration = 10.0
    time = np.arange(
        0.0,
        duration + controller.sample_time,
        controller.sample_time,
    )

    state = np.array(
        [0.0, 0.0, 1.5, np.deg2rad(5.0)]
    )

    states = []
    steering_inputs = []

    for _ in time:
        states.append(state.copy())

        steering = controller.control(state)
        steering_inputs.append(steering)

        state = (
            controller.ad @ state
            + controller.bd.flatten() * steering
        )

    states_array = np.asarray(states)
    steering_array = np.asarray(steering_inputs)

    print("LQR gain:")
    print(controller.gain)

    print("\nClosed-loop eigenvalues:")
    print(controller.closed_loop_eigenvalues)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(time, states_array[:, 2])
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("Lateral error [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        np.rad2deg(states_array[:, 3]),
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Heading error [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time,
        np.rad2deg(steering_array),
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Steering [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        "LQR state-feedback tracking-error regulation"
    )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure.savefig(
        output_directory / "lqr_tracking.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()