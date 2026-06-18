from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.controllers.lqr_controller import LQRController
from src.controllers.state_observer import StateObserver


def main() -> None:
    sample_time = 0.05
    duration = 8.0

    controller = LQRController(
        sample_time=sample_time
    )

    observer = StateObserver(
        sample_time=sample_time
    )

    time = np.arange(
        0.0,
        duration + sample_time,
        sample_time,
    )

    true_state = np.array(
        [0.3, 0.05, 1.5, np.deg2rad(5.0)]
    )

    random_generator = np.random.default_rng(42)

    true_states = []
    estimated_states = []
    estimation_errors = []

    for _ in time:
        true_states.append(true_state.copy())
        estimated_states.append(observer.state.copy())

        estimation_errors.append(
            np.linalg.norm(
                true_state - observer.state
            )
        )

        measurement_noise = np.array(
            [
                random_generator.normal(0.0, 0.01),
                random_generator.normal(
                    0.0,
                    np.deg2rad(0.1),
                ),
            ]
        )

        measurement = (
            observer.cd @ true_state
            + measurement_noise
        )

        steering = controller.control(observer.state)

        observer.update(
            measurement,
            steering,
        )

        true_state = (
            observer.ad @ true_state
            + observer.bd.flatten() * steering
        )

    true_states = np.asarray(true_states)
    estimated_states = np.asarray(estimated_states)

    print("Observer gain:")
    print(observer.gain)

    print("\nObserver error eigenvalues:")
    print(observer.error_eigenvalues)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        time,
        true_states[:, 0],
        label="True lateral velocity",
    )
    axes[0].plot(
        time,
        estimated_states[:, 0],
        linestyle="--",
        label="Estimated lateral velocity",
    )
    axes[0].set_ylabel("Lateral velocity [m/s]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time,
        true_states[:, 1],
        label="True yaw rate",
    )
    axes[1].plot(
        time,
        estimated_states[:, 1],
        linestyle="--",
        label="Estimated yaw rate",
    )
    axes[1].set_ylabel("Yaw rate [rad/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(
        time,
        estimation_errors,
    )
    axes[2].set_ylabel("Estimation error norm")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(
        "Luenberger observer with output-feedback control"
    )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure.savefig(
        output_directory / "observer_tracking.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()