from pathlib import Path

import matplotlib.pyplot as plt

from src.simulation.reference_paths import (
    double_lane_change_path,
    sinusoidal_path,
    straight_path,
)


def main() -> None:
    paths = {
        "Straight path": straight_path(),
        "Sinusoidal path": sinusoidal_path(),
        "Double-lane change": double_lane_change_path(),
    }

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 8),
        constrained_layout=True,
    )

    for axis, (name, path) in zip(axes, paths.items()):
        axis.plot(
            path.x,
            path.y,
            linewidth=2.0,
            label="Reference path",
        )

        axis.set_title(name)
        axis.set_xlabel("Longitudinal position X [m]")
        axis.set_ylabel("Lateral position Y [m]")
        axis.grid(True, alpha=0.3)
        axis.legend()

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    output_file = (
        output_directory / "reference_paths.png"
    )

    figure.savefig(output_file, dpi=200)
    print(f"Figure saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()