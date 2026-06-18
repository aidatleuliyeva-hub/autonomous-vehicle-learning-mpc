from math import radians
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.models.vehicle_parameters import VehicleParameters
from src.simulation.dataset_generator import (
    TARGET_COLUMNS,
    DatasetScenario,
    generate_scenario_data,
)
from src.simulation.reference_paths import (
    double_lane_change_path,
    sinusoidal_path,
    straight_path,
)


def vehicle_parameters(
    speed: float,
    mass: float = 1500.0,
    yaw_inertia: float = 2250.0,
    front_stiffness: float = 80_000.0,
    rear_stiffness: float = 80_000.0,
    friction: float = 0.9,
) -> VehicleParameters:
    return VehicleParameters(
        mass=mass,
        yaw_inertia=yaw_inertia,
        front_cornering_stiffness=front_stiffness,
        rear_cornering_stiffness=rear_stiffness,
        longitudinal_speed=speed,
        friction_coefficient=friction,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )


def main() -> None:
    double_lane_path = double_lane_change_path(
        length=180.0,
        spacing=0.1,
        first_change=45.0,
        second_change=125.0,
        transition_length=12.0,
    )

    sine_path = sinusoidal_path(
        length=180.0,
        spacing=0.1,
        amplitude=2.0,
        wavelength=70.0,
    )

    scenarios = [
        DatasetScenario(
            name="double_lane_mild",
            path=double_lane_path,
            nominal_parameters=vehicle_parameters(15.0),
            real_parameters=vehicle_parameters(
                speed=15.0,
                mass=1600.0,
                yaw_inertia=2400.0,
                front_stiffness=72_000.0,
                rear_stiffness=76_000.0,
                friction=0.85,
            ),
            steering_bias=radians(0.4),
            random_seed=11,
        ),
        DatasetScenario(
            name="double_lane_heavy",
            path=double_lane_path,
            nominal_parameters=vehicle_parameters(13.0),
            real_parameters=vehicle_parameters(
                speed=13.0,
                mass=1700.0,
                yaw_inertia=2550.0,
                front_stiffness=65_000.0,
                rear_stiffness=70_000.0,
                friction=0.80,
            ),
            steering_bias=radians(-0.35),
            random_seed=22,
        ),
        DatasetScenario(
            name="sine_understeer",
            path=sine_path,
            nominal_parameters=vehicle_parameters(12.0),
            real_parameters=vehicle_parameters(
                speed=12.0,
                mass=1625.0,
                yaw_inertia=2425.0,
                front_stiffness=62_000.0,
                rear_stiffness=78_000.0,
                friction=0.82,
            ),
            steering_bias=radians(0.3),
            random_seed=33,
        ),
        DatasetScenario(
            name="sine_high_speed",
            path=sine_path,
            nominal_parameters=vehicle_parameters(15.0),
            real_parameters=vehicle_parameters(
                speed=15.0,
                mass=1580.0,
                yaw_inertia=2380.0,
                front_stiffness=68_000.0,
                rear_stiffness=72_000.0,
                friction=0.84,
            ),
            steering_bias=radians(-0.25),
            random_seed=44,
        ),
        DatasetScenario(
            name="straight_steering_bias",
            path=straight_path(length=180.0, spacing=0.1,),
            nominal_parameters=vehicle_parameters(15.0),
            real_parameters=vehicle_parameters(
                speed=15.0,
                mass=1650.0,
                yaw_inertia=2450.0,
                front_stiffness=70_000.0,
                rear_stiffness=75_000.0,
                friction=0.85,
            ),
            steering_bias=radians(0.8),
            random_seed=55,
        ),
    ]

    scenario_datasets = []

    for scenario in scenarios:
        dataset = generate_scenario_data(scenario)
        scenario_datasets.append(dataset)

        maximum_error = dataset[
            "lateral_error"
        ].abs().max()

        print(
            f"{scenario.name}: "
            f"{len(dataset)} samples, "
            f"max |e_y| = {maximum_error:.4f} m"
        )

    complete_dataset = pd.concat(
        scenario_datasets,
        ignore_index=True,
    )

    output_directory = Path("data")
    output_directory.mkdir(exist_ok=True)

    output_file = (
        output_directory
        / "predictive_error_dataset.csv"
    )

    complete_dataset.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nTotal samples: {len(complete_dataset)}"
    )
    print(f"Dataset saved to: {output_file}")

    print("\nTarget statistics:")
    print(
        complete_dataset[TARGET_COLUMNS].describe()
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        constrained_layout=True,
    )

    axes[0].hist(
        complete_dataset[
            "lateral_prediction_error"
        ],
        bins=50,
    )
    axes[0].set_title(
        "Lateral predictive error"
    )
    axes[0].set_xlabel("Error [m]")
    axes[0].set_ylabel("Samples")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        complete_dataset[
            "heading_prediction_error"
        ],
        bins=50,
    )
    axes[1].set_title(
        "Heading predictive error"
    )
    axes[1].set_xlabel("Error [rad]")
    axes[1].set_ylabel("Samples")
    axes[1].grid(True, alpha=0.3)

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    figure.savefig(
        results_directory
        / "predictive_error_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()