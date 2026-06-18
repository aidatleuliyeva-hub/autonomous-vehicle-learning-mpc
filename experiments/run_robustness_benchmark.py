import json
from dataclasses import dataclass
from math import radians
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.compare_nominal_learning_mpc import (
    calculate_metrics,
    run_simulation,
)
from src.learning.extreme_learning_machine import (
    ExtremeLearningMachineRegressor,
)
from src.models.vehicle_parameters import VehicleParameters
from src.simulation.reference_paths import (
    ReferencePath,
    double_lane_change_path,
    sinusoidal_path,
    straight_path,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    path: ReferencePath
    nominal: VehicleParameters
    real: VehicleParameters
    steering_bias: float


def parameters(
    speed: float,
    mass: float = 1500.0,
    inertia: float = 2250.0,
    front_stiffness: float = 80_000.0,
    rear_stiffness: float = 80_000.0,
    friction: float = 0.9,
) -> VehicleParameters:
    return VehicleParameters(
        mass=mass,
        yaw_inertia=inertia,
        front_cornering_stiffness=front_stiffness,
        rear_cornering_stiffness=rear_stiffness,
        longitudinal_speed=speed,
        friction_coefficient=friction,
        max_steering_angle=radians(12.0),
        max_steering_rate=radians(60.0),
    )


def main() -> None:
    elm = ExtremeLearningMachineRegressor.load(
        "data/models/elm_predictive_error.joblib"
    )

    with Path(
        "data/models/compensator_config.json"
    ).open("r", encoding="utf-8") as file:
        compensator_config = json.load(file)

    scenarios = [
        BenchmarkScenario(
            name="Smooth sine",
            path=sinusoidal_path(
                length=180.0,
                spacing=0.1,
                amplitude=2.0,
                wavelength=100.0,
            ),
            nominal=parameters(13.0),
            real=parameters(
                13.0, 1600.0, 2400.0,
                70_000.0, 74_000.0, 0.84,
            ),
            steering_bias=radians(0.3),
        ),
        BenchmarkScenario(
            name="High-speed sine",
            path=sinusoidal_path(
                length=200.0,
                spacing=0.1,
                amplitude=3.0,
                wavelength=85.0,
            ),
            nominal=parameters(15.0),
            real=parameters(
                15.0, 1680.0, 2500.0,
                65_000.0, 71_000.0, 0.80,
            ),
            steering_bias=radians(-0.4),
        ),
        BenchmarkScenario(
            name="Double lane",
            path=double_lane_change_path(
                length=180.0,
                spacing=0.1,
                first_change=45.0,
                second_change=125.0,
                transition_length=11.0,
            ),
            nominal=parameters(14.0),
            real=parameters(
                14.0, 1650.0, 2475.0,
                68_000.0, 73_000.0, 0.82,
            ),
            steering_bias=radians(0.5),
        ),
        BenchmarkScenario(
            name="Heavy mismatch",
            path=double_lane_change_path(
                length=180.0,
                spacing=0.1,
                first_change=45.0,
                second_change=125.0,
                transition_length=10.0,
            ),
            nominal=parameters(12.0),
            real=parameters(
                12.0, 1720.0, 2600.0,
                62_000.0, 69_000.0, 0.78,
            ),
            steering_bias=radians(-0.35),
        ),
        BenchmarkScenario(
            name="Steering bias",
            path=straight_path(
                length=180.0,
                spacing=0.1,
            ),
            nominal=parameters(15.0),
            real=parameters(
                15.0, 1580.0, 2380.0,
                72_000.0, 76_000.0, 0.86,
            ),
            steering_bias=radians(1.0),
        ),
    ]

    rows = []

    for scenario in scenarios:
        nominal_result = run_simulation(
            path=scenario.path,
            nominal_parameters=scenario.nominal,
            real_parameters=scenario.real,
            error_predictor=None,
            steering_bias=scenario.steering_bias,
        )

        learning_result = run_simulation(
            path=scenario.path,
            nominal_parameters=scenario.nominal,
            real_parameters=scenario.real,
            error_predictor=elm,
            compensator_parameters=compensator_config,
            steering_bias=scenario.steering_bias,
        )

        nominal = calculate_metrics(nominal_result)
        learning = calculate_metrics(learning_result)

        improvement = (
            1.0
            - learning["lateral_rmse"]
            / nominal["lateral_rmse"]
        ) * 100.0

        rows.append(
            {
                "scenario": scenario.name,
                "nominal_rmse": nominal["lateral_rmse"],
                "learning_rmse": learning["lateral_rmse"],
                "improvement_percent": improvement,
                "nominal_max_error":
                    nominal["maximum_lateral_error"],
                "learning_max_error":
                    learning["maximum_lateral_error"],
                "learning_computation_ms":
                    learning["average_computation_ms"],
            }
        )

        print(
            f"{scenario.name}: "
            f"{nominal['lateral_rmse']:.6f} -> "
            f"{learning['lateral_rmse']:.6f} m, "
            f"improvement={improvement:.2f}%"
        )

    results = pd.DataFrame(rows)

    print(
        "\nMean RMSE improvement: "
        f"{results['improvement_percent'].mean():.2f}%"
    )

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    results.to_csv(
        results_directory / "robustness_benchmark.csv",
        index=False,
    )

    positions = np.arange(len(results))
    width = 0.36

    figure, axis = plt.subplots(
        figsize=(11, 5),
        constrained_layout=True,
    )

    axis.bar(
        positions - width / 2,
        results["nominal_rmse"],
        width,
        label="Nominal MPC",
    )

    axis.bar(
        positions + width / 2,
        results["learning_rmse"],
        width,
        label="MPC + ELM",
    )

    axis.set_xticks(
        positions,
        results["scenario"],
        rotation=15,
    )

    axis.set_ylabel("Lateral-error RMSE [m]")
    axis.set_title("Robustness benchmark")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)

    figure.savefig(
        results_directory
        / "robustness_benchmark.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()