import json
from math import radians
from pathlib import Path

from experiments.compare_nominal_learning_mpc import (
    calculate_metrics,
    run_simulation,
)
from src.learning.extreme_learning_machine import (
    ExtremeLearningMachineRegressor,
)
from src.models.vehicle_parameters import VehicleParameters
from src.simulation.reference_paths import sinusoidal_path


def vehicle_parameters(
    speed: float,
    mass: float,
    yaw_inertia: float,
    front_stiffness: float,
    rear_stiffness: float,
    friction: float,
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
    validation_path = sinusoidal_path(
        length=160.0,
        spacing=0.1,
        amplitude=2.3,
        wavelength=78.0,
    )

    nominal_parameters = vehicle_parameters(
        speed=14.0,
        mass=1500.0,
        yaw_inertia=2250.0,
        front_stiffness=80_000.0,
        rear_stiffness=80_000.0,
        friction=0.9,
    )

    real_parameters = vehicle_parameters(
        speed=14.0,
        mass=1640.0,
        yaw_inertia=2460.0,
        front_stiffness=67_000.0,
        rear_stiffness=73_000.0,
        friction=0.82,
    )

    elm = ExtremeLearningMachineRegressor.load(
        "data/models/elm_predictive_error.joblib"
    )

    baseline_result = run_simulation(
        path=validation_path,
        nominal_parameters=nominal_parameters,
        real_parameters=real_parameters,
        error_predictor=None,
        maximum_duration=15.0,
    )

    baseline_metrics = calculate_metrics(
        baseline_result
    )

    lateral_candidates = [
        -8.0,
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
        8.0,
    ]

    heading_candidates = [
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
    ]

    best_score = float("inf")
    best_parameters: dict[str, float] = {}
    best_metrics: dict[str, float] = {}

    total_candidates = (
        len(lateral_candidates)
        * len(heading_candidates)
    )

    candidate_number = 0

    for lateral_gain in lateral_candidates:
        for heading_gain in heading_candidates:
            candidate_number += 1

            parameters = {
                "lateral_gain": lateral_gain,
                "heading_gain": heading_gain,
                "correction_limit": radians(0.25),
                "filter_coefficient": 0.15,
            }

            result = run_simulation(
                path=validation_path,
                nominal_parameters=nominal_parameters,
                real_parameters=real_parameters,
                error_predictor=elm,
                maximum_duration=15.0,
                compensator_parameters=parameters,
            )

            metrics = calculate_metrics(result)

            lateral_ratio = (
                metrics["lateral_rmse"]
                / baseline_metrics["lateral_rmse"]
            )

            heading_ratio = (
                metrics["heading_rmse"]
                / baseline_metrics["heading_rmse"]
            )

            effort_ratio = (
                metrics["control_effort"]
                / baseline_metrics["control_effort"]
            )

            score = (
                lateral_ratio
                + 0.20 * heading_ratio
                + 0.05 * effort_ratio
            )

            if (
                metrics["maximum_lateral_error"]
                > 2.0
            ):
                score += 100.0

            if score < best_score:
                best_score = score
                best_parameters = parameters
                best_metrics = metrics

                print(
                    f"[{candidate_number}/"
                    f"{total_candidates}] "
                    "New best: "
                    f"lateral_gain={lateral_gain}, "
                    f"heading_gain={heading_gain}, "
                    f"RMSE="
                    f"{metrics['lateral_rmse']:.6f}"
                )

    improvement = (
        1.0
        - best_metrics["lateral_rmse"]
        / baseline_metrics["lateral_rmse"]
    ) * 100.0

    print("\nBaseline validation metrics:")

    for name, value in baseline_metrics.items():
        print(f"  {name}: {value:.8f}")

    print("\nBest learning validation metrics:")

    for name, value in best_metrics.items():
        print(f"  {name}: {value:.8f}")

    print("\nBest compensator parameters:")

    for name, value in best_parameters.items():
        print(f"  {name}: {value}")

    print(
        "\nValidation lateral RMSE improvement: "
        f"{improvement:.2f}%"
    )

    output_path = Path(
        "data/models/compensator_config.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_parameters,
            file,
            indent=2,
        )

    print(f"\nConfiguration saved to: {output_path}")


if __name__ == "__main__":
    main()