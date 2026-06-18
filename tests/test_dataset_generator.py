from math import radians

import numpy as np

from src.models.vehicle_parameters import VehicleParameters
from src.simulation.dataset_generator import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DatasetScenario,
    generate_scenario_data,
)
from src.simulation.reference_paths import straight_path


def test_dataset_generator_produces_finite_samples() -> None:
    nominal_parameters = VehicleParameters(
        longitudinal_speed=10.0,
    )

    real_parameters = VehicleParameters(
        mass=1650.0,
        yaw_inertia=2450.0,
        front_cornering_stiffness=70_000.0,
        rear_cornering_stiffness=75_000.0,
        longitudinal_speed=10.0,
    )

    scenario = DatasetScenario(
        name="test",
        path=straight_path(length=20.0),
        nominal_parameters=nominal_parameters,
        real_parameters=real_parameters,
        steering_bias=radians(0.5),
    )

    dataset = generate_scenario_data(
        scenario,
        maximum_duration=3.0,
    )

    required_columns = (
        FEATURE_COLUMNS + TARGET_COLUMNS
    )

    assert len(dataset) > 10
    assert all(
        column in dataset.columns
        for column in required_columns
    )

    numeric_data = dataset[required_columns].to_numpy()

    assert np.all(np.isfinite(numeric_data))

    target_values = dataset[
        TARGET_COLUMNS
    ].to_numpy()

    assert np.max(np.abs(target_values)) > 0.0