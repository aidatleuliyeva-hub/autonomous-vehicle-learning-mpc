import numpy as np

from src.simulation.path_tracking import (
    calculate_tracking_error,
    create_control_state,
    nearest_reference_index,
    wrap_angle,
)
from src.simulation.reference_paths import straight_path


def test_wrap_angle_returns_expected_range() -> None:
    assert np.isclose(wrap_angle(3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(0.5 * np.pi), 0.5 * np.pi)


def test_lateral_error_for_straight_path() -> None:
    path = straight_path()

    vehicle_state = np.array(
        [10.0, 1.5, 0.0, 0.2, 0.1]
    )

    index = nearest_reference_index(
        path,
        x_position=10.0,
        y_position=1.5,
    )

    error = calculate_tracking_error(
        vehicle_state,
        path,
        index,
    )

    assert np.isclose(error.lateral_error, 1.5)
    assert np.isclose(error.heading_error, 0.0)


def test_control_state_has_correct_order() -> None:
    path = straight_path()

    vehicle_state = np.array(
        [10.0, 1.5, 0.1, 0.2, 0.3]
    )

    error = calculate_tracking_error(
        vehicle_state,
        path,
        reference_index=20,
    )

    control_state = create_control_state(
        vehicle_state,
        error,
    )

    assert control_state.shape == (4,)
    assert np.isclose(control_state[0], 0.2)
    assert np.isclose(control_state[1], 0.3)