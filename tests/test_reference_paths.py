import numpy as np

from src.simulation.reference_paths import (
    double_lane_change_path,
    sinusoidal_path,
    straight_path,
)


def test_straight_path_has_zero_curvature() -> None:
    path = straight_path()

    np.testing.assert_allclose(path.y, 0.0)
    np.testing.assert_allclose(path.heading, 0.0)
    np.testing.assert_allclose(path.curvature, 0.0)


def test_sinusoidal_path_arrays_have_equal_length() -> None:
    path = sinusoidal_path()

    assert len(path.x) == len(path.y)
    assert len(path.x) == len(path.heading)
    assert len(path.x) == len(path.curvature)


def test_double_lane_change_returns_to_original_lane() -> None:
    path = double_lane_change_path()

    assert np.max(path.y) > 3.0
    assert abs(path.y[0]) < 0.01
    assert abs(path.y[-1]) < 0.01