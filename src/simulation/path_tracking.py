from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .reference_paths import ReferencePath


Array = NDArray[np.float64]


@dataclass(frozen=True)
class TrackingError:
    reference_index: int
    lateral_error: float
    heading_error: float


def wrap_angle(angle: float) -> float:
    return float(
        (angle + np.pi) % (2.0 * np.pi) - np.pi
    )


def nearest_reference_index(
    path: ReferencePath,
    x_position: float,
    y_position: float,
    start_index: int = 0,
    search_window: int = 100,
) -> int:
    start = max(0, start_index)
    end = min(
        len(path.x),
        start + search_window,
    )

    if start >= end:
        return len(path.x) - 1

    delta_x = path.x[start:end] - x_position
    delta_y = path.y[start:end] - y_position

    local_index = int(
        np.argmin(delta_x**2 + delta_y**2)
    )

    return start + local_index


def calculate_tracking_error(
    vehicle_state: Array,
    path: ReferencePath,
    reference_index: int,
) -> TrackingError:
    x_position, y_position, heading, _, _ = (
        vehicle_state
    )

    reference_x = path.x[reference_index]
    reference_y = path.y[reference_index]
    reference_heading = path.heading[reference_index]

    delta_x = x_position - reference_x
    delta_y = y_position - reference_y

    lateral_error = (
        -np.sin(reference_heading) * delta_x
        + np.cos(reference_heading) * delta_y
    )

    heading_error = wrap_angle(
        heading - reference_heading
    )

    return TrackingError(
        reference_index=reference_index,
        lateral_error=float(lateral_error),
        heading_error=heading_error,
    )


def create_control_state(
    vehicle_state: Array,
    tracking_error: TrackingError,
) -> Array:
    lateral_speed = vehicle_state[3]
    yaw_rate = vehicle_state[4]

    return np.array(
        [
            lateral_speed,
            yaw_rate,
            tracking_error.lateral_error,
            tracking_error.heading_error,
        ],
        dtype=float,
    )