from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ReferencePath:
    x: Array
    y: Array
    heading: Array
    curvature: Array


def _calculate_geometry(x: Array, y: Array) -> ReferencePath:
    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    heading = np.unwrap(np.arctan2(dy, dx))

    denominator = np.maximum(
        (dx**2 + dy**2) ** 1.5,
        1e-9,
    )

    curvature = (dx * ddy - dy * ddx) / denominator

    return ReferencePath(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
    )


def straight_path(
    length: float = 160.0,
    spacing: float = 0.5,
) -> ReferencePath:
    x = np.arange(0.0, length + spacing, spacing)
    y = np.zeros_like(x)

    return _calculate_geometry(x, y)


def sinusoidal_path(
    length: float = 160.0,
    spacing: float = 0.5,
    amplitude: float = 2.0,
    wavelength: float = 60.0,
) -> ReferencePath:
    x = np.arange(0.0, length + spacing, spacing)

    y = amplitude * np.sin(
        2.0 * np.pi * x / wavelength
    )

    return _calculate_geometry(x, y)


def double_lane_change_path(
    length: float = 160.0,
    spacing: float = 0.5,
    lane_width: float = 3.5,
    first_change: float = 40.0,
    second_change: float = 110.0,
    transition_length: float = 6.0,
) -> ReferencePath:
    x = np.arange(0.0, length + spacing, spacing)

    first_transition = np.tanh(
        (x - first_change) / transition_length
    )

    second_transition = np.tanh(
        (x - second_change) / transition_length
    )

    y = 0.5 * lane_width * (
        first_transition - second_transition
    )

    return _calculate_geometry(x, y)