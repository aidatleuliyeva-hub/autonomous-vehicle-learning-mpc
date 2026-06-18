from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class VehicleParameters:
    """Physical parameters of the simulated vehicle."""

    mass: float = 1500.0
    yaw_inertia: float = 2250.0

    front_axle_distance: float = 1.2
    rear_axle_distance: float = 1.6

    front_cornering_stiffness: float = 80_000.0
    rear_cornering_stiffness: float = 80_000.0

    longitudinal_speed: float = 20.0

    friction_coefficient: float = 0.9
    gravity: float = 9.81

    max_steering_angle: float = radians(25.0)
    max_steering_rate: float = radians(60.0)