import numpy as np
from numpy.typing import NDArray

from .vehicle_parameters import VehicleParameters


State = NDArray[np.float64]


class NonlinearBicycleModel:
    """Nonlinear dynamic bicycle model at constant longitudinal speed.

    State: [X, Y, psi, v_y, r]
    Input: front steering angle delta_f [rad]
    """

    def __init__(
        self,
        parameters: VehicleParameters | None = None,
    ) -> None:
        self.p = parameters or VehicleParameters()

    def derivatives(
        self,
        state: State,
        steering_angle: float,
    ) -> State:
        state = np.asarray(state, dtype=float)

        if state.shape != (5,):
            raise ValueError("state must have shape (5,)")

        _, _, psi, lateral_speed, yaw_rate = state
        p = self.p

        delta = float(
            np.clip(
                steering_angle,
                -p.max_steering_angle,
                p.max_steering_angle,
            )
        )

        alpha_front = delta - np.arctan2(
            lateral_speed + p.front_axle_distance * yaw_rate,
            p.longitudinal_speed,
        )

        alpha_rear = -np.arctan2(
            lateral_speed - p.rear_axle_distance * yaw_rate,
            p.longitudinal_speed,
        )

        wheelbase = (
            p.front_axle_distance + p.rear_axle_distance
        )

        front_normal_force = (
            p.mass
            * p.gravity
            * p.rear_axle_distance
            / wheelbase
        )

        rear_normal_force = (
            p.mass
            * p.gravity
            * p.front_axle_distance
            / wheelbase
        )

        front_limit = (
            p.friction_coefficient * front_normal_force
        )
        rear_limit = (
            p.friction_coefficient * rear_normal_force
        )

        front_lateral_force = front_limit * np.tanh(
            p.front_cornering_stiffness
            * alpha_front
            / front_limit
        )

        rear_lateral_force = rear_limit * np.tanh(
            p.rear_cornering_stiffness
            * alpha_rear
            / rear_limit
        )

        x_dot = (
            p.longitudinal_speed * np.cos(psi)
            - lateral_speed * np.sin(psi)
        )

        y_dot = (
            p.longitudinal_speed * np.sin(psi)
            + lateral_speed * np.cos(psi)
        )

        psi_dot = yaw_rate

        lateral_acceleration = (
            (
                front_lateral_force * np.cos(delta)
                + rear_lateral_force
            )
            / p.mass
            - p.longitudinal_speed * yaw_rate
        )

        yaw_acceleration = (
            p.front_axle_distance
            * front_lateral_force
            * np.cos(delta)
            - p.rear_axle_distance
            * rear_lateral_force
        ) / p.yaw_inertia

        return np.array(
            [
                x_dot,
                y_dot,
                psi_dot,
                lateral_acceleration,
                yaw_acceleration,
            ],
            dtype=float,
        )

    def step(
        self,
        state: State,
        steering_angle: float,
        dt: float,
    ) -> State:
        """Advance the model using fourth-order Runge-Kutta."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")

        state = np.asarray(state, dtype=float)

        k1 = self.derivatives(state, steering_angle)
        k2 = self.derivatives(
            state + 0.5 * dt * k1,
            steering_angle,
        )
        k3 = self.derivatives(
            state + 0.5 * dt * k2,
            steering_angle,
        )
        k4 = self.derivatives(
            state + dt * k3,
            steering_angle,
        )

        return state + dt * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0