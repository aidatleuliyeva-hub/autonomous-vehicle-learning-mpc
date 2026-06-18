import numpy as np
from numpy.typing import NDArray
from scipy.signal import cont2discrete

from .vehicle_parameters import VehicleParameters


Matrix = NDArray[np.float64]


class LinearBicycleModel:
    """Linear lateral path-tracking model.

    State: [v_y, r, e_y, e_psi]
        v_y   - lateral velocity
        r     - yaw rate
        e_y   - lateral tracking error
        e_psi - heading tracking error

    Input: steering angle delta_f
    """

    def __init__(
        self,
        parameters: VehicleParameters | None = None,
    ) -> None:
        self.p = parameters or VehicleParameters()

    def continuous_matrices(
        self,
    ) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        p = self.p

        m = p.mass
        iz = p.yaw_inertia
        lf = p.front_axle_distance
        lr = p.rear_axle_distance
        cf = p.front_cornering_stiffness
        cr = p.rear_cornering_stiffness
        vx = p.longitudinal_speed

        a_matrix = np.array(
            [
                [
                    -(cf + cr) / (m * vx),
                    (-lf * cf + lr * cr) / (m * vx) - vx,
                    0.0,
                    0.0,
                ],
                [
                    (-lf * cf + lr * cr) / (iz * vx),
                    -(lf**2 * cf + lr**2 * cr) / (iz * vx),
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    vx,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ],
            dtype=float,
        )

        b_matrix = np.array(
            [
                [cf / m],
                [lf * cf / iz],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )

        # Measured outputs: lateral and heading errors.
        c_matrix = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        d_matrix = np.zeros((2, 1), dtype=float)

        return a_matrix, b_matrix, c_matrix, d_matrix

    def curvature_disturbance_matrix(self) -> Matrix:
        """Influence of reference-path curvature on the model."""

        return np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [-self.p.longitudinal_speed],
            ],
            dtype=float,
        )

    def discrete_matrices(
        self,
        sample_time: float,
    ) -> tuple[Matrix, Matrix, Matrix, Matrix]:
        if sample_time <= 0.0:
            raise ValueError("sample_time must be positive")

        a, b, c, d = self.continuous_matrices()

        ad, bd, cd, dd, _ = cont2discrete(
            (a, b, c, d),
            sample_time,
            method="zoh",
        )

        return ad, bd, cd, dd
    
    def discrete_tracking_matrices(
        self,
        sample_time: float,
    ) -> tuple[Matrix, Matrix, Matrix]:
        """Discretize steering and curvature inputs."""

        if sample_time <= 0.0:
            raise ValueError("sample_time must be positive")

        a, b, c, _ = self.continuous_matrices()
        curvature_input = (
            self.curvature_disturbance_matrix()
        )

        combined_inputs = np.hstack(
            [b, curvature_input]
        )

        combined_feedthrough = np.zeros((2, 2))

        ad, combined_bd, _, _, _ = cont2discrete(
            (
                a,
                combined_inputs,
                c,
                combined_feedthrough,
            ),
            sample_time,
            method="zoh",
        )

        steering_bd = combined_bd[:, [0]]
        curvature_bd = combined_bd[:, [1]]

        return ad, steering_bd, curvature_bd