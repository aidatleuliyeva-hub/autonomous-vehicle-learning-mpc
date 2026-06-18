import numpy as np
from control import dlqr
from numpy.typing import NDArray

from src.models.linear_bicycle import LinearBicycleModel


Array = NDArray[np.float64]


class LQRController:
    """Discrete-time optimal state-feedback controller."""

    def __init__(
        self,
        model: LinearBicycleModel | None = None,
        sample_time: float = 0.05,
        q_matrix: Array | None = None,
        r_matrix: Array | None = None,
    ) -> None:
        self.model = model or LinearBicycleModel()
        self.sample_time = sample_time

        self.ad, self.bd, _, _ = (
            self.model.discrete_matrices(sample_time)
        )

        self.q_matrix = (
            np.diag([0.5, 0.5, 20.0, 10.0])
            if q_matrix is None
            else np.asarray(q_matrix, dtype=float)
        )

        self.r_matrix = (
            np.array([[5.0]])
            if r_matrix is None
            else np.asarray(r_matrix, dtype=float)
        )

        gain, riccati_solution, eigenvalues = dlqr(
            self.ad,
            self.bd,
            self.q_matrix,
            self.r_matrix,
        )

        self.gain = np.asarray(gain, dtype=float)
        self.riccati_solution = np.asarray(
            riccati_solution,
            dtype=float,
        )
        self.closed_loop_eigenvalues = np.asarray(
            eigenvalues,
        )

    def control(self, state: Array) -> float:
        state = np.asarray(state, dtype=float)

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")

        steering_angle = float(
            -(self.gain @ state).item()
        )

        steering_limit = (
            self.model.p.max_steering_angle
        )

        return float(
            np.clip(
                steering_angle,
                -steering_limit,
                steering_limit,
            )
        )