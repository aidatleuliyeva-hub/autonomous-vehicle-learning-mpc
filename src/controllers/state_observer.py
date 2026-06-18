import numpy as np
from control import place
from numpy.typing import NDArray

from src.models.linear_bicycle import LinearBicycleModel


Array = NDArray[np.float64]


class StateObserver:
    """Discrete-time Luenberger state observer."""

    def __init__(
        self,
        model: LinearBicycleModel | None = None,
        sample_time: float = 0.05,
        observer_poles: Array | None = None,
    ) -> None:
        self.model = model or LinearBicycleModel()
        self.sample_time = sample_time

        self.ad, self.bd, self.cd, _ = (
            self.model.discrete_matrices(sample_time)
        )

        poles = (
            np.array([0.25, 0.35, 0.45, 0.55])
            if observer_poles is None
            else np.asarray(observer_poles)
        )

        self.gain = np.asarray(
            place(self.ad.T, self.cd.T, poles).T,
            dtype=float,
        )

        self.error_eigenvalues = np.linalg.eigvals(
            self.ad - self.gain @ self.cd
        )

        self.state = np.zeros(4, dtype=float)

    def reset(self, state: Array | None = None) -> None:
        if state is None:
            self.state = np.zeros(4, dtype=float)
            return

        state = np.asarray(state, dtype=float)

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")

        self.state = state.copy()

    def update(
        self,
        measurement: Array,
        steering_angle: float,
    ) -> Array:
        measurement = np.asarray(measurement, dtype=float)

        if measurement.shape != (2,):
            raise ValueError(
                "measurement must have shape (2,)"
            )

        innovation = (
            measurement - self.cd @ self.state
        )

        self.state = (
            self.ad @ self.state
            + self.bd.flatten() * steering_angle
            + self.gain @ innovation
        )

        return self.state.copy()