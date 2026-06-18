import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from src.models.linear_bicycle import LinearBicycleModel


Array = NDArray[np.float64]


class MPCController:
    """Constrained linear Model Predictive Controller."""

    def __init__(
        self,
        model: LinearBicycleModel | None = None,
        sample_time: float = 0.05,
        prediction_horizon: int = 20,
        q_matrix: Array | None = None,
        r_weight: float = 2.0,
        steering_rate_weight: float = 20.0,
    ) -> None:
        if prediction_horizon <= 0:
            raise ValueError(
                "prediction_horizon must be positive"
            )

        self.model = model or LinearBicycleModel()
        self.sample_time = sample_time
        self.prediction_horizon = prediction_horizon

        self.ad, self.bd, self.ed = (
            self.model.discrete_tracking_matrices(
                sample_time
            )
        )

        self.q_matrix = (
            np.diag([0.5, 0.5, 25.0, 12.0])
            if q_matrix is None
            else np.asarray(q_matrix, dtype=float)
        )

        self.r_weight = r_weight
        self.steering_rate_weight = (
            steering_rate_weight
        )

        self._states = cp.Variable(
            (4, prediction_horizon + 1)
        )
        self._inputs = cp.Variable(
            (1, prediction_horizon)
        )

        self._initial_state = cp.Parameter(4)
        self._previous_input = cp.Parameter()
        self._curvature_preview = cp.Parameter(
            prediction_horizon
        )

        steering_limit = (
            self.model.p.max_steering_angle
        )

        steering_rate_step_limit = (
            self.model.p.max_steering_rate
            * sample_time
        )

        constraints = [
            self._states[:, 0]
            == self._initial_state
        ]

        objective = 0.0

        for step in range(prediction_horizon):
            objective += cp.quad_form(
                self._states[:, step],
                self.q_matrix,
            )

            objective += (
                self.r_weight
                * cp.square(self._inputs[0, step])
            )

            constraints.append(
                self._states[:, step + 1]
                == self.ad @ self._states[:, step]
                + self.bd.flatten()
                * self._inputs[0, step]
                + self.ed.flatten()
                * self._curvature_preview[step]
            )

            constraints.extend(
                [
                    self._inputs[0, step]
                    <= steering_limit,
                    self._inputs[0, step]
                    >= -steering_limit,
                ]
            )

            if step == 0:
                input_change = (
                    self._inputs[0, step]
                    - self._previous_input
                )
            else:
                input_change = (
                    self._inputs[0, step]
                    - self._inputs[0, step - 1]
                )

            objective += (
                self.steering_rate_weight
                * cp.square(input_change)
            )

            constraints.extend(
                [
                    input_change
                    <= steering_rate_step_limit,
                    input_change
                    >= -steering_rate_step_limit,
                ]
            )

        objective += cp.quad_form(
            self._states[:, prediction_horizon],
            5.0 * self.q_matrix,
        )

        self._problem = cp.Problem(
            cp.Minimize(objective),
            constraints,
        )

    def control(
        self,
        state: Array,
        previous_steering: float = 0.0,
        curvature_preview: Array | None = None,
    ) -> float:
        state = np.asarray(state, dtype=float)

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")

        self._initial_state.value = state
        self._previous_input.value = previous_steering

        if curvature_preview is None:
            curvature_preview = np.zeros(
                self.prediction_horizon
            )

        curvature_preview = np.asarray(
            curvature_preview,
            dtype=float,
        )

        if curvature_preview.shape != (
            self.prediction_horizon,
        ):
            raise ValueError(
                "curvature_preview must have shape "
                f"({self.prediction_horizon},)"
            )

        self._curvature_preview.value = curvature_preview

        self._problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            max_iter=50_000,
            eps_abs=1e-5,
            eps_rel=1e-5,
        )

        valid_statuses = {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }

        if self._problem.status not in valid_statuses:
            raise RuntimeError(
                "MPC optimization failed with status: "
                f"{self._problem.status}"
            )

        if self._inputs.value is None:
            raise RuntimeError(
                "MPC returned no control solution"
            )

        steering = float(self._inputs.value[0, 0])

        limit = self.model.p.max_steering_angle

        return float(
            np.clip(steering, -limit, limit)
        )