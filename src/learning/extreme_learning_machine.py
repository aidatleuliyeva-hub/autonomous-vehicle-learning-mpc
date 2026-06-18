from pathlib import Path
from typing import Self

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


Array = NDArray[np.float64]


class ExtremeLearningMachineRegressor:
    """Single-hidden-layer Extreme Learning Machine.

    Hidden-layer parameters are generated randomly.
    Output weights are obtained through regularized
    least squares.
    """

    def __init__(
        self,
        hidden_neurons: int = 55,
        regularization: float = 1e-2,
        random_state: int = 42,
    ) -> None:
        if hidden_neurons <= 0:
            raise ValueError(
                "hidden_neurons must be positive"
            )

        if regularization <= 0.0:
            raise ValueError(
                "regularization must be positive"
            )

        self.hidden_neurons = hidden_neurons
        self.regularization = regularization
        self.random_state = random_state

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self.input_weights: Array | None = None
        self.hidden_biases: Array | None = None
        self.output_weights: Array | None = None

        self._single_output = False

    @staticmethod
    def _activation(values: Array) -> Array:
        return np.tanh(values)

    def fit(self, features: Array, targets: Array) -> Self:
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)

        if features.ndim != 2:
            raise ValueError(
                "features must be a two-dimensional array"
            )

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
            self._single_output = True
        elif targets.ndim == 2:
            self._single_output = False
        else:
            raise ValueError(
                "targets must be one- or two-dimensional"
            )

        if len(features) != len(targets):
            raise ValueError(
                "features and targets must have equal length"
            )

        if not np.all(np.isfinite(features)):
            raise ValueError(
                "features contain non-finite values"
            )

        if not np.all(np.isfinite(targets)):
            raise ValueError(
                "targets contain non-finite values"
            )

        scaled_features = (
            self.input_scaler.fit_transform(features)
        )

        scaled_targets = (
            self.output_scaler.fit_transform(targets)
        )

        random_generator = np.random.default_rng(
            self.random_state
        )

        input_dimension = scaled_features.shape[1]

        self.input_weights = random_generator.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(input_dimension),
            size=(
                input_dimension,
                self.hidden_neurons,
            ),
        )

        self.hidden_biases = random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=self.hidden_neurons,
        )

        hidden_output = self._activation(
            scaled_features @ self.input_weights
            + self.hidden_biases
        )

        regularized_matrix = (
            hidden_output.T @ hidden_output
            + self.regularization
            * np.eye(self.hidden_neurons)
        )

        right_hand_side = (
            hidden_output.T @ scaled_targets
        )

        self.output_weights = np.linalg.solve(
            regularized_matrix,
            right_hand_side,
        )

        return self

    def predict(self, features: Array) -> Array:
        if (
            self.input_weights is None
            or self.hidden_biases is None
            or self.output_weights is None
        ):
            raise RuntimeError(
                "The ELM model must be fitted before prediction"
            )

        features = np.asarray(features, dtype=float)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError(
                "features must be a two-dimensional array"
            )

        scaled_features = (
            self.input_scaler.transform(features)
        )

        hidden_output = self._activation(
            scaled_features @ self.input_weights
            + self.hidden_biases
        )

        scaled_predictions = (
            hidden_output @ self.output_weights
        )

        predictions = (
            self.output_scaler.inverse_transform(
                scaled_predictions
            )
        )

        if self._single_output:
            return predictions.ravel()

        return predictions

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self, output_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ExtremeLearningMachineRegressor":
        model = joblib.load(path)

        if not isinstance(
            model,
            ExtremeLearningMachineRegressor,
        ):
            raise TypeError(
                "The file does not contain an ELM model"
            )

        return model