import numpy as np
import pytest

from src.learning.extreme_learning_machine import (
    ExtremeLearningMachineRegressor,
)


def test_elm_learns_simple_regression() -> None:
    random_generator = np.random.default_rng(42)

    features = random_generator.uniform(
        -0.5,
        0.5,
        size=(400, 3),
    )

    targets = np.column_stack(
        [
            2.0 * features[:, 0]
            - features[:, 1],
            features[:, 1]
            + 0.5 * features[:, 2],
        ]
    )

    model = ExtremeLearningMachineRegressor(
        hidden_neurons=100,
        regularization=1e-4,
    )

    model.fit(features, targets)
    predictions = model.predict(features)

    rmse = np.sqrt(
        np.mean((targets - predictions) ** 2)
    )

    assert predictions.shape == targets.shape
    assert rmse < 0.05


def test_elm_requires_fit_before_predict() -> None:
    model = ExtremeLearningMachineRegressor()

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 8)))


def test_elm_save_and_load(tmp_path) -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [2.0],
        ]
    )

    model = ExtremeLearningMachineRegressor(
        hidden_neurons=10
    )
    model.fit(features, targets)

    model_path = tmp_path / "elm.joblib"
    model.save(model_path)

    loaded_model = (
        ExtremeLearningMachineRegressor.load(
            model_path
        )
    )

    np.testing.assert_allclose(
        model.predict(features),
        loaded_model.predict(features),
    )