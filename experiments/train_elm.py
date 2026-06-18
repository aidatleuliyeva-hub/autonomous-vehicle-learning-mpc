from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.learning.extreme_learning_machine import (
    ExtremeLearningMachineRegressor,
)
from src.simulation.dataset_generator import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


def root_mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    return np.sqrt(
        np.mean((actual - predicted) ** 2, axis=0)
    )


def main() -> None:
    dataset = pd.read_csv(
        "data/predictive_error_dataset.csv"
    )

    features = dataset[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    targets = dataset[
        TARGET_COLUMNS
    ].to_numpy(dtype=float)

    (
        train_features,
        test_features,
        train_targets,
        test_targets,
    ) = train_test_split(
        features,
        targets,
        test_size=250,
        random_state=42,
        shuffle=True,
    )

    (
        fit_features,
        validation_features,
        fit_targets,
        validation_targets,
    ) = train_test_split(
        train_features,
        train_targets,
        test_size=200,
        random_state=43,
        shuffle=True,
    )

    hidden_candidates = [30, 55, 100, 150]

    regularization_candidates = [
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    ]

    target_scale = np.maximum(
        np.std(validation_targets, axis=0),
        1e-12,
    )

    best_score = float("inf")
    best_hidden_neurons = 55
    best_regularization = 1e-2

    for hidden_neurons in hidden_candidates:
        for regularization in regularization_candidates:
            candidate = (
                ExtremeLearningMachineRegressor(
                    hidden_neurons=hidden_neurons,
                    regularization=regularization,
                    random_state=42,
                )
            )

            candidate.fit(
                fit_features,
                fit_targets,
            )

            validation_predictions = (
                candidate.predict(
                    validation_features
                )
            )

            validation_rmse = (
                root_mean_squared_error(
                    validation_targets,
                    validation_predictions,
                )
            )

            normalized_score = float(
                np.mean(
                    validation_rmse / target_scale
                )
            )

            if normalized_score < best_score:
                best_score = normalized_score
                best_hidden_neurons = (
                    hidden_neurons
                )
                best_regularization = (
                    regularization
                )

    print(
        "Selected ELM configuration: "
        f"hidden_neurons={best_hidden_neurons}, "
        f"regularization={best_regularization}"
    )

    print(
        f"Validation normalized RMSE: "
        f"{best_score:.6f}"
    )

    model = ExtremeLearningMachineRegressor(
        hidden_neurons=best_hidden_neurons,
        regularization=best_regularization,
        random_state=42,
    )

    model.fit(
        train_features,
        train_targets,
    )

    predictions = model.predict(test_features)

    baseline_prediction = np.tile(
        np.mean(train_targets, axis=0),
        (len(test_targets), 1),
    )

    baseline_rmse = root_mean_squared_error(
        test_targets,
        baseline_prediction,
    )

    elm_rmse = root_mean_squared_error(
        test_targets,
        predictions,
    )

    elm_mae = mean_absolute_error(
        test_targets,
        predictions,
        multioutput="raw_values",
    )

    elm_r2 = r2_score(
        test_targets,
        predictions,
        multioutput="raw_values",
    )

    print(f"\nTraining samples: {len(train_features)}")
    print(f"Testing samples: {len(test_features)}")

    print("\nMean-baseline RMSE:")

    for name, value in zip(
        TARGET_COLUMNS,
        baseline_rmse,
    ):
        print(f"  {name}: {value:.8f}")

    print("\nELM RMSE:")

    for name, value in zip(
        TARGET_COLUMNS,
        elm_rmse,
    ):
        print(f"  {name}: {value:.8f}")

    print("\nELM MAE:")

    for name, value in zip(
        TARGET_COLUMNS,
        elm_mae,
    ):
        print(f"  {name}: {value:.8f}")

    print("\nELM R2:")

    for name, value in zip(
        TARGET_COLUMNS,
        elm_r2,
    ):
        print(f"  {name}: {value:.4f}")

    model_path = Path(
        "data/models/elm_predictive_error.joblib"
    )

    model.save(model_path)

    print(f"\nModel saved to: {model_path}")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    labels = [
        "Lateral prediction error [m]",
        "Heading prediction error [rad]",
    ]

    for index, axis in enumerate(axes):
        actual = test_targets[:, index]
        predicted = predictions[:, index]

        axis.scatter(
            actual,
            predicted,
            alpha=0.6,
            s=18,
        )

        minimum = min(
            actual.min(),
            predicted.min(),
        )

        maximum = max(
            actual.max(),
            predicted.max(),
        )

        axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            "r--",
            label="Ideal prediction",
        )

        axis.set_xlabel(
            f"Actual {labels[index]}"
        )

        axis.set_ylabel(
            f"Predicted {labels[index]}"
        )

        axis.set_title(
            f"ELM prediction, "
            f"R² = {elm_r2[index]:.3f}"
        )

        axis.legend()
        axis.grid(True, alpha=0.3)

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    figure.savefig(
        results_directory
        / "elm_prediction_performance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()