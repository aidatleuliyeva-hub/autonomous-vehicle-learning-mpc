from pathlib import Path
from time import perf_counter

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
    POLICY_FEATURE_COLUMNS,
    POLICY_TARGET_COLUMN,
)


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )


def main() -> None:
    dataset = pd.read_csv(
        "data/predictive_error_dataset.csv"
    )

    features = dataset[
        POLICY_FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    targets = dataset[
        POLICY_TARGET_COLUMN
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
        random_state=51,
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
        random_state=52,
        shuffle=True,
    )

    hidden_candidates = [55, 100, 150, 250]

    regularization_candidates = [
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    ]

    best_score = float("inf")
    best_hidden = 100
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

            score = rmse(
                validation_targets,
                validation_predictions,
            )

            if score < best_score:
                best_score = score
                best_hidden = hidden_neurons
                best_regularization = regularization

    print(
        "Selected policy ELM: "
        f"hidden_neurons={best_hidden}, "
        f"regularization={best_regularization}"
    )

    model = ExtremeLearningMachineRegressor(
        hidden_neurons=best_hidden,
        regularization=best_regularization,
        random_state=42,
    )

    model.fit(
        train_features,
        train_targets,
    )

    predictions = model.predict(test_features)

    steering_limit = np.deg2rad(12.0)

    predictions = np.clip(
        predictions,
        -steering_limit,
        steering_limit,
    )

    baseline = np.full(
        len(test_targets),
        np.mean(train_targets),
    )

    baseline_rmse = rmse(
        test_targets,
        baseline,
    )

    policy_rmse = rmse(
        test_targets,
        predictions,
    )

    policy_mae = mean_absolute_error(
        test_targets,
        predictions,
    )

    policy_r2 = r2_score(
        test_targets,
        predictions,
    )

    start_time = perf_counter()

    repetitions = 100

    for _ in range(repetitions):
        model.predict(test_features)

    elapsed = perf_counter() - start_time

    inference_time_ms = (
        elapsed
        / repetitions
        / len(test_features)
        * 1000.0
    )

    violations = int(
        np.sum(
            np.abs(predictions) > steering_limit
        )
    )

    print(f"\nTraining samples: {len(train_features)}")
    print(f"Testing samples: {len(test_features)}")

    print(
        "\nMean-policy baseline RMSE: "
        f"{np.rad2deg(baseline_rmse):.6f} deg"
    )

    print(
        "ELM policy RMSE: "
        f"{np.rad2deg(policy_rmse):.6f} deg"
    )

    print(
        "ELM policy MAE: "
        f"{np.rad2deg(policy_mae):.6f} deg"
    )

    print(f"ELM policy R2: {policy_r2:.6f}")

    print(
        "Average ELM inference time: "
        f"{inference_time_ms:.6f} ms/sample"
    )

    print(
        f"Steering constraint violations: "
        f"{violations}"
    )

    # Evaluate first, then refit on all available data.
    final_model = ExtremeLearningMachineRegressor(
        hidden_neurons=best_hidden,
        regularization=best_regularization,
        random_state=42,
    )

    final_model.fit(features, targets)

    model_path = Path(
        "data/models/elm_mpc_policy.joblib"
    )

    final_model.save(model_path)

    print(f"Policy model saved to: {model_path}")

    actual_degrees = np.rad2deg(test_targets)
    predicted_degrees = np.rad2deg(predictions)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    axes[0].scatter(
        actual_degrees,
        predicted_degrees,
        alpha=0.6,
        s=18,
    )

    minimum = min(
        actual_degrees.min(),
        predicted_degrees.min(),
    )

    maximum = max(
        actual_degrees.max(),
        predicted_degrees.max(),
    )

    axes[0].plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Ideal prediction",
    )

    axes[0].set_xlabel("Actual MPC steering [deg]")
    axes[0].set_ylabel("ELM steering [deg]")
    axes[0].set_title(
        f"MPC policy approximation, "
        f"R²={policy_r2:.3f}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        predicted_degrees - actual_degrees,
        bins=50,
    )

    axes[1].set_xlabel("Steering prediction error [deg]")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Policy approximation error")
    axes[1].grid(True, alpha=0.3)

    results_directory = Path("results")
    results_directory.mkdir(exist_ok=True)

    figure.savefig(
        results_directory
        / "elm_mpc_policy_approximation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()