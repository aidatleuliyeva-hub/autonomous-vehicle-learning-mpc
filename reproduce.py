import argparse
import os
import subprocess
import sys
from time import perf_counter


CORE_EXPERIMENTS = [
    "experiments.generate_predictive_error_dataset",
    "experiments.train_elm",
    "experiments.tune_compensator",
    "experiments.compare_nominal_learning_mpc",
    "experiments.run_robustness_benchmark",
    "experiments.train_policy_elm",
    "experiments.compare_mpc_policy_approximation",
]

PRELIMINARY_EXPERIMENTS = [
    "experiments.plot_reference_paths",
    "experiments.analyze_linear_model",
    "experiments.run_lqr_tracking",
    "experiments.run_observer_tracking",
    "experiments.compare_lqr_mpc",
    "experiments.run_nonlinear_mpc_tracking",
]


def run_command(
    command: list[str],
    environment: dict[str, str],
) -> None:
    print("\n" + "=" * 72)
    print("Running:", " ".join(command))
    print("=" * 72)

    subprocess.run(
        command,
        check=True,
        env=environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the numerical results of the "
            "autonomous-vehicle learning MPC project."
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also run preliminary model, LQR, observer, "
            "and reference-path experiments."
        ),
    )

    arguments = parser.parse_args()

    environment = os.environ.copy()

    # Prevent interactive plotting windows.
    environment["MPLBACKEND"] = "Agg"

    start_time = perf_counter()

    run_command(
        [sys.executable, "-m", "pytest", "-q"],
        environment,
    )

    experiments = CORE_EXPERIMENTS

    if arguments.full:
        experiments = (
            PRELIMINARY_EXPERIMENTS
            + CORE_EXPERIMENTS
        )

    for experiment in experiments:
        run_command(
            [sys.executable, "-m", experiment],
            environment,
        )

    elapsed = perf_counter() - start_time

    print("\n" + "=" * 72)
    print("Reproduction completed successfully.")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print("=" * 72)


if __name__ == "__main__":
    main()