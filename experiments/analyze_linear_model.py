from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from control import ctrb, obsv

from src.models.linear_bicycle import LinearBicycleModel


def main() -> None:
    model = LinearBicycleModel()
    a, b, c, _ = model.continuous_matrices()

    eigenvalues = np.linalg.eigvals(a)

    reachability_rank = np.linalg.matrix_rank(
        ctrb(a, b)
    )

    observability_rank = np.linalg.matrix_rank(
        obsv(a, c)
    )

    print("Continuous-time A matrix:")
    print(a)

    print("\nContinuous-time B matrix:")
    print(b)

    print("\nOpen-loop eigenvalues:")
    print(eigenvalues)

    print(
        f"\nReachability rank: "
        f"{reachability_rank}/{a.shape[0]}"
    )

    print(
        f"Observability rank: "
        f"{observability_rank}/{a.shape[0]}"
    )

    if np.all(np.real(eigenvalues) < 0.0):
        print("The open-loop system is asymptotically stable.")
    else:
        print(
            "The complete tracking-error system is not "
            "asymptotically stable."
        )
        print(
            "The two zero eigenvalues correspond to "
            "tracking-error integrators."
        )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.scatter(
        np.real(eigenvalues),
        np.imag(eigenvalues),
        marker="x",
        s=100,
        linewidths=2,
    )

    axis.axvline(0.0, color="black", linewidth=1)
    axis.axhline(0.0, color="black", linewidth=1)

    axis.set_title("Open-loop eigenvalues")
    axis.set_xlabel("Real part")
    axis.set_ylabel("Imaginary part")
    axis.grid(True, alpha=0.3)

    figure.savefig(
        output_directory / "open_loop_eigenvalues.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()