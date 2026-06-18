import numpy as np

from src.controllers.state_observer import StateObserver


def test_observer_gain_has_correct_shape() -> None:
    observer = StateObserver()

    assert observer.gain.shape == (4, 2)


def test_observer_error_dynamics_are_stable() -> None:
    observer = StateObserver()

    assert np.all(
        np.abs(observer.error_eigenvalues) < 1.0
    )


def test_observer_estimation_error_converges() -> None:
    observer = StateObserver()

    true_state = np.array(
        [0.3, -0.1, 1.0, np.deg2rad(5.0)]
    )

    initial_error = np.linalg.norm(
        true_state - observer.state
    )

    for _ in range(100):
        measurement = observer.cd @ true_state

        observer.update(
            measurement,
            steering_angle=0.0,
        )

        true_state = observer.ad @ true_state

    final_error = np.linalg.norm(
        true_state - observer.state
    )

    assert final_error < initial_error
    assert final_error < 1e-4