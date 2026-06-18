import numpy as np

from src.controllers.lqr_controller import LQRController


def test_lqr_gain_has_correct_shape() -> None:
    controller = LQRController()

    assert controller.gain.shape == (1, 4)


def test_lqr_closed_loop_is_stable() -> None:
    controller = LQRController()

    assert np.all(
        np.abs(controller.closed_loop_eigenvalues) < 1.0
    )


def test_lqr_respects_steering_limit() -> None:
    controller = LQRController()

    state = np.array([0.0, 0.0, 10.0, 1.0])
    steering = controller.control(state)

    limit = controller.model.p.max_steering_angle

    assert -limit <= steering <= limit


def test_lqr_reduces_tracking_error() -> None:
    controller = LQRController()

    state = np.array(
        [0.0, 0.0, 1.5, np.deg2rad(5.0)]
    )

    initial_error = np.linalg.norm(state[2:])

    for _ in range(200):
        steering = controller.control(state)
        state = (
            controller.ad @ state
            + controller.bd.flatten() * steering
        )

    final_error = np.linalg.norm(state[2:])

    assert final_error < initial_error