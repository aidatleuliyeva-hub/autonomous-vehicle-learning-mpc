import numpy as np

from src.controllers.mpc_controller import MPCController


def test_mpc_zero_state_produces_zero_control() -> None:
    controller = MPCController()

    steering = controller.control(np.zeros(4))

    assert abs(steering) < 1e-5


def test_mpc_respects_steering_limit() -> None:
    controller = MPCController()

    steering = controller.control(
        np.array([0.0, 0.0, 10.0, 1.0])
    )

    limit = controller.model.p.max_steering_angle

    assert -limit <= steering <= limit


def test_mpc_respects_steering_rate_limit() -> None:
    controller = MPCController()

    previous_steering = 0.0

    steering = controller.control(
        np.array([0.0, 0.0, 5.0, 0.5]),
        previous_steering=previous_steering,
    )

    allowed_change = (
        controller.model.p.max_steering_rate
        * controller.sample_time
    )

    assert (
        abs(steering - previous_steering)
        <= allowed_change + 1e-5
    )


def test_mpc_reduces_tracking_error() -> None:
    controller = MPCController()

    state = np.array(
        [0.0, 0.0, 1.5, np.deg2rad(5.0)]
    )

    initial_error = np.linalg.norm(state[2:])
    previous_steering = 0.0

    for _ in range(100):
        steering = controller.control(
            state,
            previous_steering,
        )

        state = (
            controller.ad @ state
            + controller.bd.flatten() * steering
        )

        previous_steering = steering

    final_error = np.linalg.norm(state[2:])

    assert final_error < initial_error
    assert final_error < 0.05