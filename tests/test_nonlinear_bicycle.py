import numpy as np

from src.models.nonlinear_bicycle import (
    NonlinearBicycleModel,
)


def test_zero_steering_produces_straight_motion() -> None:
    model = NonlinearBicycleModel()
    state = np.zeros(5)

    next_state = model.step(
        state,
        steering_angle=0.0,
        dt=0.1,
    )

    np.testing.assert_allclose(
        next_state,
        [2.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_positive_steering_produces_positive_yaw_rate() -> None:
    model = NonlinearBicycleModel()
    state = np.zeros(5)

    next_state = model.step(
        state,
        steering_angle=np.deg2rad(5.0),
        dt=0.1,
    )

    assert next_state[4] > 0.0