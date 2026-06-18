import numpy as np
from control import ctrb, obsv

from src.models.linear_bicycle import LinearBicycleModel


def test_continuous_matrix_dimensions() -> None:
    model = LinearBicycleModel()
    a, b, c, d = model.continuous_matrices()

    assert a.shape == (4, 4)
    assert b.shape == (4, 1)
    assert c.shape == (2, 4)
    assert d.shape == (2, 1)


def test_linear_model_is_reachable() -> None:
    model = LinearBicycleModel()
    a, b, _, _ = model.continuous_matrices()

    reachability_matrix = ctrb(a, b)

    assert np.linalg.matrix_rank(reachability_matrix) == 4


def test_linear_model_is_observable() -> None:
    model = LinearBicycleModel()
    a, _, c, _ = model.continuous_matrices()

    observability_matrix = obsv(a, c)

    assert np.linalg.matrix_rank(observability_matrix) == 4


def test_discrete_matrix_dimensions() -> None:
    model = LinearBicycleModel()
    ad, bd, cd, dd = model.discrete_matrices(0.05)

    assert ad.shape == (4, 4)
    assert bd.shape == (4, 1)
    assert cd.shape == (2, 4)
    assert dd.shape == (2, 1)