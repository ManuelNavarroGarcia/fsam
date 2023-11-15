import math

import numpy as np
import pytest

from fsam.fsam_fit import FSAM


def get_y_linear(X):
    return np.block(
        [
            [np.multiply(2, X[:, 0])],
            [np.zeros((X.shape[0],))],
            [np.multiply(1, X[:, 2])],
            [np.zeros((X.shape[0],))],
        ]
    ).T


def get_y_nonlinear(X):
    return np.block(
        [
            [np.zeros((X.shape[0],))],
            [np.power(X[:, 1], 4)],
            [np.zeros((X.shape[0],))],
            [2 * np.square(X[:, 3]) - 1],
        ]
    ).T


@pytest.mark.parametrize("optimality", [True])
def test_model_by_heuristic(optimality):
    out_obj = 43.580721310246936
    m, n = 4, 200
    train_size = 0.75
    np.random.seed(1)
    X = np.random.uniform(-1, 1, (n, m))
    y = (
        get_y_linear(X).sum(axis=1)
        + get_y_nonlinear(X).sum(axis=1)
        + np.random.normal(0, 0.5, n)
    )

    if optimality is True:
        conf_model = {"q": 2 * m, "max_iter": 1, "min_edf": 1}
    else:
        conf_model = {"q": 4, "max_iter": 10, "min_edf": 1}

    var_sec = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[30] * m)
    var_sec.fit(
        X=X,
        y=y,
        K=list(range(1, 9)),
        train_size=train_size,
        warm_start=False,
        conf_gurobi={"OutputFlag": 0, "threads": 1},
        conf_model=conf_model,
        scale_y=False,
    )

    var_sec_ = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[30] * m)
    var_sec_.fit(
        X=X,
        y=y,
        K=4,
        train_size=train_size,
        warm_start=False,
        conf_gurobi={"OutputFlag": 0, "threads": 1},
        conf_model=conf_model,
        scale_y=False,
    )
    out_obj_ = 51.59991232868817

    np.testing.assert_allclose(var_sec.sol["z"], [1, 0, 1, 0, 0, 1, 0, 1])
    np.testing.assert_allclose(var_sec.sol["obj"], out_obj, atol=1e-1)
    np.testing.assert_allclose(var_sec_.sol["z"], [1, 0, 1, 0, 0, 1, 0, 1])
    # Another objective function value is expected since the entire data set is
    # used for training (no validation)
    np.testing.assert_allclose(var_sec_.sol["obj"], out_obj_, atol=1e-1)


def test_scale_y():
    m, n = 4, 50
    np.random.seed(1)
    X = np.random.uniform(-1, 1, (n, m))
    y = get_y_linear(X).sum(axis=1) + get_y_nonlinear(X).sum(axis=1)
    if math.isclose(y.mean(), 0) and math.isclose(y.std(), 1):
        raise ValueError("`y` cannot be already standardized.")

    conf_model = {"q": 2 * m, "max_iter": 1, "min_edf": 1}
    conf_gurobi = {"OutputFlag": 0, "threads": 1}

    var_sec = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[10] * m)
    var_sec.fit(
        X=X, y=y, K=4, conf_gurobi=conf_gurobi, conf_model=conf_model, scale_y=False
    )
    y_not_scaled = var_sec.predict(X)
    var_sec.fit(
        X=X, y=y, K=4, conf_gurobi=conf_gurobi, conf_model=conf_model, scale_y=True
    )
    y_scaled = var_sec.predict(X)
    np.testing.assert_allclose(y_not_scaled, y_scaled, atol=1e-6)


@pytest.mark.parametrize("compute_coef_bounds", [True, False])
@pytest.mark.parametrize("frac_row_bounds", [0.0, 0.489, 1.0])
def test_model_by_constraints(compute_coef_bounds, frac_row_bounds):
    out_obj = 6.252350622547096
    m, n = 4, 29
    np.random.seed(1)
    X = np.random.uniform(-1, 1, (n, m))
    y = (
        get_y_linear(X).sum(axis=1)
        + get_y_nonlinear(X).sum(axis=1)
        + np.random.normal(0, 0.5, n)
    )

    var_sec = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[10] * m)
    var_sec.fit(
        X=X,
        y=y,
        K=4,
        warm_start=False,
        conf_gurobi={"OutputFlag": 0, "threads": 1},
        conf_model={"q": 2 * m, "max_iter": 1},
        scale_y=False,
        compute_coef_bounds=compute_coef_bounds,
        frac_row_bounds=frac_row_bounds,
    )
    np.testing.assert_allclose(var_sec.sol["obj"], out_obj, atol=1e-4)
