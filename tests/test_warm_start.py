import math

import numpy as np

from fsam.fsam_fit import FSAM


def dicts_almost_equal(d1: dict, d2: dict, tol: float = 1e-09):
    return d1.keys() == d2.keys() and all(
        (
            math.isclose(d1[k], d2[k], rel_tol=tol)
            if isinstance(d1[k], float)
            else d1[k] == d2[k]
        )
        for k in d1
    )


def test_warm_start():
    m, n = 4, 200
    train_size = 0.75
    np.random.seed(1)
    X = np.random.uniform(-1, 1, (n, m))
    y = 2 * X[:, 0] + np.power(X[:, 1], 4) + X[:, 2] + 2 * np.square(X[:, 3]) - 1
    conf_model = {"q": 2 * m, "max_iter": 1, "min_edf": 1}

    out1 = {"z": [1, 0, 1, 0, 1, 0, 0, 1], "obj": 14.864223031795937}
    var_sec = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[10] * m)
    var_sec.fit(
        X=X,
        y=y,
        K=[4],
        train_size=train_size,
        warm_start=True,
        conf_gurobi={"OutputFlag": 0, "threads": 1},
        conf_model=conf_model,
        scale_y=False,
    )
    assert dicts_almost_equal(
        {k: v for k, v in var_sec.init_sol.items() if k != "theta"}, out1
    )

    out2 = {"z": [1, 0, 1, 0, 0, 1, 0, 1], "obj": 8.803704859486391}
    var_sec_ = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[10] * m)
    var_sec_.fit(
        X=X,
        y=y,
        K=[4],
        train_size=train_size,
        warm_start=var_sec.sol,
        conf_gurobi={"OutputFlag": 0, "threads": 1},
        conf_model=conf_model,
        scale_y=False,
    )
    assert dicts_almost_equal(
        {k: v for k, v in var_sec_.init_sol.items() if k != "theta"}, out2
    )
