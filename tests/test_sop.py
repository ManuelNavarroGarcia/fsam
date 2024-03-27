import numpy as np
from statsmodels.tools import add_constant

from fsam.fsam_fit import FSAM
from fsam.sop import sop_fit


def test_sop():
    # This is the result obtained by saving the data and using the following
    # code from R:
    # sopfit <- sop(formula = y ~ f(x0, nseg = 10)
    #                        + f(x1, nseg = 10) + f(x2, nseg = 10),
    #            data = data,
    #            fit=TRUE)
    # print(sopfit$out$vc)
    exp_out = {
        "phi": 0.009877274,
        "tau": np.array([0.020848260, 0.014017075, 0.015953217]),
        "aic": 74.21990047583813,
        "edf": np.array([3.825383, 3.438150, 3.473035]),
    }
    # And the expected output considering the smoothing parameters
    exp_out2 = {
        "sp": np.array([0.47376971, 0.70466014, 0.61913995]),
        "aic_nonlinear": 68.2354979987828,
        "aic_lineal": 12.637679533133062,
        "edf": np.array([3.825383, 3.438150, 3.473035]),
    }
    exp_y = np.array(
        [
            0.36286385,
            0.25155531,
            1.48478864,
            0.49744108,
            1.40228542,
            1.93809025,
            1.7418926,
            0.80577551,
            1.17680397,
            0.79808525,
            0.6202125,
            0.94959137,
            0.97221624,
            0.33039621,
            1.04877948,
            1.03601552,
            0.34118997,
            1.46855592,
            0.9643729,
            0.58759908,
            1.19280688,
            0.99637524,
            1.5419435,
            1.43987575,
            1.16759189,
            1.53964945,
            0.90864388,
            0.96540528,
            0.34634908,
            1.43702627,
            0.84185524,
            0.83249471,
            1.35890048,
            1.25407387,
            1.27009263,
            0.19652358,
            1.09995147,
        ]
    )
    m, n = 3, 37
    np.random.seed(0)
    X = np.random.uniform(-1, 1, (n, m))
    y = np.sum(np.square(X), axis=1) + np.random.normal(0, 0.1, n)

    var_sec = FSAM(deg=[3] * m, ord_d=[2] * m, n_int=[10] * m, prediction=[{}] * m)
    var_sec.fit(
        X,
        y,
        K=6,
        conf_model={"criterion": "mse", "q": 6, "max_iter": 1},
        conf_gurobi={"OutputFlag": 0, "threads": 1},
    )
    S = var_sec.matrixS

    out = sop_fit(
        y=y, X=add_constant(S[:, :m], has_constant="add"), Z=S[:, m:], G=var_sec.ds
    )
    for key1, key2 in zip(out, exp_out):
        np.testing.assert_allclose(out[key1], out[key2], atol=1e-6)

    out2 = var_sec._get_sp(y=y, x_vars=range(m))
    for key1, key2 in zip(out2, exp_out2):
        np.testing.assert_allclose(out2[key1], out2[key2], atol=1e-6)

    np.testing.assert_allclose(var_sec.predict(X).flatten(), exp_y, atol=1e-5)
