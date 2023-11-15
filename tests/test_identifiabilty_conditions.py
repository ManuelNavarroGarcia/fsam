import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tools import add_constant

from fsam.fsam_fit import FSAM


# In this test we check the performance of the predictions over `train` and
# `validation` sets when the basis are centered and not centered.
def test_predictions():
    n = 200
    seed = 0
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n, 1))
    y = (np.sin(2 * np.pi * X) + 0.5).flatten()
    train_size = 0.5

    conf_model = {"q": 2, "max_iter": 1}
    conf_gurobi = {"OutputFlag": 0, "threads": 1}

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_size, shuffle=False, random_state=seed
    )

    var_sec = FSAM(deg=[3], ord_d=[2], n_int=[40])
    _ = var_sec.fit(
        X=X_train,
        y=y_train,
        K=2,
        warm_start=False,
        conf_gurobi=conf_gurobi,
        conf_model=conf_model,
    )

    S_train = add_constant(var_sec._get_matrixS(X_train), has_constant="add")
    S_val = add_constant(var_sec._get_matrixS(X_val), has_constant="add")
    beta = np.linalg.solve(S_train.T @ S_train, S_train.T @ y_train)
    y_hat_val = S_val @ beta

    np.testing.assert_allclose(var_sec.predict(X_val).flatten(), y_hat_val, atol=1e-4)
    np.testing.assert_allclose(y_val, y_hat_val, atol=1e-4)
