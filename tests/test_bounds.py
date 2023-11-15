import gurobipy as gp
import numpy as np
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from gurobipy import GRB

from fsam.fsam_fit import (
    FSAM,
    reparametrize_basis
)


@pytest.fixture
def matricesXZ(n: int = 41, m: int = 3) -> dict[str, list[np.ndarray]]:
    """Generate matrices X and Z, i.e., the unpenalized and penalized parts of
    the reparametrization of the B-spline basis, respectively.

    Parameters
    ----------
    n : int, optional
        Sample size, by default 41
    m : int, optional
        Number of features, by default 3

    Returns
    -------
    dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    """
    np.random.seed(5)
    x = np.random.uniform(-1, 1, (n, m))
    # Generate number of inner knots randomly from 5 to 10
    n_int = np.random.randint(low=5, high=10, size=(m,))

    # The first element of "X" must be the intercept term, i.e., a column of
    # ones
    Xs = [np.ones((n, 1))]
    Zs = []
    for sample, k in zip(x.T, n_int):
        # Create the B-spline basis and reparametrize it
        bspline = BsplineBasis(deg=3, xsample=sample, n_int=k)
        rep_mat = reparametrize_basis(bspline=bspline, ord_d=2, x=sample)
        # For identificability purposes, one column of `X` must be discarded
        Xs.append(rep_mat["X"][:, 1:])
        Zs.append(rep_mat["Z"])
    return {"X": Xs, "Z": Zs}


@pytest.fixture
def penalty(matricesXZ: dict[str, list[np.ndarray]]) -> np.ndarray:
    """Generate a random penalty term, which has 0s on the unpenalized parts and
    random positive numbers in the penalized parts. It is assumed that the
    penalty term is a diagonal matrix.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.

    Returns
    -------
    np.ndarray
        The penalty matrix.
    """
    return np.diag(
        np.concatenate(
            [
                np.zeros((len(matricesXZ["X"]),)),
                np.abs(
                    np.random.rand(np.add.reduce([Z.shape[1] for Z in matricesXZ["Z"]]))
                ),
            ],
        )
    )


@pytest.fixture
def y(matricesXZ: dict[str, list[np.ndarray]]) -> np.ndarray:
    """Generate a random response sample from a Gaussian distribution

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.

    Returns
    -------
    np.ndarray
        The response sample.
    """
    np.random.seed(5)
    return np.random.normal(0, 1, matricesXZ["X"][0].shape[0])


def bounds_gurobi(
    matricesXZ: dict[str, list[np.ndarray]],
    penalty: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    ub: float = 100,
    **kwargs,
) -> np.ndarray:
    """Compute the optimal solution of problems (2.14) or (2.15) in [1] directly
    implementing the optimization problems in Gurobi. The kwargs are referred to
    arguments of the optimizer.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    penalty : np.ndarray
        The penalty matrix.
    y : np.ndarray
        The response sample.
    c : np.ndarray
        The vector multiplying the decision variables in the objective function
        of (2.14) or (2.15). For the first, `c` is one element from the
        canonical basis of R^n, and for the second, `c` corresponds to one row
        from the data matrix.
    ub : float, optional
        The upper bound of the regression problem (1.1) in [1], by default 100.

    Returns
    -------
    np.ndarray
        An array containing the bounds of the problems. The first column refers
        to the minimization problem, while the second corresponds to the
        maximization problem.
    References
    ----------
    ... [1] Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset
    selection via a modern optimization lens. The annals of statistics, 44(2),
    813-852.
    """
    matrixS = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)

    M = gp.Model()
    for key, v in kwargs.items():
        _ = M.setParam(key, v)
    _ = M.setParam("OutputFlag", 0)

    theta = np.array(
        M.addVars(
            matrixS.shape[1],
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="theta",
        ).values()
    )

    M.addQConstr(
        np.dot(y - matrixS @ np.array(theta), y - matrixS @ np.array(theta))
        + np.array(theta) @ penalty @ np.array(theta)
        <= ub
    )
    bounds = np.empty((c.shape[0], 2))
    for i in range(c.shape[0]):
        for j, sense in enumerate((GRB.MINIMIZE, GRB.MAXIMIZE)):
            _ = M.setObjective(c[i, :] @ theta, sense)
            _ = M.optimize()
            bounds[i, j] = M.getObjective().getValue()

    return bounds


def bounds_lagrange(
    matricesXZ: dict[str, list[np.ndarray]],
    penalty: np.ndarray,
    y: np.ndarray,
    ub: float = 100,
    row_bounds: bool = False,
):
    """Compute the optimal solution of problems (2.15) in [1] using Lagrange
    multipliers.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    penalty : np.ndarray
        The penalty matrix.
    y : np.ndarray
        The response sample.
    ub : float, optional
        The upper bound of the regression problem (1.1) in [1], by default 100
    row_bounds: bool, optional
        If True, the bounds are computed on the rows. Otherwise, they are
        computed on the coefficients. By default, False

    Returns
    -------
    np.ndarray
        An array containing the bounds of the problems. The first column refers
        to the minimization problem, while the second corresponds to the
        maximization problem.

    References
    ----------
    ... [1] Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset
    selection via a modern optimization lens. The annals of statistics, 44(2),
    813-852.
    """
    matrixS = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)

    Q = np.dot(matrixS.T, matrixS) + penalty
    if row_bounds:
        aux = np.linalg.solve(Q, matrixS.T)
        cQd = np.dot(matrixS, np.dot(aux, y))
        dQd = np.dot(y, cQd - y) + ub
        cQc = np.diag(np.dot(matrixS, aux))
    else:
        lin = np.dot(y, matrixS)
        cQd = np.linalg.solve(Q, matrixS.T @ y)
        cQc = np.diag(np.linalg.inv(Q))
        dQd = np.dot(y, matrixS @ np.linalg.solve(Q, lin) - y) + ub
    bounds = np.tile(cQd, (2, 1)).T
    # The optimal solution of problems (2.15) in [1] using Lagrange
    # multipliers is in the form
    # `+- ( sqrt(c.T Q^-1 c * ( d.T Q^-1 d + ub)) + c.T Q^-1 d )`
    bounds[:, 0] -= np.sqrt(cQc * dQd)
    bounds[:, 1] += np.sqrt(cQc * dQd)
    return bounds


def bounds_varsec(
    matricesXZ: dict[str, list[np.ndarray]],
    penalty: np.ndarray,
    y: np.ndarray,
    ub: float = 100,
    row_bounds: bool = False,
) -> np.ndarray:
    """Compute the optimal solution of problems (2.15) in [1] using Lagrange
    multipliers using the FSAM class.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    penalty : np.ndarray
        The penalty matrix.
    y : np.ndarray
        The response sample.
    ub : float, optional
        The upper bound of the regression problem (1.1) in [1], by default 100
    row_bounds: bool, optional
        If True, the bounds are computed on the rows. Otherwise, they are
        computed on the coefficients. By default, False

    Returns
    -------
    np.ndarray
        An array containing the bounds of the problems. The first column refers
        to the minimization problem, while the second corresponds to the
        maximization problem.

    References
    ----------
    ... [1] Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset
    selection via a modern optimization lens. The annals of statistics, 44(2),
    813-852.
    """
    var_sec = FSAM(deg=[3], ord_d=[2], n_int=[40])
    matrixS = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)
    Q = np.dot(matrixS.T, matrixS) + penalty
    bounds = var_sec._get_bounds(matrixS, Q, ub, y, row_bounds).values.astype(float)
    return bounds


def test_bounds(
    matricesXZ: dict[str, list[np.ndarray]],
    penalty: np.ndarray,
    y: np.ndarray,
    ub: float = 100,
    **kwargs,
):
    """Test that the bounds for the problem (2.15) in [1] by directly
    implementing the problems in Gurobi and using Lagrange multipliers (both
    implemented directly and in the FSAM class) coincide. The kwargs are referred to
    arguments of the optimizer.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    penalty : np.ndarray
        The penalty matrix.
    y : np.ndarray
        The response sample.
    ub : float, optional
        The upper bound of the regression problem (1.1) in [1], by default 100

    Returns
    -------
    np.ndarray
        An array containing the bounds of the problems. The first column refers
        to the minimization problem, while the second corresponds to the
        maximization problem.

    References
    ----------
    ... [1] Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset
    selection via a modern optimization lens. The annals of statistics, 44(2),
    813-852.
    """
    c_rows = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)
    c_coefs = np.eye(np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1).shape[1])
    for c, row_bounds in zip([c_rows, c_coefs], [True, False]):
        bounds1 = bounds_gurobi(
            matricesXZ=matricesXZ, penalty=penalty, y=y, c=c, ub=ub, **kwargs
        )
        bounds2 = bounds_lagrange(
            matricesXZ=matricesXZ, penalty=penalty, y=y, ub=ub, row_bounds=row_bounds
        )
        bounds3 = bounds_varsec(
            matricesXZ=matricesXZ, penalty=penalty, y=y, ub=ub, row_bounds=row_bounds
        )

        assert np.allclose(bounds1, bounds2)
        assert np.allclose(bounds1, bounds3)
