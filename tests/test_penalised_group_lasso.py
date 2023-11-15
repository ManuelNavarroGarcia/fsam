from typing import Iterable, Union

import mosek.fusion
import numpy as np
import pandas as pd
import pytest
from cpsplines.psplines.bspline_basis import BsplineBasis
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant

from fsam.penalized_group_lasso import PenalisedGroupLasso
from fsam.sop import sop_fit
from fsam.fsam_fit import reparametrize_basis


class PenalisedGroupLassoMOSEK:
    def __init__(
        self, alpha_path: Union[int, float], tol_sparsity: Union[int, float] = 1e-5
    ):
        self.alpha_path = alpha_path
        self.tol_sparsity = tol_sparsity

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: Iterable[int], sp: Iterable[float]
    ):
        """Get the estimated coefficients of the Penalized Group Lasso by
        solving the problem directly with the MOSEK solver

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        y : np.ndarray
            The response sample.
        groups : Iterable[int]
            The indexes of the groups. It is assumed that, for `g` groups, the
            are given by `range(g)`.
        sp : Iterable[float]
            The smoothing parameters.
        """
        # Add a column of ones as the intercept
        self.scaler = StandardScaler(with_std=False)
        _ = self.scaler.fit(X)
        X = add_constant(self.scaler.transform(X), has_constant="add")

        group_ids = np.unique(groups)
        # Create the model and the decision variables
        M = mosek.fusion.Model()
        theta = M.variable("theta", X.shape[1], mosek.fusion.Domain.unbounded())
        t = M.variable("t", 1, mosek.fusion.Domain.greaterThan(0.0))
        u = M.variable("u", len(group_ids), mosek.fusion.Domain.greaterThan(0.0))
        v = M.variable("v", len(sp), mosek.fusion.Domain.greaterThan(0.0))

        # The least squares term, converted into a rotated second-order cone
        # constraint
        _ = M.constraint(
            "least-squares",
            mosek.fusion.Expr.vstack(
                t,
                1 / 2,
                mosek.fusion.Expr.sub(y, mosek.fusion.Expr.mul(X, theta)),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )
        # The corresponding penalty to control the smoothness of the curve,
        # converted into a rotated second-order cone constraint
        if np.abs(sp).max():
            for g in group_ids[v.getShape()[0] :]:
                gr_idx = np.where(groups == g)[0] + 1  # add intercept position
                _ = M.constraint(
                    f"penalty_{g - v.getShape()[0]}",
                    mosek.fusion.Expr.vstack(
                        v.slice(g - v.getShape()[0], g - v.getShape()[0] + 1),
                        1 / 2,
                        theta.slice(gr_idx[0], gr_idx[-1] + 1),
                    ),
                    mosek.fusion.Domain.inRotatedQCone(),
                )
        # The Group Lasso penalty, converted into a second-order cone constraint
        if self.alpha_path:
            for g in group_ids:
                gr_idx = np.where(groups == g)[0] + 1  # add intercept position
                _ = M.constraint(
                    f"gl_{g}",
                    mosek.fusion.Expr.vstack(
                        u.slice(g, g + 1),
                        mosek.fusion.Expr.mul(
                            X[:, gr_idx], theta.slice(gr_idx[0], gr_idx[-1] + 1)
                        ),
                    ),
                    mosek.fusion.Domain.inQCone(),
                )

        # The objective function
        obj = mosek.fusion.Expr.add(
            [mosek.fusion.Expr.mul(s, v.slice(g, g + 1)) for g, s in enumerate(sp)]
        )

        _ = M.objective(
            "objective",
            mosek.fusion.ObjectiveSense.Minimize,
            mosek.fusion.Expr.add(
                mosek.fusion.Expr.add(t, obj),
                mosek.fusion.Expr.mul(self.alpha_path, mosek.fusion.Expr.sum(u)),
            ),
        )

        M.solve()
        # Get the estimated coefficients, enforcing that the maximum by group
        # (in absolute value) must be greater than `self.tol_sparsity`
        non_zero_vars = (
            pd.Series(np.abs(theta.level()), index=np.r_[groups.min() - 1, groups])
            .reset_index(drop=False)
            .groupby("index")
            .max()
            .gt(self.tol_sparsity)
            .squeeze()
            .pipe(lambda x: x.index[x])
            .values
        )
        self.coef_ = np.where(
            pd.Series(theta.level(), index=np.r_[groups.min() - 1, groups]).index.isin(
                non_zero_vars
            ),
            pd.Series(theta.level(), index=np.r_[groups.min() - 1, groups]),
            0,
        )
        self.loss_ = M.primalObjValue()

    def predict(self, X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """Estimates the response variable from the data matrix `X`.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        coefs : np.ndarray
            The regression coefficients used to compute the estimated values of
            the target variable.

        Returns
        -------
        np.ndarray
            The estimated values.
        """
        return np.dot(add_constant(self.scaler.transform(X), has_constant="add"), coefs)


class PenalizedGroupLassoOld(PenalisedGroupLasso):
    def __init__(
        self,
        alpha_path: Iterable[Union[int, float]] = None,
        tol: Union[int, float] = 1e-3,
        max_iter: int = 100,
        tol_sparsity: Union[int, float] = 1e-5,
        n_alphas: int = 30,
        eps: Union[int, float] = 1e-3,
    ):
        super(PenalizedGroupLassoOld, self).__init__(
            alpha_path=alpha_path,
            tol=tol,
            max_iter=max_iter,
            tol_sparsity=tol_sparsity,
            n_alphas=n_alphas,
            eps=eps,
        )

    def _solve_linear_model_old(
        self, X: np.ndarray, residual: np.ndarray, alpha: Union[int, float]
    ) -> np.ndarray:
        """Find the estimated coefficients for the Group Lasso algorithm with
        just one group using MOSEK solver.

        Parameters
        ----------
        X : np.ndarray
            The design matrix.
        residual : np.ndarray
            The response variable.
        alpha : np.ndarray
            The shrinkage parameter.

        Returns
        -------
        np.ndarray
            The estimated coefficients.
        """
        # Create the model and the decision variables
        M = mosek.fusion.Model()
        theta = M.variable("theta", X.shape[1], mosek.fusion.Domain.unbounded())
        t = M.variable("t", 1, mosek.fusion.Domain.greaterThan(0.0))
        u = M.variable("u", 1, mosek.fusion.Domain.greaterThan(0.0))

        # The least squares term, converted into a rotated second-order cone
        # constraint
        _ = M.constraint(
            "least-squares",
            mosek.fusion.Expr.vstack(
                t,
                1 / 2,
                mosek.fusion.Expr.sub(residual, mosek.fusion.Expr.mul(X, theta)),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )
        # The Group Lasso penalty, converted into a second-order cone constraint
        _ = M.constraint(
            "gl",
            mosek.fusion.Expr.vstack(u, mosek.fusion.Expr.mul(X, theta)),
            mosek.fusion.Domain.inQCone(),
        )
        # The objective function
        _ = M.objective(
            "objective",
            mosek.fusion.ObjectiveSense.Minimize,
            mosek.fusion.Expr.add(t, mosek.fusion.Expr.mul(alpha, u)),
        )

        M.solve()
        return theta.level()

    def _solve_nonlinear_model_old(
        self,
        Q: np.ndarray,
        R_inv: np.ndarray,
        residual: np.ndarray,
        alpha: Union[int, float],
        sp: Union[int, float],
    ) -> np.ndarray:
        """Find the estimated coefficients for the adapted problem (4.2) in [1]
        using MOSEK solver.

        Parameters
        ----------
        Q : np.ndarray
            The orthogonal matrix of the QR decomposition of the design matrix.
        R_inv : np.ndarray
            The inverse of the upper triangular matrix of the QR decomposition
            of the design matrix.
        residual : np.ndarray
            The response variable.
        alpha : np.ndarray
            The shrinkage parameter.
        sp : Union[int, float]
            The smoothing parameter.

        Returns
        -------
        np.ndarray
            The estimated coefficients.

        References
        ----------
        ... [1] Chouldechova, A., & Hastie, T. (2015). Generalized additive
        model selection. arXiv preprint arXiv:1506.03850.
        """
        # Create the model and the decision variables
        M = mosek.fusion.Model()
        theta = M.variable("theta", R_inv.shape[1], mosek.fusion.Domain.unbounded())
        t = M.variable("t", 1, mosek.fusion.Domain.greaterThan(0.0))
        u = M.variable("u", 1, mosek.fusion.Domain.greaterThan(0.0))
        v = M.variable("v", 1, mosek.fusion.Domain.greaterThan(0.0))

        # The least squares term, converted into a rotated second-order cone
        # constraint
        _ = M.constraint(
            "least-squares",
            mosek.fusion.Expr.vstack(
                t,
                1 / 2,
                mosek.fusion.Expr.sub(residual, mosek.fusion.Expr.mul(Q, theta)),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )
        # The corresponding penalty to control the smoothness of the curve,
        # converted into a rotated second-order cone constraint
        _ = M.constraint(
            "penalty",
            mosek.fusion.Expr.vstack(v, 1 / 2, mosek.fusion.Expr.mul(R_inv, theta)),
            mosek.fusion.Domain.inRotatedQCone(),
        )
        # The Group Lasso penalty, converted into a second-order cone constraint
        _ = M.constraint(
            "gl", mosek.fusion.Expr.vstack(u, theta), mosek.fusion.Domain.inQCone()
        )

        # The objective function
        _ = M.objective(
            "objective",
            mosek.fusion.ObjectiveSense.Minimize,
            mosek.fusion.Expr.add(
                mosek.fusion.Expr.add(t, mosek.fusion.Expr.mul(sp, v)),
                mosek.fusion.Expr.mul(alpha, u),
            ),
        )

        M.solve()
        return theta.level()


def get_y(X: np.ndarray) -> np.ndarray:
    """Generate the response variable sample

    Parameters
    ----------
    X : np.ndarray
        The data matrix

    Returns
    -------
    np.ndarray
        The response variable sample.
    """
    return (
        np.multiply(2, X[:, 0])
        + np.multiply(2, X[:, 2])
        + np.multiply(1, np.power(X[:, 3], 2))
        + np.multiply(3, np.power(X[:, 4], 2))
    )


def get_matricesXZ(
    X: np.ndarray, deg: int, n_int: int, ord_d: int
) -> dict[str, list[np.ndarray]]:
    """Given the matrix `X` containing the sample data from explanatory
    variables, reparamatrize the constructed B-splines bases and returns a
    dictionary with lists of the reparametrized bases for each regressor.

    Parameters
    ----------
    X : np.ndarray
        The data matrix.
    deg : int
        The degree of the B-spline bases.
    n_int : int
        The number of intervals used to defined the knot sequences of the
        B-spline bases.
    ord_d : int
        The penalty order.

    Returns
    -------
    dict[str, list[np.ndarray]]
        A dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    """
    Xs = []
    Zs = []
    for x in X.T:
        bspline = BsplineBasis(deg=deg, xsample=x, n_int=n_int)
        rep_mat = reparametrize_basis(bspline=bspline, ord_d=ord_d, x=x)
        Xs.append(rep_mat["X"][:, 1:])
        Zs.append(rep_mat["Z"])
    return {"X": Xs, "Z": Zs}


def get_sp(
    matricesXZ: dict[str, list[np.ndarray]], y: np.ndarray, ds: np.ndarray
) -> list[float]:
    """Get the smoothing parameters using the separation of overlapping
    precision matrices (SOP) method.

    Parameters
    ----------
    matricesXZ : dict[str, list[np.ndarray]]
        Dictionary with keys "X" and "Z" and with values lists with the
        matrices `X` and `Z` for each regressor.
    y : np.ndarray
        The response variable sample
    ds : np.ndarray
        List with the diagonal elements of the precision matrices for each
        variance component in the model.

    Returns
    -------
    list[float]
        A list containing the smoothing parameters resulting from the SOP
        algorithm
    """
    sp = []
    for X, Z in zip(*matricesXZ.values()):
        sp_dict = sop_fit(X=add_constant(X, has_constant="add"), Z=Z, y=y, G=[ds])
        sp.append(np.clip(sp_dict["phi"] / sp_dict["tau"], a_max=1e5, a_min=None)[0])
    return sp


def get_precision_elements(n_int: int, deg: int, ord_d: int) -> np.ndarray:
    """Computes the diagonal elements of the precision matrices for each
    variance component.

    Parameters
    ----------
    n_int : int
        The number of intervals used to defined the knot sequences of the
        B-spline bases.
    deg : int
        The degree of the B-spline bases
    ord_d : int
        The penalty order.

    Returns
    -------
    np.ndarray
        An array with the eighen values of the matrix of differences
    """
    D = np.diff(np.eye(n_int + deg + ord_d, dtype=np.int32), n=ord_d)[ord_d:-ord_d, :]
    ds = np.linalg.eigh(D.T @ D)[0][ord_d:][::-1]
    return ds


def test_penalised_group_lasso(
    deg: int = 3, ord_d: int = 2, n_int: int = 8, m: int = 5
):
    alpha_path = [0.305790]
    expected_vars = [0, 2, 8, 9]
    expected_obj_function = 23.98179502470704
    expected_coefs = np.array(
        [
            [
                1.19551991,
                1.97653599,
                0,
                1.99958236,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -0.00128767302,
                0.000362429084,
                -0.0000606982127,
                -0.00194269084,
                -0.00661797892,
                -0.00629346661,
                0.0365133453,
                -0.00635552350,
                0.188015940,
                0.00145404149,
                0.00369342191,
                -0.0124912874,
                0.00503338222,
                -0.0591352733,
                -0.00777180302,
                0.0755918205,
                -0.0289824884,
                0.823625641,
            ]
        ]
    )
    expected_ps_coefs = np.array(
        [
            1.19551991,
            1.99873692,
            0.0,
            2.01824655,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.00107044,
            0.00032749,
            0.00001953,
            -0.00184687,
            -0.0066995,
            -0.00546325,
            0.03864611,
            -0.00426528,
            0.19667668,
            0.00114233,
            0.00172702,
            -0.01258258,
            0.00604227,
            -0.0593211,
            -0.00640414,
            0.07863077,
            -0.03099771,
            0.8376977,
        ]
    )

    np.random.seed(0)
    X = np.random.uniform(-1, 1, (149, m))
    y = get_y(X=X)

    ds = get_precision_elements(deg=deg, ord_d=ord_d, n_int=n_int)
    matricesXZ = get_matricesXZ(X=X, deg=deg, ord_d=ord_d, n_int=n_int)
    sp = get_sp(matricesXZ, y, ds)

    groups = (
        pd.Series(
            [
                np.repeat(i, A.shape[1])
                for i, A in enumerate(matricesXZ["X"] + matricesXZ["Z"])
            ]
        )
        .explode()
        .astype(int)
        .values
    )
    S = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)

    pgl = PenalisedGroupLasso(alpha_path=alpha_path, tol_sparsity=1e-4)
    pgl.fit(S, y, groups=groups, sp=sp, scale_y=False)
    pglMOSEK = PenalisedGroupLassoMOSEK(alpha_path=alpha_path)
    pglMOSEK.fit(S, y, groups=groups, sp=sp)
    pgl_df = pgl.coef_.loc[alpha_path, 0:].set_axis(["coefs"]).T
    pgl_vars = np.unique(pgl_df["coefs"].loc[lambda x: np.abs(x) > 0.0].index).tolist()
    # Check the coefficients of PGL, PGLMOSEK and the expected values coincide
    assert np.allclose(pgl.coef_.values, pglMOSEK.coef_, atol=1e-4)
    assert np.allclose(pgl.coef_.values, expected_coefs, atol=1e-4)
    assert np.allclose(pgl.ps_coef_.values, expected_ps_coefs, atol=1e-4)
    # Check the loss function of PGL, PGLMOSEK and the expected value coincide
    assert np.allclose(pglMOSEK.loss_, expected_obj_function, atol=1e-4)
    assert np.allclose(pgl.loss_, expected_obj_function, atol=1e-4)
    # Check the variables chosen by PGL coincide with the expected variables
    # (since the coefficients from PGL and PGLMOSEK must coincide at this point,
    # it is only necessary to check this with PGL)
    assert np.allclose(pgl_vars, expected_vars)

    # Check that the closed-form solution for each block coincide with the
    # solution of the MOSEK solver
    pglOLD = PenalizedGroupLassoOld(alpha_path=alpha_path, tol_sparsity=1e-4)
    # Linear blocks
    for i in range(m):
        # For the sake of simplicity, we assume that the residual is `y`
        out1 = pglOLD._solve_linear_model_old(
            np.expand_dims(X[:, i], axis=1), y, alpha_path[0]
        )
        out2 = pglOLD._solve_linear_model(
            np.expand_dims(X[:, i], axis=1), y, alpha_path[0]
        )
        assert np.allclose(out1, out2, atol=1e-3)
    # Non-linear blocks
    for i, sp_ in enumerate(sp):
        S_ = S[:, np.where(groups == i + m)[0]]
        Q_, R_ = np.linalg.qr(S_)
        _, L_, S_inv_ = np.linalg.svd(
            2 * (sp_ * np.linalg.inv(R_).T @ np.linalg.inv(R_) + np.eye(R_.shape[0])),
            hermitian=True,
        )
        out1 = pglOLD._solve_nonlinear_model_old(
            Q=Q_, R_inv=np.linalg.inv(R_), residual=y, alpha=alpha_path[0], sp=sp_
        )
        beta_norm_ = pglOLD._get_beta_norm(
            Q=Q_,
            S_inv=S_inv_,
            L=L_,
            residual=y,
            alpha=alpha_path[0],
            init_sol=1e-5,
        )
        assert np.allclose(beta_norm_, np.linalg.norm(out1), atol=1e-3)
        out2 = pglOLD._solve_nonlinear_model(
            Q=Q_,
            R_inv=np.linalg.inv(R_),
            residual=y,
            alpha=alpha_path[0],
            sp=sp_,
            beta_norm=beta_norm_,
        )
        assert np.allclose(out1, out2, atol=1e-3)


@pytest.mark.parametrize("slope", [0.25, 2])
def test_alpha_max(
    slope: float,
    deg: int = 3,
    ord_d: int = 2,
    n_int: int = 8,
    m: int = 5,
    n: int = 100,
):
    np.random.seed(0)
    X = np.random.uniform(-1, 1, (n, m))
    y = slope * X[:, 0] + np.square(X[:, 1])

    ds = get_precision_elements(deg=deg, ord_d=ord_d, n_int=n_int)
    matricesXZ = get_matricesXZ(X=X, deg=deg, ord_d=ord_d, n_int=n_int)
    sp = get_sp(matricesXZ, y, ds)

    groups = (
        pd.Series(
            [
                np.repeat(i, A.shape[1])
                for i, A in enumerate(matricesXZ["X"] + matricesXZ["Z"])
            ]
        )
        .explode()
        .astype(int)
        .values
    )
    S = np.concatenate(matricesXZ["X"] + matricesXZ["Z"], axis=1)

    gap = 1e-2
    pgl = PenalisedGroupLasso(n_alphas=1)
    pgl.fit(S, y, groups=groups, sp=sp)

    pgl_non_null = PenalisedGroupLasso(alpha_path=[pgl.alpha_path[0] - gap])
    pgl_non_null.fit(S, y, groups=groups, sp=sp)
    pgl_df = pgl_non_null.coef_.loc[pgl_non_null.alpha_path, 0:].set_axis(["coefs"]).T
    pgl_vars = np.unique(pgl_df["coefs"].loc[lambda x: np.abs(x) > 0.0].index).tolist()

    pgl_null = PenalisedGroupLasso(alpha_path=[pgl.alpha_path[0] + gap])
    pgl_null.fit(S, y, groups=groups, sp=sp)

    # `pgl_vars` counts the groups of variables that have at least one
    # coefficient different from 0
    assert len(pgl_vars) == 1
    # `coef_` counts intercept
    assert np.count_nonzero(pgl_null.coef_.values) == 1
