#!/usr/bin/env python
import copy
import functools
import itertools
import logging
import math
import operator
import os
import pickle
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import gurobipy as gp
import numpy as np
import pandas as pd
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from gurobipy import GRB
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant
from tqdm.auto import tqdm

from fsam.penalized_group_lasso import get_solutions_pgl
from fsam.sop import sop_fit


def reparametrize_basis(
    bspline: BsplineBasis, ord_d: int, x: np.ndarray
) -> dict[str, np.ndarray]:
    """Reparametrize a B-spline basis. The first matrix, `X`, is a Vandermonde
    matrix with the same columns as `ord_d`. The second matrix, `Z`, contains
    the eigenvectors related to the non-zero eigenvalues of D.T @ D, where D is
    the penalization matrix, divided by the square root of this non-zero
    eigenvalues.

    Parameters
    ----------
    bspline : BsplineBasis
        The `BsplineBasis` object.
    ord_d : int
        The order of the penalization, by default 2.
    x : np.ndarray
        The real covariate used to generate the `BsplineBasis`

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with keys "X" and "Z" containing the matrices resulting
        from the reparametrization. Also, the non-zero eigenvalues of the
        penalty matrix are returned with key "d"
    """
    # The BsplineBasis object must have the design matrix as an atribute
    if not hasattr(bspline, "matrixB"):
        _ = bspline.get_matrix_B()

    B = bspline.bspline_basis(x=x)
    D = PenaltyMatrix(bspline=bspline).get_diff_matrix(ord_d=ord_d)
    u, d, _ = np.linalg.svd(D.T @ D)
    U_Z = u[:, :-ord_d] @ np.diag(1 / np.sqrt(d[:-ord_d]))
    # Generate `X` and `Z` matrices
    Z = B @ U_Z
    X = np.vander(x, N=ord_d, increasing=True)
    return {"X": X, "Z": Z, "d": d[:-ord_d]}


class FSAM:
    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (40,),
        prediction: Optional[Iterable[dict[str, Union[int, float]]]] = None,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.prediction = prediction

    def _get_bspline_bases(self, X: np.ndarray):
        """Given the matrix `X` containing the sample data from explanatory
        variables, construct the B-spline basis and the number of linear and
        non-linear parameters that will be associated with each feature.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.

        Attributes
        ----------
        bspline_bases : List[BsplineBasis]
            The list of B-spline bases on each axis resulting from
            `_get_bspline_bases`.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with keys `X` and `Z` containing the matrices resulting
            from the reparametrization.
        """
        n_nonlin = []
        n_lin = []
        bspline_bases = []
        prediction_ = self.prediction.copy()
        # For each covariate, generate a B-spline basis
        for i, (d, n, o, p) in enumerate(
            zip(self.deg, self.n_int, self.ord_d, prediction_)
        ):
            if p:
                if p["backwards"] == X[:, i].min():
                    _ = p.pop("backwards", None)
                if p["forward"] == X[:, i].max():
                    _ = p.pop("forward", None)
            bspline = BsplineBasis(deg=d, xsample=X[:, i], n_int=n, prediction=p)
            _ = bspline.get_matrix_B()
            bspline_bases.append(bspline)
            n_lin.append(o - 1)
            n_nonlin.append(n + bspline.int_forw + bspline.int_back + d - o)
        self.range_vars = np.concatenate(
            (
                np.cumsum(n_lin) - 1,
                np.sum(n_lin, keepdims=True),
                np.cumsum(n_nonlin) + np.sum(n_lin),
            )
        )
        self.bspline_bases = bspline_bases
        return None

    def _get_matrixS(self, X: np.ndarray) -> np.ndarray:
        """Given the matrix `X` containing the sample data from explanatory
        variables, reparametrize the B-spline bases and returns the matrices
        concatenated of the reparametrized bases.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.

        Attributes
        ----------
        ds : list[np.ndarray]
            The list of non-zero eigenvalues of the penalty matrix for each
            smooth term.

        Returns
        -------
        np.ndarray
            The concatenation of the matrices resulting from the
            reparametrization.
        """
        Xs, Zs, ds = [], [], []
        # For each covariate, generate a B-spline basis and reparametrize it
        for i, (bspline, o) in enumerate(zip(self.bspline_bases, self.ord_d)):
            rep_mat = reparametrize_basis(bspline=bspline, ord_d=o, x=X[:, i])
            # The X given by `reparametrize_basis` has a column of 1s. We are
            # not interested in that column at that point, so we retain only the
            # right columns
            Xs.append(rep_mat["X"][:, 1:])
            Zs.append(rep_mat["Z"])
            ds.append(rep_mat["d"])
        # Concatenate matrices X and add a column of 1s to the left
        self.ds = ds
        return np.concatenate(Xs + Zs, axis=1)

    def _get_sp(
        self,
        y: np.ndarray,
        x_vars: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """Get the smoothing parameters fitted simultaneously using the SOP
        algorithm.
        Parameters
        ----------
        y : np.ndarray
            The response variable sample
        x_vars : Iterable[int], optional
            The indices of the variables whose smoothing parameters are
            estimated, by default None.
        """
        if x_vars is None:
            x_vars = range(self.m)

        range_lin, range_nonlin, ds = [], [], []
        for i in x_vars:
            range_lin.append(range(self.range_vars[i], self.range_vars[i + 1]))
            range_nonlin.append(
                range(self.range_vars[self.m + i], self.range_vars[self.m + i + 1])
            )
            ds.append(self.ds[i])
        # Compute the smoothing parameter
        two_terms_output = sop_fit(
            y=y,
            X=add_constant(
                self.matrixS[:, list(itertools.chain(*range_lin))], has_constant="add"
            ),
            Z=self.matrixS[:, list(itertools.chain(*range_nonlin))],
            G=ds,
        )

        # Compute the AIC for the linear term
        modelo = LinearRegression(fit_intercept=True)
        modelo.fit(self.matrixS[:, list(itertools.chain(*range_lin))], y)
        y_pred = modelo.predict(self.matrixS[:, list(itertools.chain(*range_lin))])

        aic_lineal = np.sum(np.square(y - y_pred)) + 4

        # Compute the AIC for the non-linear term (X is only the intercept)
        non_linear_output = sop_fit(
            y=y,
            X=np.ones((len(self.matrixS), 1)),
            Z=self.matrixS[:, list(itertools.chain(*range_nonlin))],
            G=ds,
        )

        return {
            "sp": two_terms_output["phi"] / two_terms_output["tau"],
            "aic_nonlinear": non_linear_output["aic"],
            "aic_lineal": aic_lineal,
            "edf": two_terms_output["edf"],
        }

    def _complete_initial_solution(
        self,
        dict_theta: dict[int, Iterable[int]],
        k: int,
        y: np.ndarray,
        init_sol: Optional[dict[str, Union[float, np.ndarray]]] = None,
    ) -> dict[str, Union[float, np.ndarray]]:
        """Completes/generates a feasible solution for the variable selection
        problem. It is the best solution among:
        * Randomly adds more terms up to the predefined sparsity parameter `k`
        from a initial solution `init_sol`.
        * Chooses the `k` terms related with the lowest AIC values.
        * The PGL solution for sparsity parameter `k`.

        Parameters
        ----------
        dict_theta : dict[int, Iterable[int]]
            A dictionary containing the index of the terms as keys and a list of
            their corresponding columns as values.
        k : int
            The sparsity parameter.
        y : np.ndarray
            The response variable sample.
        init_sol : Optional[dict[str, Union[float, np.ndarray]]]
            A feasible solution. By default, None.

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            A dictionary containing the indicator variables "z", the regression
            coefficients "theta" and objective function value "obj".
        """
        if init_sol is None:
            init_sol = {
                "obj": 0,
                "theta": np.zeros(self.matrixS.shape[1] + 1),
                "z": np.zeros(2 * self.m),
            }
        last_k_init_sol = init_sol.copy()

        # Create solution by using the incomplete initial solution
        np.random.seed(0)
        # Check that the initial solution is subset in the variable that can
        # enter into the model
        assert all(np.isin(np.flatnonzero(init_sol["z"]), list(dict_theta.keys())))
        # Add terms only from variables that can enter into the model
        terms_to_add = np.sort(
            np.random.choice(
                np.setdiff1d(list(dict_theta.keys()), np.flatnonzero(init_sol["z"])),
                int(k - init_sol["z"].sum()),
                replace=False,
            )
        )
        last_k_init_sol["z"][terms_to_add] = 1
        last_k_init_sol = self._get_sol(
            self.matrixS, np.flatnonzero(last_k_init_sol["z"]), dict_theta, y
        )
        assert last_k_init_sol["z"].sum() == k

        # Create solution by using the lowest AIC associated variables
        best_aic = (
            pd.Series(self.aic[list(dict_theta.keys())], index=dict_theta.keys())
            .sort_values()[:k]
            .sort_index()
            .index.tolist()
        )
        first_aics_init_sol = self._get_sol(self.matrixS, best_aic, dict_theta, y)
        assert first_aics_init_sol["z"].sum() == k

        # Create solution by using the PGL solution for sparsity parameter `k`.
        # If the solution for `k` is not available, randomly choose `k` terms
        # from the solution with larger `k` available
        if k in self.pgl_sols.loc[pd.isna(self.pgl_sols["vars"]), :].index:
            next_k_df = (
                self.pgl_sols[self.pgl_sols.index > k]
                .dropna()["vars"]
                .first_valid_index()
            )
            # If `next_k_df` is empty, then choose them randomly
            if next_k_df is None:
                next_k_vars = list(dict_theta.keys())
            else:
                next_k_vars = self.pgl_sols.loc[next_k_df, "vars"]
            pgl_vars = np.random.choice(next_k_vars, k, replace=False)
        else:
            pgl_vars = self.pgl_sols.loc[k, "vars"]
            if len(pgl_vars) < k:
                terms_to_add = np.random.choice(
                    np.setdiff1d(list(dict_theta.keys()), pgl_vars),
                    k - len(pgl_vars),
                    replace=False,
                )
                pgl_vars = np.sort(np.r_[terms_to_add, pgl_vars])
        pgl_init_sol = self._get_sol(self.matrixS, pgl_vars, dict_theta, y)
        assert pgl_init_sol["z"].sum() == k

        # Select the solution with the lowest objective function value
        init_sol = min(
            [first_aics_init_sol, last_k_init_sol, pgl_init_sol],
            key=lambda x: x.get("obj"),
        )

        return init_sol

    def _get_sol(
        self,
        X: np.ndarray,
        init_sol_vars: Iterable[int],
        dict_theta: dict[int, Iterable[int]],
        y: np.ndarray,
        pen: Optional[np.ndarray] = None,
    ) -> dict[str, Union[float, np.ndarray]]:
        """Given a set of terms `init_sol_vars`, computes the regression
        coefficients and the objective function value using the P-splines
        framework.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        init_sol_vars : Iterable[int]
            Variables that determine the initial solution.
        dict_theta : dict[int, Iterable[int]]
            A dictionary containing the index of the terms as keys and a list of
            their corresponding columns as values.
        y : np.ndarray
            The response variable sample.
        pen : Optional[np.ndarray], optional
            The penalty matrix. If None, the penalty matrix of the additive
            model is used. By default, None.

        Returns
        -------
        dict[str, Union[float, np.ndarray]]
            A dictionary containing the indicator variables "z", the regression
            coefficients "theta" and objective function value "obj".
        """
        if pen is None:
            pen = self.pen.copy()
        # Get the related columns for the variables
        init_sol_cols = functools.reduce(
            operator.iconcat, [dict_theta[i] for i in init_sol_vars], []
        )
        assert (
            X.shape[1] == self.matrixS.shape[1]
        ), "The matrices `X` and `matrixS` must have the same number of columns."

        # Solve the subproblem using the least squares criterion
        S = add_constant(X[:, init_sol_cols], has_constant="add")
        pen = block_diag(*(np.zeros(1), np.diag(pen[init_sol_cols, init_sol_cols])))
        Q = S.T @ S + pen
        init_theta = np.linalg.solve(Q, np.dot(S.T, y))

        # Get the optimal indicator variables
        init_sol_z = (
            pd.Series(1, index=init_sol_vars)
            .reindex(range(2 * self.m))
            .fillna(0)
            .values.astype(int)
        )
        # Get the fitted regression coefficients
        init_sol_theta = (
            pd.Series(init_theta, index=[-1] + init_sol_cols)
            .reindex(range(-1, X.shape[1]))
            .fillna(0)
            .values
        )

        # Get the optimal objective function value
        init_sol_obj = (
            np.dot(y - S @ init_theta, y - S @ init_theta)
            + init_theta @ pen @ init_theta
        )
        init_sol = {"z": init_sol_z, "theta": init_sol_theta, "obj": init_sol_obj}
        return init_sol

    def _get_bounds(
        self, S: np.ndarray, Q: np.ndarray, ub: float, y: np.ndarray, row_bounds: bool
    ) -> pd.DataFrame:
        """Compute the optimal solution of problems (2.15) in [1] using Lagrange
        multipliers.

        Parameters
        ----------
        S : np.ndarray
            The reparametrized data matrix.
        Q : np.ndarray
            The quadratic part of the objective function of the problem.
        ub : float
            The upper bound of the regression problem (1.1) in [1].
        y : np.ndarray
            The response sample.
        row_bounds: bool, optional
            If True, the bounds are computed on the rows. Otherwise, they are
            computed on the coefficients. By default, False

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the bounds of the problems. The "lower"
            column refers to the minimization problem, while the "upper"
            corresponds to the maximization problem.

        Raises
        ------
        ValueError
            Negative values encountered in `numpy.sqrt()` on the formula of
            Lagrange multipliers (tolerance: 1e-12).

        References
        -------
        ... [1] Bertsimas, D., King, A., & Mazumder, R. (2016). Best subset
        selection via a modern optimization lens. The annals of statistics, 44
        (2), 813-852.
        """
        if row_bounds:
            aux = np.linalg.solve(Q, S.T)
            cQd = np.dot(S, np.dot(aux, y))
            dQd = np.dot(y, cQd - y) + ub
            cQc = np.diag(np.dot(S, aux))
        else:
            cQd = np.linalg.solve(Q, S.T @ y)
            cQc = np.diag(np.linalg.inv(Q))
            dQd = np.dot(y, S @ cQd - y) + ub
        if dQd < 0:
            if math.isclose(dQd, 0, abs_tol=1e-12):
                dQd = 0
            else:
                raise ValueError(f"Negative values encountered in numpy.sqrt(): {dQd}.")
        cQc_dQd = np.sqrt(cQc * dQd)
        bounds = np.tile(cQd, (2, 1)).T
        # The optimal solution of problems (2.15) in [1] using Lagrange
        # multipliers is in the form
        # `+- ( sqrt(c.T Q^-1 c * ( d.T Q^-1 d + ub)) + c.T Q^-1 d )`
        bounds[:, 0] -= cQc_dQd
        bounds[:, 1] += cQc_dQd
        return pd.DataFrame(bounds, columns=["lower", "upper"])

    def _initialize_model(
        self,
        y: np.ndarray,
        dict_theta: dict[int, Iterable[int]],
        L: int,
        init_sol: dict[str, Union[float, np.ndarray]],
        compute_coef_bounds: bool,
        frac_row_bounds: float,
        **kwargs,
    ) -> gp.Model:
        """Initializes the variable selection optimization model with sparsity
        parameter `L` for the variables with index the keys of `dict_theta`.

        Parameters
        ----------
        y : np.ndarray
            The response variable sample.
        dict_theta : dict[int, Iterable[int]]
            A dictionary containing the index of the terms not fixed involved
            in the subproblem as keys and a list of their corresponding columns
            as values.
        L : int
            Sparsity parameter (sum of linear and non-linear terms that may
            enter into the model).
        init_sol : dict[str, Union[float, np.ndarray]]
            An initial solution.
        compute_coef_bounds: bool, optional
            When set to True, computes the bounds for the coefficients.
            Otherwise, SOS-1 constraints are used. By default, True
        frac_row_bounds: float
            The proportion of rows in which the bounds for the linear
            combinations of the regression coefficients are computed. It must be
            a float between 0 and 1. If 0, the bounds are not computed.

        Returns
        -------
        gp.Model
            The Gurobi model reduced to the destroyed variables.
        """
        BOUNDS_TOL = 1e-3
        # Select columns that will enter into the model
        select_cols = np.sort(
            functools.reduce(operator.iconcat, list(dict_theta.values()), [])
        )
        # Slice the S matrix by the columns related with the variables entering
        # the model
        S = self.matrixS[:, select_cols]

        # Construct the penalty term with the columns related with the variables
        # entering the model
        pen = np.diag(self.pen[select_cols, select_cols])

        # Now, we compute the upper bound of the objective function value. To do
        # so, we select the ones that have been destroyed
        intersect_z = np.intersect1d(
            list(dict_theta.keys()), np.flatnonzero(init_sol["z"])
        )
        # If the number of ones that have been destroyed is less than `L`, we
        # randomly add more terms up to `L` from the destroyed variables
        if len(intersect_z) < L:
            intersect_z = np.sort(
                np.r_[
                    intersect_z,
                    np.random.choice(
                        np.setdiff1d(list(dict_theta.keys()), intersect_z),
                        size=L - len(intersect_z),
                        replace=False,
                    ),
                ]
            )
        assert len(intersect_z) == L
        # Calculate the objective function value of the ones that have been
        # destroyed. This value will serve as an upper bound
        sol_intersect_z = self._get_sol(self.matrixS, intersect_z, dict_theta, y)
        ub = sol_intersect_z["obj"]

        S_intercept = add_constant(S, has_constant="add")
        pen_intercept = block_diag(*(np.zeros(1), pen))

        # Create model
        M = gp.Model()
        for key, v in kwargs.items():
            _ = M.setParam(key, v)

        # The regression coefficients. The lower variable is set to minus
        # infinity since Gurobi assumes the lower bound to be zero for all
        # variables by default, so we need to explicitly set the lower bound
        theta = M.addMVar(
            (S.shape[1],), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta"
        )
        # Indicator variables for the terms (1 if the term does not enter the
        # model, 0 otherwise)
        z = M.addVars(len(dict_theta.keys()), vtype=GRB.BINARY, name="z")
        if not compute_coef_bounds:
            t = M.addVars(len(dict_theta.keys()), vtype=GRB.BINARY, name="t")

        # Provide the initial solution to the decision variables
        for var, value in zip(
            z.values(), sol_intersect_z["z"][list(dict_theta.keys())]
        ):
            var.Start = value
        for var, value in zip(theta, sol_intersect_z["theta"][1:][select_cols]):
            var.Start = value

        # For the variables entering the model, reset the name of the columns
        # they occupy (since we need to index the variable `theta`)
        count = 0
        dict_theta_ = copy.deepcopy(dict_theta)
        for key, value in dict_theta_.items():
            dict_theta_[key] = list(range(count, count + len(value)))
            count += len(value)

        # Build the SOS-1 (exclusivity) constraints on the coefficients and the
        # variables using the big-M previously calculated
        if compute_coef_bounds:
            # Compute the bounds for the coefficients. The first element is
            # discarded as it corresponds to the intercept, which is not estimated
            coef_bounds = (
                self._get_bounds(
                    S_intercept,
                    S_intercept.T @ S_intercept + pen_intercept,
                    ub,
                    y,
                    False,
                )
                .iloc[1:, :]
                .reset_index(drop=True)
            )

            # Build the SOS-1 (exclusivity) constraints on the coefficients and the
            # variables
            for i, val in enumerate(dict_theta_.values()):
                for v in val:
                    _ = M.addConstr(coef_bounds.loc[v, "lower"] * z[i] <= theta[v])
                    _ = M.addConstr(theta[v] <= coef_bounds.loc[v, "upper"] * z[i])
        else:
            for i, val in enumerate(dict_theta_.values()):
                _ = M.addLConstr(t[i] + z[i], "==", 1)
                for v in val:
                    # It is needed to use `.tolist()` since `theta` is a `MVar`,
                    # and `addSOS` only accepts `Var` objects
                    _ = M.addSOS(GRB.SOS_TYPE1, [theta[v].tolist(), t[i]])

        # Set the sparsity constraint
        _ = M.addConstr(gp.quicksum(z) <= L, "sparsity")

        # Implement bounds on linear combinations of the regression coefficients
        # only if the difference between the lower and the upper is greater
        # than `bounds_tol`
        rows_restricted_idx = np.sort(
            pd.Series(range(S.shape[0]))
            .sample(frac=frac_row_bounds, random_state=0)
            .values
        )
        if len(rows_restricted_idx) > 0:
            # It is crucial to compute the bounds over the entire data set and
            # then restrict them to the rows in `rows_restricted_idx`
            # (otherwise, the results are not correct)
            row_bounds = self._get_bounds(
                S_intercept, S_intercept.T @ S_intercept + pen_intercept, ub, y, True
            )
            if np.abs(row_bounds.diff(1, 1).dropna(axis=1)).gt(BOUNDS_TOL).all().values:
                _ = M.addMConstr(
                    S[rows_restricted_idx, :],
                    theta,
                    GRB.GREATER_EQUAL,
                    row_bounds.loc[rows_restricted_idx, "lower"].values - y.mean(),
                )
                _ = M.addMConstr(
                    S[rows_restricted_idx, :],
                    theta,
                    GRB.LESS_EQUAL,
                    row_bounds.loc[rows_restricted_idx, "upper"].values - y.mean(),
                )

        # Construct the objective function
        obj = (
            theta.T @ (S.T @ S + pen) @ theta
            - 2 * (y - y.mean()).T @ S @ theta
            + np.dot(y - y.mean(), y - y.mean())
        )
        _ = M.setObjective(obj, GRB.MINIMIZE)
        return M

    def _run_optimizer(
        self,
        k: int,
        y: np.ndarray,
        init_sol: dict[str, Union[float, np.ndarray]],
        warm_start: Union[bool, dict[str, Any]],
        min_edf: int,
        compute_coef_bounds: bool,
        frac_row_bounds: float,
        **kwargs,
    ) -> gp.Model:
        """Runs the optimizer to find a solution that minimizes the variable
        selection problem with sparsity parameter `k`.

        Parameters
        ----------
        k : int
            The sparsity parameter.
        y : np.ndarray
            The response variable sample.
        init_sol : dict[str, Union[float, np.ndarray]]
            An initial solution containing the indicator variables "z", the
            regression coefficients "theta" and objective function value "obj".
        warm_start : bool
            When set to True, completes an initial solution for the problem from
            `init_sol` by adding variables to it. When it is dictionary, it is assumed
            that is in an input initial solution.
        min_edf : int
            The minimum effective degrees of freedom required so a non-linear
            term is not screened out.
        compute_coef_bounds: bool
            When set to True, computes the bounds for the coefficients.
            Otherwise, SOS-1 constraints are used.
        frac_row_bounds: float
            The proportion of rows in which the bounds for the linear
            combinations of the regression coefficients are computed. It must be
            a float between 0 and 1. If 0, the bounds are not computed.

        Returns
        -------
        gp.Model
            The fitted Gurobi model.
        """

        self._runs_time = 0
        # The probability assigned to the terms with very low effective degrees
        # of freedom
        PROB_ZERO = 1e-10
        # Define the destroy probability for each term
        # We add a small tolerance so the worst model does not have a zero
        # probability of being destroyed
        distances = np.abs(self.aic - (self.aic.max() + 1e-3))
        distances[np.where(self.edf < min_edf)[0] + self.m] = PROB_ZERO
        probs = np.divide(distances, np.linalg.norm(distances, ord=1))

        if self.conf_model["q"] != 2 * self.m:
            n_active_terms = np.flatnonzero(probs > PROB_ZERO)
            if len(n_active_terms) < self.conf_model["q"]:
                self.conf_model["q"] = len(n_active_terms)
        else:
            n_active_terms = np.arange(0, 2 * self.m)

        # Group the regression coefficients by the variables they belong to
        dict_theta = {}
        count_theta = 0
        for i, p in enumerate(np.diff(self.range_vars)):
            if i in n_active_terms:
                dict_theta[i] = list(range(count_theta, count_theta + p))
                count_theta += p

        # Complete the initial solution if `warm_start` is True
        if isinstance(warm_start, bool) and warm_start:
            init_sol = self._complete_initial_solution(
                dict_theta=dict_theta, k=k, y=y, init_sol=init_sol
            )
        elif isinstance(warm_start, dict):
            if isinstance(warm_start["z"], list):
                warm_start["z"] = np.array(warm_start["z"], dtype=int)
            init_sol = warm_start.copy()
        else:
            init_sol = {
                "z": np.zeros((2 * self.m,), dtype=int),
                "theta": np.zeros((count_theta + 1,)),
                "obj": 1e9,
            }
        self.initial_solution = init_sol.copy()
        self.initial_solution["z"] = list(self.initial_solution["z"])

        # Set column names in `evolution_df`
        evolution_cols = [
            "Best_objective",
            "Current_objective",
            "Times",
            "Iterations",
            "Destroyed_vars",
        ]
        # Create a DataFrame to record how the solution evolves
        evolution_df = pd.DataFrame(
            np.array(
                [
                    [
                        init_sol["obj"],
                        init_sol["obj"],
                        0.0,
                        -1,
                        list(np.where(init_sol["z"] == 1)[0]),
                    ]
                ],
                dtype=object,
            ),
            index=[-1],
            columns=evolution_cols,
        )

        # `wait` variable counts number of iterations without a new incumbent
        wait = 0
        init_time = time.time()
        for it in range(self.conf_model["max_iter"]):
            # Stop the algorithm if the maximum time is reached
            if (
                self.conf_model["max_time"] is not None
                and (time.time() - init_time) > self.conf_model["max_time"]
            ):
                logging.info(f"Time limit reached at iteration {it}.")
                break

            # Distinguish between optimality and heuristic:
            # * Full optimization problem: Every term is considered, so
            #   `to_destroy` is just a sequence of numbers from 0 to 2m.
            # * Heuristic: Only `q` terms are considered, so `to_destroy` is a
            #   randomly selected subset of the terms with probability `probs`
            if self.conf_model["q"] == 2 * self.m:
                to_destroy = np.array(range(0, 2 * self.m), dtype=int)
            else:
                to_destroy = np.sort(
                    np.random.choice(
                        n_active_terms,
                        size=self.conf_model["q"],
                        p=probs[n_active_terms],
                        replace=False,
                    )
                )

            # The parameter `L` must be upper bounded by the size of the
            # subproblems `q`
            dict_theta_ = {
                j: dict_theta[j] for j in dict_theta.keys() if j in to_destroy
            }
            L = min(
                k - len(np.setdiff1d(np.flatnonzero(init_sol["z"]), to_destroy)),
                self.conf_model["q"],
            )
            M = self._initialize_model(
                y=y,
                dict_theta=dict_theta_,
                L=L,
                init_sol=init_sol,
                compute_coef_bounds=compute_coef_bounds,
                frac_row_bounds=frac_row_bounds,
                **kwargs,
            )
            _ = M.optimize()

            self._runs_time += M.Runtime
            evolution_df.loc[it, "Times"] = self._runs_time

            provisional_z = (
                pd.Series(
                    [M.getVarByName(f"z[{i}]").X for i in range(len(to_destroy))],
                    index=np.sort(to_destroy),
                )
                .reindex(range(2 * self.m))
                .fillna(pd.Series(init_sol["z"]))
                .round()
                .astype(int)
                .values
            )
            assert provisional_z.sum() <= k

            provisional_sol = self._get_sol(
                self.matrixS, np.flatnonzero(provisional_z), dict_theta, y
            )

            if init_sol["obj"] > provisional_sol["obj"]:
                init_sol = provisional_sol.copy()
                # Resets wait counter
                wait = 0
            else:
                wait += 1
            evolution_df.loc[
                it,
                ["Best_objective", "Current_objective", "Iterations", "Destroyed_vars"],
            ] = np.array(
                [init_sol["obj"], provisional_sol["obj"], it, list(to_destroy)],
                dtype=object,
            )
            # Stop the algorithm if the solution does not change in the last
            # `n_iter_no_change` iterations (if it is not None)
            if (
                self.conf_model["n_iter_no_change"] is not None
                and wait >= self.conf_model["n_iter_no_change"]
            ):
                logging.info(
                    f"The algorithm ended at iteration {it + 1} since the solution",
                    f"did not changed in {self.conf_model['n_iter_no_change']} epochs.",
                    sep=os.linesep,
                )
                break

        M._evolution = evolution_df

        # Save the solution for the next k
        M._init_sol = init_sol

        M._sol = init_sol.copy()
        M._sol["z"] = M._sol["z"].tolist()
        return M

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K: Union[list[int], int],
        conf_gurobi: Optional[dict[str, Union[int, float]]] = None,
        conf_model: Optional[dict[str, Any]] = None,
        train_size: Optional[Union[int, float]] = 0.7,
        random_state: int = 0,
        warm_start: Union[bool, dict[str, Any]] = True,
        scale_y: bool = False,
        compute_coef_bounds: bool = True,
        frac_row_bounds: float = 1.0,
    ):
        """Given a training and validation set and the threshold list `K`, fits the
        variable selection model, returning the solution and the times by
        cutoff. Besides, returns the best k obtained in the validation set. The
        kwargs are referred to Gurobi parameters.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        y : np.ndarray
            The response variable sample.
        K : Union[list[int], int]
            The sparsity parameter, i.e., the number of non-zero terms allowed
            in the model.
            - If it is a list of integers, the data is split into train and
            validation, and the best sparsity parameter is the best according to
            a criterion in the validation set.
            - If it is an integer, fits the model over the entire dataset `X`
            and `y`. In this case, the `train_size` argument is ignored.
        conf_gurobi : Optional[dict[str,  Union[int,float]]], optional
            Parameters that control the operation of the Gurobi solvers. The
            keys of the dictionary are the names of the parameters while its
            values correspond to the given input. By default None
        conf_model : Optional[dict[str, Union[int, bool, str]]], optional
            Parameters that control the optimization problem which is going to
            be solved. These can be:
                - criterion : Indicates the criterion under the best k is
                  selected. By default `mse`
                - q : The size of the problems to be solved using the
                  matheuristic.
                - max_iter : The number of problems to be solved. By default 50.
                - max_time : The maximum allowed amount of time of each problem
                  to be solved. By default 10000.
                - n_iter_no_change : Number of iterations with no improvement
                  after which a partial problem will be stopped. By default 20.
                - patience : Number of sparsity parameters tested with no
                  improvement on the validation set. By default None.
                - ps_output : Whether to refit the PGL solution via P-splines.
                  By default False.
                - min_edf : The minimum effective degrees of freedom to consider
                  a non-linear term significant. By default 1.
            If any of these parameters is not inputted, a default parameter will
            be assigned. By default None
        train_size : Optional[Union[int, float]], optional
            A number between 0 and 1 that indicates the fraction `X` to be used
            for training. By default 0.7
                - If float, should be between 0.0 and 1.0 and represent the
                  proportion of the dataset to include in the train split.
                - If int, represents the absolute number of train samples.
                - If None, all samples are used in the fitting.
        random_state : int, optional
            Controls the shuffling applied to the data before applying the
            split. Pass an int for reproducible output across multiple function
            calls. By default 0
        warm_start: Union[bool, dict[str, Any]], optional
            When set to True, computes an initial solution for the problem,
            otherwise, just fit a whole new problem. The initial solution is
            computed for each k. When it is dictionary, it is assumed that is in an
            input initial solution. By default, True
        scale_y: bool, optional
            When set to True, scales the response variable `y` to have zero mean
            and unit variance. By default, False
        compute_coef_bounds: bool, optional
            When set to True, computes the bounds for the coefficients.
            Otherwise, SOS-1 constraints are used. By default, True
        frac_row_bounds: float, optional
            The proportion of rows in which the bounds for the linear
            combinations of the regression coefficients are computed. It must be
            a float between 0 and 1. If 0, the bounds are not computed. By
            default, 1.0

        Raises
        ------
        ValueError
            The number of elements used to construct the B-spline basis must agree
            with the number of variables.
        ValueError
            The sparsity parameter `K` must be an integer or a list.
        ValueError
            The parameter `frac_row_bounds` must be a float between 0 and 1.
        """

        if self.prediction is None:
            self.prediction = [{} for _ in range(X.shape[1])]

        common_dim_args = [self.deg, self.ord_d, self.n_int, self.prediction, X.T]
        if len({len(i) for i in common_dim_args}) != 1:
            raise ValueError(
                "Lengths of `deg`, `ord_d`, `n_int`, `prediction` and `x` must agree."
            )
        if not isinstance(K, (int, list)):
            raise ValueError("`K` must be an integer or list")
        if not 0 <= frac_row_bounds <= 1:
            raise ValueError("`frac_row_bounds` must be a float between 0 and 1.")

        if isinstance(K, int):
            if train_size is not None:
                logging.warning(
                    "Since `K` is an integer, the `train_size` argument is ignored."
                )
            train_size = None

        # Filling the arguments of model configuration
        self.conf_model = conf_model
        _ = self._fill_conf_model_args()
        if conf_gurobi is None:
            conf_gurobi = {"OutputFlag": 0, "threads": 1}
        # Construct the B-spline bases
        _ = self._get_bspline_bases(X=X)

        # Validation is not required if `K` is an integer
        if train_size is not None:
            X, X_val, y, y_val = train_test_split(
                X, y, train_size=train_size, shuffle=False, random_state=random_state
            )
        # Define design matrix of linear mixed model and center it
        self.matrixS = self._get_matrixS(X=X)
        self.scaler = StandardScaler(with_std=False)
        _ = self.scaler.fit(self.matrixS)
        self.matrixS = self.scaler.transform(self.matrixS)
        if scale_y:
            self.scaler_y = StandardScaler(with_std=False)
            _ = self.scaler_y.fit(y.reshape(1, -1).T)
            y = self.scaler_y.transform(y.reshape(1, -1).T).flatten()
            if train_size is not None:
                y_val = self.scaler_y.transform(y_val.reshape(1, -1).T).flatten()

        self.m = X.shape[1]
        sp_aic = (
            pd.DataFrame([self._get_sp(y=y, x_vars=[i]) for i in range(self.m)])
            .explode("sp")
            .explode("edf")
            .astype(np.float32)
        )
        linear_aic = sp_aic["aic_lineal"].values
        nonlinear_aic = sp_aic["aic_nonlinear"].values
        self.sp = sp_aic["sp"].values
        self.edf = sp_aic["edf"].values

        if self.conf_model["q"] >= 2 * self.m:
            if self.conf_model["q"] > 2 * self.m:
                logging.warning(
                    f"q = {self.conf_model['q']} surpasses the number of terms {2 * self.m}."
                )
                logging.warning(f"`q` is set to {2 * self.m}.")
                self.conf_model["q"] = 2 * self.m
            if self.conf_model["max_iter"] != 1:
                logging.warning("`max_iter` is set to 1.")
                self.conf_model["max_iter"] = 1

        if isinstance(warm_start, bool) and warm_start:
            start_time = time.time()
            # Get group lasso initial solutions
            self.pgl_groups = np.concatenate(
                [np.repeat(i, rep) for i, rep in enumerate(np.diff(self.range_vars))]
            )
            active_groups = np.where(self.edf >= self.conf_model["min_edf"])[0] + self.m
            self.pgl_sols = get_solutions_pgl(
                X=self._get_matrixS(X=X),
                y=y,
                X_val=self._get_matrixS(X=X_val) if train_size is not None else None,
                y_val=y_val if train_size is not None else None,
                K=K,
                sp=self.sp,
                groups=self.pgl_groups,
                criterion=self.conf_model["criterion"],
                max_iter=self.conf_model["n_iter_pgl"],
                tol=self.conf_model["tol_pgl"],
                n_alphas=self.conf_model["n_alphas"],
                eps=self.conf_model["eps"],
                ps_output=self.conf_model["ps_output"],
                active_groups=active_groups,
            )
            end_time = time.time()
            self.pgl_time = end_time - start_time

        # Construct the penalty term, setting big smoothing parameters to zero
        ls = [np.zeros((self.range_vars[self.m],))]
        for s, n_rep in zip(self.sp, np.diff(self.range_vars)[self.m :]):
            ls.append(np.repeat(s, n_rep))

        self.pen = np.diag(np.array(np.concatenate(ls)))

        sol = {}
        initial_solution = {}
        init_sol = None
        evolution = {}

        self.aic = np.concatenate((linear_aic, nonlinear_aic))

        wait = 0
        criterion_score = [np.inf]
        # Since `K` may be a list or an integer, we transform it to a np.ndarray
        # before iterating
        for k in tqdm(
            np.array([K]).flatten().tolist(),
            desc=f"\nK: {K} ",
            leave=False,
            colour="RED",
        ):
            if k > 2 * self.m:
                logging.warning(f"k = {k} surpasses the number of terms {2 * self.m}.")
                logging.warning(f"The value of `k` is set to {2 * self.m}.")
                k = 2 * self.m
            self.model = self._run_optimizer(
                y=y,
                k=k,
                init_sol=init_sol,
                warm_start=warm_start,
                min_edf=self.conf_model["min_edf"],
                compute_coef_bounds=compute_coef_bounds,
                frac_row_bounds=frac_row_bounds,
                **conf_gurobi,
            )

            # Get initial solution for next K's
            init_sol = self.model._init_sol
            # Save solution
            sol[k] = copy.deepcopy(self.model._sol)
            # Save initial solution
            initial_solution[k] = copy.deepcopy(self.initial_solution)
            # Save evolution
            evolution[k] = copy.deepcopy(self.model._evolution)
            if train_size is not None:
                if scale_y:
                    y_val_ = self.scaler_y.inverse_transform(
                        y_val.reshape(1, -1).T
                    ).flatten()
                else:
                    y_val_ = y_val.copy()
                y_pred_val = self.predict(X_val, sol[k]["theta"])
                if self.conf_model["criterion"] == "mse":
                    criterion_k = mean_squared_error(y_val_, y_pred_val)
                elif self.conf_model["criterion"] == "mae":
                    criterion_k = mean_absolute_error(y_val_, y_pred_val)
                else:
                    raise ValueError("Criterion not chosen correctly.")

                if np.min(criterion_score) > criterion_k:
                    wait = 0
                else:
                    wait += 1
                criterion_score.append(criterion_k)
                if (self.conf_model["patience"] is not None) and (
                    wait >= self.conf_model["patience"]
                ):
                    logging.info(
                        f"The algorithm stopped at sparsity {k} since the solution has",
                        f"not improved in {self.conf_model['patience']} iterations.",
                        sep=os.linesep,
                    )
                    break

        # Save best k
        self.k = K[np.argmin(criterion_score[1:])] if train_size is not None else K
        self.obj_evolution = evolution
        self.sol = sol[self.k]
        self.init_sol = initial_solution[self.k]
        self.ind_vars = (
            pd.DataFrame({key: sol[key]["z"] for key in sol.keys()})
            .T.rename(
                lambda i: f"l{i+1}" if i < self.m else f"nl{i%self.m+1}",
                axis="columns",
            )
            .rename_axis("K", axis=0)
        )
        self.reg_coefficients = {key: sol[key]["theta"] for key in sol.keys()}
        self.criterion_score = criterion_score[1:]
        return None

    def save(self, path: Path):
        """Saves results given by FSAM class in a pickle file. It saves the following
        data: objective evolution function in train set, solutions for every k, best k,
        indicator variables for every sparsity parameter, criterion score for validation
        set, configuration of the model, AICs scores and smoothing parameters.

        Parameters
        ----------
        path : Path
            Path where to save the pickle
        """
        dict_to_save = {
            "obj_evolution": self.obj_evolution,
            "regression_coefficients": self.reg_coefficients,
            "k": self.k,
            "ind_vars": self.ind_vars,
            "criterion_scores": self.criterion_score,
            "conf": self.conf_model,
            "aic": self.aic,
            "sp": self.sp,
            "edf": self.edf,
        }
        if not path.parent.exists():
            logging.info(f"Directory {path.parent} does not exist. Creating a new one.")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(path) + ".pkl", "wb") as handle:
            pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def predict(self, X: np.ndarray, sol: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimates the response variable from the data matrix `X`.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        sol : Optional[np.ndarray], optional
            The regression coefficients used to compute the estimated values of
            the target variable. If None, the coefficients to be used are the
            optimal parameters estimated during the fitting procedure, if this
            process was carried out. By default, None.

        Returns
        -------
        np.ndarray
            The estimated values for the best k of the response variable.

        Raises
        ------
        AttributeError
            If `sol` is None and the fitting procedure was not carried out, and
            hence `self.sol["theta"]` does not exists, the prediction cannot be
            performed.
        ValueError
            If some of the coordinates are outside the definition range of the
            B-spline bases.
        """
        bsp_min = np.array([bsp.knots[bsp.deg] for bsp in self.bspline_bases])
        bsp_max = np.array([bsp.knots[-bsp.deg] for bsp in self.bspline_bases])
        # If some coordinates are outside the range where the B-spline bases
        # were defined, the problem must be fitted again
        if (X.min(axis=0) < bsp_min).sum() > 0:
            bad_vars = np.where(X.min(axis=0) < bsp_min)[0]
            raise ValueError(
                f"Predictors {bad_vars} contain samples outside the definition range."
            )
        if (X.max(axis=0) > bsp_max).sum() > 0:
            bad_vars = np.where(X.max(axis=0) > bsp_max)[0]
            raise ValueError(
                f"Predictors {bad_vars} contain samples outside the definition range."
            )

        if sol is None:
            try:
                sol = self.sol["theta"]
            except AttributeError:
                raise ValueError("Provide a valid set of fitted coefficients `sol`.")
        # Transform the data matrix using reparametrized B-splines
        S = add_constant(
            self.scaler.transform(self._get_matrixS(X=X)), has_constant="add"
        )

        # Get the estimated values of y for best_k obtained in fit
        out = np.expand_dims(np.dot(S, sol), axis=1)
        if hasattr(self, "scaler_y"):
            out = self.scaler_y.inverse_transform(out)
        return out

    def _fill_conf_model_args(self):
        """Fill the `conf_model` dictionary by default parameters if they are not
        provided.
        """
        if self.conf_model is None:
            self.conf_model = {}
        self.conf_model["q"] = self.conf_model.get("q", 12)
        self.conf_model["max_iter"] = self.conf_model.get("max_iter", 50)
        self.conf_model["max_time"] = self.conf_model.get("max_time", 10000)
        self.conf_model["criterion"] = self.conf_model.get("criterion", "mse")
        self.conf_model["n_iter_no_change"] = self.conf_model.get(
            "n_iter_no_change", 20
        )
        self.conf_model["n_iter_pgl"] = self.conf_model.get("n_iter_pgl", 10000)
        self.conf_model["tol_pgl"] = self.conf_model.get("tol_pgl", 1e-5)
        self.conf_model["eps"] = self.conf_model.get("eps", 1e-3)
        self.conf_model["n_alphas"] = self.conf_model.get("n_alphas", 10)
        self.conf_model["ps_output"] = self.conf_model.get("ps_output", False)
        self.conf_model["patience"] = self.conf_model.get("patience", None)
        self.conf_model["min_edf"] = self.conf_model.get("min_edf", 1)
