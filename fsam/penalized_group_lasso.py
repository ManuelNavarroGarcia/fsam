#!/usr/bin/env python
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tools import add_constant


class PenalisedGroupLasso:
    def __init__(
        self,
        alpha_path: Iterable[Union[int, float]] = None,
        tol: Union[int, float] = 1e-3,
        max_iter: int = 100,
        tol_sparsity: Union[int, float] = 1e-5,
        n_alphas: int = 30,
        eps: Union[int, float] = 1e-3,
    ):
        self.alpha_path = alpha_path
        self.tol = tol
        self.max_iter = max_iter
        self.tol_sparsity = tol_sparsity
        self.n_alphas = n_alphas
        self.eps = eps

    def _solve_linear_model(
        self,
        X: np.ndarray,
        residual: np.ndarray,
        alpha: Union[int, float],
    ) -> np.ndarray:
        """Find the estimated coefficients for the Group Lasso algorithm with
        just one group using a closed form solution. The problem

        argmin_{beta_j} ||y - X_j beta_j||_2^2 + alpha * ||X_j beta_j||_2

        has the following closed-form solution:

        beta_j = (1 - 0.5 * alpha/||X_j^T @ y||_2)_+ (X_j^T @ y) / ||X_j^T||_2

        Parameters
        ----------
        X : np.ndarray
            The linear design matrix.
        residual : np.ndarray
            The response variable.
        alpha : np.ndarray
            The shrinkage parameter.

        Returns
        -------
        np.ndarray
            The estimated coefficients.
        """
        aux_dot = np.dot(X.T, residual)
        aux_max = max(0, 1 - 0.5 * alpha * np.linalg.norm(X) / np.abs(aux_dot))
        return aux_max * aux_dot / np.linalg.norm(X) ** 2

    def _solve_nonlinear_model(
        self,
        Q: np.ndarray,
        R_inv: np.ndarray,
        residual: np.ndarray,
        alpha: Union[int, float],
        sp: Union[int, float],
        beta_norm: Union[int, float],
    ) -> np.ndarray:
        """Find the estimated coefficients for the adapted problem (4.2) in [1]
        using a closed form solution.

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
        beta_norm : Union[int, float]
            The value of the norm of the estimated coefficient.

        Returns
        -------
        np.ndarray
            The estimated coefficients.

        References
        ----------
        ... [1] Chouldechova, A., & Hastie, T. (2015). Generalized additive
        model selection. arXiv preprint arXiv:1506.03850.
        """
        aux = 2 * np.dot(Q.T, residual)
        if np.linalg.norm(aux) < alpha:
            out = 0
        else:
            out = np.linalg.solve(
                (2 + alpha / beta_norm) * np.eye(R_inv.shape[1])
                + 2 * sp * R_inv.T @ R_inv,
                aux,
            )
        return out

    def _get_beta_norm(
        self,
        Q: np.ndarray,
        residual: np.ndarray,
        S_inv: np.ndarray,
        L: np.ndarray,
        alpha: Union[int, float],
        init_sol: Union[int, float],
    ) -> Union[int, float]:
        """Find the norm of the estimated coefficient for the adapted problem
        (4.2) in [1] using a one-dimensional line search.

        Parameters
        ----------
        Q : np.ndarray
            The orthogonal matrix of the QR decomposition of the design matrix.
        residual : np.ndarray
            The response variable.
        S_inv : np.ndarray
            The inverse of the upper triangular matrix of the QR decomposition
            of the matrix 2 * (sp * R.T @ R + np.eye(R.shape[0])).
        L : np.ndarray
            The diagonal matrix of the eigenvalues of the QR decomposition of
            the matrix 2 * (sp * R.T @ R + np.eye(R.shape[0])).
        alpha : Union[int, float]
            The shrinkage parameter.
        init_sol : Union[int, float]
            The initial guess for the norm of the solution.

        Returns
        -------
        Union[int,float]
            The norm of the coefficient.

        References
        ----------
        ... [1] Chouldechova, A., & Hastie, T. (2015). Generalized additive
        model selection. arXiv preprint arXiv:1506.03850.
        """
        aux_ = np.dot(Q.T, residual)
        if np.linalg.norm(2 * aux_) < alpha:
            beta_norm = 0.0
        else:
            aux = S_inv @ aux_
            beta_norm = minimize(
                lambda s: np.square(
                    np.sum(np.square((np.divide(aux, (L * s + alpha))))) - 0.25
                ),
                x0=init_sol,
                method="L-BFGS-B",
                bounds=[(1e-5, 1e6)],
                options={"ftol": 1e-12, "gtol": 1e-12},
            ).x[0]
        return beta_norm

    def _get_loss_function(
        self,
        XQs: Iterable[np.ndarray],
        R_invs: Iterable[np.ndarray],
        y: np.ndarray,
        groups: Iterable[int],
        coefs: np.ndarray,
        intercept: Union[int, float],
        sp: Union[int, float],
        alpha: Union[int, float],
    ) -> float:
        """Get the loss function value when the predictors are centered and the
        design matrices of the non-linear parts are decomposed into a QR
        decompositions.

        Parameters
        ----------
        XQs : Iterable[np.ndarray]
            An iterable with the linear design matrices and the orthogonal
            matrices of the QR decompositions of the non-linear design matrices.
        R_invs : Iterable[np.ndarray],
            An iterable with the inverse of the upper triangular matrices of the
            QR decompositions of the non-linear design matrices.
        y : np.ndarray
            The response variable.
        groups : Iterable[int]
            An array containing the indexes of the groups.
        coefs : np.ndarray
            The estimated coefficients.
        intercept : Union[int, float]
            The estimated intercept.
        sp : Union[int, float]
            The smoothing paramater.
        alpha : Union[int, float]
            The shrinkage parameter.

        Returns
        -------
        float
            The value of the loss function.
        """
        m = len(sp)
        penalty_ps = 0
        penalty_gl = 0

        for j in range(m):
            linear_ = np.flatnonzero(groups == j)
            non_linear_ = np.flatnonzero(groups == (j + m))
            coefs_nonlin = coefs[non_linear_]
            if len(np.flatnonzero(coefs_nonlin)) > 0:
                penalty_gl += np.linalg.norm(
                    np.multiply(XQs[:, j], coefs[linear_]), ord=2
                )

            if len(np.flatnonzero(coefs_nonlin)):
                penalty_ps += sp[j] * np.square(
                    np.linalg.norm(np.dot(R_invs[j], coefs_nonlin), ord=2),
                )
                penalty_gl += np.linalg.norm(coefs_nonlin, ord=2)

        fitting = np.dot(XQs, coefs)
        loss = (
            np.square(np.linalg.norm(y - intercept - fitting, ord=2))
            + penalty_ps
            + penalty_gl * alpha
        )
        return loss

    def _get_final_loss_function(
        self,
        S: np.ndarray,
        y: np.ndarray,
        groups: Iterable[int],
        coefs: np.ndarray,
        intercept: Union[int, float],
        sp: Union[int, float],
        alpha: Union[int, float],
    ) -> float:
        """Get the loss function value.

        Parameters
        ----------
        S : np.ndarray
            The design matrix.
        y : np.ndarray
            The response variable.
        groups : Iterable[int]
            An array containing the indexes of the groups.
        coefs : np.ndarray
            The estimated coefficients.
        intercept : Union[int, float]
            The estimated intercept.
        sp : Union[int, float]
            The smoothing paramater.
        alpha : Union[int, float]
            The shrinkage parameter.

        Returns
        -------
        float
            The value of the loss function.
        """
        m = len(sp)
        penalty_ps = 0
        penalty_gl = 0

        for j in range(m):
            linear_ = np.where(groups == j)[0]
            non_linear_ = np.where(groups == (j + m))[0]
            coefs_nonlin = coefs[non_linear_]
            penalty_ps += sp[j] * np.square(np.linalg.norm(coefs_nonlin, ord=2))
            penalty_gl += np.linalg.norm(
                np.dot(S[:, linear_], coefs[linear_]), ord=2
            ) + np.linalg.norm(np.dot(S[:, non_linear_], coefs_nonlin), ord=2)

        loss = (
            np.square(np.linalg.norm(y - intercept - np.dot(S, coefs), ord=2))
            + penalty_ps
            + penalty_gl * alpha
        )
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Iterable[int],
        sp: Iterable[float],
        scale_y: bool = False,
        active_groups: Optional[np.ndarray] = None,
    ):
        """Get the estimated coefficients of the Penalized Group Lasso. If not
        given, computes `alpha_path` with a logscale, with the maximum value
        the one that gives at least one non-zero regression coefficient, and
        the minimum is given by alpha_max * eps.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.
        y : np.ndarray
            The response sample.
        groups : Iterable[int]
            The indexes of the groups. It is assumed that, for `g` groups, they
            are given by `range(g)`.
        sp : Iterable[float]
            The smoothing parameters.
        scale_y: bool, optional
            When set to True, scales the response variable `y` to have zero
            mean. By default, False.
        active_groups
            An array of group indexes whose regression coefficients will be
            computed. By default, None.

        Raises
        ------
        ValueError
            If any group has more features than observations.
        """
        # Define a group for the intercept
        INTERCEPT_COL = -1
        # Center the data
        self.scaler = StandardScaler(with_std=False)
        _ = self.scaler.fit(X)
        X = self.scaler.transform(X)
        if scale_y:
            self.scaler_y = StandardScaler(with_std=False)
            _ = self.scaler_y.fit(y.reshape(1, -1).T)
            y = self.scaler_y.transform(y.reshape(1, -1).T).flatten()
        groups_index = np.unique(groups)
        m = int(len(groups_index) / 2)
        if active_groups is None:
            active_groups = np.array(range(m, 2 * m))

        Xs = [x.reshape(1, -1).T for x in X[:, :m].T]
        # Get the QR decomposition of the non-linear design matrix for each
        # regressor
        Qs, R_invs = [], []
        for group in groups_index[m:]:
            Q, R = np.linalg.qr(X[:, np.where(groups == group)[0]])
            Qs.append(Q)
            if len(set(R.shape)) > 1:
                raise ValueError(f"More features than observations for group: {group}")
            R_invs.append(np.linalg.inv(R))
        # Get the SVD decomposition of the matrices 2 * (sp * R.T @ R +
        # np.eye(R.shape[0])) for each smoothing parameter and upper triangular
        # matrix R of the non-linear design matrix
        SVD_nonlin = [
            np.linalg.svd(
                2 * (s * R.T @ R + np.eye(R.shape[0])),
                hermitian=True,
            )
            for s, R in zip(sp, R_invs)
        ]
        XQs = np.concatenate(Xs + Qs, axis=1)
        # Initialize the norm of the coefficients for each group
        beta_norm = np.ones(m) * 1e-5

        if self.alpha_path is None:
            Qs_lin = [x / np.linalg.norm(x) for x in Xs]
            Qs_nonlin = list(map(Qs.__getitem__, active_groups - m))
            alpha_max = np.max(
                [
                    np.linalg.norm(2 * np.dot(Q.T, y - y.mean()))
                    for Q in Qs_lin + Qs_nonlin
                ]
            )
            self.alpha_path = np.logspace(
                np.log10(alpha_max),
                np.log10(alpha_max * self.eps),
                num=self.n_alphas,
            ).tolist()

        # Initialize to zero the regression coefficients
        coefs_alphas = (
            pd.DataFrame(
                index=[INTERCEPT_COL] + self.alpha_path,
                columns=groups,
                dtype=np.float64,
            )
            .T.fillna({INTERCEPT_COL: 0})
            .T
        )
        intercept_alphas = pd.Series(index=self.alpha_path, dtype=np.float64)
        losses_alphas = pd.Series(index=self.alpha_path, dtype=np.float64)
        active_set = list(
            np.array(list(zip(groups_index[:m], active_groups))).flatten()
        )

        for i, alpha in enumerate(self.alpha_path):
            # Initialize the loss function that we want to minimize
            losses = [np.inf]
            # Get the optimal coefficients for the previous shrinkage parameter
            # to accelerate convergence
            coefs_est = coefs_alphas.iloc[i, :].values.astype(float)
            coefs_old = coefs_est.copy()
            # As at the end of every alpha the coefficients are transformed
            # back and clipped by sparsity we need to recompute this.
            XQ_coef = np.dot(XQs, coefs_est)

            for it in range(self.max_iter):
                for j in active_set:
                    # Compute the estimate of the linear coefficient
                    coefs_est_ = coefs_est.copy()
                    coefs_est_[np.where(groups == j)[0]] = 0.0
                    if not np.all(coefs_old == coefs_est_):
                        XQ_coef = np.dot(XQs, coefs_est_)
                    residuals = y - (y.mean() + XQ_coef)

                    if j in groups_index[:m]:
                        coefs_ = self._solve_linear_model(
                            X=Xs[j], residual=residuals, alpha=alpha
                        )

                    # Compute the estimate of the non-linear coefficient
                    if j in active_groups:
                        k = j - m
                        beta_norm_ = self._get_beta_norm(
                            Q=Qs[k],
                            S_inv=SVD_nonlin[k][2],
                            L=SVD_nonlin[k][1],
                            residual=residuals,
                            alpha=alpha,
                            init_sol=beta_norm[k],
                        )
                        beta_norm[k] = beta_norm_
                        coefs_ = self._solve_nonlinear_model(
                            Q=Qs[k],
                            R_inv=R_invs[k],
                            residual=residuals,
                            alpha=alpha,
                            sp=sp[k],
                            beta_norm=beta_norm[k],
                        )
                    # Copy the old coefficients to avoid computing `XQ_coef`
                    # the times the coefficients do not change
                    coefs_old = coefs_est.copy()
                    coefs_est[np.where(groups == j)[0]] = (
                        coefs_ if np.abs(coefs_).max() > self.tol_sparsity else 0
                    )

                losses.append(
                    self._get_loss_function(
                        XQs, R_invs, y, groups, coefs_est, y.mean(), sp, alpha
                    )
                )
                # Check convergence
                if np.abs(losses[it] - losses[it + 1]) < self.tol:
                    print(f"Convergence reached at iteration {it}.")
                    break

                if it == (self.max_iter - 1):
                    print(
                        "The algorithm has not converged. Actual loss:",
                        f"{np.abs(losses[it] - losses[it + 1])} ",
                    )
            # After the convergence, get the original regression coefficients
            # for the non-linear terms (as we benefit of the QR decomposition
            # for the design matrix)
            for j in groups_index[m:]:
                coefs_transformed = np.dot(
                    R_invs[j - m], coefs_est[np.where(groups == j)[0]]
                )
                coefs_est[np.where(groups == j)[0]] = (
                    coefs_transformed
                    if np.abs(coefs_transformed).max() > self.tol_sparsity
                    else 0
                )
            # Get the transformed intercept and the value of the original loss
            # function for the estimated coefficients
            intercept = (y - np.dot(X, coefs_est)).mean()
            intercept_alphas.loc[alpha] = intercept
            coefs_alphas.loc[alpha, :] = coefs_est
            losses_alphas.loc[alpha] = self._get_final_loss_function(
                X,
                y,
                groups,
                coefs_est,
                intercept,
                sp,
                alpha,
            )
        # Save the regression coefficients and the loss function values for each
        # shrinkage parameter
        self.coef_ = pd.concat(
            (intercept_alphas.to_frame(name=INTERCEPT_COL), coefs_alphas.iloc[1:, :]),
            axis=1,
        )
        self.ps_coef_ = self.get_ps_coefficients(X, y, sp)
        self.loss_ps_ = pd.Series(
            [
                self._get_final_loss_function(
                    X,
                    y,
                    groups,
                    coefs,
                    intercept,
                    sp,
                    alpha,
                )
                for coefs in self.ps_coef_.values[:, 1:]
            ],
            index=self.alpha_path,
        )
        self.loss_ = losses_alphas.copy()

    def predict(self, X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """
        Estimates the response variable from the data matrix `X`.

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
        X = add_constant(self.scaler.transform(X), has_constant="add")
        out = np.expand_dims(np.dot(X, coefs), axis=1)
        if hasattr(self, "scaler_y"):
            out = self.scaler_y.inverse_transform(out)
        return out

    def get_ps_coefficients(
        self, X: np.ndarray, y: np.ndarray, sp: Iterable[float]
    ) -> pd.DataFrame:
        """Estimate via P-splines the regression coefficients for each `alpha`
        during the fitting procedure.

           Parameters
           ----------
           X : np.ndarray
               The data matrix.
           y : np.ndarray
               The response variable sample.
           sp : Iterable[float]
               The smoothing parameters.

           Returns
           -------
           pd.DataFrame
               A similar DataFrame as `self.coef_` but containing the P-splines
               regression coefficients
        """
        # Make a copy of `self.coef_` (we transpose it by convinience)
        aux_df = self.coef_.T.copy()
        m = len(sp)
        # Iterate over each row of `self.coef_` (not the intercept)
        for alpha, ser in self.coef_.iloc[:, 1:].iterrows():
            # Select the non-zero coefficients and add a column of ones
            X_ = add_constant(X[:, np.where(ser)[0]], has_constant="add")
            # Get the groups that have non-zero coefficients
            nonzero_groups = ser[ser.ne(0)].index.unique()
            # Create the penalization matrix (the +1 is to consider intercept)
            pen_ = np.diag(
                np.concatenate(
                    [np.zeros((nonzero_groups < m).sum() + 1)]
                    + [
                        np.repeat(sp[g - m], len(ser.loc[g]))
                        for g in nonzero_groups[nonzero_groups >= m]
                    ]
                )
            )
            ser_ = ser.reset_index(drop=True)
            # Compute the P-splines regression coefficients that are non-zero,
            # and then fill the series with zeros when null contribution is
            # expected
            ps_coef = (
                pd.Series(
                    np.linalg.solve(X_.T @ X_ + pen_, np.dot(X_.T, y)),
                    index=np.r_[
                        -1,
                        ser_[ser_.ne(0)].index,
                    ],
                )
                .reindex(range(-1, self.coef_.shape[1] - 1))
                .fillna(0)
            )
            # We assign the DataFrame a column with the name "alpha_", which
            # will be assigned at the rightmost place. Then, we drop the
            # leftmost column (the one with the coefficients for `alpha`)
            aux_df = aux_df.assign(
                **{str(alpha) + "_": ps_coef.set_axis(self.coef_.columns)}
            ).drop(columns=aux_df.columns[0], axis=1)
        # Finally, we get the original columns
        return aux_df.T.set_axis(self.coef_.index, axis=0)


def get_solutions_pgl(
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    K: Union[list[int], int],
    groups: Iterable[int],
    sp: Iterable[Union[int, float]],
    criterion: str = "mse",
    alpha_path: Optional[list[Union[int, float]]] = None,
    max_iter: int = 10000,
    tol: float = 1e-5,
    n_alphas: int = 10,
    eps: float = 1e-3,
    ps_output: bool = False,
    active_groups: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Select the best feature for each sparsity parameter in `K` based on the
    penalized group lasso algorithm for a set of shrinkage parameters.

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    y : np.ndarray
        The response variable sample
    X_val : Optional[np.ndarray]
        The validation data matrix. If is not None, the best sparsity parameter
        is chosen by the shrinkage parameter that minimizes `criterion` in the
        validation set. Otherwise, no validation is done.
    y_val : Optional[np.ndarray]
        The validation response variable sample. The same behaviour applies as
        for `X_val`.
    K : Union[list[int], int]
        An iterable with different values for the sparsity parameter.
    groups : Iterable[int]
        Contains the feature indices of the corresponding group.
    sp : Iterable[Union[int, float]]
        The smoothing parameter vector of training data.
    criterion : str
        Indicates the criterion under the best alpha is selected. By default
        `mse`.
    alpha_path : Optional[list[Union[int, float]]]
        The list of candidates for the shrinkage parameter, by default None.
    max_iter : int
        Maximum number of iterations to be executed by the group-lasso
        algorithm. By default 10000.
    tol : Union[int, float], optional
        The convergence tolerance. The optimisation algorithm will stop once
        ||x_{n+1} - x_n|| < ``tol``. By default 1e-5.
    n_alphas : int, optional
        Number of shrinkage parameters along the regularization path, by
        default 10.
    eps : float, optional
        Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.
        By default 1e-3.
    ps_output : bool, optional
        If True, the `criterion` is applied over the P-splines regression
        coefficients computed from the chosen variables by the penalized group
        lasso algorithm. By default False.
    active_groups
        An array of group indexes whose regression coefficients will be
        computed. By default, None.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with index `K` containing a list with the selected
        variables.

    Raises
    ------
    ValueError
        The criterion must be a string valued "mse" or "mae".
    ValueError
        `X_val` and `y_val` must share the type.
    ValueError
        `K` must be an integer or list.
    """
    if criterion not in ("mse", "mae"):
        raise ValueError("Criterion not chosen correctly.")
    if (X_val is None) != (y_val is None):
        raise ValueError("`X_val` and `y_val` types must agree.")
    if not isinstance(K, (int, list)):
        raise ValueError("`K` must be an integer or list")

    pgl = PenalisedGroupLasso(
        alpha_path=alpha_path, max_iter=max_iter, tol=tol, n_alphas=n_alphas, eps=eps
    )
    pgl.fit(X, y, groups, sp=sp, active_groups=active_groups)

    df_coefs = pgl.ps_coef_.copy() if ps_output else pgl.coef_.copy()
    chosen_vars, val_scores = [], []
    for alpha, row in df_coefs.iterrows():
        # Discard variables with zero coefficients. Note that the first index is
        # the intercept, so we discard it beforehand
        chosen_vars.append(
            row.iloc[1:]
            .reset_index(drop=False)
            .groupby("index")
            .filter(lambda g: g[alpha].ne(0).any())["index"]
            .unique()
        )
        if X_val is not None:
            if criterion == "mse":
                val_score = mean_squared_error(
                    y_val, pgl.predict(X_val, row.values).flatten()
                )
            elif criterion == "mae":
                val_score = mean_absolute_error(
                    y_val, pgl.predict(X_val, row.values).flatten()
                )
            val_scores.append(val_score)
        else:
            val_scores = None
    df = pd.DataFrame(
        {
            "loss": pgl.loss_ps_ if ps_output else pgl.loss_,
            "vars": chosen_vars,
            "k": [len(a) for a in chosen_vars],
            "score": val_scores,
        }
    )

    # Get a DataFrame with index `k` and values the selected variables for each
    # k. Since `K` may be a list or an integer, we transform it to a np.ndarray
    # before sorting
    K_ord = np.sort(np.array([K]).flatten()).tolist()
    df_init_sols = (
        df.query("k > 0")  # Get rows with at least one variable
        .sort_values(["k", "score"], ascending=False)
        .rename_axis("alpha")
        .reset_index()
        .drop_duplicates("k", keep="last")
        .set_index("k")
        .reindex(range(1, K_ord[-1] + 1))
        .sort_index()
        .ffill()
        .reindex(K_ord)
    )

    return df_init_sols
