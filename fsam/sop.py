import copy
import logging
from typing import Iterable, Optional, Union

import numpy as np


def compute_deviance(
    Quad: np.ndarray,
    G: np.ndarray,
    n: int,
    phi: Union[int, float],
    ssr: Union[int, float],
    edf: Union[int, float],
) -> float:
    """Estimates the restricted maximum likelihood (REML) deviance for variance
    components using the approach developed in [1].

    Parameters
    ----------
    Quad : np.ndarray
        The matrix resulting from multiplying `[X:Z].T * [X:Z]` plus the
        variance components associated to the nonlinear terms, i.e, the
        penalization matrix.
    G : np.ndarray
        The elements in the diagonal of the covariance matrix of the random
        effects.
    n : int
        The length of the response data.
    phi : Union[int, float]
        Dispersion parameter of the SOP algorithm.
    ssr : Union[int, float]
        Squared sum of the residuals.
    edf : Union[int, float]
        The effective degrees of freedom.

    Returns
    -------
    float
        Deviance value for the given data.

    References
    -------
    ... [1] https://cran.r-project.org/web/packages/SOP/SOP.pdf (Maria Xose
        Rodriguez-Alvarez).
    """
    _, log_det_Quad = np.linalg.slogdet(Quad)
    _, log_det_G = np.linalg.slogdet(G)
    return log_det_Quad + log_det_G + n * np.log(phi) + ssr / phi + edf


def sop_fit(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    G: list[np.ndarray],
    max_iter: int = 200,
    tol: Union[int, float] = 1e-6,
    phi_guess: Optional[Union[int, float]] = None,
    tau_guess: Optional[Iterable[Union[int, float]]] = None,
) -> dict[str, float]:
    """Get the smoothing parameters using the separation of overlapping
    precision matrices (SOP) method. This algorithm is an adaptation of
    the SOP R package [1].

    Parameters
    ----------
    y : np.ndarray
        Vector of observations (length n).
    X : np.ndarray
        Design matrix for the fixed effects (shape n x p).
    Z : np.ndarray
        Design matrix for the random effects (shape n x q).
    G : list[np.ndarray]
        List with the diagonal elements of the precision matrices for each
        variance component in the model. Each element of the list is a vector of
        the same length as the number of columns in Z.
    max_iter : int, optional
        Maximum number of iterations the SOP algorithm will perform, by default
        200.
    tol : Union[int, float], optional
        Tolerance for the convergence criterion, by default 1e-6.
    phi_guess : Union[int, float], optional
        Dispersion parameter. If None, it is initialized to 1. By default None.
    tau_guess : Iterable[Union[int, float]], optional
        Variance components. If None, it is initialized to a vector of 1s. By
        default None.

    Returns
    -------
    dict[str, float]
        A dictionary with keys `phi`,`tau` and `aic` containing the variance and
        deviance parameters resulting from the SOP algorithm and the Akaike
        Information Criterion of the given model.

    References
    -------
    .. [1] https://cran.r-project.org/web/packages/SOP/SOP.pdf (Maria Xose
        Rodriguez-Alvarez).
    """
    # If X unidimensional, convert it to bidimensional
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)

    # Compute [X:Z]*[X:Z].T, quadratic term and linear term from the basis
    S = np.concatenate((X, Z), axis=1)
    Quad = S.T @ S
    lin = np.dot(y, S)

    # Get number of linear and nonlinear variables and terms
    n = len(y)
    n_vars = len(G)
    nonlin_n = list(map(lambda x: np.count_nonzero(x), G))
    lin_terms = X.shape[1]

    # Minimum value for clipping to avoid numerical problems
    min_value = 1e-6

    # Initialize algorithm values in case they are not.
    phi = 1 if phi_guess is None else phi_guess
    tau = np.ones((n_vars)) if tau_guess is None else tau_guess
    ed = np.ones((n_vars))
    devold = np.inf

    for it in range(max_iter):
        # Compute only diagonal elements of the variance components
        # of the model
        G = np.repeat(tau, nonlin_n)
        Ginv = 1 / G
        # Compute the left side matrix of the Henderson equations as the
        # quadratic term plus the variance components
        C = np.multiply(1 / phi, Quad) + np.diag(
            np.concatenate((np.zeros(lin_terms), Ginv))
        )
        try:
            Cinv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Cinv = np.linalg.pinv(C)

        # Compute the solutions of the Henderson equations in a closed form
        theta = np.multiply(1 / phi, np.dot(Cinv, lin))
        # Get the associated nonlinear terms
        alpha_square = np.square(theta[lin_terms:])

        # Compute the diagonal elements of variance components minus each
        # partition of the C inverse corresponding to the k-th random component
        # alpha_k
        aux = G - np.diag(Cinv)[lin_terms:]
        # Estimated effective degrees of freedom associated
        # with each random component
        n_cumsum = np.cumsum([0] + nonlin_n[:-1])
        ed = np.clip(
            a=np.divide(np.add.reduceat(aux, n_cumsum), tau),
            a_min=min_value,
            a_max=None,
        )
        # Estimated variance parameters
        tau = np.clip(
            np.divide(np.add.reduceat(alpha_square, n_cumsum), ed),
            a_min=min_value,
            a_max=None,
        )

        # Compute the error term
        err = y - np.dot(S, theta)
        # Compute the residual sum of squares
        ssr = np.linalg.norm(err) ** 2
        dev = compute_deviance(
            Quad=C,
            G=np.diag(G),
            n=n,
            phi=phi,
            ssr=ssr,
            edf=np.dot(alpha_square, Ginv),
        )
        # Dispersion parameter
        phi = ssr / (n - lin_terms - ed.sum())
        # Convergence criterion based on REML
        if np.abs(devold - dev) < tol:
            break
        devold_ = copy.copy(devold)
        devold = dev.copy()
    # In case SOP algorithm has reached `max_iter` without converging.
    if it == max_iter - 1:
        logging.warning(
            "SOP algorithm has not converged."
            f"Convergence score is {np.abs(devold_ - dev)} while `tol` is {tol}."
        )
    # Compute AICs: the deviance is the sum of squared residuals, and we penalize them
    # with twice the effective degrees of freedom of the system
    aic = 2 * ed.sum() + ssr

    # `ed` gives the effective degrees of freedom of the non-linear tems
    return {"phi": phi, "tau": tau, "aic": aic, "edf": ed}
