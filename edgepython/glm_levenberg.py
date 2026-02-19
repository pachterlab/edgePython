# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Levenberg-Marquardt GLM fitting for negative binomial models.

Port of edgeR's mglmLevenberg and nbinomDeviance (C/C++ code reimplemented in NumPy).
"""

import numpy as np
from .compressed_matrix import (CompressedMatrix, compress_offsets,
                                 compress_weights, compress_dispersions)


def mglm_levenberg(y, design, dispersion=0, offset=0, weights=None,
                   coef_start=None, start_method='null', maxit=200, tol=1e-06):
    """Fit genewise negative binomial GLMs using Levenberg damping.

    Port of edgeR's mglmLevenberg.

    Parameters
    ----------
    y : ndarray
        Count matrix (genes x samples).
    design : ndarray
        Design matrix (samples x coefficients).
    dispersion : float or ndarray
        NB dispersions.
    offset : float, ndarray, or CompressedMatrix
        Log-scale offsets.
    weights : ndarray, optional
        Observation weights.
    coef_start : ndarray, optional
        Starting coefficient values.
    start_method : str
        'null' or 'y' for initialization.
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict with 'coefficients', 'fitted.values', 'deviance', 'iter', 'failed'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ngenes, nlibs = y.shape

    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    ncoefs = design.shape[1]

    # Handle empty design
    if ncoefs == 0:
        offset_mat = _expand_compressed(offset, y.shape)
        fitted = np.exp(offset_mat)
        dev = nbinom_deviance(y, fitted, dispersion, weights)
        return {
            'coefficients': np.zeros((ngenes, 0)),
            'fitted.values': fitted,
            'deviance': dev,
            'iter': np.zeros(ngenes, dtype=int),
            'failed': np.zeros(ngenes, dtype=bool)
        }

    # Expand offset, dispersion, weights
    offset_mat = _expand_compressed(offset, y.shape)
    disp_mat = _expand_compressed(dispersion, y.shape)
    if np.any(np.asarray(disp_mat, dtype=np.float64) < 0):
        raise ValueError("Negative dispersions not allowed")
    if weights is not None:
        w_mat = _expand_compressed(weights, y.shape)
    else:
        w_mat = np.ones_like(y)

    # Initialize coefficients
    if coef_start is not None:
        beta = np.asarray(coef_start, dtype=np.float64)
        if beta.ndim == 1:
            beta = np.tile(beta, (ngenes, 1))
    else:
        beta = _get_levenberg_start(y, offset_mat, disp_mat, w_mat, design, start_method == 'null')

    # Levenberg-Marquardt iteration for each gene
    coefficients = np.zeros((ngenes, ncoefs))
    fitted_values = np.zeros_like(y)
    deviance = np.zeros(ngenes)
    n_iter = np.zeros(ngenes, dtype=int)
    failed = np.zeros(ngenes, dtype=bool)

    for g in range(ngenes):
        beta_g = beta[g].copy()
        y_g = y[g]
        offset_g = offset_mat[g]
        disp_g = disp_mat[g] if disp_mat.ndim == 2 else disp_mat
        w_g = w_mat[g] if w_mat.ndim == 2 else w_mat

        if np.isscalar(disp_g):
            disp_g = np.full(nlibs, disp_g)
        if np.isscalar(w_g):
            w_g = np.full(nlibs, w_g)

        converged = False
        lev = 1e-3  # Levenberg damping parameter

        for it in range(maxit):
            # Compute mu
            eta = design @ beta_g + offset_g
            mu = np.exp(np.clip(eta, -500, 500))
            mu = np.maximum(mu, 1e-300)

            # Working weights
            denom = 1 + disp_g * mu
            working_w = w_g * mu ** 2 / (mu * denom)
            working_w = np.maximum(working_w, 1e-300)

            # Working residuals
            z = (y_g - mu) / mu

            # Weighted least squares
            W = np.diag(working_w)
            XtWX = design.T @ W @ design
            XtWz = design.T @ (working_w * z)

            # Add Levenberg damping
            XtWX_lev = XtWX + lev * np.diag(np.diag(XtWX) + 1e-10)

            try:
                delta = np.linalg.solve(XtWX_lev, XtWz)
            except np.linalg.LinAlgError:
                failed[g] = True
                break

            # Compute deviance before update
            dev_old = _unit_deviance_sum(y_g, mu, disp_g, w_g)

            # Trial update
            beta_new = beta_g + delta
            eta_new = design @ beta_new + offset_g
            mu_new = np.exp(np.clip(eta_new, -500, 500))
            mu_new = np.maximum(mu_new, 1e-300)
            dev_new = _unit_deviance_sum(y_g, mu_new, disp_g, w_g)

            if dev_new <= dev_old:
                # Accept and decrease damping
                beta_g = beta_new
                lev = max(lev / 10, 1e-10)
                if abs(dev_old - dev_new) < tol * (abs(dev_old) + 0.1):
                    converged = True
                    n_iter[g] = it + 1
                    break
            else:
                # Reject and increase damping
                lev = min(lev * 10, 1e10)

        if not converged and not failed[g]:
            n_iter[g] = maxit
            # Use last good values
            eta = design @ beta_g + offset_g
            mu = np.exp(np.clip(eta, -500, 500))

        coefficients[g] = beta_g
        eta_final = design @ beta_g + offset_g
        fitted_values[g] = np.exp(np.clip(eta_final, -500, 500))

    deviance = nbinom_deviance(y, fitted_values, dispersion, weights)

    return {
        'coefficients': coefficients,
        'fitted.values': fitted_values,
        'deviance': deviance,
        'iter': n_iter,
        'failed': failed
    }


def _get_levenberg_start(y, offset, dispersion, weights, design, use_null):
    """Get starting values for Levenberg-Marquardt."""
    ngenes, nlibs = y.shape
    ncoefs = design.shape[1]
    beta = np.zeros((ngenes, ncoefs))

    if use_null:
        # Start from null model (intercept only via offset)
        for g in range(ngenes):
            lib_size = np.exp(offset[g] if offset.ndim == 2 else offset)
            total = np.sum(y[g])
            total_lib = np.sum(lib_size)
            if total > 0 and total_lib > 0:
                mu_hat = total / total_lib
                # Solve for beta[0] such that exp(X*beta + offset) ≈ y
                # With null start, set all beta to 0 except intercept
                beta[g, 0] = np.log(mu_hat) if mu_hat > 0 else -20
    else:
        # Start from y values
        for g in range(ngenes):
            lib_size = np.exp(offset[g] if offset.ndim == 2 else offset)
            y_norm = y[g] / np.maximum(lib_size, 1e-300)
            y_norm = np.maximum(y_norm, 1e-300)
            log_y = np.log(y_norm)
            try:
                beta[g] = np.linalg.lstsq(design, log_y, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta[g, 0] = np.mean(log_y)

    return beta


def nbinom_deviance(y, mean, dispersion=0, weights=None):
    """Residual deviances for row-wise negative binomial GLMs.

    Port of edgeR's nbinomDeviance. Fully vectorized over genes.
    """
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(1, -1)
        mean = mean.reshape(1, -1)

    ngenes, nlibs = y.shape
    mean = np.maximum(mean, 1e-300)

    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if dispersion.size == 1:
        disp = dispersion[0]
    elif dispersion.ndim == 1 and len(dispersion) == ngenes:
        disp = dispersion
    elif isinstance(dispersion, np.ndarray) and dispersion.shape == y.shape:
        disp = dispersion
    else:
        disp = np.broadcast_to(dispersion, ngenes).copy()

    if weights is not None:
        w = _expand_compressed(weights, y.shape)
    else:
        w = None

    # Compute unit deviance for entire matrix at once
    scalar_disp = np.isscalar(disp) or (isinstance(disp, np.ndarray) and disp.ndim == 0)
    if scalar_disp:
        d = float(disp)
    else:
        d = disp

    if scalar_disp and d == 0:
        # Poisson case
        unit_dev = np.zeros_like(y)
        pos = y > 0
        unit_dev[pos] = 2 * (y[pos] * np.log(y[pos] / mean[pos]) - (y[pos] - mean[pos]))
        unit_dev[~pos] = 2 * mean[~pos]
    elif scalar_disp:
        # Scalar NB dispersion - most common case
        unit_dev = np.zeros_like(y)
        pos = y > 0
        if np.any(pos):
            unit_dev[pos] = 2 * (y[pos] * np.log(y[pos] / mean[pos]) -
                                  (y[pos] + 1.0 / d) * np.log((1 + d * y[pos]) /
                                                                (1 + d * mean[pos])))
        zero = ~pos
        if np.any(zero):
            unit_dev[zero] = 2.0 / d * np.log(1 + d * mean[zero])
    else:
        # Per-gene or per-element dispersion
        if d.ndim == 1:
            d_mat = d[:, None]  # (ngenes, 1)
        else:
            d_mat = d  # (ngenes, nlibs)
        unit_dev = np.zeros_like(y)
        pos = y > 0
        if np.any(pos):
            d_pos = np.broadcast_to(d_mat, y.shape)[pos]
            unit_dev[pos] = 2 * (y[pos] * np.log(y[pos] / mean[pos]) -
                                  (y[pos] + 1.0 / d_pos) * np.log((1 + d_pos * y[pos]) /
                                                                    (1 + d_pos * mean[pos])))
        zero = ~pos
        if np.any(zero):
            d_zero = np.broadcast_to(d_mat, y.shape)[zero]
            unit_dev[zero] = 2.0 / d_zero * np.log(1 + d_zero * mean[zero])

    unit_dev = np.maximum(unit_dev, 0)
    if w is not None:
        return np.sum(w * unit_dev, axis=1)
    return np.sum(unit_dev, axis=1)


def nbinom_unit_deviance(y, mean, dispersion=0):
    """Unit deviance for the negative binomial distribution.

    Port of edgeR's nbinomUnitDeviance.
    """
    return _unit_nb_deviance(y, mean, dispersion)


def _unit_nb_deviance(y, mu, dispersion):
    """Compute unit negative binomial deviance."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.maximum(mu, 1e-300)

    if np.isscalar(dispersion):
        disp = dispersion
    else:
        disp = np.asarray(dispersion, dtype=np.float64)

    # Poisson case
    if np.isscalar(disp) and disp == 0:
        dev = np.zeros_like(y)
        pos = y > 0
        dev[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos]))
        dev[~pos] = 2 * mu[~pos]
        return dev

    # NB case
    dev = np.zeros_like(y)
    pos = y > 0
    zero = ~pos

    if np.isscalar(disp):
        # y > 0 part
        if np.any(pos):
            dev[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) -
                            (y[pos] + 1 / disp) * np.log((1 + disp * y[pos]) /
                                                          (1 + disp * mu[pos])))
        # y == 0 part
        if np.any(zero):
            dev[zero] = 2 / disp * np.log(1 + disp * mu[zero])
    else:
        if np.any(pos):
            dev[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) -
                            (y[pos] + 1 / disp[pos]) * np.log((1 + disp[pos] * y[pos]) /
                                                               (1 + disp[pos] * mu[pos])))
        if np.any(zero):
            d_z = disp[zero]
            dev[zero] = 2 / d_z * np.log(1 + d_z * mu[zero])

    return np.maximum(dev, 0)


def _unit_deviance_sum(y, mu, disp, weights):
    """Sum of weighted unit deviances."""
    ud = _unit_nb_deviance(y, mu, disp)
    return np.sum(weights * ud)


def _expand_compressed(x, shape):
    """Expand a scalar, vector, or CompressedMatrix to full matrix."""
    if isinstance(x, CompressedMatrix):
        return x.as_matrix()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.size == 1:
        return np.full(shape, x.ravel()[0])
    if x.ndim == 1:
        if len(x) == shape[1]:
            return np.tile(x, (shape[0], 1))
        elif len(x) == shape[0]:
            return np.tile(x.reshape(-1, 1), (1, shape[1]))
    if x.shape == shape:
        return x
    return np.broadcast_to(x, shape).copy()
