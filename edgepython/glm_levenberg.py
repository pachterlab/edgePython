# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Levenberg-Marquardt GLM fitting for negative binomial models.

Port of edgeR's mglmLevenberg and nbinomDeviance (C/C++ code reimplemented in NumPy).
"""

import numpy as np
from .compressed_matrix import CompressedMatrix


def mglm_levenberg(y, design, dispersion=0, offset=0, weights=None,
                   coef_start=None, start_method='null', maxit=200, tol=1e-06):
    """Fit genewise negative binomial GLMs using Levenberg damping.

    Port of edgeR's mglmLevenberg. Vectorized over all genes simultaneously
    using batch matrix operations (einsum, batched linalg.solve, active mask).

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

    # Expand offset, dispersion, weights to full (ngenes, nlibs) matrices
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

    # Vectorized Levenberg-Marquardt iteration over all genes
    beta = beta.copy()
    n_iter = np.zeros(ngenes, dtype=int)
    failed = np.zeros(ngenes, dtype=bool)
    active = np.ones(ngenes, dtype=bool)
    lev = np.full(ngenes, 1e-3)
    coef_idx = np.arange(ncoefs)

    # Precompute design outer-product matrix: design_outer[j, k*ncoefs+l] = X[j,k]*X[j,l]
    # XtWX[g] = working_w_a[g] @ design_outer  (reshaped to (ncoefs, ncoefs))
    # This converts the 3-operand einsum to a single BLAS-3 matmul per iteration.
    design_outer = (design[:, :, None] * design[:, None, :]).reshape(nlibs, ncoefs * ncoefs)
    design_T = design.T  # (ncoefs, nlibs) — precomputed for eta computation

    for it in range(maxit):
        if not np.any(active):
            break
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            a = active
            n_a = np.count_nonzero(a)

            # Compute mu for active genes: eta = design @ beta + offset
            # beta[a] is (n_a, ncoefs), design is (nlibs, ncoefs)
            # eta[g, j] = sum_k(design[j,k] * beta[g,k]) + offset[g,j]
            eta_a = beta[a] @ design_T + offset_mat[a]
            mu_a = np.exp(np.clip(eta_a, -500, 500))
            mu_a = np.maximum(mu_a, 1e-300)

            # Working weights: w * mu / (1 + disp * mu)
            denom_a = 1.0 + disp_mat[a] * mu_a
            working_w_a = w_mat[a] * mu_a / denom_a
            working_w_a = np.maximum(working_w_a, 1e-300)

            # Working residuals
            z_a = (y[a] - mu_a) / mu_a

            # Batch XtWX: (n_a, ncoefs, ncoefs)
            # working_w_a @ design_outer: (n_a, nlibs) @ (nlibs, p²) → (n_a, p²)
            XtWX = (working_w_a @ design_outer).reshape(n_a, ncoefs, ncoefs)

            # Batch XtWz: (n_a, ncoefs)
            # (n_a, nlibs) @ (nlibs, ncoefs) → (n_a, ncoefs)
            XtWz = (working_w_a * z_a) @ design

            # Add per-gene Levenberg damping to diagonal
            diag_vals = np.diagonal(XtWX, axis1=1, axis2=2)
            XtWX_lev = XtWX.copy()
            XtWX_lev[:, coef_idx, coef_idx] += lev[a, np.newaxis] * (diag_vals + 1e-10)

            # Batch solve: (n_a, ncoefs)
            delta = np.linalg.solve(XtWX_lev, XtWz[..., np.newaxis])[..., 0]

            # Deviance before update (inline, NOT using _unit_deviance_sum)
            ud_old = _unit_nb_deviance(y[a], mu_a, disp_mat[a])
            dev_old = np.sum(w_mat[a] * ud_old, axis=1)

            # Trial update
            beta_new_a = beta[a] + delta
            eta_new_a = beta_new_a @ design_T + offset_mat[a]
            mu_new_a = np.exp(np.clip(eta_new_a, -500, 500))
            mu_new_a = np.maximum(mu_new_a, 1e-300)

            # Deviance after trial update
            ud_new = _unit_nb_deviance(y[a], mu_new_a, disp_mat[a])
            dev_new = np.sum(w_mat[a] * ud_new, axis=1)

            # Accept/reject per gene (within active set)
            accept_local = dev_new <= dev_old
            reject_local = ~accept_local

            # Update beta only for accepted genes
            active_indices = np.where(a)[0]
            accept_global = active_indices[accept_local]
            reject_global = active_indices[reject_local]

            beta[accept_global] = beta_new_a[accept_local]

            # Update Levenberg damping per gene
            lev[accept_global] = np.maximum(lev[accept_global] / 10.0, 1e-10)
            lev[reject_global] = np.minimum(lev[reject_global] * 10.0, 1e10)

            # Check convergence among accepted genes
            if np.any(accept_local):
                rel_change = np.abs(dev_old[accept_local] - dev_new[accept_local])
                threshold = tol * (np.abs(dev_old[accept_local]) + 0.1)
                converged_local = rel_change < threshold
                converged_global = accept_global[converged_local]
                n_iter[converged_global] = it + 1
                active[converged_global] = False

    # Genes that never converged get n_iter = maxit
    n_iter[active] = maxit

    # Compute final fitted values for all genes
    eta_final = beta @ design.T + offset_mat
    fitted_values = np.exp(np.clip(eta_final, -500, 500))

    deviance = nbinom_deviance(y, fitted_values, dispersion, weights)

    return {
        'coefficients': beta,
        'fitted.values': fitted_values,
        'deviance': deviance,
        'iter': n_iter,
        'failed': failed
    }


def _get_levenberg_start(y, offset, dispersion, weights, design, use_null):
    """Get starting values for Levenberg-Marquardt (vectorized over genes)."""
    ngenes, nlibs = y.shape
    ncoefs = design.shape[1]
    beta = np.zeros((ngenes, ncoefs))

    if use_null:
        # Vectorized null start: estimate intercept from total counts / total lib size
        total_y = np.sum(y, axis=1)  # (ngenes,)
        if offset.ndim == 2:
            total_lib = np.sum(np.exp(offset), axis=1)  # (ngenes,)
        else:
            total_lib = np.full(ngenes, np.sum(np.exp(offset)))
        valid = (total_y > 0) & (total_lib > 0)
        mu_hat = np.where(valid, total_y / np.maximum(total_lib, 1e-300), 0.0)
        beta[:, 0] = np.where(valid & (mu_hat > 0), np.log(np.maximum(mu_hat, 1e-300)), -20.0)
    else:
        # Start from y values — per-gene lstsq (infrequently called)
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
        unit_dev = _unit_deviance_by_element(y, mean, np.broadcast_to(d_mat, y.shape))

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
        dev = _unit_deviance_by_element(y, mu, np.broadcast_to(disp, y.shape))

    return np.maximum(dev, 0)


def _unit_deviance_by_element(y, mu, disp):
    """Unit deviance with a dispersion per element.

    As in edgeR, a zero dispersion gives the Poisson deviance for that
    element, so an array of zeros is a Poisson fit rather than 0/0.
    """
    dev = np.zeros_like(y)
    pos = y > 0
    pois = disp == 0
    m = pos & pois
    dev[m] = 2 * (y[m] * np.log(y[m] / mu[m]) - (y[m] - mu[m]))
    m = ~pos & pois
    dev[m] = 2 * mu[m]
    m = pos & ~pois
    d = disp[m]
    dev[m] = 2 * (y[m] * np.log(y[m] / mu[m]) -
                  (y[m] + 1 / d) * np.log((1 + d * y[m]) / (1 + d * mu[m])))
    m = ~pos & ~pois
    d = disp[m]
    dev[m] = 2 / d * np.log(1 + d * mu[m])
    return dev


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
