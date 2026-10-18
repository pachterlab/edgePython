# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Levenberg-Marquardt GLM fitting for negative binomial models.

Port of edgeR's mglmLevenberg and nbinomDeviance. The genewise fitting and
deviance loops are numba ports of edgeR's C code, in glm_kernels.py.
"""

import numpy as np
from .compressed_matrix import CompressedMatrix
from .glm_kernels import (fit_leven_mat, leven_start, nb_deviance_sum_mat,
                          unit_nb_deviance_mat)


def mglm_levenberg(y, design, dispersion=0, offset=0, weights=None,
                   coef_start=None, start_method='null', maxit=200, tol=1e-06):
    """Fit genewise negative binomial GLMs using Levenberg damping.

    Port of edgeR's mglmLevenberg, following edgeR 5.0's C fitter: additive
    damping started at 1e-6 times the largest Fisher information, raised until
    the deviance decreases, and convergence when dl'dbeta < tol. Genes whose
    counts are all zero get NA coefficients and zero fitted values.

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
        beta = leven_start(y, offset_mat, disp_mat, w_mat, design, start_method == 'null')

    # Levenberg-Marquardt fit of each gene, as in edgeR's C code
    beta, fitted_values, deviance, n_iter, failed = fit_leven_mat(
        np.ascontiguousarray(y), np.ascontiguousarray(offset_mat, dtype=np.float64),
        np.ascontiguousarray(disp_mat, dtype=np.float64),
        np.ascontiguousarray(w_mat, dtype=np.float64), np.ascontiguousarray(design),
        np.ascontiguousarray(beta, dtype=np.float64), int(maxit), float(tol))

    return {
        'coefficients': beta,
        'fitted.values': fitted_values,
        'deviance': deviance,
        'iter': n_iter,
        'failed': failed
    }


def nbinom_deviance(y, mean, dispersion=0, weights=None):
    """Residual deviances for row-wise negative binomial GLMs.

    Port of edgeR's nbinomDeviance, which sums edgeR's C unit deviance: a
    small value is added to y and mean, and dispersions below 1e-4 (including
    0, the Poisson case) use a Taylor expansion.
    """
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
        mean = mean.reshape(1, -1)
    disp = _expand_dispersion(dispersion, y.shape)
    if weights is not None:
        w = np.ascontiguousarray(_expand_compressed(weights, y.shape), dtype=np.float64)
    else:
        w = np.ones_like(y)
    return nb_deviance_sum_mat(np.ascontiguousarray(y),
                               np.ascontiguousarray(np.broadcast_to(mean, y.shape)), disp, w)


def _expand_dispersion(dispersion, shape):
    """Dispersion as a contiguous (ngenes, nlibs) matrix: scalar, genewise or full."""
    if isinstance(dispersion, CompressedMatrix):
        return np.ascontiguousarray(dispersion.as_matrix(), dtype=np.float64)
    d = np.asarray(dispersion, dtype=np.float64)
    if d.ndim == 0 or d.size == 1:
        return np.full(shape, float(d.ravel()[0]))
    if d.ndim == 1 and len(d) == shape[0]:
        return np.ascontiguousarray(np.broadcast_to(d[:, None], shape))
    return np.ascontiguousarray(np.broadcast_to(d, shape))


def nbinom_unit_deviance(y, mean, dispersion=0):
    """Unit deviance for the negative binomial distribution.

    Port of edgeR's nbinomUnitDeviance, using edgeR's C unit deviance.
    """
    y = np.asarray(y, dtype=np.float64)
    mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), y.shape)
    if y.ndim == 1:
        disp = np.broadcast_to(np.asarray(dispersion, dtype=np.float64), y.shape)
        return unit_nb_deviance_mat(y[None, :].copy(), mean[None, :].copy(),
                                    disp[None, :].copy())[0]
    return unit_nb_deviance_mat(np.ascontiguousarray(y), np.ascontiguousarray(mean),
                                _expand_dispersion(dispersion, y.shape))


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
