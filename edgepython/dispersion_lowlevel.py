# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Low-level dispersion estimation functions for edgePython.

Port of edgeR's adjustedProfileLik, maximizeInterpolant,
condLogLikDerDelta, condLogLikDerSize, dispCoxReid,
dispCoxReidInterpolateTagwise, dispCoxReidSplineTrend, dispBinTrend, etc.
"""

import numpy as np
import warnings
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import minimize
from numba import njit

from .utils import (expand_as_matrix, systematic_subset, moving_average_by_col,
                    cut_with_min_n)
from .expression import ave_log_cpm
from .glm_kernels import adj_profile_ll_mat


def adjusted_profile_lik_grid(grid_dispersions, y, design, offset, weights=None):
    """Adjusted profile log-likelihoods at each of a grid of dispersions.

    Equivalent to calling adjusted_profile_lik once per grid point, as
    edgeR's estimateDisp does. grid_dispersions is (ngrid,) for common grid
    values or (ngenes, ngrid) for genewise ones. Returns (ngenes, ngrid).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    grid_dispersions = np.asarray(grid_dispersions, dtype=np.float64)
    ngrid = grid_dispersions.shape[-1]
    apl = np.empty((y.shape[0], ngrid), dtype=np.float64)
    for gi in range(ngrid):
        d = grid_dispersions[:, gi] if grid_dispersions.ndim == 2 else grid_dispersions[gi]
        apl[:, gi] = adjusted_profile_lik(d, y, design, offset, weights=weights)
    return apl


def adjusted_profile_lik(dispersion, y, design, offset, weights=None,
                         start=None, get_coef=False):
    """Tagwise Cox-Reid adjusted profile log-likelihoods for the dispersion.

    Port of edgeR's adjustedProfileLik (C code reimplemented).

    Parameters
    ----------
    dispersion : float or ndarray
        Dispersion value(s).
    y : ndarray
        Count matrix (genes x samples).
    design : ndarray
        Design matrix.
    offset : ndarray
        Offset matrix.
    weights : ndarray, optional
        Observation weights.
    start : ndarray, optional
        Starting coefficients for GLM fit.
    get_coef : bool
        If True, return coefficients along with APL.

    Returns
    -------
    ndarray of adjusted profile log-likelihoods (one per gene),
    or dict with 'apl' and 'beta' if get_coef=True.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ngenes, nlibs = y.shape
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)

    offset = np.asarray(offset, dtype=np.float64)
    if offset.ndim == 1:
        offset = np.tile(offset, (ngenes, 1))

    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if len(dispersion) == 1:
        disp = np.full(ngenes, dispersion[0])
    else:
        disp = dispersion

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = np.tile(w, (ngenes, 1))
    else:
        w = np.ones_like(y)

    # Fitted values from edgeR's GLM fit (fit_apl): one-way or Levenberg,
    # without prior counts
    from .glm_fit import glm_fit
    fit = glm_fit(y, design=design, dispersion=disp, offset=offset,
                  weights=weights, prior_count=0, start=start)
    mu = np.ascontiguousarray(fit['fitted.values'], dtype=np.float64)
    beta = fit['coefficients']

    disp_mat = np.ascontiguousarray(np.broadcast_to(disp[:, None], y.shape), dtype=np.float64)
    apl = adj_profile_ll_mat(np.ascontiguousarray(y), mu, disp_mat,
                             np.ascontiguousarray(np.broadcast_to(w, y.shape), dtype=np.float64),
                             np.ascontiguousarray(design), True)

    if get_coef:
        return {'apl': apl, 'beta': beta}
    return apl


@njit(cache=True)
def _fmm_spline(n, x, y, b, c, d):
    """Forsythe-Malcolm-Moler cubic spline (matches R's splines.c / edgeR's fmm_spline).

    Computes coefficients b, c, d such that in segment i:
        S(t) = y[i] + b[i]*t + c[i]*t^2 + d[i]*t^3
    where t = x_eval - x[i].
    """
    if n < 2:
        return
    if n < 3:
        t = (y[1] - y[0]) / (x[1] - x[0])
        b[0] = t
        b[1] = t
        c[0] = c[1] = d[0] = d[1] = 0.0
        return

    nm1 = n - 1

    # Set up tridiagonal system
    # Using d for offdiagonal, b for diagonal, c for RHS
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]
    for i in range(1, nm1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    # End conditions (FMM: match third derivatives)
    b[0] = -d[0]
    b[nm1] = -d[nm1 - 1]
    c[0] = 0.0
    c[nm1] = 0.0
    if n > 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[nm1] = c[nm1 - 1] / (x[nm1] - x[nm1 - 2]) - c[nm1 - 2] / (x[nm1 - 1] - x[nm1 - 3])
        c[0] = c[0] * d[0] * d[0] / (x[3] - x[0])
        c[nm1] = -c[nm1] * d[nm1 - 1] * d[nm1 - 1] / (x[nm1] - x[nm1 - 3])

    # Gaussian elimination
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]

    # Backward substitution
    c[nm1] = c[nm1] / b[nm1]
    for i in range(nm1 - 1, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    # Compute polynomial coefficients
    b[nm1] = (y[nm1] - y[nm1 - 1]) / d[nm1 - 1] + d[nm1 - 1] * (c[nm1 - 1] + 2.0 * c[nm1])
    for i in range(nm1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]
    c[nm1] = 3.0 * c[nm1]
    d[nm1] = d[nm1 - 1]


@njit(cache=True)
def _maximize_interpolant_kernel(x, y_mat, ngenes, npts, result):
    """Numba kernel: FMM spline + analytical max (matches edgeR's C find_max).

    For each gene, fits an FMM cubic spline, finds the grid point with the
    highest value, then analytically solves for the maximum on the two
    neighbouring segments by finding roots of the derivative (a quadratic).
    This is O(npts) per gene with no discretisation artifacts.
    """
    b = np.empty(npts)
    c = np.empty(npts)
    d = np.empty(npts)
    y_row = np.empty(npts)

    for g in range(ngenes):
        # Copy row (fmm_spline modifies y in-place via c)
        for i in range(npts):
            y_row[i] = y_mat[g, i]

        # Find coarse grid maximum
        maxed = y_row[0]
        maxed_at = 0
        for i in range(1, npts):
            if y_row[i] > maxed:
                maxed = y_row[i]
                maxed_at = i
        x_max = x[maxed_at]

        # Fit FMM spline: S(t) = y[i] + b[i]*t + c[i]*t^2 + d[i]*t^3
        _fmm_spline(npts, x, y_row, b, c, d)

        # Check left segment (maxed_at - 1)
        if maxed_at > 0:
            seg = maxed_at - 1
            lb = b[seg]
            lc = c[seg]
            ld = d[seg]

            # Derivative: b + 2c*t + 3d*t^2 = 0
            # Discriminant: (2c)^2 - 4*(3d)*b = 4*(c^2 - 3*d*b)
            delta = lc * lc - 3.0 * ld * lb
            if delta >= 0.0:
                # Solution for maximum (not minimum)
                numerator = -lc - np.sqrt(delta)
                chosen_sol = numerator / (3.0 * ld) if ld != 0.0 else 0.0

                seg_width = x[maxed_at] - x[seg]
                if chosen_sol > 0.0 and chosen_sol < seg_width:
                    temp = ((ld * chosen_sol + lc) * chosen_sol + lb) * chosen_sol + y_row[seg]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[seg]

        # Check right segment (maxed_at)
        if maxed_at < npts - 1:
            seg = maxed_at
            rb = b[seg]
            rc = c[seg]
            rd = d[seg]

            delta = rc * rc - 3.0 * rd * rb
            if delta >= 0.0:
                numerator = -rc - np.sqrt(delta)
                chosen_sol = numerator / (3.0 * rd) if rd != 0.0 else 0.0

                seg_width = x[seg + 1] - x[seg]
                if chosen_sol > 0.0 and chosen_sol < seg_width:
                    temp = ((rd * chosen_sol + rc) * chosen_sol + rb) * chosen_sol + y_row[seg]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[seg]

        result[g] = x_max


def maximize_interpolant(x, y):
    """Find the maximum of an interpolated function for each row.

    Port of edgeR's maximizeInterpolant. Uses FMM cubic spline fitting
    followed by analytical maximum finding on neighbouring segments,
    matching R's C implementation exactly.

    Parameters
    ----------
    x : ndarray
        Grid points (sorted, unique).
    y : ndarray
        Log-likelihood matrix (genes x grid points).

    Returns
    -------
    ndarray of maximizing x values (one per gene).
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    ngenes = y.shape[0]
    npts = len(x)

    result = np.empty(ngenes, dtype=np.float64)
    _maximize_interpolant_kernel(x, y, ngenes, npts, result)
    return result


def cond_log_lik_der_size(y, r, der=0):
    """Derivatives of conditional log-likelihood w.r.t. r=1/dispersion.

    Port of edgeR's condLogLikDerSize.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    n = y.shape[1]
    m = np.mean(y, axis=1)

    if der == 0:
        # Log-likelihood
        return (np.sum(gammaln(y + r[:, None]), axis=1) +
                gammaln(n * r) - gammaln(n * (m + r)) - n * gammaln(r))
    elif der == 1:
        # First derivative
        return (np.sum(digamma(y + r[:, None]), axis=1) +
                n * digamma(n * r) - n * digamma(n * (m + r)) - n * digamma(r))
    elif der == 2:
        # Second derivative
        return (np.sum(polygamma(1, y + r[:, None]), axis=1) +
                n**2 * polygamma(1, n * r) - n**2 * polygamma(1, n * (m + r)) -
                n * polygamma(1, r))
    else:
        raise ValueError(f"der must be 0, 1, or 2, got {der}")


def cond_log_lik_der_delta(y, delta, der=0):
    """Derivatives of conditional log-likelihood w.r.t. delta=dispersion/(1+dispersion).

    Port of edgeR's condLogLikDerDelta.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    delta = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    r = (1.0 / delta) - 1.0

    if der == 0:
        return cond_log_lik_der_size(y, r, der=0)
    elif der == 1:
        return cond_log_lik_der_size(y, r, der=1) * (-delta**(-2))
    elif der == 2:
        return (cond_log_lik_der_size(y, r, der=1) * 2 * delta**(-3) +
                cond_log_lik_der_size(y, r, der=2) * delta**(-4))
    else:
        raise ValueError(f"der must be 0, 1, or 2, got {der}")


def common_cond_log_lik_der_delta(y_split, delta, der=0):
    """Sum of conditional log-likelihoods across groups.

    Port of edgeR's commonCondLogLikDerDelta.
    """
    total = 0.0
    for y_group in y_split:
        total += np.sum(cond_log_lik_der_delta(y_group, delta, der=der))
    return total


def _cond_log_lik_grid(y, r_vals):
    """Vectorized conditional log-likelihood (der=0) for a grid of r values.

    Equivalent to stacking cond_log_lik_der_size(y, r, der=0) for each r in
    r_vals, but computed with a single 3-D gammaln broadcast — eliminating the
    Python loop over grid points.

    Parameters
    ----------
    y : ndarray (ngenes, nsamples)
    r_vals : ndarray (ngrid,)  —  r = 1/dispersion values (same for all genes)

    Returns
    -------
    ndarray (ngenes, ngrid)
    """
    y = np.asarray(y, dtype=np.float64)
    r_vals = np.asarray(r_vals, dtype=np.float64)
    n = y.shape[1]
    m = np.mean(y, axis=1)  # (ngenes,)

    # y[:, :, None] + r_vals[None, None, :] -> (ngenes, nsamples, ngrid)
    ll = (np.sum(gammaln(y[:, :, None] + r_vals[None, None, :]), axis=1)  # (ngenes, ngrid)
          + gammaln(n * r_vals)[None, :]                                    # (1, ngrid)
          - gammaln(n * (m[:, None] + r_vals[None, :]))                     # (ngenes, ngrid)
          - n * gammaln(r_vals)[None, :])                                   # (1, ngrid)
    return ll


def disp_cox_reid(y, design=None, offset=None, weights=None, ave_log_cpm_vals=None,
                  interval=(0, 4), tol=1e-5, min_row_sum=5, subset=10000):
    """Cox-Reid APL estimator of common dispersion.

    Port of edgeR's dispCoxReid.

    Parameters
    ----------
    y : ndarray
        Count matrix.
    design : ndarray, optional
        Design matrix.
    offset : ndarray, optional
        Offset.
    weights : ndarray, optional
        Weights.
    ave_log_cpm_vals : ndarray, optional
        Pre-computed AveLogCPM values.
    interval : tuple
        Search interval for dispersion.
    tol : float
        Optimization tolerance.
    min_row_sum : int
        Minimum row sum.
    subset : int
        Number of genes to subset.

    Returns
    -------
    float : estimated common dispersion.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    if design is None:
        design = np.ones((y.shape[1], 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

    if offset is None:
        offset = np.log(y.sum(axis=0))
    offset = expand_as_matrix(np.asarray(offset, dtype=np.float64), y.shape)

    if interval[0] < 0:
        raise ValueError("please give a non-negative interval for the dispersion")

    # Apply min row count
    row_sums = y.sum(axis=1)
    keep = row_sums >= min_row_sum
    if not np.all(keep):
        y = y[keep]
        offset = offset[keep]
        if weights is not None:
            weights = np.asarray(weights)
            if weights.ndim == 2:
                weights = weights[keep]
        if ave_log_cpm_vals is not None:
            ave_log_cpm_vals = ave_log_cpm_vals[keep]

    if y.shape[0] < 1:
        raise ValueError("no data rows with required number of counts")

    # Subsetting
    if subset is not None and subset <= y.shape[0] / 2:
        if ave_log_cpm_vals is None:
            ave_log_cpm_vals = ave_log_cpm(y, offset=offset, weights=weights)
        i = systematic_subset(subset, ave_log_cpm_vals)
        y = y[i]
        offset = offset[i]
        if weights is not None and weights.ndim == 2:
            weights = weights[i]

    def fun(par):
        return -np.sum(adjusted_profile_lik(par ** 4, y, design, offset, weights=weights))

    # Optimize over par = dispersion^0.25 with R's optimize(), as edgeR does
    from .utils import optimize_brent
    par = optimize_brent(fun, min(interval) ** 0.25, max(interval) ** 0.25, tol=tol)
    return par ** 4


def disp_cox_reid_interpolate_tagwise(y, design, offset=None, dispersion=None,
                                       trend=True, ave_log_cpm_vals=None,
                                       min_row_sum=5, prior_df=10, span=0.3,
                                       grid_npts=11, grid_range=(-6, 6),
                                       weights=None):
    """Estimate tagwise NB dispersions using Cox-Reid APL with interpolation.

    Port of edgeR's dispCoxReidInterpolateTagwise.

    Parameters
    ----------
    y : ndarray
        Count matrix.
    design : ndarray
        Design matrix.
    offset : ndarray, optional
        Offset.
    dispersion : float or ndarray
        Starting dispersion(s).
    trend : bool
        Use trend.
    ave_log_cpm_vals : ndarray, optional
        Average log CPM.
    min_row_sum : int
        Minimum row sum.
    prior_df : float
        Prior degrees of freedom.
    span : float
        Span for moving average.
    grid_npts : int
        Number of grid points.
    grid_range : tuple
        Range for grid.
    weights : ndarray, optional
        Weights.

    Returns
    -------
    ndarray of tagwise dispersions.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    ncoefs = design.shape[1]

    if offset is None:
        offset = np.log(y.sum(axis=0))
    offset = expand_as_matrix(np.asarray(offset, dtype=np.float64), y.shape)

    if ave_log_cpm_vals is None:
        # As edgeR: library sizes are column sums unless offsets were given
        ave_log_cpm_vals = ave_log_cpm(y)

    # A copy: the dispersions are updated below and must not change the
    # caller's (e.g. a DGEList's trended.dispersion)
    dispersion = np.array(dispersion, dtype=np.float64, ndmin=1)
    if len(dispersion) == 1:
        dispersion = np.full(ntags, dispersion[0])
    elif len(dispersion) != ntags:
        raise ValueError("length of dispersion doesn't match nrow(y)")

    # Apply min_row_sum
    row_sums = y.sum(axis=1)
    keep = row_sums >= min_row_sum
    if not np.all(keep):
        if np.any(keep):
            dispersion[keep] = disp_cox_reid_interpolate_tagwise(
                y[keep], design, offset=offset[keep],
                dispersion=dispersion[keep],
                ave_log_cpm_vals=ave_log_cpm_vals[keep],
                grid_npts=grid_npts, min_row_sum=0,
                prior_df=prior_df, span=span, trend=trend,
                weights=weights[keep] if weights is not None and np.ndim(weights) == 2 else weights)
        return dispersion

    # Posterior profile likelihood
    prior_n = prior_df / (nlibs - ncoefs)
    spline_pts = np.linspace(grid_range[0], grid_range[1], grid_npts)

    # Build (ntags, ngrid) dispersion matrix and evaluate APL in one batched call,
    # avoiding repeated setup overhead (gammaln(y+1), group structure, etc.)
    disp_grid = dispersion[:, None] * (2.0 ** spline_pts[None, :])  # (ntags, ngrid)
    apl = adjusted_profile_lik_grid(disp_grid, y, design, offset, weights=weights)

    if trend:
        o = np.argsort(ave_log_cpm_vals)
        oo = np.argsort(o)
        width = int(np.floor(span * ntags))
        width = max(width, 1)
        apl_smooth = moving_average_by_col(apl[o], width=width)[oo]
    else:
        apl_smooth = np.tile(np.mean(apl, axis=0), (ntags, 1))

    apl_smooth = (apl + prior_n * apl_smooth) / (1 + prior_n)

    # Tagwise maximization
    d = maximize_interpolant(spline_pts, apl_smooth)
    return dispersion * 2 ** d


def _ns_basis_with_knots(x, internal_knots, boundary_knots):
    """Create natural cubic spline basis matching R's cbind(1, ns(x, knots=knots)).

    Uses the truncated power basis from ESL (Hastie et al.) eq 5.4-5.5.

    Parameters
    ----------
    x : array
        Data values.
    internal_knots : array
        Internal knot positions.
    boundary_knots : array of length 2
        [lower, upper] boundary knots.

    Returns
    -------
    ndarray of shape (n, len(internal_knots) + 2)
        Basis matrix including intercept column.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    internal_knots = np.asarray(internal_knots, dtype=np.float64)

    all_knots = np.sort(np.concatenate([[boundary_knots[0]],
                                         internal_knots,
                                         [boundary_knots[1]]]))
    K = len(all_knots)
    ncols = K  # = len(internal_knots) + 2

    basis = np.zeros((n, ncols))
    basis[:, 0] = 1.0
    basis[:, 1] = x

    if K > 2:
        xi_K = all_knots[-1]
        xi_Km1 = all_knots[-2]

        def d_func(xi_j):
            return (np.maximum(x - xi_j, 0) ** 3 -
                    np.maximum(x - xi_K, 0) ** 3) / (xi_K - xi_j)

        d_Km1 = d_func(xi_Km1)
        for j in range(K - 2):
            basis[:, 2 + j] = d_func(all_knots[j]) - d_Km1

    return basis


def _optim(fun, par0, method):
    """R's optim(par0, fun, method = method): Nelder-Mead is ported exactly."""
    if method == 'Nelder-Mead':
        from .utils import optim_nelder_mead
        return optim_nelder_mead(fun, par0)[0]
    return minimize(fun, par0, method=method).x


def _ns_r(x, knots, boundary_knots, intercept=False):
    """R's splines::ns(x, knots = knots, intercept = intercept) for x within
    the boundary knots.

    A cubic B-spline basis on the boundary knots (each four times) and the
    interior knots, with the first function dropped unless intercept, made
    natural by projecting out the second derivatives at the boundaries with
    the QR decomposition of the constraint matrix, as ns() does.
    """
    from scipy.interpolate import BSpline
    x = np.asarray(x, dtype=np.float64)
    bk = np.sort(np.asarray(boundary_knots, dtype=np.float64))
    if np.any((x < bk[0]) | (x > bk[1])):
        raise ValueError("x outside the boundary knots")
    aknots = np.sort(np.concatenate([np.repeat(bk, 4), np.asarray(knots, dtype=np.float64)]))
    nb = len(aknots) - 4

    def design(points, deriv):
        # splineDesign(); x is within the knots, where extrapolate=True
        # also covers the right boundary knot
        out = np.empty((len(points), nb))
        for j in range(nb):
            spl = BSpline(aknots, np.eye(nb)[j], 3, extrapolate=True)
            out[:, j] = (spl.derivative(deriv) if deriv else spl)(points)
        return out

    basis = design(x, 0)
    const = design(bk, 2)
    if not intercept:
        basis, const = basis[:, 1:], const[:, 1:]
    q, _ = np.linalg.qr(const.T, mode='complete')
    # Some BLAS builds raise spurious floating-point flags in matmul
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return (q.T @ basis.T).T[:, 2:]


def disp_cox_reid_spline_trend(y, design, offset=None, df=5, subset=10000,
                                ave_log_cpm_vals=None, method_optim='Nelder-Mead'):
    """Estimate spline trend dispersion.

    Faithful port of edgeR's dispCoxReidSplineTrend.
    Fits: dispersion = exp(X @ par - abundance) where X is a natural spline
    basis, optimized via Nelder-Mead on adjusted profile likelihood.

    Returns
    -------
    dict with 'dispersion' and 'AveLogCPM'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if offset is None:
        offset = np.zeros(nlibs)
    offset = expand_as_matrix(np.asarray(offset, dtype=np.float64), y.shape)

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset)

    all_zero = y.sum(axis=1) == 0
    abundance_nonzero = ave_log_cpm_vals[~all_zero]
    y_nonzero = y[~all_zero]
    offset_nonzero = offset[~all_zero]

    i = systematic_subset(subset, abundance_nonzero)

    if len(abundance_nonzero) < 2:
        common_disp = disp_cox_reid(y_nonzero, design, offset=offset_nonzero)
        disp = np.full(ntags, common_disp)
        return {'dispersion': disp, 'AveLogCPM': ave_log_cpm_vals}

    # Knot placement matching R: weighted mix of quantile and equally-spaced
    p1 = np.arange(1, df) / df
    knots1 = np.quantile(abundance_nonzero, p1)
    r = np.array([np.min(abundance_nonzero), np.max(abundance_nonzero)])
    knots2 = r[0] + p1 * (r[1] - r[0])
    knots = 0.3 * knots1 + 0.7 * knots2

    # Natural spline basis exactly as R's cbind(1, ns(abundance, knots=knots)):
    # Nelder-Mead's path depends on the basis, not just its span
    X = np.column_stack([np.ones(len(abundance_nonzero)), _ns_r(abundance_nonzero, knots, r)])

    # Objective: negative sum of adjusted profile likelihoods
    def fun(par, y_sub, design, offset_sub, abundance_sub, X_sub):
        eta = X_sub @ par
        dispersion = np.exp(eta - abundance_sub)
        try:
            apl = adjusted_profile_lik(dispersion, y_sub, design, offset_sub)
            return -np.sum(apl)
        except Exception:
            return 1e10

    # Initial parameters matching R
    par0 = np.zeros(df + 1)
    par0[0] = np.median(abundance_nonzero[i]) + np.log(0.1)

    par = _optim(lambda p: fun(p, y_nonzero[i], design, offset_nonzero[i],
                               abundance_nonzero[i], X[i]), par0, method_optim)

    # Evaluate fitted dispersions for all genes
    disp_nonzero = np.exp(X @ par - abundance_nonzero)

    disp = np.full(ntags, np.nan)
    disp[all_zero] = disp_nonzero[np.argmin(abundance_nonzero)] if len(disp_nonzero) > 0 else 0.1
    disp[~all_zero] = disp_nonzero

    return {'dispersion': disp, 'AveLogCPM': ave_log_cpm_vals}


def disp_cox_reid_power_trend(y, design, offset=None, ave_log_cpm_vals=None,
                               subset=10000, method_optim='Nelder-Mead'):
    """Estimate power trend dispersion.

    Faithful port of edgeR's dispCoxReidPowerTrend.
    Fits the parametric model: dispersion = exp(a + b*AveLogCPM) + exp(c)
    by maximizing the Cox-Reid adjusted profile likelihood via Nelder-Mead.

    Returns
    -------
    dict with 'dispersion' and 'AveLogCPM'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags = y.shape[0]

    if offset is None:
        offset = 0.0
    offset = expand_as_matrix(np.asarray(offset, dtype=np.float64), y.shape)

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset)

    abundance_full = ave_log_cpm_vals

    # Exclude all-zero rows
    all_zero = y.sum(axis=1) == 0
    abundance_nonzero = abundance_full[~all_zero]
    y_nonzero = y[~all_zero]
    offset_nonzero = offset[~all_zero]

    # Systematic subset for efficiency
    i = systematic_subset(subset, abundance_nonzero)

    # Objective: negative sum of adjusted profile likelihoods
    def fun(par, y_sub, design, offset_sub, abundance_sub):
        dispersion = np.exp(par[0] + par[1] * abundance_sub) + np.exp(par[2])
        try:
            apl = adjusted_profile_lik(dispersion, y_sub, design, offset_sub)
            return -np.sum(apl)
        except Exception:
            return 1e10

    par0 = np.array([np.log(0.1), 0.0, -5.0])
    par = _optim(lambda p: fun(p, y_nonzero[i], design, offset_nonzero[i],
                               abundance_nonzero[i]), par0, method_optim)

    # Compute dispersion for all genes using fitted parameters
    dispersion = np.exp(par[0] + par[1] * abundance_full) + np.exp(par[2])

    return {'dispersion': dispersion, 'AveLogCPM': abundance_full}


def _approx_rule2(x, y, xout):
    """R's approxfun(x, y, rule = 2, ties = mean)(xout)."""
    xu, inv = np.unique(np.asarray(x, dtype=np.float64), return_inverse=True)
    yu = np.bincount(inv, weights=np.asarray(y, dtype=np.float64)) / np.bincount(inv)
    return np.interp(xout, xu, yu)


def _trend_from_bins(bin_a, bin_d, x, method, df=5, span=0.3):
    """Dispersion trend over x from binned estimates, as edgeR's binned
    trends compute it: sqrt(dispersion) is fitted by a natural spline (as
    ns(bin.A, knots, intercept = TRUE) with lm.fit, extended linearly
    beyond the bins as predict() does) or by loessFit, then squared.
    """
    root = np.sqrt(bin_d)
    if method == 'spline':
        p1 = np.arange(1, df) / df
        r = np.array([np.min(bin_a), np.max(bin_a)])
        knots = 0.3 * np.quantile(bin_a, p1) + 0.7 * (r[0] + p1 * (r[1] - r[0]))
        beta = np.linalg.lstsq(_ns_basis_with_knots(bin_a, knots, boundary_knots=r), root,
                               rcond=None)[0]
        return (_ns_basis_with_knots(x, knots, boundary_knots=r) @ beta) ** 2
    from .limma_port import _loess_fit
    fitted = _loess_fit(root, bin_a, span=span, iterations=1)
    return _approx_rule2(bin_a, fitted, x) ** 2


def disp_bin_trend(y, design=None, offset=None, df=5, span=0.3,
                   min_n=400, method_bin='CoxReid', method_trend='spline',
                   ave_log_cpm_vals=None, weights=None):
    """Estimate dispersion trend by binning.

    Port of edgeR's dispBinTrend.

    Returns
    -------
    dict with 'dispersion', 'AveLogCPM', 'bin.AveLogCPM', 'bin.dispersion'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    pos = y.sum(axis=1) > 0
    if not np.any(pos):
        return {'AveLogCPM': ave_log_cpm_vals,
                'dispersion': np.zeros(ntags)}
    npostags = np.sum(pos)

    if design is None:
        design = np.ones((nlibs, 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

    if offset is None:
        offset = np.log(y.sum(axis=0))
    offset = expand_as_matrix(np.asarray(offset, dtype=np.float64), y.shape)

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset, weights=weights)

    # Define bins
    if npostags < 100:
        nbins = 1
    else:
        nbins = int(np.floor(npostags ** 0.4))
        nbins = min(nbins, 1000)
        min_n = min(min_n, npostags // nbins)
    if min_n < 50:
        nbins = npostags // 50
        min_n = 50

    nbins = max(nbins, 1)

    if nbins == 1:
        d = disp_cox_reid(y[pos], design, offset=offset[pos],
                          weights=weights[pos] if weights is not None and np.ndim(weights) == 2 else weights,
                          min_row_sum=0, ave_log_cpm_vals=ave_log_cpm_vals[pos])
        return {'AveLogCPM': ave_log_cpm_vals,
                'dispersion': np.full(ntags, d),
                'bin.AveLogCPM': np.array([np.mean(ave_log_cpm_vals[pos])]),
                'bin.dispersion': np.array([d])}

    groups = np.zeros(ntags, dtype=int)
    bins_info = cut_with_min_n(ave_log_cpm_vals[pos], intervals=nbins, min_n=min_n)
    groups[pos] = bins_info['group']

    bin_d = np.zeros(nbins)
    bin_a = np.zeros(nbins)
    for i in range(1, nbins + 1):
        bin_mask = groups == i
        if np.sum(bin_mask) == 0:
            continue
        bin_ave = ave_log_cpm_vals[bin_mask]
        w_bin = None
        if weights is not None and np.ndim(weights) == 2:
            w_bin = weights[bin_mask]
        try:
            bin_d[i - 1] = disp_cox_reid(y[bin_mask], design, offset=offset[bin_mask],
                                          weights=w_bin, min_row_sum=0,
                                          ave_log_cpm_vals=bin_ave)
        except Exception:
            bin_d[i - 1] = 0.1
        bin_a[i - 1] = np.mean(bin_ave)

    # Few bins: interpolate sqrt(dispersion) linearly, held constant beyond
    if nbins < 7:
        dispersion = _approx_rule2(bin_a, np.sqrt(bin_d), ave_log_cpm_vals) ** 2
    else:
        dispersion = _trend_from_bins(bin_a, bin_d, ave_log_cpm_vals, method_trend,
                                      df=df, span=span)
    return {'AveLogCPM': ave_log_cpm_vals, 'dispersion': dispersion,
            'bin.AveLogCPM': bin_a, 'bin.dispersion': bin_d}


def _filter_and_subset(y, offset, ave_log_cpm_vals, min_row_sum, subset):
    """Drop rows with small sums, then take edgeR's systematic subset.

    As in dispPearson; AveLogCPM is filtered with the counts. (edgeR's
    dispDeviance subsets with the unfiltered AveLogCPM, a bug.)
    """
    keep = y.sum(axis=1) >= min_row_sum
    y, offset = y[keep], offset[keep]
    if ave_log_cpm_vals is not None:
        ave_log_cpm_vals = np.asarray(ave_log_cpm_vals)[keep]
    if y.shape[0] < 1:
        raise ValueError("no data rows with required number of counts")
    if subset is not None and subset <= y.shape[0] / 2:
        if ave_log_cpm_vals is None:
            ave_log_cpm_vals = ave_log_cpm(y, offset=offset)
        i = systematic_subset(subset, ave_log_cpm_vals)
        y, offset = y[i], offset[i]
    return y, offset


def disp_pearson(y, design=None, offset=None, subset=10000, ave_log_cpm_vals=None,
                 min_row_sum=5, tol=1e-6, initial_dispersion=0.1):
    """Pearson estimator of the common dispersion.

    Port of edgeR's dispPearson: fit at initial_dispersion, then solve
    mean((y - mu)^2 / (mu (1 + phi mu))) = df.residual / nlibs over the
    cells with mu > 0 by Newton's method from phi = 0.
    """
    from .glm_fit import glm_fit
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    design = np.ones((y.shape[1], 1)) if design is None else np.asarray(design, dtype=np.float64)
    offset = expand_as_matrix(np.asarray(0.0 if offset is None else offset, dtype=np.float64), y.shape)
    y, offset = _filter_and_subset(y, offset, ave_log_cpm_vals, min_row_sum, subset)

    mu = glm_fit(y, design=design, dispersion=initial_dispersion, offset=offset,
                 prior_count=0)['fitted.values']
    one = (y.shape[1] - design.shape[1]) / y.shape[1]
    pos = mu > 0
    y, mu = y.T[pos.T], mu.T[pos.T]
    s2 = (y - mu) ** 2
    phi = 0.0
    for it in range(1, 102):
        Q = np.mean(s2 / mu / (1 + phi * mu))
        dQ = np.mean(s2 / (1 + phi * mu) ** 2)
        dif = (Q - one) / dQ
        if dif < 0:
            break
        phi += dif
        if dif < tol:
            break
        if it > 100:
            warnings.warn("iteration limit reached")
            break
    return phi


def disp_deviance(y, design=None, offset=None, subset=10000, ave_log_cpm_vals=None,
                  interval=(0, 4), tol=1e-5, min_row_sum=5, robust=False):
    """Deviance estimator of the common dispersion.

    Port of edgeR's dispDeviance: the dispersion at which the mean (or with
    robust, a quantile) of the genewise deviances equals the residual df,
    found by R's uniroot() over dispersion^0.25. As in edgeR, the bounds are
    first checked at par = interval[0] and interval[1], i.e. at dispersions
    interval^4.
    """
    from .glm_fit import glm_fit
    from .utils import uniroot_brent
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    design = np.ones((y.shape[1], 1)) if design is None else np.asarray(design, dtype=np.float64)
    offset = expand_as_matrix(np.asarray(0.0 if offset is None else offset, dtype=np.float64), y.shape)
    if min(interval) < 0:
        raise ValueError("please give a non-negative interval for the dispersion")
    y, offset = _filter_and_subset(y, offset, ave_log_cpm_vals, min_row_sum, subset)

    df = y.shape[1] - design.shape[1]
    if robust:
        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.cdf(df, df)
        bias = lambda dev: np.quantile(dev, p) - df
    else:
        bias = lambda dev: np.mean(dev) - df

    def fun(par):
        dev = glm_fit(y, design=design, dispersion=par ** 4, offset=offset,
                      prior_count=0)['deviance']
        return bias(dev)

    if fun(interval[0]) <= 0:
        return interval[0]
    if fun(interval[1]) > 0:
        warnings.warn("dispersion estimate above interval upper limit")
        return interval[1]
    root = uniroot_brent(fun, interval[0] ** 0.25, interval[1] ** 0.25, tol=tol)
    return root ** 4
