# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Smoothing functions for edgePython.

Port of edgeR's locfitByCol and loessByCol.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _locfit_degree0_point(i, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted):
    """Compute locfit degree=0 for a single point i."""
    lo = i
    hi = i + 1
    while hi - lo < nn:
        can_left = lo > 0
        can_right = hi < n
        if can_left and can_right:
            if x_sorted[i] - x_sorted[lo - 1] <= x_sorted[hi] - x_sorted[i]:
                lo -= 1
            else:
                hi += 1
        elif can_left:
            lo -= 1
        elif can_right:
            hi += 1
        else:
            break

    h = 0.0
    for k in range(lo, hi):
        d = abs(x_sorted[k] - x_sorted[i])
        if d > h:
            h = d
    h += 1e-10

    sw = 0.0
    for k in range(lo, hi):
        u = abs(x_sorted[k] - x_sorted[i]) / h
        if u >= 1.0:
            wk = 0.0
        else:
            t = 1.0 - u * u * u
            wk = t * t * t
        wk *= w_sorted[k]
        sw += wk
        if wk > 0.0:
            for j in range(ncols):
                result_sorted[i, j] += wk * y_sorted[k, j]

    if sw > 0.0:
        for j in range(ncols):
            result_sorted[i, j] /= sw
    else:
        for j in range(ncols):
            result_sorted[i, j] = y_sorted[i, j]


@njit(cache=True, parallel=True)
def _locfit_degree0_kernel(x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted):
    """Numba kernel for locfit degree=0 (Nadaraya-Watson) with nearest-neighbor bandwidth."""
    for i in prange(n):
        idx = np.int64(i)
        _locfit_degree0_point(idx, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted)


@njit(cache=True)
def _locfit_degree0_grid_point(i, x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid):
    """Compute locfit degree=0 at eval point x_eval[i] using sorted data."""
    xv = x_eval[i]
    # Binary search to find insertion point in x_sorted
    lo_bs = 0
    hi_bs = n
    while lo_bs < hi_bs:
        mid = (lo_bs + hi_bs) // 2
        if x_sorted[mid] < xv:
            lo_bs = mid + 1
        else:
            hi_bs = mid
    # lo_bs is the insertion point; expand window to nn neighbors
    lo = lo_bs
    hi = lo_bs
    while hi - lo < nn:
        can_left = lo > 0
        can_right = hi < n
        if can_left and can_right:
            if xv - x_sorted[lo - 1] <= x_sorted[hi] - xv:
                lo -= 1
            else:
                hi += 1
        elif can_left:
            lo -= 1
        elif can_right:
            hi += 1
        else:
            break

    h = 0.0
    for k in range(lo, hi):
        d = abs(x_sorted[k] - xv)
        if d > h:
            h = d
    h += 1e-10

    sw = 0.0
    for k in range(lo, hi):
        u = abs(x_sorted[k] - xv) / h
        if u >= 1.0:
            wk = 0.0
        else:
            t = 1.0 - u * u * u
            wk = t * t * t
        wk *= w_sorted[k]
        sw += wk
        if wk > 0.0:
            for j in range(ncols):
                result_grid[i, j] += wk * y_sorted[k, j]

    if sw > 0.0:
        for j in range(ncols):
            result_grid[i, j] /= sw


@njit(cache=True, parallel=True)
def _locfit_degree0_grid_kernel(x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid):
    """Evaluate locfit degree=0 at M grid points using sorted data."""
    m = len(x_eval)
    for i in prange(m):
        idx = np.int64(i)
        _locfit_degree0_grid_point(idx, x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid)


@njit(cache=True)
def _locfit_degree1_point(i, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted):
    """Compute locfit degree=1 for a single point i."""
    lo = i
    hi = i + 1
    while hi - lo < nn:
        can_left = lo > 0
        can_right = hi < n
        if can_left and can_right:
            if x_sorted[i] - x_sorted[lo - 1] <= x_sorted[hi] - x_sorted[i]:
                lo -= 1
            else:
                hi += 1
        elif can_left:
            lo -= 1
        elif can_right:
            hi += 1
        else:
            break

    h = 0.0
    for k in range(lo, hi):
        d = abs(x_sorted[k] - x_sorted[i])
        if d > h:
            h = d
    h += 1e-10

    sum_w = 0.0
    sum_w_dx = 0.0
    sum_w_dx2 = 0.0

    for k in range(lo, hi):
        u = abs(x_sorted[k] - x_sorted[i]) / h
        if u >= 1.0:
            wk = 0.0
        else:
            t = 1.0 - u * u * u
            wk = t * t * t
        wk *= w_sorted[k]
        dx = x_sorted[k] - x_sorted[i]
        sum_w += wk
        sum_w_dx += wk * dx
        sum_w_dx2 += wk * dx * dx

    det = sum_w * sum_w_dx2 - sum_w_dx * sum_w_dx
    if abs(det) < 1e-300:
        for j in range(ncols):
            result_sorted[i, j] = y_sorted[i, j]
    else:
        for j in range(ncols):
            rhs0 = 0.0
            rhs1 = 0.0
            for k in range(lo, hi):
                u = abs(x_sorted[k] - x_sorted[i]) / h
                if u >= 1.0:
                    wk = 0.0
                else:
                    t2 = 1.0 - u * u * u
                    wk = t2 * t2 * t2
                wk *= w_sorted[k]
                dx = x_sorted[k] - x_sorted[i]
                rhs0 += wk * y_sorted[k, j]
                rhs1 += wk * dx * y_sorted[k, j]
            result_sorted[i, j] = (sum_w_dx2 * rhs0 - sum_w_dx * rhs1) / det


@njit(cache=True, parallel=True)
def _locfit_degree1_kernel(x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted):
    """Numba kernel for locfit degree=1 (local linear regression) with nearest-neighbor bandwidth."""
    for i in prange(n):
        idx = np.int64(i)
        _locfit_degree1_point(idx, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted)


@njit(cache=True)
def _locfit_degree1_grid_point(i, x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid):
    """Compute locfit degree=1 at eval point x_eval[i] using sorted data."""
    xv = x_eval[i]
    # Binary search to find insertion point in x_sorted
    lo_bs = 0
    hi_bs = n
    while lo_bs < hi_bs:
        mid = (lo_bs + hi_bs) // 2
        if x_sorted[mid] < xv:
            lo_bs = mid + 1
        else:
            hi_bs = mid
    lo = lo_bs
    hi = lo_bs
    while hi - lo < nn:
        can_left = lo > 0
        can_right = hi < n
        if can_left and can_right:
            if xv - x_sorted[lo - 1] <= x_sorted[hi] - xv:
                lo -= 1
            else:
                hi += 1
        elif can_left:
            lo -= 1
        elif can_right:
            hi += 1
        else:
            break

    h = 0.0
    for k in range(lo, hi):
        d = abs(x_sorted[k] - xv)
        if d > h:
            h = d
    h += 1e-10

    sum_w = 0.0
    sum_w_dx = 0.0
    sum_w_dx2 = 0.0

    for k in range(lo, hi):
        u = abs(x_sorted[k] - xv) / h
        if u >= 1.0:
            wk = 0.0
        else:
            t = 1.0 - u * u * u
            wk = t * t * t
        wk *= w_sorted[k]
        dx = x_sorted[k] - xv
        sum_w += wk
        sum_w_dx += wk * dx
        sum_w_dx2 += wk * dx * dx

    det = sum_w * sum_w_dx2 - sum_w_dx * sum_w_dx
    if abs(det) < 1e-300:
        # Fallback: find nearest data point
        best_k = lo
        best_d = abs(x_sorted[lo] - xv)
        for k in range(lo + 1, hi):
            d = abs(x_sorted[k] - xv)
            if d < best_d:
                best_d = d
                best_k = k
        for j in range(ncols):
            result_grid[i, j] = y_sorted[best_k, j]
    else:
        for j in range(ncols):
            rhs0 = 0.0
            rhs1 = 0.0
            for k in range(lo, hi):
                u = abs(x_sorted[k] - xv) / h
                if u >= 1.0:
                    wk = 0.0
                else:
                    t2 = 1.0 - u * u * u
                    wk = t2 * t2 * t2
                wk *= w_sorted[k]
                dx = x_sorted[k] - xv
                rhs0 += wk * y_sorted[k, j]
                rhs1 += wk * dx * y_sorted[k, j]
            result_grid[i, j] = (sum_w_dx2 * rhs0 - sum_w_dx * rhs1) / det


@njit(cache=True, parallel=True)
def _locfit_degree1_grid_kernel(x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid):
    """Evaluate locfit degree=1 at M grid points using sorted data."""
    m = len(x_eval)
    for i in prange(m):
        idx = np.int64(i)
        _locfit_degree1_grid_point(idx, x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid)


def locfit_by_col(y, x=None, weights=1, span=0.5, degree=0):
    """Local regression smoother for columns of a matrix.

    Port of edgeR's locfitByCol. Uses a simple local weighted regression
    since Python doesn't have a direct locfit equivalent.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, ncols = y.shape

    weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), n).copy()
    if x is None:
        x = np.arange(1, n + 1, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if span * n < 2 or n <= 1:
        return y.copy()

    # Sort by x for efficient windowing
    order = np.argsort(x)
    x_sorted = x[order].copy()
    y_sorted = y[order].copy()
    w_sorted = weights[order].copy()

    nn = max(2, int(round(span * n)))

    # Adaptive grid path for large n: evaluate at M grid points, interpolate
    if n > 1000:
        M = 200
        x_eval = np.linspace(x_sorted[0], x_sorted[-1], M)
        result_grid = np.zeros((M, ncols), dtype=np.float64)
        if degree == 0:
            _locfit_degree0_grid_kernel(x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid)
        else:
            _locfit_degree1_grid_kernel(x_eval, x_sorted, y_sorted, w_sorted, n, ncols, nn, result_grid)
        # Interpolate grid results to all data points
        result_sorted = np.empty((n, ncols), dtype=np.float64)
        for j in range(ncols):
            result_sorted[:, j] = np.interp(x_sorted, x_eval, result_grid[:, j])
    else:
        result_sorted = np.zeros((n, ncols), dtype=np.float64)
        if degree == 0:
            _locfit_degree0_kernel(x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted)
        else:
            _locfit_degree1_kernel(x_sorted, y_sorted, w_sorted, n, ncols, nn, result_sorted)

    # Unsort
    result = np.empty_like(result_sorted)
    result[order] = result_sorted
    return result


@njit(cache=True)
def _loess_point(i, x, y, n, ncols, nspan, order, rank, fitted, leverages):
    """Compute loess for a single point i."""
    ri = rank[i]
    lo_cand = ri
    hi_cand = ri + 1
    while hi_cand - lo_cand < nspan:
        can_go_left = lo_cand > 0
        can_go_right = hi_cand < n
        if can_go_left and can_go_right:
            dl = abs(x[order[lo_cand - 1]] - x[i])
            dr = abs(x[order[hi_cand]] - x[i])
            if dl <= dr:
                lo_cand -= 1
            else:
                hi_cand += 1
        elif can_go_left:
            lo_cand -= 1
        elif can_go_right:
            hi_cand += 1
        else:
            break

    max_dist = 0.0
    for k in range(lo_cand, hi_cand):
        d = abs(x[order[k]] - x[i])
        if d > max_dist:
            max_dist = d
    max_dist += 1e-10

    sw = 0.0
    for k in range(lo_cand, hi_cand):
        idx = order[k]
        u = abs(x[idx] - x[i]) / max_dist
        if u >= 1.0:
            wk = 0.0
        else:
            t = 1.0 - u * u * u
            wk = t * t * t
        sw += wk

    self_w_norm = 0.0
    if sw > 0.0:
        for k in range(lo_cand, hi_cand):
            idx = order[k]
            u = abs(x[idx] - x[i]) / max_dist
            if u >= 1.0:
                wk = 0.0
            else:
                t = 1.0 - u * u * u
                wk = t * t * t
            w_norm = wk / sw
            for j in range(ncols):
                fitted[i, j] += w_norm * y[idx, j]
            if idx == i:
                self_w_norm = w_norm
        leverages[i] = self_w_norm
    else:
        for j in range(ncols):
            fitted[i, j] = y[i, j]
        leverages[i] = 1.0


@njit(cache=True, parallel=True)
def _loess_kernel(x, y, n, ncols, nspan, fitted, leverages):
    """Numba kernel for loess_by_col: degree-0 local regression with leverages."""
    order = np.argsort(x)
    rank = np.empty(n, dtype=np.int64)
    for k in range(n):
        rank[order[k]] = k

    for i in prange(n):
        idx = np.int64(i)
        _loess_point(idx, x, y, n, ncols, nspan, order, rank, fitted, leverages)


def loess_by_col(y, x=None, span=0.5):
    """Fit a lowess curve of degree 0 to each column of a matrix.

    Port of edgeR's loessByCol. Returns fitted values and leverages.

    Parameters
    ----------
    y : array-like
        Matrix of values.
    x : array-like, optional
        Covariate (defaults to 1:nrow).
    span : float
        Span for smoothing.

    Returns
    -------
    dict with 'fitted_values' and 'leverages'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]
    ncols = y.shape[1]

    if x is None:
        x = np.arange(1, n + 1, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).copy()

    nspan = min(int(span * n), n)
    if nspan <= 1:
        return {
            'fitted_values': y.copy(),
            'leverages': np.ones(n)
        }

    fitted = np.zeros((n, ncols), dtype=np.float64)
    leverages = np.zeros(n, dtype=np.float64)

    _loess_kernel(x, y, n, ncols, nspan, fitted, leverages)

    return {
        'fitted_values': fitted,
        'leverages': leverages
    }
