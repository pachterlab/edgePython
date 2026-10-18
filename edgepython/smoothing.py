# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Smoothing functions for edgePython.

Port of edgeR's locfitByCol and loessByCol.
"""

import numpy as np
from numba import njit, prange


def _locfit_tree(x, y, w, span, degree=0, cut=0.8):
    """Fitted values of locfit(y ~ x, weights = w, alpha = span, deg = degree).

    Port of locfit's default evaluation structure (ev = "tree"; locfit
    1.5-9.12, ev_atree.c, ev_interp.c, lf_nbhd.c). locfit does not fit at
    every point. It fits at the vertices of an adaptive tree over the range
    of x, splitting a cell at its midpoint while the cell is wider than
    `cut` times the smaller bandwidth of its two vertices. With degree 0 it
    interpolates linearly between the two vertices of each point's cell;
    with degree 1 each vertex also has a slope and the interpolation is
    cubic Hermite (rectcell_interp). The columns of y share the tree, which
    depends only on x, w and span.
    """
    n = x.shape[0]
    k = int(n * span + 1e-12)
    max_vertices = int((5 / min(span, 1.0) / cut + 1) * 2)

    def bandwidth(v):
        dist = np.sort(np.abs(x - v))
        if k > n:
            return dist[-1] * span
        return dist[max(k, 1) - 1]

    vx = [x.min(), x.max()]
    vh = [bandwidth(vx[0]), bandwidth(vx[1])]
    # Cells as (left vertex, right vertex, left child, right child)
    cells = [[0, 1, -1, -1]]

    def split(c):
        lo, hi = cells[c][0], cells[c][1]
        h = min(vh[lo], vh[hi])
        if h <= 0 or not cut * h < vx[hi] - vx[lo] or len(vx) >= max_vertices:
            return
        mid = len(vx)
        vx.append((vx[lo] + vx[hi]) / 2)
        vh.append(bandwidth(vx[mid]))
        cells[c][2], cells[c][3] = len(cells), len(cells) + 1
        cells.append([lo, mid, -1, -1])
        cells.append([mid, hi, -1, -1])
        split(cells[c][2])
        split(cells[c][3])

    split(0)
    vx = np.array(vx)
    vh = np.array(vh)
    cells = np.array(cells)

    # Tricube-weighted local mean (degree 0) or local line (degree 1) at
    # each vertex
    dx = x[None, :] - vx[:, None]
    u = np.abs(dx) / np.where(vh > 0, vh, np.inf)[:, None]
    kern = np.where(u < 1, (1 - u ** 3) ** 3, 0.0) * w[None, :]
    kern[vh <= 0] = np.abs(x[None, :] - vx[vh <= 0, None]) == 0
    sw = kern.sum(axis=1)
    # Some BLAS builds (e.g. Apple Accelerate) raise spurious floating-point
    # flags in matmul on finite input
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        t0 = kern @ y
        t1 = (kern * dx) @ y if degree == 1 else None
    fit = t0 / np.where(sw > 0, sw, 1.0)[:, None]
    fit[sw <= 0] = 0.0
    if degree == 1:
        s1 = (kern * dx).sum(axis=1)
        s2 = (kern * dx * dx).sum(axis=1)
        det = sw * s2 - s1 * s1
        ok = det > 1e-12 * np.maximum(sw * s2, 1e-300)
        safe_det = np.where(ok, det, 1.0)[:, None]
        slope = np.where(ok[:, None], (sw[:, None] * t1 - s1[:, None] * t0) / safe_det, 0.0)
        fit = np.where(ok[:, None], (s2[:, None] * t0 - s1[:, None] * t1) / safe_det, fit)

    # Descend to each point's cell, then interpolate between its vertices
    cell = np.zeros(n, dtype=np.int64)
    while True:
        inner = cells[cell, 2] >= 0
        if not inner.any():
            break
        lo = vx[cells[cell, 0]]
        width = vx[cells[cell, 1]] - lo
        right = ~(2 * (x - lo) < width)
        child = np.where(right, cells[cell, 3], cells[cell, 2])
        cell = np.where(inner, child, cell)
    lo, hi = cells[cell, 0], cells[cell, 1]
    width = vx[hi] - vx[lo]
    step = x - vx[lo]
    safe = np.where(width > 0, width, 1.0)
    if degree == 0:
        out = ((width - step)[:, None] * fit[lo] + step[:, None] * fit[hi]) / safe[:, None]
    else:
        # hermite2(): cubic in t = step / width, slopes scaled by the width
        t = np.clip(step / safe, 0.0, 1.0)[:, None]
        phi1 = t * t * (3 - 2 * t)
        out = ((1 - phi1) * fit[lo] + phi1 * fit[hi]
               + width[:, None] * (t * (1 - t) ** 2 * slope[lo] + t * t * (t - 1) * slope[hi]))
    out[width <= 0] = fit[lo[width <= 0]]
    return out


def locfit_by_col(y, x=None, weights=1, span=0.5, degree=0):
    """Local regression smoother for columns of a matrix.

    Port of edgeR's locfitByCol, following locfit's adaptive-tree
    evaluation for degree 0 and degree 1.
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

    return _locfit_tree(x, y, weights, span, degree=degree)


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
