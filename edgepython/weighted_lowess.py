# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Port of limma's weightedLowess C code.

Weighted local regression with delta-based binning to approximately npts
seed points. Matches the behavior of limma's weighted_lowess() C function
in src/weighted_lowess.c.

The algorithm:
1. Sort data by x
2. Select ~npts seed points spaced at least delta apart
3. For each seed, find the span window where cumulative weight >= span * total_weight
4. Fit local weighted linear regression (tricube kernel) at each seed
5. Linearly interpolate between seeds
6. Optionally iterate with bisquare robustness weights
"""

import numpy as np
from numba import njit

_THRESHOLD = 1e-7


def weighted_lowess(x, y, weights=None, span=0.3, iterations=4, npts=200, delta=None):
    """Weighted lowess smoothing matching limma's C implementation.

    Parameters
    ----------
    x : array-like
        Covariate values.
    y : array-like
        Response values.
    weights : array-like, optional
        Prior weights (default: all ones).
    span : float
        Proportion of total weight to use in each local regression window.
    iterations : int
        Total number of fitting passes (1 = no robustness iterations).
    npts : int
        Approximate number of seed points for binning.
    delta : float, optional
        Minimum distance between seed points. Computed from npts if None.

    Returns
    -------
    dict with keys 'fitted', 'residuals', 'weights' (robustness weights), 'delta'.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).copy()

    if n < 2:
        raise ValueError("Need at least two points")

    # Sort by x (mergesort for stable ordering matching R's order())
    o = np.argsort(x, kind='mergesort')
    xs = x[o].copy()
    ys = y[o].copy()
    ws = weights[o].copy()

    # Compute delta if not provided (matching R wrapper logic)
    if delta is None:
        npts = int(npts + 0.5)
        if npts >= n:
            delta = 0.0
        else:
            dx = np.sort(np.diff(xs))
            cumrange = np.cumsum(dx)
            numclusters = np.arange(npts)
            # R 1-based to Python 0-based index conversion
            indices = len(dx) - 1 - numclusters
            delta = float(np.min(cumrange[indices] / (npts - numclusters)))

    delta = float(delta)

    # Compute total weight and span weight
    total_weight = np.sum(ws)
    span_weight = total_weight * span
    subrange = (xs[-1] - xs[0]) / n

    # Find seed points (binned to ~npts)
    seed_idx, nseeds = _find_seeds(xs, n, delta)

    # Find span limits for each seed
    frame_start, frame_end, max_dist = _find_limits(
        seed_idx, nseeds, xs, ws, n, span_weight)

    # Initialize fitted values and robustness weights
    fitted = np.zeros(n, dtype=np.float64)
    rob_w = np.ones(n, dtype=np.float64)

    # Run iterations in compiled code
    _lowess_iterations(xs, ys, ws, fitted, rob_w, seed_idx, nseeds,
                       frame_start, frame_end, max_dist, total_weight,
                       subrange, iterations)

    # Map back to original (unsorted) order
    fitted_orig = np.empty(n, dtype=np.float64)
    fitted_orig[o] = fitted
    rob_orig = np.empty(n, dtype=np.float64)
    rob_orig[o] = rob_w

    return {
        'fitted': fitted_orig,
        'residuals': y - fitted_orig,
        'weights': rob_orig,
        'delta': delta
    }


def _find_seeds(xs, n, delta):
    """Find seed point indices for delta-based binning.

    Always includes first and last points. Interior points are included
    if they are more than delta away from the last included point.
    """
    if delta <= 0 or n <= 2:
        return np.arange(n, dtype=np.intp), n

    seeds = [0]
    last_pt = 0
    for pt in range(1, n - 1):
        if xs[pt] - xs[last_pt] > delta:
            seeds.append(pt)
            last_pt = pt
    seeds.append(n - 1)

    return np.array(seeds, dtype=np.intp), len(seeds)


@njit(cache=True)
def _find_limits(seed_idx, nseeds, xs, ws, n, span_weight):
    """Find span window [start, end] and max distance for each seed.

    For each seed point, extends the window left and right (choosing
    the closer direction each step) until the cumulative weight in the
    window reaches span_weight. Then extends to include ties.
    """
    frame_start = np.empty(nseeds, dtype=np.intp)
    frame_end = np.empty(nseeds, dtype=np.intp)
    max_dist = np.empty(nseeds, dtype=np.float64)

    for s in range(nseeds):
        curpt = seed_idx[s]
        left = curpt
        right = curpt
        cur_w = ws[curpt]
        at_start = (left == 0)
        at_end = (right == n - 1)
        mdist = 0.0

        while cur_w < span_weight and (not at_end or not at_start):
            if at_end:
                # Can only extend left
                left -= 1
                cur_w += ws[left]
                if left == 0:
                    at_start = True
                ldist = xs[curpt] - xs[left]
                if mdist < ldist:
                    mdist = ldist
            elif at_start:
                # Can only extend right
                right += 1
                cur_w += ws[right]
                if right == n - 1:
                    at_end = True
                rdist = xs[right] - xs[curpt]
                if mdist < rdist:
                    mdist = rdist
            else:
                # Extend in direction of closer point
                ldist = xs[curpt] - xs[left - 1]
                rdist = xs[right + 1] - xs[curpt]
                if ldist < rdist:
                    left -= 1
                    cur_w += ws[left]
                    if left == 0:
                        at_start = True
                    if mdist < ldist:
                        mdist = ldist
                else:
                    right += 1
                    cur_w += ws[right]
                    if right == n - 1:
                        at_end = True
                    if mdist < rdist:
                        mdist = rdist

        # Extend to ties
        while left > 0 and xs[left] == xs[left - 1]:
            left -= 1
        while right < n - 1 and xs[right] == xs[right + 1]:
            right += 1

        frame_start[s] = left
        frame_end[s] = right
        max_dist[s] = mdist

    return frame_start, frame_end, max_dist


@njit(cache=True)
def _lowess_fit(xs, ys, ws, rw, curpt, left, right, dist):
    """Local weighted linear regression at a single point."""
    threshold = 1e-7
    allweight = 0.0
    xmean = 0.0
    ymean = 0.0

    if dist < threshold:
        for i in range(left, right + 1):
            w = ws[i] * rw[i]
            allweight += w
        if allweight == 0.0:
            return 0.0
        val = 0.0
        for i in range(left, right + 1):
            val += ys[i] * ws[i] * rw[i]
        return val / allweight

    for i in range(left, right + 1):
        u = abs(xs[curpt] - xs[i]) / dist
        tricube = (1.0 - u * u * u)
        tricube = tricube * tricube * tricube
        w = tricube * ws[i] * rw[i]
        allweight += w
        xmean += w * xs[i]
        ymean += w * ys[i]

    if allweight == 0.0:
        return 0.0

    xmean /= allweight
    ymean /= allweight

    var = 0.0
    covar = 0.0
    for i in range(left, right + 1):
        u = abs(xs[curpt] - xs[i]) / dist
        tricube = (1.0 - u * u * u)
        tricube = tricube * tricube * tricube
        w = tricube * ws[i] * rw[i]
        temp = xs[i] - xmean
        var += temp * temp * w
        covar += temp * (ys[i] - ymean) * w

    if var < threshold:
        return ymean

    slope = covar / var
    return slope * xs[curpt] + ymean - slope * xmean


@njit(cache=True)
def _lowess_iterations(xs, ys, ws, fitted, rob_w, seed_idx, nseeds,
                       frame_start, frame_end, max_dist, total_weight,
                       subrange, iterations):
    """Run the full lowess iteration loop in compiled code."""
    n = len(xs)
    threshold = 1e-7

    for _it in range(iterations):
        fitted[0] = _lowess_fit(xs, ys, ws, rob_w,
                                0, frame_start[0], frame_end[0], max_dist[0])
        last_pt = 0
        for s in range(1, nseeds):
            pt = seed_idx[s]
            fitted[pt] = _lowess_fit(xs, ys, ws, rob_w,
                                     pt, frame_start[s], frame_end[s],
                                     max_dist[s])
            if pt - last_pt > 1:
                dx_interp = xs[pt] - xs[last_pt]
                if dx_interp > threshold * subrange:
                    slope = (fitted[pt] - fitted[last_pt]) / dx_interp
                    intercept = fitted[pt] - slope * xs[pt]
                    for j in range(last_pt + 1, pt):
                        fitted[j] = slope * xs[j] + intercept
                else:
                    avg = 0.5 * (fitted[pt] + fitted[last_pt])
                    for j in range(last_pt + 1, pt):
                        fitted[j] = avg
            last_pt = pt

        # Compute absolute residuals
        abs_resid = np.empty(n)
        resid_sum = 0.0
        for i in range(n):
            abs_resid[i] = abs(ys[i] - fitted[i])
            resid_sum += abs_resid[i]
        resid_scale = resid_sum / n

        # Sort residuals
        ror = np.argsort(abs_resid)
        sorted_resid = abs_resid[ror]

        cumw = 0.0
        half_weight = total_weight / 2.0
        cmad = 0.0
        for i in range(n):
            cumw += ws[ror[i]]
            if cumw == half_weight and i < n - 1:
                cmad = 3.0 * (sorted_resid[i] + sorted_resid[i + 1])
                break
            elif cumw > half_weight:
                cmad = 6.0 * sorted_resid[i]
                break

        if cmad <= threshold * resid_scale:
            break

        for i in range(n):
            rob_w[i] = 0.0
        for i in range(n):
            if sorted_resid[i] < cmad:
                u = sorted_resid[i] / cmad
                rob_w[ror[i]] = (1.0 - u * u) * (1.0 - u * u)
            else:
                break
