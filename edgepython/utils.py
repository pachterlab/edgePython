# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Utility functions for edgePython.

Port of edgeR utility functions: expandAsMatrix, addPriorCount, movingAverageByCol,
predFC, goodTuring, thinCounts, gini, cutWithMinN, sumTechReps, systematicSubset,
nearestReftoX, getPriorN, zscoreNBinom, binomTest, dropEmptyLevels, etc.
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from .compressed_matrix import CompressedMatrix, compress_offsets, compress_prior
import warnings


def expand_as_matrix(x, dim=None, byrow=True):
    """Convert scalar, row/column vector, or matrix to a full matrix.

    Port of edgeR's expandAsMatrix.
    """
    if dim is None:
        return np.atleast_2d(np.asarray(x, dtype=np.float64))
    dim = (int(dim[0]), int(dim[1]))

    if isinstance(x, CompressedMatrix):
        return expand_as_matrix(x.as_matrix(), dim=dim, byrow=byrow)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.size == 1:
        return np.full(dim, x.ravel()[0])
    if x.ndim <= 1:
        lx = len(x)
        if lx == dim[0] and lx == dim[1]:
            return np.tile(x.reshape(-1, 1) if not byrow else x.reshape(1, -1),
                           (1, dim[1]) if not byrow else (dim[0], 1))
        if lx == dim[1]:
            return np.tile(x.reshape(1, -1), (dim[0], 1))
        if lx == dim[0]:
            return np.tile(x.reshape(-1, 1), (1, dim[1]))
        raise ValueError("x of unexpected length")
    if x.ndim == 2:
        if x.shape == tuple(dim):
            return x.copy()
        raise ValueError("x is matrix of wrong size")
    raise ValueError("x has wrong dimensions")


def add_prior_count(y, lib_size=None, offset=None, prior_count=1):
    """Add library-size-adjusted prior counts.

    Port of edgeR's addPriorCount.

    Returns
    -------
    dict with 'y' (adjusted counts) and 'offset' (adjusted offsets).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    if offset is None:
        if lib_size is None:
            lib_size = y.sum(axis=0)
        offset = np.log(lib_size)
    offset = np.atleast_1d(np.asarray(offset, dtype=np.float64))

    prior_count = np.atleast_1d(np.asarray(prior_count, dtype=np.float64))

    # Expand offset and prior_count to matrix form
    if offset.ndim == 1:
        offset_mat = np.tile(offset, (y.shape[0], 1))
    else:
        offset_mat = offset.copy()

    if prior_count.ndim == 0 or prior_count.size == 1:
        prior_mat = np.full(y.shape, prior_count.ravel()[0])
    elif prior_count.ndim == 1 and len(prior_count) == y.shape[0]:
        prior_mat = np.tile(prior_count.reshape(-1, 1), (1, y.shape[1]))
    else:
        prior_mat = expand_as_matrix(prior_count, dim=y.shape, byrow=False)

    # Effective library sizes from offset
    lib_size_eff = np.exp(offset_mat)

    # Scale prior counts to be proportional to library size
    avg_lib = np.mean(lib_size_eff, axis=0) if lib_size_eff.ndim == 2 else np.mean(lib_size_eff)
    if lib_size_eff.ndim == 2:
        # prior is scaled by lib_size / mean_lib_size
        mean_lib = np.mean(lib_size_eff)
        scaled_prior = prior_mat * lib_size_eff / mean_lib
    else:
        scaled_prior = prior_mat

    y_aug = y + scaled_prior
    offset_aug = np.log(lib_size_eff + 2 * np.mean(scaled_prior, axis=0, keepdims=True) * np.mean(lib_size_eff) / lib_size_eff)

    # Simplified: match edgeR C code behavior
    # offset_aug = log(lib_size + 2*prior_count_scaled)
    if offset.ndim == 1:
        lib = np.exp(offset)
        pc = prior_count.ravel()[0] if prior_count.size == 1 else np.mean(prior_count)
        offset_aug = np.log(lib + 2.0 * pc * lib / np.mean(lib))
        scaled_prior_simple = prior_count.ravel()[0] * lib / np.mean(lib) if prior_count.size == 1 else prior_count.reshape(-1, 1) * lib / np.mean(lib)
        y_aug = y + (scaled_prior_simple if np.ndim(scaled_prior_simple) == 2 else np.tile(scaled_prior_simple, (y.shape[0], 1)))
        offset_aug_mat = np.tile(offset_aug, (y.shape[0], 1)) if offset_aug.ndim == 1 else offset_aug

    return {'y': y_aug, 'offset': offset_aug}


def moving_average_by_col(x, width=5, full_length=True):
    """Moving average smoother for columns of a matrix.

    Port of edgeR's movingAverageByCol.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    width = int(width)
    if width <= 1:
        return x
    n, m = x.shape
    if width > n:
        width = n

    if full_length:
        half1 = (width + 1) // 2
        half2 = width // 2
        x_pad = np.vstack([np.zeros((half1, m)), x, np.zeros((half2, m))])
    else:
        if width == n:
            return np.tile(x.mean(axis=0), (1, 1))
        x_pad = np.vstack([np.zeros((1, m)), x])

    cs = np.cumsum(x_pad, axis=0)
    n2 = cs.shape[0]
    result = cs[width:n2] - cs[:n2 - width]
    n3 = result.shape[0]

    w = np.full(n3, width, dtype=np.float64)
    if full_length:
        if half1 > 1:
            w[:half1 - 1] = width - np.arange(half1 - 1, 0, -1)
        w[n3 - half2:] = width - np.arange(1, half2 + 1)

    return result / w.reshape(-1, 1)


def pred_fc(y, design, prior_count=0.125, offset=None, dispersion=0, weights=None):
    """Predicted fold changes with shrinkage.

    Port of edgeR's predFC.
    """
    from .glm_fit import glm_fit

    out = add_prior_count(y, offset=offset, prior_count=prior_count)
    design = np.asarray(design, dtype=np.float64)
    g = glm_fit(out['y'], design, offset=out['offset'], dispersion=dispersion,
                prior_count=0, weights=weights)
    return g['coefficients'] / np.log(2)


def good_turing(x, conf=1.96):
    """Simple Good-Turing frequency estimation.

    Faithful port of edgeR's goodTuring (R wrapper + C code in good_turing.c).
    """
    x = np.asarray(x, dtype=int)

    # Tabulate frequencies — matches R's goodTuring R wrapper
    max_x = x.max()
    if max_x < len(x):
        # np.bincount(x): index i = count of value i in x
        bc = np.bincount(x)
        n0 = bc[0] if len(bc) > 0 else 0
        n = bc[1:]  # counts for values 1, 2, ..., max_x
        r = np.arange(1, len(n) + 1)
        mask = n > 0
        r = r[mask]
        n = n[mask]
    else:
        r_unique, counts = np.unique(x, return_counts=True)
        sort_idx = np.argsort(r_unique)
        r_unique = r_unique[sort_idx]
        counts = counts[sort_idx]
        if r_unique[0] == 0:
            n0 = counts[0]
            r = r_unique[1:]
            n = counts[1:]
        else:
            n0 = 0
            r = r_unique
            n = counts

    if len(r) == 0:
        return {'count': r, 'n': n, 'n0': n0, 'proportion': np.array([]),
                'P0': 0.0}

    r = r.astype(np.int64)
    n = n.astype(np.int64)
    nr = len(r)
    last = nr - 1

    # --- Port of good_turing.c ---
    # Compute bigN, Z values, and linear regression in one pass
    bigN = 0.0
    log_obs = np.log(r.astype(float))
    meanX = 0.0
    meanY = 0.0
    XYs = 0.0
    Xsquares = 0.0

    for i in range(nr):
        bigN += float(r[i]) * float(n[i])

        prev_obs = 0 if i == 0 else r[i - 1]
        logO = log_obs[i]

        xx = (2 * (r[i] - prev_obs)) if i == last else (r[i + 1] - prev_obs)
        logZ = np.log(2.0 * n[i]) - np.log(float(xx))

        meanX += logO
        meanY += logZ
        XYs += logO * logZ
        Xsquares += logO * logO

    meanX /= nr
    meanY /= nr
    XYs -= meanX * meanY * nr
    Xsquares -= meanX * meanX * nr

    slope = XYs / Xsquares if Xsquares != 0 else 0.0

    # P0: only nonzero if first observed count is 1
    P0 = 0.0 if (nr == 0 or r[0] != 1) else float(n[0]) / bigN

    # Compute r* values with indiffValsSeen logic
    out = np.zeros(nr)
    bigNprime = 0.0
    indiff_vals_seen = False

    for i in range(nr):
        next_obs = r[i] + 1
        # Turing estimate (intercept cancels out)
        y = float(next_obs) * np.exp(slope * (np.log(float(next_obs)) - log_obs[i]))

        if i == last or r[i + 1] != next_obs:
            indiff_vals_seen = True

        if not indiff_vals_seen:
            # Direct estimate
            x_direct = float(next_obs) * float(n[i + 1]) / float(n[i])
            if abs(x_direct - y) <= conf * x_direct * np.sqrt(
                    1.0 / float(n[i + 1]) + 1.0 / float(n[i])):
                indiff_vals_seen = True
            else:
                out[i] = x_direct

        if indiff_vals_seen:
            out[i] = y

        bigNprime += out[i] * float(n[i])

    # Normalize to proportions
    factor = (1.0 - P0) / bigNprime if bigNprime > 0 else 0.0
    proportion = out * factor

    return {
        'count': r,
        'n': n,
        'n0': n0,
        'proportion': proportion,
        'P0': P0
    }


def good_turing_proportions(counts):
    """Transform counts using Good-Turing proportions.

    Port of edgeR's goodTuringProportions.
    """
    counts = np.asarray(counts, dtype=int)
    z = counts.astype(float).copy()
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    nlibs = z.shape[1]
    for i in range(nlibs):
        g = good_turing(counts[:, i] if counts.ndim == 2 else counts)
        p0 = g['P0'] / g['n0'] if g['n0'] > 0 else 0
        zero = z[:, i] == 0
        z[zero, i] = p0
        nonzero = ~zero
        if np.any(nonzero):
            m = np.searchsorted(g['count'], z[nonzero, i].astype(int))
            m = np.clip(m, 0, len(g['proportion']) - 1)
            z[nonzero, i] = g['proportion'][m]
    return z


def thin_counts(x, prob=None, target_size=None):
    """Binomial or multinomial thinning of counts.

    Port of edgeR's thinCounts.
    """
    x = np.asarray(x, dtype=int).copy()
    if prob is not None:
        x = np.random.binomial(x, prob)
    else:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if target_size is None:
            target_size = x.sum(axis=0).min()
        target_size = np.atleast_1d(np.asarray(target_size, dtype=int))
        if len(target_size) == 1:
            target_size = np.full(x.shape[1], target_size[0])
        actual_size = x.sum(axis=0)
        if np.any(target_size > actual_size):
            raise ValueError("target_size bigger than actual size")
        for j in range(x.shape[1]):
            diff = actual_size[j] - target_size[j]
            if diff > 0:
                probs = x[:, j].astype(float)
                probs /= probs.sum()
                remove = np.random.multinomial(diff, probs)
                x[:, j] -= remove
        x = np.maximum(x, 0)
    return x


def gini(x):
    """Gini diversity index for columns of a matrix.

    Port of edgeR's gini.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    result = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        xs = np.sort(x[:, j])
        i = np.arange(1, n + 1)
        m = 0.75 * n
        s1 = np.sum((i - m) * xs)
        s2 = np.sum(xs)
        if s2 > 0:
            result[j] = (2 * (s1 / s2 + m) - n - 1) / n
    return result


def cut_with_min_n(x, intervals=2, min_n=1):
    """Cut numeric x into intervals with minimum count per bin.

    Port of edgeR's cutWithMinN.
    """
    x = np.asarray(x, dtype=np.float64)
    isna = np.isnan(x)
    if np.any(isna):
        group = np.full(len(x), np.nan)
        out = cut_with_min_n(x[~isna], intervals=intervals, min_n=min_n)
        group[~isna] = out['group']
        return {'group': group, 'breaks': out['breaks']}

    intervals = int(intervals)
    min_n = int(min_n)
    nx = len(x)

    if nx < intervals * min_n:
        raise ValueError("too few observations: length(x) < intervals*min_n")

    if intervals == 1:
        return {'group': np.ones(nx, dtype=int), 'breaks': None}

    # Add jitter
    x_jit = x + 1e-10 * (np.random.uniform(size=nx) - 0.5)

    # Try equally spaced
    breaks = np.linspace(x_jit.min() - 1, x_jit.max() + 1, intervals + 1)
    z = np.digitize(x_jit, breaks[1:-1])
    n = np.bincount(z, minlength=intervals)
    if np.all(n >= min_n):
        return {'group': z + 1, 'breaks': breaks}

    # Try quantile-based
    quantiles = np.quantile(x_jit, np.linspace(0, 1, intervals + 1))
    quantiles[0] -= 1
    quantiles[-1] += 1

    for w in np.linspace(0.1, 1.0, 10):
        brk = w * quantiles + (1 - w) * breaks
        z = np.digitize(x_jit, brk[1:-1])
        n = np.bincount(z, minlength=intervals)
        if np.all(n >= min_n):
            return {'group': z + 1, 'breaks': brk}

    # Fallback: order by x
    o = np.argsort(x_jit)
    n_per = nx // intervals
    nresid = nx - intervals * n_per
    sizes = np.full(intervals, n_per)
    if nresid > 0:
        sizes[:nresid] += 1
    z = np.zeros(nx, dtype=int)
    z[o] = np.repeat(np.arange(1, intervals + 1), sizes)
    return {'group': z, 'breaks': quantiles}


def sum_tech_reps(x, ID=None):
    """Sum over technical replicate columns.

    Port of edgeR's sumTechReps.
    """
    if isinstance(x, dict) and 'counts' in x:
        # DGEList-like
        if ID is None:
            raise ValueError("No sample IDs")
        ID = np.asarray(ID)
        unique_ids, inverse = np.unique(ID, return_inverse=True)
        if len(unique_ids) == len(ID):
            return x

        from copy import deepcopy
        y = deepcopy(x)
        # Sum counts
        new_counts = np.zeros((x['counts'].shape[0], len(unique_ids)))
        for i, uid in enumerate(unique_ids):
            mask = ID == uid
            new_counts[:, i] = x['counts'][:, mask].sum(axis=1)
        y['counts'] = new_counts

        # Average lib.size and norm.factors
        if 'samples' in y:
            new_samples = pd.DataFrame(index=unique_ids)
            for col in y['samples'].columns:
                vals = y['samples'][col].values
                if np.issubdtype(type(vals[0]), np.number) if not isinstance(vals[0], str) else False:
                    new_vals = np.array([vals[ID == uid].sum() for uid in unique_ids])
                    if col == 'norm.factors':
                        counts_per_id = np.array([np.sum(ID == uid) for uid in unique_ids])
                        new_vals = new_vals / counts_per_id
                    new_samples[col] = new_vals
                else:
                    new_samples[col] = [vals[ID == uid][0] for uid in unique_ids]
            y['samples'] = new_samples
        return y
    else:
        # Matrix
        x = np.asarray(x, dtype=np.float64)
        if ID is None:
            raise ValueError("No sample IDs")
        ID = np.asarray(ID)
        unique_ids = np.unique(ID)
        result = np.zeros((x.shape[0], len(unique_ids)))
        for i, uid in enumerate(unique_ids):
            mask = ID == uid
            result[:, i] = x[:, mask].sum(axis=1)
        return result


def systematic_subset(n, order_by):
    """Take a systematic subset of indices stratified by a ranking variable.

    Port of edgeR's systematicSubset.
    """
    order_by = np.asarray(order_by)
    ntotal = len(order_by)
    sampling_ratio = ntotal // n
    if sampling_ratio <= 1:
        return np.arange(ntotal)
    i1 = sampling_ratio // 2
    indices = np.arange(i1, ntotal, sampling_ratio)
    o = np.argsort(order_by)
    return o[indices]


def nearest_ref_to_x(x, reference):
    """Find nearest element of reference for each element of x.

    Port of edgeR's nearestReftoX.
    """
    reference = np.sort(reference)
    midpt = (reference[:-1] + reference[1:]) / 2
    return np.searchsorted(midpt, x)


def get_prior_n(y, design=None, prior_df=20):
    """Determine prior.n to keep prior degrees of freedom fixed.

    Port of edgeR's getPriorN.
    """
    if isinstance(y, dict):
        nlibs = y['counts'].shape[1] if 'counts' in y else 0
        if design is None:
            npar = len(y['samples']['group'].unique()) if 'samples' in y else 1
        else:
            npar = design.shape[1]
    else:
        if design is None:
            raise ValueError("design must be provided for matrix input")
        nlibs = np.asarray(y).shape[1]
        npar = design.shape[1]

    residual_df = nlibs - npar
    if residual_df <= 0:
        return prior_df
    return prior_df / residual_df


def zscore_nbinom(q, size, mu, method='midp'):
    """Z-score equivalents for negative binomial deviates.

    Port of edgeR's zscoreNBinom.
    """
    q = np.asarray(q, dtype=np.float64)
    size = np.atleast_1d(np.asarray(size, dtype=np.float64))
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    n = len(q)
    size = np.broadcast_to(size, n).copy()
    mu = np.broadcast_to(mu, n).copy()

    z = np.zeros(n)
    qr = np.round(q).astype(int)

    for i in range(n):
        if mu[i] <= 0 or size[i] <= 0:
            z[i] = 0
            continue
        logd = stats.nbinom.logpmf(qr[i], size[i], size[i] / (size[i] + mu[i]))
        if qr[i] == 0:
            w = (q[i] - qr[i]) + 0.5
            logp = logd + np.log(max(w, 1e-300))
            z[i] = stats.norm.ppf(np.exp(logp)) if np.exp(logp) < 1 else 0
        elif q[i] >= mu[i]:
            logp_tail = stats.nbinom.logsf(qr[i], size[i], size[i] / (size[i] + mu[i]))
            w = 0.5 - (q[i] - qr[i])
            from .limma_port import logsumexp
            logp = logsumexp(logp_tail, logd + np.log(max(w, 1e-300)))
            z[i] = -stats.norm.ppf(np.exp(logp)) if np.exp(logp) < 1 else 0
        else:
            logp_tail = stats.nbinom.logcdf(max(qr[i] - 1, 0), size[i], size[i] / (size[i] + mu[i]))
            w = (q[i] - qr[i]) + 0.5
            from .limma_port import logsumexp
            logp = logsumexp(logp_tail, logd + np.log(max(w, 1e-300)))
            z[i] = stats.norm.ppf(np.exp(logp)) if np.exp(logp) < 1 else 0

    return z


def binom_test(y1, y2, n1=None, n2=None, p=None):
    """Multiple exact binomial tests.

    Port of edgeR's binomTest.
    """
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have same length")

    if n1 is None:
        n1 = np.sum(y1)
    if n2 is None:
        n2 = np.sum(y2)
    if p is None:
        p = n1 / (n1 + n2)

    size = y1 + y2
    pvalue = np.ones(len(y1))

    if p == 0.5:
        for i in range(len(y1)):
            if size[i] > 0:
                k = min(y1[i], y2[i])
                pvalue[i] = min(2 * stats.binom.cdf(k, size[i], 0.5), 1.0)
        return pvalue

    for i in range(len(y1)):
        if size[i] == 0:
            pvalue[i] = 1.0
            continue
        if size[i] > 10000:
            table = np.array([[y1[i], y2[i]], [n1 - y1[i], n2 - y2[i]]])
            _, pv, _, _ = stats.chi2_contingency(table, correction=False)
            pvalue[i] = pv
        else:
            # Method of small probabilities
            d = stats.binom.pmf(np.arange(size[i] + 1), size[i], p)
            d_obs = stats.binom.pmf(y1[i], size[i], p)
            pvalue[i] = np.sum(d[d <= d_obs + 1e-15])

    return np.minimum(pvalue, 1.0)


def drop_empty_levels(x):
    """Drop unused factor levels.

    Port of edgeR's dropEmptyLevels.
    """
    if isinstance(x, pd.Categorical):
        return x.remove_unused_categories()
    return pd.Categorical(x)


def design_as_factor(design):
    """Construct a factor from the unique rows of a design matrix.

    Port of edgeR's designAsFactor.
    """
    design = np.asarray(design, dtype=np.float64)
    z = (np.e + np.pi) / 5
    powers = z ** np.arange(design.shape[1])
    row_vals = design @ powers
    _, inverse = np.unique(row_vals, return_inverse=True)
    return inverse


def residual_df(zero_fit, design):
    """Calculate effective residual DF adjusted for exact zeros.

    Port of edgeR's .residDF.
    """
    zero_fit = np.asarray(zero_fit, dtype=bool)
    nlibs = zero_fit.shape[1] if zero_fit.ndim == 2 else len(zero_fit)
    ncoefs = design.shape[1]
    base_df = nlibs - ncoefs

    if zero_fit.ndim == 1:
        n_zeros = np.sum(zero_fit)
        return max(base_df - n_zeros, 0)

    # Group rows with same zero pattern
    ngenes = zero_fit.shape[0]
    df = np.full(ngenes, base_df, dtype=np.float64)

    for i in range(ngenes):
        zf = zero_fit[i]
        n_zeros = np.sum(zf)
        if n_zeros == 0:
            continue
        if n_zeros >= nlibs - 1:
            df[i] = 0
            continue
        # Reduce design matrix
        keep = ~zf
        design_sub = design[keep]
        rank_sub = np.linalg.matrix_rank(design_sub)
        df[i] = np.sum(keep) - rank_sub

    return df


def scale_offset(y, offset):
    """Scale offsets to be consistent with library sizes.

    Port of edgeR's scaleOffset.
    """
    if isinstance(y, dict) and 'counts' in y:
        lib_size = y['samples']['lib.size'].values * y['samples']['norm.factors'].values
        y['offset'] = scale_offset(lib_size, offset)
        return y

    if isinstance(y, np.ndarray) and y.ndim == 2:
        lib_size = y.sum(axis=0)
    else:
        lib_size = np.asarray(y, dtype=np.float64)

    offset = np.asarray(offset, dtype=np.float64)

    if offset.ndim == 2:
        adj = offset.mean(axis=1, keepdims=True)
    else:
        adj = np.mean(offset)

    return np.mean(np.log(lib_size)) + offset - adj


def _model_matrix_group(group):
    """Create a model matrix from a group factor (model.matrix(~group) equivalent).

    Returns an intercept + dummy-coded design matrix.
    """
    group = np.asarray(group)
    unique_groups = np.unique(group)
    n = len(group)
    ngroups = len(unique_groups)

    if ngroups <= 1:
        return np.ones((n, 1))

    # Intercept + (ngroups - 1) dummy columns
    design = np.zeros((n, ngroups))
    design[:, 0] = 1.0  # intercept
    for i in range(1, ngroups):
        design[group == unique_groups[i], i] = 1.0

    return design


def model_matrix(formula, data=None):
    """Create a design matrix from an R-style formula.

    Uses patsy to parse the formula and build the design matrix,
    matching R's ``model.matrix(formula, data)`` behaviour.

    Parameters
    ----------
    formula : str
        R-style formula, e.g. ``'~ group'``, ``'~ batch + condition'``,
        ``'~ 0 + group'`` (no intercept).
    data : DataFrame, dict, ndarray, scipy.sparse, or Series
        Sample-level data.  Column names are used as variables in
        the formula.

        - **DataFrame**: used directly.
        - **dict**: converted to DataFrame (keys → column names).
        - **ndarray**: columns named ``x0, x1, …`` automatically.
        - **scipy.sparse**: densified, then treated as ndarray.
        - **Series**: wrapped in a single-column DataFrame whose
          column name is the Series ``.name`` (or ``x0``).

    Returns
    -------
    ndarray
        Design matrix (samples x coefficients), dtype float64.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'group': ['A','A','B','B'], 'batch': [1,2,1,2]})
    >>> model_matrix('~ group', df)
    array([[1., 0.],
           [1., 0.],
           [1., 1.],
           [1., 1.]])
    >>> model_matrix('~ 0 + group', df)   # no intercept
    array([[1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.]])
    """
    try:
        import patsy
    except ImportError:
        raise ImportError(
            "patsy package required for formula interface. "
            "Install with: pip install patsy"
        )

    if data is None:
        raise ValueError("data must be provided for formula-based design")

    # Convert various types to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        name = data.name if data.name is not None else 'x0'
        data = pd.DataFrame({name: data.values})
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = pd.DataFrame({'x0': data})
        else:
            cols = {f'x{i}': data[:, i] for i in range(data.shape[1])}
            data = pd.DataFrame(cols)
    elif not isinstance(data, pd.DataFrame):
        # scipy.sparse or other array-like
        if hasattr(data, 'toarray'):
            data = data.toarray()
        data = np.asarray(data)
        if data.ndim == 1:
            data = pd.DataFrame({'x0': data})
        else:
            cols = {f'x{i}': data[:, i] for i in range(data.shape[1])}
            data = pd.DataFrame(cols)

    design_info = patsy.dmatrix(formula, data=data, return_type='dataframe')
    return np.asarray(design_info, dtype=np.float64)


def _resolve_design(design, y):
    """Resolve design argument: formula string → numpy array.

    If *design* is a string it is treated as an R-style formula and
    evaluated against the sample metadata in *y* (which must be a
    DGEList with a 'samples' key).  Otherwise *design* is returned
    as-is.
    """
    if not isinstance(design, str):
        return design

    if not (isinstance(y, dict) and 'samples' in y):
        raise ValueError(
            "Formula design requires a DGEList with sample metadata. "
            "Pass a DGEList or use model_matrix() explicitly."
        )
    return model_matrix(design, y['samples'])


def model_matrix_meth(object, design=None):
    """Create expanded design matrix for BS-seq methylation analysis.

    Port of edgeR's ``modelMatrixMeth``.

    Takes a sample-level design matrix (``nsamples x p``) and expands it
    for a DGEList produced by :func:`read_bismark2dge`, which has
    ``2 * nsamples`` columns arranged as
    ``S1-Me, S1-Un, S2-Me, S2-Un, ...``.

    The returned design matrix has ``2 * nsamples`` rows and
    ``nsamples + p`` columns:

    * **Left block** (``nsamples`` columns): sample indicator matrix.
      Each sample gets a 1 in both its Me and Un rows.
    * **Right block** (``p`` columns): treatment design for Me rows
      (odd rows), zeros for Un rows (even rows).

    Parameters
    ----------
    object : DGEList or ndarray
        Either a DGEList (in which case the sample-level design is taken
        from ``design`` or built from the group factor) or a numpy array
        to use directly as the sample-level treatment design matrix.
    design : ndarray, optional
        Sample-level design matrix (``nsamples x p``).  Used when
        *object* is a DGEList.  If None and *object* is a DGEList,
        a ``~group`` design is created from the sample metadata.

    Returns
    -------
    ndarray
        Expanded design matrix, shape ``(2 * nsamples, nsamples + p)``.
    """
    if isinstance(object, np.ndarray):
        design_treatments = object.copy()
    elif isinstance(object, dict):
        # DGEList
        if design is not None:
            design_treatments = np.asarray(design, dtype=np.float64)
        else:
            # Build ~group design from samples
            if 'samples' in object and 'group' in object['samples'].columns:
                group = object['samples']['group'].values
                # Only use first half (Me samples)
                ncols = object['counts'].shape[1]
                nsamples = ncols // 2
                group_half = group[:nsamples] if len(group) > nsamples else group
                design_treatments = _model_matrix_group(group_half)
            else:
                raise ValueError(
                    "No design provided and DGEList has no group factor"
                )
    else:
        raise TypeError("object must be a DGEList or a numpy array")

    nsamples = design_treatments.shape[0]
    nparam = design_treatments.shape[1]

    # Sample indicator: gl(nsamples, 2) → [0,0,1,1,2,2,...]
    # model.matrix(~0+Sample) → identity matrix indexed by sample
    design_samples = np.zeros((2 * nsamples, nsamples), dtype=np.float64)
    for i in range(nsamples):
        design_samples[2 * i, i] = 1.0
        design_samples[2 * i + 1, i] = 1.0

    # Expand treatment design: duplicate each row for Me and Un
    design_expanded = np.zeros((2 * nsamples, nparam), dtype=np.float64)
    for i in range(nsamples):
        design_expanded[2 * i, :] = design_treatments[i, :]
        design_expanded[2 * i + 1, :] = design_treatments[i, :]

    # Methylation indicator: 1 for Me (even rows), 0 for Un (odd rows)
    meth_indicator = np.zeros(2 * nsamples, dtype=np.float64)
    for i in range(nsamples):
        meth_indicator[2 * i] = 1.0

    # Right block: treatment design * methylation indicator
    design_right = design_expanded * meth_indicator[:, np.newaxis]

    return np.hstack([design_samples, design_right])


def nearest_tss(chr, locus, tss_data=None, species="Hs"):
    """Find nearest transcription start site for genomic coordinates.

    Port of edgeR's ``nearestTSS``.

    For each query position ``(chr[i], locus[i])``, finds the nearest
    TSS on the same chromosome and returns information about the
    corresponding gene.

    Parameters
    ----------
    chr : array-like of str
        Chromosome names for query positions.
    locus : array-like of int
        Genomic positions for query positions.
    tss_data : DataFrame, optional
        TSS annotation with columns: ``chr``, ``tss``, ``gene_id``,
        ``gene_name``, ``strand``.  If None, attempts to fetch from
        Ensembl BioMart using ``pybiomart`` (requires internet).
    species : str
        Species code for BioMart query (default ``"Hs"`` for human).
        Only used when ``tss_data`` is None.

    Returns
    -------
    DataFrame
        With columns: ``gene_id``, ``gene_name``, ``strand``, ``tss``,
        ``width``, ``distance``.  ``distance`` is positive when the
        query locus is downstream of the TSS on the gene's strand.
    """
    chr_arr = np.asarray(chr, dtype=str)
    locus_arr = np.asarray(locus, dtype=np.int64)
    n = len(chr_arr)

    if len(locus_arr) == 1:
        locus_arr = np.full(n, locus_arr[0], dtype=np.int64)
    elif len(locus_arr) != n:
        raise ValueError("Length of locus doesn't agree with length of chr")

    # Handle NAs
    na_mask = np.array([(c == '' or c == 'nan' or c == 'None')
                        for c in chr_arr])

    if tss_data is None:
        tss_data = _fetch_tss_biomart(species)

    # Ensure tss_data has required columns
    required = {'chr', 'tss', 'gene_id', 'gene_name', 'strand'}
    missing = required - set(tss_data.columns)
    if missing:
        raise ValueError(f"tss_data missing columns: {missing}")

    # Sort tss_data by chromosome and TSS position
    tss_data = tss_data.sort_values(['chr', 'tss']).reset_index(drop=True)

    # Group by chromosome
    tss_by_chr = {}
    for chrom, grp in tss_data.groupby('chr'):
        tss_by_chr[chrom] = grp

    # Prepare output
    out_gene_id = np.full(n, np.nan, dtype=object)
    out_gene_name = np.full(n, np.nan, dtype=object)
    out_strand = np.full(n, np.nan, dtype=object)
    out_tss = np.full(n, np.nan, dtype=np.float64)
    out_width = np.full(n, np.nan, dtype=np.float64)
    out_distance = np.full(n, np.nan, dtype=np.float64)

    # Check if query chr values start with "chr" but tss_data doesn't (or vice versa)
    query_has_chr = any(c.startswith('chr') for c in chr_arr if c)
    tss_has_chr = any(str(c).startswith('chr') for c in tss_data['chr'].values[:10])

    for chrom_name in tss_by_chr:
        grp = tss_by_chr[chrom_name]
        tss_positions = grp['tss'].values.astype(np.float64)

        # Match query chromosomes to this reference chromosome
        if query_has_chr and not tss_has_chr:
            query_chrom = 'chr' + str(chrom_name)
        elif not query_has_chr and tss_has_chr:
            query_chrom = str(chrom_name).replace('chr', '')
        else:
            query_chrom = str(chrom_name)

        iinc = np.where((chr_arr == query_chrom) & ~na_mask)[0]
        if len(iinc) == 0:
            continue

        which = nearest_ref_to_x(locus_arr[iinc].astype(np.float64),
                                  tss_positions)

        for j, qi in enumerate(iinc):
            ref_idx = which[j]
            row = grp.iloc[ref_idx]
            out_gene_id[qi] = row['gene_id']
            out_gene_name[qi] = row['gene_name']
            out_strand[qi] = row['strand']
            out_tss[qi] = row['tss']
            if 'width' in grp.columns:
                out_width[qi] = row['width']
            # distance: signed distance, positive = downstream of TSS
            dist = locus_arr[qi] - int(row['tss'])
            if row['strand'] == '-':
                dist = -dist
            out_distance[qi] = dist

    result = pd.DataFrame({
        'gene_id': out_gene_id,
        'gene_name': out_gene_name,
        'strand': out_strand,
        'tss': pd.array(out_tss, dtype=pd.Int64Dtype()),
        'width': pd.array(out_width, dtype=pd.Int64Dtype()),
        'distance': pd.array(out_distance, dtype=pd.Int64Dtype()),
    })
    return result


def _fetch_tss_biomart(species="Hs"):
    """Fetch TSS data from Ensembl BioMart.

    Requires the ``pybiomart`` package.
    """
    try:
        from pybiomart import Server
    except ImportError:
        raise ImportError(
            "pybiomart package required to fetch TSS data from Ensembl. "
            "Install with: pip install pybiomart\n"
            "Alternatively, pass tss_data as a DataFrame with columns: "
            "chr, tss, gene_id, gene_name, strand"
        )

    species_map = {
        'Hs': 'hsapiens_gene_ensembl',
        'Mm': 'mmusculus_gene_ensembl',
        'Rn': 'rnorvegicus_gene_ensembl',
        'Dm': 'dmelanogaster_gene_ensembl',
        'Dr': 'drerio_gene_ensembl',
    }

    dataset_name = species_map.get(species)
    if dataset_name is None:
        raise ValueError(
            f"Unknown species code '{species}'. Known: {list(species_map.keys())}"
        )

    server = Server(host='http://www.ensembl.org')
    dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets[dataset_name]

    result = dataset.query(
        attributes=[
            'chromosome_name',
            'transcription_start_site',
            'ensembl_gene_id',
            'external_gene_name',
            'strand',
            'transcript_length',
        ]
    )

    result.columns = ['chr', 'tss', 'gene_id', 'gene_name', 'strand_int',
                       'width']
    result['strand'] = np.where(result['strand_int'] > 0, '+', '-')
    result = result.drop(columns=['strand_int'])

    # Keep one TSS per gene (the one with smallest TSS per chromosome)
    result = result.sort_values(['chr', 'tss']).drop_duplicates(
        subset=['chr', 'gene_id'], keep='first'
    ).reset_index(drop=True)

    return result
