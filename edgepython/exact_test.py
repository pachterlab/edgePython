# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Exact tests for differential expression in edgePython.

Port of edgeR's exactTest, equalizeLibSizes, q2qnbinom, splitIntoGroups.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm
from scipy.special import gammaln, logsumexp
from numba import njit

from .utils import drop_empty_levels, binom_test


@njit(cache=True)
def _nb_logpmf(k, size, prob):
    """Numba-compatible NB log-PMF using math.lgamma."""
    return (math.lgamma(k + size) - math.lgamma(k + 1.0) - math.lgamma(size)
            + size * math.log(prob) + k * math.log(1.0 - prob))


@njit(cache=True)
def _logsumexp_1d(arr, n):
    """Numba-compatible logsumexp for a buffer of length n."""
    if n == 0:
        return -math.inf
    x_max = arr[0]
    for i in range(1, n):
        if arr[i] > x_max:
            x_max = arr[i]
    if x_max == -math.inf:
        return -math.inf
    s = 0.0
    for i in range(n):
        s += math.exp(arr[i] - x_max)
    return x_max + math.log(s)


@njit(cache=True)
def _nb_exact_loop(s1, s2, s, mu1, dispersion, remaining_mask, n1, n2, pvals, work_buf):
    """Numba kernel for the NB exact test per-gene loop."""
    log2 = math.log(2.0)
    ntags = len(s1)
    n_total = n1 + n2

    for g in range(ntags):
        if not remaining_mask[g]:
            continue
        if s[g] == 0:
            pvals[g] = 1.0
            continue

        d = dispersion[g]
        size1 = n1 / d
        size2 = n2 / d
        prob1 = size1 / (size1 + mu1[g])
        mu2_g = n2 * s[g] / n_total
        prob2 = size2 / (size2 + mu2_g)
        size_total = n_total / d
        mu_total = s[g]  # mu[g] * n_total = s[g]/n_total * n_total = s[g]
        prob_total = size_total / (size_total + mu_total)

        if s1[g] < mu1[g]:
            # Left tail: x = 0..s1[g]
            count = s1[g] + 1
            for x in range(count):
                work_buf[x] = (_nb_logpmf(x, size1, prob1)
                               + _nb_logpmf(s[g] - x, size2, prob2))
            log_sum_top = _logsumexp_1d(work_buf, count)
            log_p_bot = _nb_logpmf(s[g], size_total, prob_total)
            log_pval = log2 + log_sum_top - log_p_bot
            p = math.exp(min(log_pval, 0.0))
            pvals[g] = min(p, 1.0)
        elif s1[g] > mu1[g]:
            # Right tail: x = s1[g]..s[g]
            count = s[g] - s1[g] + 1
            for idx in range(count):
                x = s1[g] + idx
                work_buf[idx] = (_nb_logpmf(x, size1, prob1)
                                 + _nb_logpmf(s[g] - x, size2, prob2))
            log_sum_top = _logsumexp_1d(work_buf, count)
            log_p_bot = _nb_logpmf(s[g], size_total, prob_total)
            log_pval = log2 + log_sum_top - log_p_bot
            p = math.exp(min(log_pval, 0.0))
            pvals[g] = min(p, 1.0)
        # else s1[g] == mu1[g]: pvals[g] stays 1.0


def exact_test(y, pair=None, dispersion='auto', rejection_region='doubletail',
               big_count=900, prior_count=0.125):
    """Exact test for differential expression between two groups.

    Port of edgeR's exactTest.

    Parameters
    ----------
    y : DGEList
        DGEList object.
    pair : list of length 2, optional
        Groups to compare. Default is first two groups.
    dispersion : str, float, or ndarray
        'auto', 'common', 'trended', 'tagwise', or numeric.
    rejection_region : str
        'doubletail', 'deviance', or 'smallp'.
    big_count : int
        Threshold for beta approximation.
    prior_count : float
        Prior count for logFC calculation.

    Returns
    -------
    dict (DGEExact-like) with 'table', 'comparison', 'genes'.
    """
    if not (isinstance(y, dict) and 'counts' in y):
        raise ValueError("Currently only supports DGEList objects.")

    group = np.asarray(y['samples']['group'].values)
    unique_groups = np.unique(group)

    if pair is None:
        pair = unique_groups[:2].tolist()
    elif len(pair) != 2:
        raise ValueError("pair must be of length 2.")

    group = np.array([str(g) for g in group])
    unique_group_labels = np.array([str(g) for g in unique_groups])

    # edgeR-style convenience: allow pair as integer level indices (e.g. [0, 1]).
    if len(pair) == 2 and all(isinstance(p, (int, np.integer)) for p in pair):
        p0, p1 = int(pair[0]), int(pair[1])
        if p0 < 0 or p1 < 0 or p0 >= len(unique_group_labels) or p1 >= len(unique_group_labels):
            raise ValueError(
                f"pair indices out of range for {len(unique_group_labels)} groups: {pair}"
            )
        pair = [unique_group_labels[p0], unique_group_labels[p1]]
    else:
        pair = [str(p) for p in pair]

    # Get dispersion
    if dispersion is None or dispersion == 'auto':
        from .dgelist import get_dispersion
        dispersion = get_dispersion(y)
        if dispersion is None:
            raise ValueError("No dispersion values found in DGEList object.")
    elif isinstance(dispersion, str):
        valid = ('common', 'trended', 'tagwise')
        if dispersion not in valid:
            raise ValueError(f"dispersion must be one of {valid}")
        dispersion = y.get(f'{dispersion}.dispersion')
        if dispersion is None:
            raise ValueError("Specified dispersion not found in object")

    ntags = y['counts'].shape[0]
    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if len(dispersion) == 1:
        dispersion = np.full(ntags, dispersion[0])

    # Reduce to two groups
    j = np.isin(group, pair)
    if np.sum(j) == 0:
        raise ValueError(
            f"No samples matched pair={pair}. Available groups: {unique_group_labels.tolist()}"
        )
    counts = y['counts'][:, j]
    lib_size = y['samples']['lib.size'].values[j]
    norm_factors = y['samples']['norm.factors'].values[j]
    group_sub = group[j]

    lib_size = lib_size * norm_factors
    offset = np.log(lib_size)
    lib_size_average = np.exp(np.mean(offset))

    # logFC with prior counts
    pc = prior_count * lib_size / np.mean(lib_size)
    offset_aug = np.log(lib_size + 2 * pc)

    j1 = group_sub == pair[0]
    n1 = np.sum(j1)
    y1 = counts[:, j1]

    j2 = group_sub == pair[1]
    n2 = np.sum(j2)
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"Both pair groups must have >=1 sample. pair={pair}, n1={int(n1)}, n2={int(n2)}"
        )
    y2 = counts[:, j2]

    from .glm_fit import mglm_one_group
    abundance1 = mglm_one_group(
        y1 + np.tile(pc[j1], (ntags, 1)),
        offset=offset_aug[j1], dispersion=dispersion)
    abundance2 = mglm_one_group(
        y2 + np.tile(pc[j2], (ntags, 1)),
        offset=offset_aug[j2], dispersion=dispersion)
    logFC = (abundance2 - abundance1) / np.log(2)

    # Equalize library sizes
    abundance = mglm_one_group(counts, dispersion=dispersion, offset=offset)
    e = np.exp(abundance)

    input_mean1 = np.outer(e, lib_size[j1])
    output_mean1 = np.outer(e, np.full(n1, lib_size_average))
    y1_eq = q2q_nbinom(y1.astype(float), input_mean1, output_mean1, dispersion)

    input_mean2 = np.outer(e, lib_size[j2])
    output_mean2 = np.outer(e, np.full(n2, lib_size_average))
    y2_eq = q2q_nbinom(y2.astype(float), input_mean2, output_mean2, dispersion)

    # Exact p-values
    if rejection_region == 'doubletail':
        exact_pvals = exact_test_double_tail(y1_eq, y2_eq, dispersion=dispersion,
                                              big_count=big_count)
    elif rejection_region == 'deviance':
        exact_pvals = exact_test_by_deviance(y1_eq, y2_eq, dispersion=dispersion)
    else:
        exact_pvals = exact_test_by_small_p(y1_eq, y2_eq, dispersion=dispersion)

    from .expression import ave_log_cpm
    alc = y.get('AveLogCPM')
    if alc is None:
        alc = ave_log_cpm(y)

    table = pd.DataFrame({
        'logFC': logFC,
        'logCPM': alc,
        'PValue': exact_pvals
    })
    rn = None
    if hasattr(y['counts'], 'index'):
        rn = y['counts'].index
    if rn is not None:
        table.index = rn

    return {
        'table': table,
        'comparison': pair,
        'genes': y.get('genes')
    }


def exact_test_double_tail(y1, y2, dispersion=0, big_count=900):
    """Double-tail exact test for NB distribution.

    Port of edgeR's exactTestDoubleTail.
    """
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    if y1.ndim == 1:
        y1 = y1.reshape(-1, 1)
    if y2.ndim == 1:
        y2 = y2.reshape(-1, 1)

    ntags = y1.shape[0]
    n1 = y1.shape[1]
    n2 = y2.shape[1]

    s1 = np.round(y1.sum(axis=1)).astype(int)
    s2 = np.round(y2.sum(axis=1)).astype(int)

    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if len(dispersion) == 1:
        dispersion = np.full(ntags, dispersion[0])

    s = s1 + s2
    mu = s / (n1 + n2)
    mu1 = n1 * mu
    mu2 = n2 * mu

    pvals = np.ones(ntags)

    # Poisson case
    pois = dispersion <= 0
    if np.any(pois):
        pvals[pois] = binom_test(s1[pois], s2[pois], p=n1 / (n1 + n2))

    # Beta approximation for large counts
    big = (s1 > big_count) & (s2 > big_count)
    if np.any(big):
        pvals[big] = _exact_test_beta_approx(y1[big], y2[big], dispersion[big])

    # NB exact test for remaining (use log-scale to avoid underflow)
    remaining = ~pois & ~big
    if np.any(remaining):
        max_s = int(np.max(s[remaining])) + 1
        work_buf = np.empty(max(max_s, 1), dtype=np.float64)
        _nb_exact_loop(s1.astype(np.int64), s2.astype(np.int64),
                       s.astype(np.int64), mu1, dispersion,
                       remaining, n1, n2, pvals, work_buf)

    return np.minimum(pvals, 1.0)


def exact_test_by_deviance(y1, y2, dispersion=0):
    """Exact test using deviance as rejection region.

    Simplified port: uses double-tail as fallback.
    """
    return exact_test_double_tail(y1, y2, dispersion=dispersion)


def exact_test_by_small_p(y1, y2, dispersion=0):
    """Exact test using small-p as rejection region.

    Simplified port: uses double-tail as fallback.
    """
    return exact_test_double_tail(y1, y2, dispersion=dispersion)


def _exact_test_beta_approx(y1, y2, dispersion):
    """Beta approximation for exact test with large counts.

    Faithful port of edgeR's exactTestBetaApprox.
    """
    from scipy.stats import beta as beta_dist

    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    if y1.ndim == 1:
        y1 = y1.reshape(-1, 1)
    if y2.ndim == 1:
        y2 = y2.reshape(-1, 1)

    n1 = y1.shape[1]
    n2 = y2.shape[1]
    s1 = y1.sum(axis=1)
    s2 = y2.sum(axis=1)
    y = s1 + s2

    ntags = len(s1)
    dispersion = np.broadcast_to(np.atleast_1d(np.asarray(dispersion, dtype=np.float64)),
                                 (ntags,)).copy()

    mu = y / (n1 + n2)
    pvals = np.ones(ntags)
    all_zero = y <= 0

    alpha1 = n1 * mu / (1.0 + dispersion * mu)
    alpha2 = (n2 / n1) * alpha1

    # Median of the beta distribution
    med = np.zeros(ntags)
    nz = ~all_zero
    if np.any(nz):
        med[nz] = beta_dist.median(alpha1[nz], alpha2[nz])

    # Left tail with continuity correction
    left = ((s1 + 0.5) / y < med) & ~all_zero
    if np.any(left):
        pvals[left] = 2.0 * beta_dist.cdf((s1[left] + 0.5) / y[left],
                                            alpha1[left], alpha2[left])

    # Right tail with continuity correction
    right = ((s1 - 0.5) / y > med) & ~all_zero
    if np.any(right):
        pvals[right] = 2.0 * beta_dist.sf((s1[right] - 0.5) / y[right],
                                            alpha1[right], alpha2[right])

    return np.minimum(pvals, 1.0)


def equalize_lib_sizes(y, group=None, dispersion=None, lib_size=None):
    """Equalize library sizes using quantile-to-quantile transformation.

    Port of edgeR's equalizeLibSizes.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix or DGEList.
    group : array-like, optional
        Group factor.
    dispersion : float, optional
        Dispersion.
    lib_size : ndarray, optional
        Library sizes.

    Returns
    -------
    dict with 'pseudo.counts' and 'pseudo.lib.size'.
    """
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist, get_dispersion
        dge = valid_dgelist(dge)
        if dispersion is None:
            dispersion = get_dispersion(dge)
        ls = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values
        out = equalize_lib_sizes(dge['counts'], group=dge['samples']['group'].values,
                                  dispersion=dispersion, lib_size=ls)
        dge['pseudo.counts'] = out['pseudo.counts']
        dge['pseudo.lib.size'] = out['pseudo.lib.size']
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if group is None:
        group = np.ones(nlibs, dtype=int)
    group = np.asarray(group)

    if dispersion is None:
        dispersion = 0.05

    if lib_size is None:
        lib_size = y.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    common_lib_size = np.exp(np.mean(np.log(lib_size)))
    unique_groups = np.unique(group)

    input_mean = np.zeros_like(y)
    output_mean = np.zeros_like(y)

    from .glm_fit import mglm_one_group
    for grp in unique_groups:
        j = group == grp
        beta = mglm_one_group(y[:, j], dispersion=dispersion,
                              offset=np.log(lib_size[j]))
        lam = np.exp(beta)
        input_mean[:, j] = np.outer(lam, lib_size[j])
        output_mean[:, j] = np.outer(lam, np.full(np.sum(j), common_lib_size))

    pseudo = q2q_nbinom(y, input_mean, output_mean, dispersion)
    pseudo = np.maximum(pseudo, 0)

    return {'pseudo.counts': pseudo, 'pseudo.lib.size': common_lib_size}


def q2q_nbinom(x, input_mean, output_mean, dispersion=0):
    """Quantile-to-quantile mapping between negative-binomial distributions.

    Port of edgeR's q2qnbinom. Uses average of normal and gamma approximations.
    """
    from scipy.stats import norm, gamma as gamma_dist

    x = np.asarray(x, dtype=np.float64)
    input_mean = np.asarray(input_mean, dtype=np.float64)
    output_mean = np.asarray(output_mean, dtype=np.float64)
    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))

    if dispersion.size == 1:
        d = dispersion[0]
    else:
        d = dispersion

    eps = 1e-14
    zero = (input_mean < eps) | (output_mean < eps)
    input_mean = np.where(zero, input_mean + 0.25, input_mean)
    output_mean = np.where(zero, output_mean + 0.25, output_mean)

    if np.isscalar(d):
        d_arr = d
    else:
        if d.ndim == 1 and x.ndim == 2:
            d_arr = d[:, None]
        else:
            d_arr = d

    ri = 1 + d_arr * input_mean
    vi = input_mean * ri
    ro = 1 + d_arr * output_mean
    vo = output_mean * ro

    i = x >= input_mean
    j = ~i

    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    q1 = np.zeros_like(x)
    q2 = np.zeros_like(x)

    # Upper tail (x >= input_mean)
    if np.any(i):
        p1[i] = norm.logsf(x[i], loc=input_mean[i], scale=np.sqrt(np.maximum(vi[i], eps)))
        shape_i = input_mean[i] / np.maximum(ri[i], eps)
        scale_i = ri[i]
        p2[i] = gamma_dist.logsf(x[i], a=shape_i, scale=scale_i)
        q1[i] = norm.isf(np.exp(p1[i]), loc=output_mean[i], scale=np.sqrt(np.maximum(vo[i], eps)))
        shape_o = output_mean[i] / np.maximum(ro[i], eps)
        scale_o = ro[i]
        q2[i] = gamma_dist.isf(np.exp(p2[i]), a=shape_o, scale=scale_o)

    # Lower tail (x < input_mean)
    if np.any(j):
        p1[j] = norm.logcdf(x[j], loc=input_mean[j], scale=np.sqrt(np.maximum(vi[j], eps)))
        shape_i = input_mean[j] / np.maximum(ri[j], eps)
        scale_i = ri[j]
        p2[j] = gamma_dist.logcdf(x[j], a=shape_i, scale=scale_i)
        q1[j] = norm.ppf(np.exp(p1[j]), loc=output_mean[j], scale=np.sqrt(np.maximum(vo[j], eps)))
        shape_o = output_mean[j] / np.maximum(ro[j], eps)
        scale_o = ro[j]
        q2[j] = gamma_dist.ppf(np.exp(p2[j]), a=shape_o, scale=scale_o)

    return (q1 + q2) / 2


def split_into_groups(y, group=None):
    """Split a count matrix into a list of matrices by group.

    Port of edgeR's splitIntoGroups.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    if group is None:
        return [y]

    group = np.asarray(group)
    unique_groups = np.unique(group)

    result = []
    for g in unique_groups:
        mask = group == g
        sub = y[:, mask]
        if sub.ndim == 1:
            sub = sub.reshape(-1, 1)
        result.append(sub)

    return result


def split_into_groups_pseudo(pseudo, group, pair):
    """Extract data for two groups from pseudo-count matrix.

    Port of edgeR's splitIntoGroupsPseudo.
    """
    pseudo = np.asarray(pseudo, dtype=np.float64)
    group = np.asarray(group)

    y1 = pseudo[:, group == pair[0]]
    y2 = pseudo[:, group == pair[1]]
    if y1.ndim == 1:
        y1 = y1.reshape(-1, 1)
    if y2.ndim == 1:
        y2 = y2.reshape(-1, 1)

    return {'y1': y1, 'y2': y2}
