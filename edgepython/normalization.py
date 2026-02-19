# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Normalization methods for edgePython.

Port of edgeR's calcNormFactors/normLibSizes (TMM, TMMwsp, RLE, upperquartile)
and ChIP-seq normalization (normalizeChIPtoInput, calcNormOffsetsforChIP).
"""

import numpy as np
import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests


def calc_norm_factors(counts, lib_size=None, method='TMM', ref_column=None,
                      logratio_trim=0.3, sum_trim=0.05, do_weighting=True,
                      a_cutoff=-1e10, p=0.75):
    """Calculate normalization factors for a count matrix.

    Port of edgeR's calcNormFactors / normLibSizes.

    Parameters
    ----------
    counts : array-like or DGEList
        Count matrix (genes x samples), or DGEList object.
    lib_size : array-like, optional
        Library sizes. Defaults to column sums.
    method : str
        One of 'TMM', 'TMMwsp', 'RLE', 'upperquartile', 'none'.
    ref_column : int, optional
        Reference column for TMM/TMMwsp.
    logratio_trim : float
        Amount of trim for log-ratios (TMM).
    sum_trim : float
        Amount of trim for sums (TMM).
    do_weighting : bool
        Use precision weights in TMM.
    a_cutoff : float
        Abundance cutoff for TMM.
    p : float
        Quantile for upper-quartile method.

    Returns
    -------
    DGEList (if input is DGEList) or ndarray of normalization factors.
    """
    # Handle DGEList input
    if isinstance(counts, dict) and 'counts' in counts:
        y = counts
        if y.get('offset') is not None:
            warnings.warn("object contains offsets, which take precedence over library "
                          "sizes and norm factors (and which will not be recomputed).")
        ls = y['samples']['lib.size'].values
        nf = _calc_norm_factors_default(
            y['counts'], lib_size=ls, method=method, ref_column=ref_column,
            logratio_trim=logratio_trim, sum_trim=sum_trim,
            do_weighting=do_weighting, a_cutoff=a_cutoff, p=p)
        y['samples']['norm.factors'] = nf
        return y

    return _calc_norm_factors_default(
        counts, lib_size=lib_size, method=method, ref_column=ref_column,
        logratio_trim=logratio_trim, sum_trim=sum_trim,
        do_weighting=do_weighting, a_cutoff=a_cutoff, p=p)


# Alias
norm_lib_sizes = calc_norm_factors


def _calc_norm_factors_default(x, lib_size=None, method='TMM', ref_column=None,
                               logratio_trim=0.3, sum_trim=0.05, do_weighting=True,
                               a_cutoff=-1e10, p=0.75):
    """Core normalization factor calculation for count matrices."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)):
        raise ValueError("NA counts not permitted")
    nsamples = x.shape[1]

    if lib_size is None:
        lib_size = x.sum(axis=0)
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA lib.sizes not permitted")
        if len(lib_size) != nsamples:
            if len(lib_size) > 1:
                warnings.warn("length(lib_size) doesn't match number of samples")
            lib_size = np.full(nsamples, lib_size[0] if len(lib_size) == 1 else lib_size.mean())

    # Backward compatibility
    if method == 'TMMwzp':
        method = 'TMMwsp'

    valid_methods = ('TMM', 'TMMwsp', 'RLE', 'upperquartile', 'none')
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Remove all-zero rows
    allzero = np.sum(x > 0, axis=1) == 0
    if np.any(allzero):
        x = x[~allzero]

    # Degenerate cases
    if x.shape[0] == 0 or nsamples == 1:
        method = 'none'

    if method == 'TMM':
        f = _calc_tmm(x, lib_size, ref_column, logratio_trim, sum_trim, do_weighting, a_cutoff)
    elif method == 'TMMwsp':
        f = _calc_tmmwsp(x, lib_size, ref_column, logratio_trim, sum_trim, do_weighting, a_cutoff)
    elif method == 'RLE':
        f = _calc_factor_rle(x) / lib_size
    elif method == 'upperquartile':
        f = _calc_factor_quantile(x, lib_size, p)
    else:
        f = np.ones(nsamples)

    # Normalize so factors multiply to one
    f = f / np.exp(np.mean(np.log(f)))

    return f


def _calc_tmm(x, lib_size, ref_column, logratio_trim, sum_trim, do_weighting, a_cutoff):
    """TMM normalization."""
    nsamples = x.shape[1]
    if ref_column is None:
        f75 = _calc_factor_quantile(x, lib_size, 0.75)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.median(f75) < 1e-20:
                ref_column = np.argmax(np.sum(np.sqrt(x), axis=0))
            else:
                ref_column = np.argmin(np.abs(f75 - np.mean(f75)))

    f = np.full(nsamples, np.nan)
    for i in range(nsamples):
        f[i] = _calc_factor_tmm(
            obs=x[:, i], ref=x[:, ref_column],
            libsize_obs=lib_size[i], libsize_ref=lib_size[ref_column],
            logratio_trim=logratio_trim, sum_trim=sum_trim,
            do_weighting=do_weighting, a_cutoff=a_cutoff)
    return f


def _calc_tmmwsp(x, lib_size, ref_column, logratio_trim, sum_trim, do_weighting, a_cutoff):
    """TMMwsp normalization."""
    nsamples = x.shape[1]
    if ref_column is None:
        ref_column = np.argmax(np.sum(np.sqrt(x), axis=0))

    f = np.full(nsamples, np.nan)
    for i in range(nsamples):
        f[i] = _calc_factor_tmmwsp(
            obs=x[:, i], ref=x[:, ref_column],
            libsize_obs=lib_size[i], libsize_ref=lib_size[ref_column],
            logratio_trim=logratio_trim, sum_trim=sum_trim,
            do_weighting=do_weighting, a_cutoff=a_cutoff)
    return f


def _calc_factor_rle(data):
    """Scale factors as in Anders et al (2010)."""
    with np.errstate(divide='ignore'):
        gm = np.exp(np.mean(np.log(data.astype(float)), axis=1))
    pos = gm > 0
    result = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        ratio = data[pos, j] / gm[pos]
        result[j] = np.median(ratio)
    return result


def _calc_factor_quantile(data, lib_size, p=0.75):
    """Upper-quartile normalization."""
    f = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        f[j] = np.quantile(data[:, j], p)
    if np.min(f) == 0:
        warnings.warn("One or more quantiles are zero")
    return f / lib_size


def _calc_factor_tmm(obs, ref, libsize_obs=None, libsize_ref=None,
                     logratio_trim=0.3, sum_trim=0.05, do_weighting=True,
                     a_cutoff=-1e10):
    """TMM between two libraries."""
    obs = np.asarray(obs, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    if libsize_obs is None:
        nO = np.sum(obs)
    else:
        nO = libsize_obs
    if libsize_ref is None:
        nR = np.sum(ref)
    else:
        nR = libsize_ref

    with np.errstate(divide='ignore', invalid='ignore'):
        logR = np.log2(obs / nO) - np.log2(ref / nR)
        absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2
        v = (nO - obs) / nO / obs + (nR - ref) / nR / ref

    # Remove infinite values
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > a_cutoff)
    logR = logR[fin]
    absE = absE[fin]
    v = v[fin]

    if len(logR) == 0 or np.max(np.abs(logR)) < 1e-6:
        return 1.0

    n = len(logR)
    loL = int(np.floor(n * logratio_trim)) + 1
    hiL = n + 1 - loL
    loS = int(np.floor(n * sum_trim)) + 1
    hiS = n + 1 - loS

    rank_logR = _rank(logR)
    rank_absE = _rank(absE)
    keep = ((rank_logR >= loL) & (rank_logR <= hiL) &
            (rank_absE >= loS) & (rank_absE <= hiS))

    if do_weighting:
        denom = np.sum(1 / v[keep])
        if denom > 0 and np.isfinite(denom):
            f = np.sum(logR[keep] / v[keep]) / denom
        else:
            f = np.nanmean(logR[keep])
    else:
        f = np.nanmean(logR[keep])

    if np.isnan(f):
        f = 0.0

    return 2 ** f


def _calc_factor_tmmwsp(obs, ref, libsize_obs=None, libsize_ref=None,
                        logratio_trim=0.3, sum_trim=0.05, do_weighting=True,
                        a_cutoff=-1e10):
    """TMM with singleton pairing."""
    obs = np.asarray(obs, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    eps = 1e-14

    pos_obs = obs > eps
    pos_ref = ref > eps
    npos = 2 * pos_obs.astype(int) + pos_ref.astype(int)

    # Remove double zeros and NAs
    keep = (npos != 0) & ~np.isnan(npos)
    obs = obs[keep]
    ref = ref[keep]
    npos = npos[keep]

    if libsize_obs is None:
        libsize_obs = np.sum(obs)
    if libsize_ref is None:
        libsize_ref = np.sum(ref)

    # Pair singleton positives
    zero_obs = npos == 1
    zero_ref = npos == 2
    k = zero_obs | zero_ref
    n_eligible = min(np.sum(zero_obs), np.sum(zero_ref))

    if n_eligible > 0:
        refk = np.sort(ref[k])[::-1][:n_eligible]
        obsk = np.sort(obs[k])[::-1][:n_eligible]
        obs = np.concatenate([obs[~k], obsk])
        ref = np.concatenate([ref[~k], refk])
    else:
        obs = obs[~k]
        ref = ref[~k]

    n = len(obs)
    if n == 0:
        return 1.0

    obs_p = obs / libsize_obs
    ref_p = ref / libsize_ref
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.log2(obs_p / ref_p)
        A = 0.5 * np.log2(obs_p * ref_p)

    if np.max(np.abs(M[np.isfinite(M)])) < 1e-6:
        return 1.0

    # Sort by M with tie-breaking
    obs_p_shrunk = (obs + 0.5) / (libsize_obs + 0.5)
    ref_p_shrunk = (ref + 0.5) / (libsize_ref + 0.5)
    M_shrunk = np.log2(obs_p_shrunk / ref_p_shrunk)
    o_M = np.lexsort((M_shrunk, M))
    o_A = np.argsort(A)

    loM = int(n * logratio_trim) + 1
    hiM = n - loM
    keep_M = np.zeros(n, dtype=bool)
    keep_M[o_M[loM:hiM]] = True

    loA = int(n * sum_trim) + 1
    hiA = n - loA
    keep_A = np.zeros(n, dtype=bool)
    keep_A[o_A[loA:hiA]] = True

    keep = keep_M & keep_A
    M_keep = M[keep]

    if do_weighting:
        obs_p_k = obs_p[keep]
        ref_p_k = ref_p[keep]
        v = (1 - obs_p_k) / obs_p_k / libsize_obs + (1 - ref_p_k) / ref_p_k / libsize_ref
        w = (1 + 1e-6) / (v + 1e-6)
        TMM = np.sum(w * M_keep) / np.sum(w)
    else:
        TMM = np.mean(M_keep) if len(M_keep) > 0 else 0

    return 2 ** TMM


def _rank(x):
    """Compute ranks (1-based, average ties)."""
    from scipy.stats import rankdata
    return rankdata(x, method='average')


# =====================================================================
# ChIP-seq normalization
# =====================================================================

def normalize_chip_to_input(input_counts, response, dispersion=0.01, niter=6,
                            loss='p', verbose=False):
    """Normalize ChIP-Seq read counts to input and test for enrichment.

    Port of edgeR's normalizeChIPtoInput.  For a single sample, aligns
    ChIP-Seq mark counts to input control counts under a negative binomial
    model, iteratively estimating a scaling factor and the proportion of
    enriched features.

    Parameters
    ----------
    input_counts : array-like
        Non-negative input control counts for each genomic feature.
    response : array-like
        Non-negative integer ChIP-Seq mark counts for each feature.
    dispersion : float
        Negative binomial dispersion (must be positive).
    niter : int
        Number of iterations for estimating scaling factor and
        proportion enriched.
    loss : str
        Loss function: ``'p'`` for cumulative probabilities,
        ``'z'`` for z-values.
    verbose : bool
        If True, print working estimates at each iteration.

    Returns
    -------
    dict with keys:
        ``p_value`` : ndarray – upper-tail p-values for enrichment.
        ``pmid_value`` : ndarray – mid-p-values.
        ``scaling_factor`` : float – scaling factor aligning response
        to input for unenriched features.
        ``prop_enriched`` : float – estimated proportion of enriched
        features.
    """
    input_counts = np.asarray(input_counts, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)

    if len(input_counts) != len(response):
        raise ValueError("input and response must be same length")
    if np.any(input_counts < 0) or np.any(response < 0):
        raise ValueError("negative values not allowed")
    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    # Remove features where both input and response are zero
    zero = (input_counts <= 0) & (response <= 0)
    if np.any(zero):
        p_value = np.ones(len(zero))
        pmid_value = np.ones(len(zero))
        out = normalize_chip_to_input(
            input_counts[~zero], response[~zero],
            dispersion=dispersion, niter=niter, loss=loss, verbose=verbose,
        )
        p_value[~zero] = out['p_value']
        pmid_value[~zero] = out['pmid_value']
        return {
            'p_value': p_value,
            'pmid_value': pmid_value,
            'scaling_factor': out['scaling_factor'],
            'prop_enriched': out['prop_enriched'],
        }

    n = len(response)

    # Special cases
    if n == 0:
        return {'p_value': np.array([]), 'pmid_value': np.array([]),
                'scaling_factor': np.nan, 'prop_enriched': np.nan}
    if np.all(input_counts == 0):
        return {'p_value': np.zeros(n), 'pmid_value': np.zeros(n),
                'scaling_factor': 0.0, 'prop_enriched': 1.0}
    if n == 1:
        # Avoid inf for response=0 in single-feature inputs.
        sf = 0.0 if response[0] <= 0 else float(input_counts[0] / response[0])
        return {'p_value': np.array([1.0]), 'pmid_value': np.array([1.0]),
                'scaling_factor': sf,
                'prop_enriched': 0.0}

    # Replace zero inputs with minimum positive value
    inp = input_counts.copy()
    inp[inp == 0] = np.min(inp[inp > 0])

    size = 1.0 / dispersion  # NB size parameter

    if loss not in ('p', 'z'):
        raise ValueError("loss must be 'p' or 'z'")

    def _nb_p_and_d(resp, mu):
        """Upper-tail p and pmf for NB(mu, size)."""
        p_val = stats.nbinom.sf(resp.astype(int), size, size / (size + mu))
        d_val = stats.nbinom.pmf(resp.astype(int), size, size / (size + mu))
        return p_val, d_val

    def _objective_p(sf, inp_v, resp_v, prop_enrich):
        mu = sf * inp_v
        p = stats.nbinom.cdf(resp_v.astype(int), size, size / (size + mu))
        d = stats.nbinom.pmf(resp_v.astype(int), size, size / (size + mu))
        pmid = p - d / 2
        n_not_enriched = max(round(len(resp_v) * (1 - prop_enrich)), 1)
        p_sorted = np.partition(pmid, n_not_enriched - 1)[:n_not_enriched]
        return abs(np.mean(p_sorted) - 0.5)

    def _objective_z(sf, inp_v, resp_v, prop_enrich):
        from .utils import zscore_nbinom
        mu = sf * inp_v
        z = zscore_nbinom(resp_v, size=size, mu=mu)
        n_not_enriched = max(round(len(resp_v) * (1 - prop_enrich)), 1)
        z_sorted = np.partition(np.abs(z), n_not_enriched - 1)[:n_not_enriched]
        return np.mean(z_sorted)

    objective = _objective_p if loss == 'p' else _objective_z

    # Starting values
    prop_enriched = 0.5
    ratios = response / inp
    sf_interval = (np.percentile(ratios, 10), np.percentile(ratios, 80))

    if sf_interval[0] == sf_interval[1]:
        scaling_factor = sf_interval[0]
        p, d = _nb_p_and_d(response, scaling_factor * inp)
        pmid = p - d / 2
        _, adj_p, _, _ = multipletests(pmid, method='holm')
        enriched = adj_p < 0.5
        prop_enriched = np.sum(enriched) / n
        if verbose:
            print(f"prop.enriched: {prop_enriched}  scaling.factor: {scaling_factor}")
    else:
        from scipy.optimize import minimize_scalar
        for _ in range(niter):
            res = minimize_scalar(
                objective, bounds=sf_interval, method='bounded',
                args=(inp, response, prop_enriched),
            )
            scaling_factor = res.x
            p, d = _nb_p_and_d(response, scaling_factor * inp)
            pmid = p - d / 2
            _, adj_p, _, _ = multipletests(pmid, method='holm')
            enriched = adj_p < 0.5
            prop_enriched = np.sum(enriched) / n
            if verbose:
                print(f"prop.enriched: {prop_enriched}  scaling.factor: {scaling_factor}")

    return {
        'p_value': p,
        'pmid_value': pmid,
        'scaling_factor': float(scaling_factor),
        'prop_enriched': float(prop_enriched),
    }


def calc_norm_offsets_for_chip(input_counts, response, dispersion=0.01,
                               niter=6, loss='p', verbose=False):
    """Compute normalization offsets for ChIP-Seq relative to input.

    Port of edgeR's calcNormOffsetsforChIP.  Calls
    :func:`normalize_chip_to_input` for each sample and returns a matrix
    of offsets (log-scale) suitable for edgePython's GLM framework.

    Parameters
    ----------
    input_counts : array-like
        Input control count matrix (genes x samples), or a single
        column that is shared across all samples.
    response : array-like or DGEList
        ChIP-Seq mark count matrix (genes x samples), or a DGEList.
    dispersion : float
        Negative binomial dispersion (must be positive).
    niter : int
        Number of iterations.
    loss : str
        Loss function (``'p'`` or ``'z'``).
    verbose : bool
        If True, print working estimates.

    Returns
    -------
    If *response* is a DGEList, returns the DGEList with the ``offset``
    field set.  Otherwise returns a numeric matrix of offsets
    (genes x samples).
    """
    is_dgelist = isinstance(response, dict) and 'counts' in response

    if is_dgelist:
        resp_mat = np.asarray(response['counts'], dtype=np.float64)
    else:
        resp_mat = np.asarray(response, dtype=np.float64)

    inp_mat = np.asarray(input_counts, dtype=np.float64)
    if inp_mat.ndim == 1:
        inp_mat = inp_mat[:, np.newaxis]
    if resp_mat.ndim == 1:
        resp_mat = resp_mat[:, np.newaxis]

    if inp_mat.shape[0] != resp_mat.shape[0]:
        raise ValueError("nrows of input and response disagree")
    if inp_mat.shape[1] == 1 and resp_mat.shape[1] > 1:
        inp_mat = np.broadcast_to(inp_mat, resp_mat.shape).copy()
    if inp_mat.shape[1] != resp_mat.shape[1]:
        raise ValueError("ncols of input and response disagree")

    offset = np.empty_like(resp_mat, dtype=np.float64)
    for j in range(resp_mat.shape[1]):
        out = normalize_chip_to_input(
            inp_mat[:, j], resp_mat[:, j],
            dispersion=dispersion, niter=niter, loss=loss, verbose=verbose,
        )
        offset[:, j] = np.log(out['scaling_factor'] * inp_mat[:, j])

    if is_dgelist:
        response = dict(response)
        response['offset'] = offset
        return response
    return offset
