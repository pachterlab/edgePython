# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Gene set testing for edgePython.

Port of edgeR's camera, fry, roast, mroast, romer, goana, kegga.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import t as t_dist, norm as norm_dist, beta as beta_dist, rankdata
from statsmodels.stats.multitest import multipletests
from numba import njit, prange


def _zscore_t_hill(x, df):
    """Convert t-statistics to z-scores using Hill's approximation.

    Port of limma's .zscoreTHill. This is the method used by R's camera
    when approx=TRUE, method="hill".
    """
    x = np.asarray(x, dtype=np.float64)
    df = np.minimum(df, 1e100)
    A = df - 0.5
    B = 48.0 * A * A
    z = A * np.log1p(x / df * x)
    z = (((((-0.4 * z - 3.3) * z - 24.0) * z - 85.5) / (0.8 * z * z + 100.0 + B) + z + 3.0) / B + 1.0) * np.sqrt(z)
    return z * np.sign(x)


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------

def _zscore_glm(y, design, contrast):
    """Convert DGEGLM counts to NB z-scores under null model.

    Port of edgeR's .zscoreGLM.
    """
    from .glm_fit import glm_fit
    from .utils import zscore_nbinom

    counts = y['counts'].copy().astype(np.float64)

    # QL scaling
    if y.get('average.ql.dispersion') is not None:
        s2_prior = np.atleast_1d(np.asarray(y.get('s2.prior', 1.0), dtype=np.float64))
        if s2_prior.ndim == 0 or s2_prior.size == 1:
            s2_prior = np.full(counts.shape[0], float(s2_prior.ravel()[0]))
        counts = counts / np.maximum(1.0, s2_prior)[:, np.newaxis]

    design = np.asarray(design, dtype=np.float64)
    p = design.shape[1]

    # Build null design by removing the contrast column
    if isinstance(contrast, (int, np.integer)):
        contrast_idx = int(contrast)
        cols = [i for i in range(p) if i != contrast_idx]
        design0 = design[:, cols]
    else:
        # contrast is a vector - remove last column (after contrastAsCoef)
        design0 = design[:, :-1]

    dispersion = y.get('dispersion', 0.05)
    offset = y.get('offset')
    w = y.get('weights')

    # Fit null model
    fit_null = glm_fit(counts, design=design0, dispersion=dispersion,
                       offset=offset, weights=w, prior_count=0)

    mu = np.maximum(fit_null['fitted.values'], 1e-17)

    # size parameter = 1/dispersion
    disp = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if disp.size == 1:
        disp = np.full(counts.shape[0], float(disp.ravel()[0]))

    # Compute z-scores column by column
    ngenes, nsamples = counts.shape
    z = np.zeros_like(counts)
    for j in range(nsamples):
        z[:, j] = zscore_nbinom(counts[:, j], size=1.0 / disp, mu=mu[:, j])

    return z


def _zscore_dge(y, design, contrast):
    """Convert DGEList counts to NB z-scores under null model.

    Port of edgeR's .zscoreDGE. Fits a null GLM (without contrast column)
    and converts raw counts to standard normal z-scores using the mid-p
    negative binomial quantile residual method.
    """
    from .glm_fit import glm_fit
    from .dgelist import get_dispersion, get_offset
    from .utils import zscore_nbinom
    from .limma_port import contrast_as_coef

    counts = y['counts'].copy().astype(np.float64)
    design = np.asarray(design, dtype=np.float64)
    p = design.shape[1]

    if p < 2:
        raise ValueError("design matrix must have at least two columns")

    # Get dispersion
    dispersion = get_dispersion(y)
    if dispersion is None:
        raise ValueError("Dispersion estimate not found. "
                         "Please estimate dispersions before gene set testing.")

    # Build null design by removing the contrast column
    if isinstance(contrast, (int, np.integer)):
        contrast_idx = int(contrast)
        cols = [i for i in range(p) if i != contrast_idx]
        design0 = design[:, cols]
    else:
        # Contrast is a vector: use contrastAsCoef to reparametrize,
        # then drop the last column
        cac = contrast_as_coef(design, contrast, first=False)
        design_reparametrized = cac['design']
        design0 = design_reparametrized[:, :-1]

    # Get offset from DGEList
    offset = get_offset(y)

    # Fit null model
    fit_null = glm_fit(counts, design=design0, dispersion=dispersion,
                       offset=offset, prior_count=0)

    mu = np.maximum(fit_null['fitted.values'], 1e-17)

    # size parameter = 1/dispersion
    disp = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if disp.size == 1:
        disp = np.full(counts.shape[0], float(disp.ravel()[0]))

    # Compute z-scores column by column
    ngenes, nsamples = counts.shape
    z = np.zeros_like(counts)
    for j in range(nsamples):
        z[:, j] = zscore_nbinom(counts[:, j], size=1.0 / disp, mu=mu[:, j])

    return z


def _resolve_input(y, design, contrast):
    """Resolve input type and return z-score matrix, design, contrast.

    Used by fry, roast, mroast, romer to dispatch DGEList/DGEGLM/matrix.
    """
    is_dgeglm = isinstance(y, dict) and 'coefficients' in y and 'dispersion' in y
    is_dgelist = isinstance(y, dict) and 'counts' in y and 'coefficients' not in y

    if design is None and isinstance(y, dict):
        design = y.get('design')
    if design is None:
        raise ValueError("design matrix must be provided")
    design = np.asarray(design, dtype=np.float64)

    if contrast is None:
        contrast = design.shape[1] - 1

    if is_dgeglm:
        expr = _zscore_glm(y, design=design, contrast=contrast)
    elif is_dgelist:
        expr = _zscore_dge(y, design=design, contrast=contrast)
    else:
        expr = np.asarray(y, dtype=np.float64)

    return expr, design, contrast


def _extract_effects(y, design, contrast):
    """QR decomposition of design to extract contrast effect and residuals.

    Port of limma's .lmEffects (internal).

    Returns
    -------
    dict with:
        unscaledt : ndarray (G,) - unscaled t-statistics (contrast effect)
        U : ndarray (df_residual, G) - residual effects
        sigma2 : ndarray (G,) - residual variances
        df_residual : int - residual degrees of freedom
    """
    y = np.asarray(y, dtype=np.float64)
    G, n = y.shape
    design = np.asarray(design, dtype=np.float64)
    p = design.shape[1]
    df_residual = n - p

    # Reorder design so contrast column is last
    if isinstance(contrast, (int, np.integer)):
        contrast_idx = int(contrast)
        if contrast_idx < p - 1:
            j = [i for i in range(p) if i != contrast_idx] + [contrast_idx]
            design = design[:, j]
    else:
        contrast_vec = np.asarray(contrast, dtype=np.float64)
        if contrast_vec.ndim == 1 and len(contrast_vec) == p:
            nonzero = np.where(contrast_vec != 0)[0]
            if len(nonzero) == 1 and contrast_vec[nonzero[0]] == 1:
                contrast_idx = nonzero[0]
                if contrast_idx < p - 1:
                    j = [i for i in range(p) if i != contrast_idx] + [contrast_idx]
                    design = design[:, j]
            else:
                QR_c = np.linalg.qr(contrast_vec.reshape(-1, 1), mode='complete')
                design = (QR_c[0].T @ design.T).T
                if QR_c[1][0, 0] < 0:
                    design[:, 0] = -design[:, 0]
                design = np.column_stack([design[:, 1:], design[:, 0]])

    # QR decomposition of design
    Q_full, R_full = np.linalg.qr(design, mode='complete')
    effects = Q_full.T @ y.T  # n x G

    unscaledt = effects[p - 1, :]  # contrast row
    # Check sign
    R_reduced = np.linalg.qr(design, mode='reduced')[1]
    if R_reduced[p - 1, p - 1] < 0:
        unscaledt = -unscaledt

    # Residual effects
    U = effects[p:, :]  # (n-p) x G
    sigma2 = np.mean(U ** 2, axis=0)

    return {
        'unscaledt': unscaledt,
        'U': U,
        'sigma2': sigma2,
        'df_residual': df_residual,
    }


# -----------------------------------------------------------------------
# Shared helpers for camera and fry
# -----------------------------------------------------------------------

@njit(cache=True)
def _householder_qty(X, y):
    """Q'y for the Householder QR of X (full column rank, no pivoting).

    Returns Q'y and the sign of R[p-1, p-1]; the reflectors are those of
    R's qr() (LINPACK dqrdc2) and LAPACK, so Q'y matches qr.qty().
    """
    n, p = X.shape
    A = X.copy()
    b = y.copy()
    v = np.empty(n)
    for k in range(p):
        alpha = A[k, k]
        xnorm2 = 0.0
        for i in range(k + 1, n):
            xnorm2 += A[i, k] * A[i, k]
        if xnorm2 == 0.0:
            continue
        beta = -np.copysign(np.sqrt(alpha * alpha + xnorm2), alpha)
        tau = (beta - alpha) / beta
        v[k] = 1.0
        for i in range(k + 1, n):
            v[i] = A[i, k] / (alpha - beta)
        for j in range(k + 1, p):
            w = 0.0
            for i in range(k, n):
                w += v[i] * A[i, j]
            for i in range(k, n):
                A[i, j] -= tau * w * v[i]
        w = 0.0
        for i in range(k, n):
            w += v[i] * b[i]
        for i in range(k, n):
            b[i] -= tau * w * v[i]
        A[k, k] = beta
    return b, np.sign(A[p - 1, p - 1])


@njit(cache=True)
def _weighted_qty_gene(X, y, sw, L_inv, use_block):
    n, p = X.shape
    wX = np.empty((n, p))
    wy = np.empty(n)
    for i in range(n):
        wy[i] = y[i] * sw[i]
        for j in range(p):
            wX[i, j] = X[i, j] * sw[i]
    if use_block:
        wy = L_inv @ wy
        wX = L_inv @ wX
    return _householder_qty(wX, wy)


@njit(parallel=True, cache=True)
def _weighted_qty(X, Y, W, L_inv, use_block):
    """Per-gene Q'(sqrt(w) y) for the weighted design sqrt(w) X."""
    G, n = Y.shape
    effects = np.empty((G, n))
    signs = np.empty(G)
    for g in prange(G):
        eff, s = _weighted_qty_gene(X, Y[g], np.sqrt(W[g]), L_inv, use_block)
        effects[g] = eff
        signs[g] = s
    return effects, signs


def _get_eawp(y):
    """Expression values, weights, design and gene IDs (limma's getEAWP)."""
    if isinstance(y, dict) and 'E' in y:
        E = y['E']
        weights, design = y.get('weights'), y.get('design')
    else:
        E, weights, design = y, None, None
    ids = E.index if isinstance(E, pd.DataFrame) else None
    if isinstance(y, dict) and ids is None and isinstance(y.get('genes'), pd.DataFrame):
        ids = y['genes'].index
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
    return E, weights, design, (None if ids is None else np.asarray(ids))


def _default_design(y, design):
    """design, else y['design'], else model.matrix(~group) for a DGEList."""
    if design is None and isinstance(y, dict):
        design = y.get('design')
        if design is None and 'samples' in y:
            from .utils import _model_matrix_group
            design = _model_matrix_group(np.asarray(y['samples']['group']))
    if design is None:
        raise ValueError("design matrix not specified")
    return np.asarray(design, dtype=np.float64)


def _parse_sets(index, ids, G, default_name):
    """Gene sets as a list of (name, 0-based indices, gene weights or None).

    index is a dict or list of sets, or a single set. A set is a list of
    0-based indices or gene IDs, or a DataFrame whose first column holds
    indices or IDs and whose second column holds gene weights.
    """
    if isinstance(index, pd.DataFrame) or not isinstance(index, (dict, list, tuple)):
        index = {'set1': index}
    elif isinstance(index, (list, tuple)) and len(index) and np.ndim(index[0]) == 0 \
            and not isinstance(index[0], pd.DataFrame):
        index = {'set1': index}
    if isinstance(index, dict):
        items = list(index.items())
    else:
        items = [(f'{default_name}{i + 1}', s) for i, s in enumerate(index)]
    if len(items) == 0:
        raise ValueError("index is empty")

    sets = []
    for i, (name, iset) in enumerate(items):
        iw = None
        if isinstance(iset, pd.DataFrame):
            if iset.shape[1] < 2 or not np.issubdtype(iset.iloc[:, 1].dtype, np.number):
                raise ValueError(f"index {i + 1} is a data.frame but doesn't contain gene weights")
            iw = iset.iloc[:, 1].to_numpy(dtype=np.float64)
            iset = iset.iloc[:, 0].to_numpy()
            if iset.dtype.kind in 'OUS':
                if len(np.unique(iset)) < len(iset):
                    raise ValueError(f"Duplicate gene ids in set {i + 1}")
                pos = {g: k for k, g in enumerate(ids)} if ids is not None else {}
                j = np.array([pos.get(g, -1) for g in iset], dtype=np.int64)
                if np.any(j < 0):
                    warnings.warn(f"index {i + 1} contains gene IDs not in dataset")
                iset, iw = j[j >= 0], iw[j >= 0]
        else:
            iset = np.asarray(iset)
            if iset.dtype.kind in 'OUS':
                iset = np.flatnonzero(np.isin(ids, iset)) if ids is not None else np.zeros(0, int)
        iset = np.asarray(iset, dtype=np.int64)
        if iset.size and (iset.max() >= G or iset.min() < 0):
            raise ValueError(f"set {i + 1} contains indices out of range")
        sets.append((name, iset, iw))
    return sets


def _contrast_last(design, contrast):
    """Rotate or reorder design so the contrast is its last coefficient (camera)."""
    p = design.shape[1]
    contrast = np.atleast_1d(np.asarray(contrast))
    if contrast.size == 1:
        c = int(contrast[0])
        if c < p - 1:
            design = design[:, [j for j in range(p) if j != c] + [c]]
        return design
    Q, R = np.linalg.qr(contrast.astype(np.float64).reshape(-1, 1), mode='complete')
    design = design @ Q
    if R[0, 0] < 0:
        design[:, 0] = -design[:, 0]
    return np.column_stack([design[:, 1:], design[:, 0]])


def _inv_sqrt_cor_times(B, u, rho):
    """limma's .invSqrtCorTimes: premultiply B by the inverse square root of
    the equicorrelation matrix (signed by u) on the rows where u != 0."""
    i = np.flatnonzero(u != 0)
    k = len(i)
    if k < 2:
        return B
    s = u[i]
    sa = np.sqrt(1 - rho)
    sd = np.sqrt(1 + (k - 1) * rho)
    gamma = -rho / (sa * sd * (sa + sd))
    Bs = B[i]
    B = B.copy()
    B[i] = Bs / sa + np.outer(gamma * s, s @ Bs)
    return B


def _bh(p):
    return multipletests(p, method='fdr_bh')[1]


# -----------------------------------------------------------------------
# camera.default
# -----------------------------------------------------------------------

def _camera_default(y, index, design=None, contrast=None, weights=None,
                    use_ranks=False, allow_neg_cor=False, inter_gene_cor=0.01,
                    trend_var=False, sort=True, directional=True):
    """Competitive gene set test. Port of limma 3.99's camera.default."""
    from .limma_port import squeeze_var

    E, ew, edesign, ids = _get_eawp(y)
    G, n = E.shape
    if G < 3:
        raise ValueError("Too few genes in dataset: need at least 3")
    if design is None:
        design = edesign
    if design is None:
        raise ValueError("design matrix not specified")
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != n:
        raise ValueError("row dimension of design matrix must match column dimension of data")
    p = design.shape[1]
    df_residual = n - p
    if df_residual < 1:
        raise ValueError("No residual df: cannot compute t-tests")
    if contrast is None:
        contrast = p - 1
    if weights is None:
        weights = ew

    fixed_cor = not (inter_gene_cor is None or np.isnan(inter_gene_cor))
    if not directional and not use_ranks:
        use_ranks = True
        warnings.warn("Setting `use_ranks=True` for non-directional tests")
    if fixed_cor:
        df_camera = np.inf if use_ranks else G - 2
    else:
        df_camera = min(df_residual, G - 2)

    # Array weights rescale the data; other weights need a fit per gene
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if np.any(weights <= 0):
            raise ValueError("weights must be positive")
        if weights.size == n:
            sw = np.sqrt(weights.ravel())
            E = E * sw
            design = design * sw[:, None]
            weights = None
        elif weights.size == G:
            weights = np.repeat(weights.reshape(-1, 1), n, axis=1)
        if weights is not None and weights.shape != E.shape:
            raise ValueError("weights not conformal with y")

    design = _contrast_last(design, contrast)
    if weights is None:
        if np.linalg.matrix_rank(design) < p:
            raise ValueError("design matrix is not of full rank")
        Q, R = np.linalg.qr(design, mode='complete')
        effects = Q.T @ E.T
        unscaledt = effects[p - 1] * (-1.0 if R[p - 1, p - 1] < 0 else 1.0)
        U = effects[p:]
    else:
        eff, signs = _weighted_qty(np.ascontiguousarray(design), np.ascontiguousarray(E),
                                   np.ascontiguousarray(weights), np.eye(n), False)
        unscaledt = eff[:, p - 1] * np.where(signs < 0, -1.0, 1.0)
        U = eff[:, p:].T

    sigma2 = np.mean(U ** 2, axis=0)
    U = U.T / np.sqrt(np.maximum(sigma2, 1e-8))[:, None]

    A = np.mean(E, axis=1) if trend_var else None
    sv = squeeze_var(sigma2, np.full(G, float(df_residual)), covariate=A)
    modt = unscaledt / np.sqrt(sv['var_post'])
    if use_ranks:
        Stat = modt if directional else np.abs(modt)
    else:
        df_prior = np.atleast_1d(sv['df_prior'])
        df_total = np.minimum(df_residual + df_prior, G * df_residual)
        Stat = _zscore_t_hill(modt, df_total)
        meanStat = np.mean(Stat)
        varStat = np.var(Stat, ddof=1)

    sets = _parse_sets(index, ids, G, 'Set')
    nsets = len(sets)
    NGenes = np.zeros(nsets, dtype=int)
    Correlation = np.full(nsets, np.nan)
    Down = np.ones(nsets)
    Up = np.ones(nsets)
    tstat = np.full(nsets, np.nan)
    for i, (_, iset, iw) in enumerate(sets):
        m = len(iset)
        NGenes[i] = m
        if iw is not None:
            # Weighted gene set: regression of Stat on the gene weights
            if fixed_cor:
                correlation = inter_gene_cor
            elif m > 1:
                Uset = np.sign(iw)[:, None] * U[iset]
                vif = m * np.mean(np.mean(Uset, axis=0) ** 2)
                correlation = (vif - 1) / (m - 1)
            else:
                correlation = np.nan
            Correlation[i] = correlation
            if m:
                x = np.zeros(G)
                x[iset] = iw
                Z = _inv_sqrt_cor_times(np.column_stack([np.ones(G), x, Stat]), np.sign(x), correlation)
                Qz, Rz = np.linalg.qr(Z[:, :2], mode='complete')
                eff = Qz.T @ Z[:, 2]
                s2 = np.mean(eff[2:] ** 2)
                t_stat = np.sign(Rz[1, 1]) * eff[1] / np.sqrt(s2)
                Down[i] = t_dist.cdf(t_stat, G - 2)
                Up[i] = t_dist.sf(t_stat, G - 2)
            continue

        if fixed_cor:
            correlation = inter_gene_cor
            vif = 1 + (m - 1) * correlation
        elif m > 1:
            Uset = U[iset]
            vif = m * np.mean(np.mean(Uset, axis=0) ** 2)
            correlation = (vif - 1) / (m - 1)
        else:
            vif = 1
            correlation = np.nan
        Correlation[i] = correlation
        if use_ranks:
            if not allow_neg_cor:
                correlation = max(0, correlation)
            Down[i], Up[i] = _rank_sum_test_with_correlation(iset, Stat, correlation, df_camera)
        else:
            if not allow_neg_cor:
                vif = max(1, vif)
            m2 = G - m
            delta = G / m2 * (np.mean(Stat[iset]) - meanStat)
            varStatPooled = ((G - 1) * varStat - delta ** 2 * m * m2 / G) / (G - 2)
            tstat[i] = delta / np.sqrt(varStatPooled * (vif / m + 1 / m2))

    # pt() evaluated once for all unweighted parametric sets
    ok = ~np.isnan(tstat)
    if np.any(ok):
        Down[ok] = t_dist.cdf(tstat[ok], df_camera)
        Up[ok] = t_dist.sf(tstat[ok], df_camera)

    tab = pd.DataFrame({'NGenes': NGenes}, index=[s[0] for s in sets])
    if not fixed_cor:
        tab['Correlation'] = Correlation
    if directional:
        tab['Direction'] = np.where(Down < Up, 'Down', 'Up')
        tab['PValue'] = 2 * np.minimum(Down, Up)
    else:
        tab['PValue'] = Up
    tab['FDR'] = _bh(tab['PValue'].to_numpy()) if nsets > 1 else tab['PValue']
    if sort and nsets > 1:
        tab = tab.sort_values('PValue', kind='mergesort')
    return tab


def _rank_sum_test_with_correlation(iset, statistics, correlation, df):
    """Port of limma's rankSumTestWithCorrelation.

    Wilcoxon rank-sum test adjusted for inter-gene correlation,
    using the arcsin-based variance formula from limma.
    """
    n = len(statistics)
    n1 = len(iset)
    n2 = n - n1

    ranks = rankdata(statistics, method='average')
    r1 = ranks[iset]

    # U statistic (R convention: U = n1*n2 + n1*(n1+1)/2 - sum(r1))
    U = n1 * n2 + n1 * (n1 + 1) / 2.0 - np.sum(r1)
    mu = n1 * n2 / 2.0

    # Variance formula using arcsin (matches R's limma exactly)
    if correlation == 0 or n1 == 1:
        sigma2 = n1 * n2 * (n + 1) / 12.0
    else:
        sigma2 = (np.arcsin(1.0) * n1 * n2
                  + np.arcsin(0.5) * n1 * n2 * (n2 - 1)
                  + np.arcsin(correlation / 2.0) * n1 * (n1 - 1) * n2 * (n2 - 1)
                  + np.arcsin((correlation + 1.0) / 2.0) * n1 * (n1 - 1) * n2)
        sigma2 = sigma2 / (2.0 * np.pi)

    # Ties adjustment
    _, nties = np.unique(ranks, return_counts=True)
    if len(nties) < len(ranks):
        adjustment = np.sum(nties * (nties + 1) * (nties - 1)) / (n * (n + 1) * (n - 1))
        sigma2 = sigma2 * (1.0 - adjustment)

    sigma2 = max(sigma2, 1e-15)

    # Continuity correction (matching R)
    z_lower = (U + 0.5 - mu) / np.sqrt(sigma2)
    z_upper = (U - 0.5 - mu) / np.sqrt(sigma2)

    if np.isinf(df):
        p_down = norm_dist.sf(z_upper)  # less = P(T > z_upper)
        p_up = norm_dist.cdf(z_lower)   # greater = P(T < z_lower)
    else:
        p_down = t_dist.sf(z_upper, df)
        p_up = t_dist.cdf(z_lower, df)

    return p_down, p_up


# -----------------------------------------------------------------------
# fry: .lmEffects, fry.default and .fryEffects
# -----------------------------------------------------------------------

def _lm_effects(E, design, contrast, array_weights=None, gene_weights=None,
                weights=None, block=None, correlation=None):
    """Contrast and residual effects per gene. Port of limma's .lmEffects.

    Returns a genes x (n - p + 1) matrix whose first column is the signed
    contrast effect and whose other columns are the residual effects.
    """
    from scipy.linalg import solve_triangular
    from .limma_port import contrast_as_coef

    G, n = E.shape
    p = design.shape[1]
    if n <= p:
        raise ValueError("No residual degrees of freedom")
    if np.any(np.isnan(E)):
        raise ValueError("All y values must be finite and non-NA")
    contrast = np.atleast_1d(np.asarray(contrast))
    if np.all(contrast == 0):
        raise ValueError("contrast all zero")
    if contrast.size == 1:
        c = int(contrast[0])
        X = design[:, [j for j in range(p) if j != c] + [c]] if c < p - 1 else design
    else:
        if contrast.size != p:
            raise ValueError("length of contrast must match column dimension of design")
        X = contrast_as_coef(design, contrast.astype(np.float64), first=False)['design']

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if array_weights is None and weights.size == n:
            array_weights, weights = weights.ravel(), None
        elif gene_weights is None and weights.size == G:
            gene_weights, weights = weights.ravel(), None
    if array_weights is not None:
        array_weights = np.asarray(array_weights, dtype=np.float64)
        if array_weights.size != n:
            raise ValueError("Length of array.weights doesn't match number of arrays")
        if np.any(~(array_weights > 0)):
            raise ValueError("array.weights must be positive")
    if gene_weights is not None:
        gene_weights = np.asarray(gene_weights, dtype=np.float64)
        if gene_weights.size != G:
            raise ValueError("Length of gene.weights doesn't match number of genes")
        if np.any(~(gene_weights > 0)):
            raise ValueError("gene.weights must be positive")
    if weights is not None:
        if weights.shape != (G, n):
            raise ValueError("weights must have same dimensions as y")
        if np.any(~(weights > 0)):
            raise ValueError("weights must be positive")

    if array_weights is not None:
        ws = np.sqrt(array_weights)
        X = X * ws[:, None]
        E = E * ws

    L_inv = np.eye(n)
    if block is not None:
        if correlation is None:
            raise ValueError("correlation must be set")
        block = np.asarray(block)
        if block.size != n:
            raise ValueError("Length of block does not match number of arrays")
        cormatrix = correlation * (block[:, None] == block[None, :])
        np.fill_diagonal(cormatrix, 1.0)
        L = np.linalg.cholesky(cormatrix)
        if weights is None:
            E = solve_triangular(L, E.T, lower=True).T
            X = solve_triangular(L, X, lower=True)
        else:
            L_inv = solve_triangular(L, np.eye(n), lower=True)

    if np.linalg.matrix_rank(X) < p:
        raise ValueError("design must be full column rank")
    if weights is None:
        Q, R = np.linalg.qr(X, mode='complete')
        effects = (Q.T @ E.T).T[:, p - 1:]
        if R[p - 1, p - 1] < 0:
            effects[:, 0] = -effects[:, 0]
    else:
        effects, signs = _weighted_qty(np.ascontiguousarray(X), np.ascontiguousarray(E),
                                       np.ascontiguousarray(weights), L_inv, block is not None)
        effects = effects[:, p - 1:]
        effects[:, 0] *= signs
    if gene_weights is not None:
        effects = np.sqrt(gene_weights)[:, None] * effects
    return effects


def _fry_effects(effects, sets, sort='directional'):
    """Port of limma's .fryEffects, for sets parsed by _parse_sets."""
    nsets = len(sets)
    df_residual = effects.shape[1] - 1
    NGenes = np.zeros(nsets, dtype=int)
    t_stat = np.zeros(nsets)
    PValue_Mixed = np.zeros(nsets)
    for i, (_, iset, iw) in enumerate(sets):
        EffectsSet = effects[iset]
        if iw is not None:
            EffectsSet = iw[:, None] * EffectsSet
        MeanEffectsSet = EffectsSet.mean(axis=0)
        t_stat[i] = MeanEffectsSet[0] / np.sqrt(np.mean(MeanEffectsSet[1:] ** 2))
        NGenes[i] = EffectsSet.shape[0]
        if NGenes[i] > 1:
            A = np.linalg.svd(EffectsSet, compute_uv=False) ** 2
            d1 = len(A)
            d = d1 - 1
            beta_mean = 1 / d1
            beta_var = d / d1 / d1 / (d1 / 2 + 1)
            Fobs = (np.sum(EffectsSet[:, 0] ** 2) - A[d1 - 1]) / (A[0] - A[d1 - 1])
            Frb_mean = (np.sum(A) * beta_mean - A[d1 - 1]) / (A[0] - A[d1 - 1])
            COV = np.full((d1, d1), -beta_var / d)
            np.fill_diagonal(COV, beta_var)
            Frb_var = (A @ COV @ A) / (A[0] - A[d1 - 1]) ** 2
            alphaplusbeta = Frb_mean * (1 - Frb_mean) / Frb_var - 1
            alpha = alphaplusbeta * Frb_mean
            beta = alphaplusbeta - alpha
            PValue_Mixed[i] = beta_dist.sf(Fobs, alpha, beta)

    PValue = 2 * t_dist.sf(np.abs(t_stat), df_residual)
    PValue_Mixed[NGenes == 1] = PValue[NGenes == 1]
    tab = pd.DataFrame({
        'NGenes': NGenes,
        'Direction': np.where(t_stat < 0, 'Down', 'Up'),
        'PValue': PValue,
        'FDR': _bh(PValue) if nsets > 1 else PValue,
        'PValue.Mixed': PValue_Mixed,
        'FDR.Mixed': _bh(PValue_Mixed) if nsets > 1 else PValue_Mixed,
    }, index=[s[0] for s in sets])

    if isinstance(sort, (bool, np.bool_)):
        sort = 'directional' if sort else 'none'
    if sort not in ('directional', 'mixed', 'none'):
        raise ValueError("sort must be 'directional', 'mixed' or 'none'")
    if sort == 'directional':
        o = np.lexsort((PValue_Mixed, -NGenes, PValue))
    elif sort == 'mixed':
        o = np.lexsort((PValue, -NGenes, PValue_Mixed))
    else:
        return tab
    return tab.iloc[o]


def _gauss_legendre_unit(n):
    """Nodes and weights of n-point Gauss-Legendre quadrature on [0, 1]
    (statmod's gauss.quad.prob(n, "uniform"))."""
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def _fry_default(E, index, design, contrast, ids=None, geneid=None, gene_weights=None,
                 standardize='posterior.sd', sort='directional', array_weights=None,
                 weights=None, block=None, correlation=None, trend_var=False,
                 robust=False, winsor_tail_p=(0.05, 0.1)):
    """Port of limma 3.99's fry.default."""
    from scipy.stats import chi2
    from .limma_port import (squeeze_var, _posterior_var, _fit_f_dist,
                             _fit_f_dist_trend, _fit_f_dist_robustly)

    G = E.shape[0]
    if gene_weights is not None:
        gene_weights = np.asarray(gene_weights, dtype=np.float64)
        if gene_weights.size != G:
            raise ValueError("length of gene.weights should equal nrow(y)")
    if block is not None and correlation is None:
        raise ValueError("Intra-block correlation must be specified")
    covariate = E.mean(axis=1) if trend_var else None

    effects = _lm_effects(E, design, contrast, array_weights=array_weights,
                          weights=weights, block=block, correlation=correlation)

    if standardize not in ('none', 'residual.sd', 'posterior.sd', 'p2'):
        raise ValueError("standardize must be 'none', 'residual.sd', 'posterior.sd' or 'p2'")
    if standardize != 'none':
        nodes, qweights = _gauss_legendre_unit(128)
        df_residual = effects.shape[1] - 1
        Eu2max = np.sum((df_residual + 1) * nodes ** df_residual
                        * chi2.ppf(nodes, df=1) * qweights)
        E2 = effects ** 2
        s2_robust = (E2.sum(axis=1) - E2.max(axis=1)) / (df_residual + 1 - Eu2max)
        if standardize == 'p2':
            s2_robust = squeeze_var(s2_robust, 0.92 * df_residual, covariate=covariate,
                                    robust=robust, winsor_tail_p=winsor_tail_p)['var_post']
        elif standardize == 'posterior.sd':
            s2 = E2[:, 1:].mean(axis=1)
            dfr = np.full(G, float(df_residual))
            if robust:
                fit = _fit_f_dist_robustly(s2, dfr, covariate=covariate, winsor_tail_p=winsor_tail_p)
                scale, df_prior = fit['scale'], fit['df2_shrunk']
            elif covariate is None:
                fit = _fit_f_dist(s2, dfr)
                scale, df_prior = fit['s2'], fit['df2']
            else:
                fit = _fit_f_dist_trend(s2, dfr, covariate)
                scale, df_prior = fit['var_prior'], fit['df_prior']
            s2_robust = _posterior_var(s2_robust, np.full(G, 0.92 * df_residual), scale, df_prior)
        effects = effects / np.sqrt(s2_robust)[:, None]

    if geneid is not None:
        ids = np.asarray(geneid)
        if ids.size != G:
            raise ValueError("geneid vector should be of length nrow(y)")
    sets = _parse_sets(index if index is not None else {'set1': np.arange(G)}, ids, G, 'Set')
    if gene_weights is not None:
        sets = [(name, iset, gene_weights[iset] if iw is None else iw) for name, iset, iw in sets]
    return _fry_effects(effects, sets, sort=sort)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def _count_input(y):
    """'glm' for a DGEGLM, 'dge' for a DGEList, None otherwise."""
    if isinstance(y, dict) and 'coefficients' in y and 'dispersion' in y:
        return 'glm'
    if isinstance(y, dict) and 'counts' in y and 'coefficients' not in y:
        return 'dge'
    return None


def _count_zscores(y, kind, design, contrast):
    """NB z-scores of a DGEList/DGEGLM, as a DataFrame when gene IDs are known."""
    z = (_zscore_glm if kind == 'glm' else _zscore_dge)(y, design=design, contrast=contrast)
    ids = None
    if isinstance(y.get('counts'), pd.DataFrame):
        ids = y['counts'].index
    elif isinstance(y.get('genes'), pd.DataFrame):
        ids = y['genes'].index
    return z if ids is None else pd.DataFrame(z, index=ids)


def camera(y, index, design=None, contrast=None, weights=None,
           use_ranks=False, allow_neg_cor=False, inter_gene_cor=0.01,
           sort=True, trend_var=False, directional=True):
    """Competitive gene set test accounting for inter-gene correlation.

    Port of limma 3.99's camera.default and edgeR's camera.DGEList.

    Parameters
    ----------
    y : ndarray, DataFrame, EList-like dict, DGEList-like dict or DGEGLM-like dict
        Log-expression matrix, or an EList (dict with 'E' and optionally
        'weights' and 'design', e.g. from voom). A DGEList or DGEGLM is
        converted to NB z-scores under the null model first, as edgeR does.
    index : dict, list or DataFrame
        Gene sets. A set is a list of 0-based row indices or gene IDs, or
        a DataFrame whose first column holds indices or IDs and whose
        second column holds gene weights (a weighted gene set). A dict
        names the sets; a single set may be given on its own.
    design : ndarray, optional
        Design matrix. Defaults to y['design'], or for a DGEList to a
        design for y['samples']['group'].
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector. Defaults to the last
        column.
    weights : ndarray, optional
        Array weights (length n), gene weights (length G) or observation
        weights (G x n). Defaults to y['weights'] for an EList.
    use_ranks : bool
        Use a rank-based test instead of the parametric one.
    allow_neg_cor : bool
        Allow negative inter-gene correlation.
    inter_gene_cor : float or None
        Fixed inter-gene correlation (default 0.01). None or NaN estimates
        it from each set and adds a Correlation column.
    sort : bool
        Sort results by p-value.
    trend_var : bool
        Use a mean-variance trend when squeezing the variances.
    directional : bool
        If False, test for any change regardless of direction (implies
        use_ranks).

    Returns
    -------
    DataFrame with columns NGenes, (Correlation), Direction, PValue, FDR.
    """
    kind = _count_input(y)
    if kind is not None:
        design = _default_design(y, design)
        if contrast is None:
            contrast = design.shape[1] - 1
        y = _count_zscores(y, kind, design, contrast)
        trend_var = False
    return _camera_default(y, index, design=design, contrast=contrast, weights=weights,
                           use_ranks=use_ranks, allow_neg_cor=allow_neg_cor,
                           inter_gene_cor=inter_gene_cor, trend_var=trend_var,
                           sort=sort, directional=directional)


def fry(y, index=None, design=None, contrast=None, sort='directional', geneid=None,
        gene_weights=None, standardize='posterior.sd', array_weights=None,
        weights=None, block=None, correlation=None, trend_var=False,
        robust=False, winsor_tail_p=(0.05, 0.1)):
    """Fast approximation to roast with infinite rotations.

    Port of limma 3.99's fry.default and edgeR's fry.DGEList. A DGEList or
    DGEGLM is converted to NB z-scores under the null model and tested
    with standardize='none', as edgeR does.

    Parameters
    ----------
    y : ndarray, DataFrame, EList-like dict, DGEList-like dict or DGEGLM-like dict
        Expression data (see camera).
    index : dict, list or DataFrame, optional
        Gene sets, as for camera. Defaults to all genes.
    design : ndarray, optional
        Design matrix. Defaults to y['design'], or for a DGEList to a
        design for y['samples']['group'].
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector. Defaults to the last
        column.
    sort : str or bool
        'directional' (default), 'mixed' or 'none'.
    geneid : array-like, optional
        Gene IDs used to match character gene sets.
    gene_weights : array-like, optional
        Weights applied to the genes of every unweighted set.
    standardize : str
        'posterior.sd' (default), 'residual.sd', 'p2' or 'none'.
    array_weights, weights : array-like, optional
        Array weights, or observation weights (G x n; defaults to
        y['weights'] for an EList).
    block : array-like, optional
        Blocks of correlated samples; needs correlation.
    correlation : float, optional
        Intra-block correlation.
    trend_var, robust : bool
        Trended / robust estimation of the prior variance.
    winsor_tail_p : tuple
        Winsorization tail proportions when robust.

    Returns
    -------
    DataFrame with columns NGenes, Direction, PValue, FDR, PValue.Mixed,
    FDR.Mixed.
    """
    kind = _count_input(y)
    if kind is not None:
        design = _default_design(y, design)
        if contrast is None:
            contrast = design.shape[1] - 1
        y = _count_zscores(y, kind, design, contrast)
        standardize = 'none'
    E, ew, edesign, ids = _get_eawp(y)
    if design is None:
        design = edesign
    if design is None:
        raise ValueError("design matrix not specified")
    design = np.asarray(design, dtype=np.float64)
    if design.shape[0] != E.shape[1]:
        raise ValueError("row dimension of design matrix must match column dimension of data")
    if contrast is None:
        contrast = design.shape[1] - 1
    if weights is None:
        weights = ew
    return _fry_default(E, index, design, contrast, ids=ids, geneid=geneid,
                        gene_weights=gene_weights, standardize=standardize, sort=sort,
                        array_weights=array_weights, weights=weights, block=block,
                        correlation=correlation, trend_var=trend_var, robust=robust,
                        winsor_tail_p=winsor_tail_p)


def roast(y, index, design=None, contrast=None, nrot=999,
          set_statistic='mean', sort=True):
    """Rotation gene set test for a single or multiple gene sets.

    Port of edgeR's roast.DGEList → limma's roast.default.

    For DGEList/DGEGLM input, counts are first converted to NB z-scores,
    then roast is applied with var.prior=1, df.prior=Inf (since z-scores
    are already standardized).

    Parameters
    ----------
    y : ndarray, DGEList-like dict, or DGEGLM-like dict
        Expression data.
    index : dict, list of lists, or list of ints
        Gene set indices (0-based). If dict or list of lists, tests first set.
        If list of ints, treats as single gene set.
    design : ndarray, optional
        Design matrix.
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector.
    nrot : int
        Number of rotations (default 999).
    set_statistic : str
        'mean' (default), 'floormean', or 'mean50'.
    sort : bool
        Sort results by p-value.

    Returns
    -------
    DataFrame with columns Active.Prop, P.Value for Down/Up/UpOrDown/Mixed.
    """
    expr, design, contrast = _resolve_input(y, design, contrast)

    # Handle index format - roast tests a single gene set
    if isinstance(index, dict):
        first_key = list(index.keys())[0]
        idx = np.asarray(index[first_key], dtype=int)
    elif isinstance(index, list):
        if len(index) > 0 and isinstance(index[0], (list, np.ndarray)):
            idx = np.asarray(index[0], dtype=int)
        else:
            idx = np.asarray(index, dtype=int)
    else:
        idx = np.asarray(index, dtype=int)

    eff = _extract_effects(expr, design, contrast)
    unscaledt = eff['unscaledt']
    U = eff['U']
    df_residual = eff['df_residual']
    G = len(unscaledt)

    # For DGEList z-scores: var.prior=1, df.prior=Inf => var_post=1
    # So modt = unscaledt / 1 = unscaledt
    modt = unscaledt.copy()

    # Compute set statistics for observed data
    m = len(idx)
    t_set = modt[idx]

    # Active proportions
    p_thresh = 0.05
    # Two-sided p-values for each gene
    gene_pvals = 2 * t_dist.sf(np.abs(modt), df_residual)
    active_down = np.sum((gene_pvals[idx] < p_thresh) & (modt[idx] < 0)) / m
    active_up = np.sum((gene_pvals[idx] < p_thresh) & (modt[idx] > 0)) / m

    # Observed set statistics
    obs_mean_up = np.mean(t_set)
    obs_mean_down = -obs_mean_up
    obs_mean_mixed = np.mean(np.abs(t_set))

    # Rotation loop
    count_up = 0
    count_down = 0
    count_upordown = 0
    count_mixed = 0

    rng = np.random.default_rng()
    for _ in range(nrot):
        # Random rotation in the residual space
        # Generate random unit vector in R^(df_residual)
        rand_vec = rng.standard_normal(df_residual)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)

        # Rotated residuals projected onto random direction
        rotated_resid = rand_vec @ U  # (G,)

        # Rotated moderated t: combine original contrast effect direction
        # with rotated residual (simulating rotation in the space)
        # Under the rotation framework, we rotate the entire effects space
        # For DGEList with var.prior=1: rotated modt = Q_contrast @ rotated_effects
        rot_t = rotated_resid  # Since var_post=1, this is already the statistic

        rot_t_set = rot_t[idx]
        rot_mean_up = np.mean(rot_t_set)
        rot_mean_down = -rot_mean_up
        rot_mean_mixed = np.mean(np.abs(rot_t_set))

        if rot_mean_up >= obs_mean_up:
            count_up += 1
        if rot_mean_down >= obs_mean_down:
            count_down += 1
        if max(rot_mean_up, rot_mean_down) >= max(obs_mean_up, obs_mean_down):
            count_upordown += 1
        if rot_mean_mixed >= obs_mean_mixed:
            count_mixed += 1

    # P-values
    p_up = (count_up + 1) / (nrot + 1)
    p_down = (count_down + 1) / (nrot + 1)
    p_upordown = (count_upordown + 1) / (nrot + 1)
    p_mixed = (count_mixed + 1) / (nrot + 1)

    result = pd.DataFrame({
        'Active.Prop': [active_down, active_up, max(active_down, active_up), np.nan],
        'P.Value': [p_down, p_up, p_upordown, p_mixed],
    }, index=['Down', 'Up', 'UpOrDown', 'Mixed'])

    # Add ngenes as metadata
    result.attrs['ngenes'] = m

    return result


def mroast(y, index, design=None, contrast=None, nrot=999,
           set_statistic='mean', adjust_method='BH', midp=True, sort=True):
    """Rotation gene set test for multiple gene sets.

    Port of edgeR's mroast.DGEList → limma's mroast.default.

    Tests multiple gene sets simultaneously using shared rotations for
    proper FDR correction.

    Parameters
    ----------
    y : ndarray, DGEList-like dict, or DGEGLM-like dict
        Expression data.
    index : dict or list of lists
        Gene set indices (0-based).
    design : ndarray, optional
        Design matrix.
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector.
    nrot : int
        Number of rotations (default 999).
    set_statistic : str
        'mean' (default), 'floormean', or 'mean50'.
    adjust_method : str
        P-value adjustment method (default 'BH').
    midp : bool
        Use mid-p adjustment (default True).
    sort : bool
        Sort results by p-value.

    Returns
    -------
    DataFrame with columns NGenes, PropDown, PropUp, Direction, PValue, FDR,
    PValue.Mixed, FDR.Mixed.
    """
    expr, design, contrast = _resolve_input(y, design, contrast)

    # Convert index format
    if isinstance(index, dict):
        set_names = list(index.keys())
        set_indices = [np.asarray(v, dtype=int) for v in index.values()]
    elif isinstance(index, list):
        set_names = [f'Set{i+1}' for i in range(len(index))]
        set_indices = [np.asarray(v, dtype=int) for v in index]
    else:
        raise ValueError("index must be a dict or list of lists")

    nsets = len(set_names)

    eff = _extract_effects(expr, design, contrast)
    unscaledt = eff['unscaledt']
    U = eff['U']
    df_residual = eff['df_residual']
    G = len(unscaledt)

    # For DGEList z-scores: var.prior=1, df.prior=Inf => var_post=1
    modt = unscaledt.copy()

    # Compute observed statistics and proportions for each set
    p_thresh = 0.05
    gene_pvals = 2 * t_dist.sf(np.abs(modt), df_residual)

    obs_up = np.zeros(nsets)
    obs_down = np.zeros(nsets)
    obs_mixed = np.zeros(nsets)
    prop_down = np.zeros(nsets)
    prop_up = np.zeros(nsets)
    set_sizes = np.zeros(nsets, dtype=int)

    for s in range(nsets):
        idx = set_indices[s]
        m = len(idx)
        set_sizes[s] = m
        t_set = modt[idx]
        obs_up[s] = np.mean(t_set)
        obs_down[s] = -obs_up[s]
        obs_mixed[s] = np.mean(np.abs(t_set))
        prop_down[s] = np.sum((gene_pvals[idx] < p_thresh) & (modt[idx] < 0)) / m
        prop_up[s] = np.sum((gene_pvals[idx] < p_thresh) & (modt[idx] > 0)) / m

    # Shared rotation loop
    count_up = np.zeros(nsets)
    count_down = np.zeros(nsets)
    count_mixed = np.zeros(nsets)

    rng = np.random.default_rng()
    for _ in range(nrot):
        rand_vec = rng.standard_normal(df_residual)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)
        rot_t = rand_vec @ U  # (G,)

        for s in range(nsets):
            idx = set_indices[s]
            rot_t_set = rot_t[idx]
            rot_mean = np.mean(rot_t_set)

            if rot_mean >= obs_up[s]:
                count_up[s] += 1
            if -rot_mean >= obs_down[s]:
                count_down[s] += 1
            if np.mean(np.abs(rot_t_set)) >= obs_mixed[s]:
                count_mixed[s] += 1

    # P-values
    if midp:
        p_up_vals = (count_up + 0.5) / (nrot + 1)
        p_down_vals = (count_down + 0.5) / (nrot + 1)
        p_mixed_vals = (count_mixed + 0.5) / (nrot + 1)
    else:
        p_up_vals = (count_up + 1) / (nrot + 1)
        p_down_vals = (count_down + 1) / (nrot + 1)
        p_mixed_vals = (count_mixed + 1) / (nrot + 1)

    # Two-sided directional p-value and direction
    p_dir = np.minimum(2 * np.minimum(p_up_vals, p_down_vals), 1.0)
    directions = np.where(p_up_vals < p_down_vals, 'Up', 'Down')

    # FDR correction
    method_map = {'BH': 'fdr_bh', 'bonferroni': 'bonferroni',
                  'holm': 'holm', 'hochberg': 'simes-hochberg',
                  'BY': 'fdr_by', 'fdr': 'fdr_bh'}
    sm_method = method_map.get(adjust_method, 'fdr_bh')

    if nsets > 1:
        _, fdr_dir, _, _ = multipletests(p_dir, method=sm_method)
        _, fdr_mixed, _, _ = multipletests(p_mixed_vals, method=sm_method)
    else:
        fdr_dir = p_dir
        fdr_mixed = p_mixed_vals

    result_df = pd.DataFrame({
        'NGenes': set_sizes,
        'PropDown': prop_down,
        'PropUp': prop_up,
        'Direction': directions,
        'PValue': p_dir,
        'FDR': fdr_dir,
        'PValue.Mixed': p_mixed_vals,
        'FDR.Mixed': fdr_mixed,
    }, index=set_names)

    if sort and nsets > 1:
        result_df = result_df.sort_values('PValue')

    return result_df


def romer(y, index, design=None, contrast=None, nrot=9999):
    """Rank-based rotation gene set enrichment test.

    Port of edgeR's romer.DGEList → limma's romer.default.

    For DGEList/DGEGLM input, counts are first converted to NB z-scores,
    then romer is applied. Unlike roast/mroast/fry, romer lets squeezeVar
    estimate its own variance prior from the z-score data.

    Parameters
    ----------
    y : ndarray, DGEList-like dict, or DGEGLM-like dict
        Expression data.
    index : dict or list of lists
        Gene set indices (0-based).
    design : ndarray, optional
        Design matrix.
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector.
    nrot : int
        Number of rotations (default 9999).

    Returns
    -------
    DataFrame with columns NGenes, Up, Down, Mixed (p-values).
    """
    from .limma_port import squeeze_var

    expr, design, contrast = _resolve_input(y, design, contrast)

    # Convert index format
    if isinstance(index, dict):
        set_names = list(index.keys())
        set_indices = [np.asarray(v, dtype=int) for v in index.values()]
    elif isinstance(index, list):
        set_names = [f'Set{i+1}' for i in range(len(index))]
        set_indices = [np.asarray(v, dtype=int) for v in index]
    else:
        raise ValueError("index must be a dict or list of lists")

    nsets = len(set_names)

    eff = _extract_effects(expr, design, contrast)
    unscaledt = eff['unscaledt']
    U = eff['U']
    sigma2 = eff['sigma2']
    df_residual = eff['df_residual']
    G = len(unscaledt)

    # squeezeVar to estimate prior (romer does its own variance moderation)
    sv = squeeze_var(sigma2, np.full(G, float(df_residual)))
    var_post = sv['var_post']
    df_prior_val = sv['df_prior']

    # Moderated t-statistics
    sd_post = np.sqrt(np.maximum(var_post, 1e-15))
    modt = unscaledt / sd_post

    # Shrink residuals (as R's romer does with shrink.resid=TRUE)
    if np.isscalar(df_prior_val):
        dp = float(df_prior_val)
    else:
        dp = float(np.median(df_prior_val))
    s0 = np.sqrt(np.maximum(sv.get('var_prior', 1.0), 1e-15))
    if np.isscalar(s0):
        s0 = float(s0)
    else:
        s0 = float(np.median(s0))

    # Shrink residuals: U_shrunk = U * s0 / sd_unshrunk
    sd_unshrunk = np.sqrt(np.maximum(sigma2, 1e-15))
    shrink_factor = s0 / np.maximum(sd_unshrunk, 1e-15)
    U_shrunk = U * shrink_factor[np.newaxis, :]

    # Compute ranks for observed data
    # Up: high t -> high rank (ascending ranks)
    # Down: low t -> high rank (descending ranks)
    # Mixed: high |t| -> high rank
    up_ranks = rankdata(modt)
    down_ranks = rankdata(-modt)
    mixed_ranks = rankdata(np.abs(modt))

    # Observed mean ranks per set
    obs_up = np.zeros(nsets)
    obs_down = np.zeros(nsets)
    obs_mixed = np.zeros(nsets)
    set_sizes = np.zeros(nsets, dtype=int)

    for s in range(nsets):
        idx = set_indices[s]
        m = len(idx)
        set_sizes[s] = m
        obs_up[s] = np.mean(up_ranks[idx])
        obs_down[s] = np.mean(down_ranks[idx])
        obs_mixed[s] = np.mean(mixed_ranks[idx])

    # Rotation loop
    count_up = np.zeros(nsets)
    count_down = np.zeros(nsets)
    count_mixed = np.zeros(nsets)

    rng = np.random.default_rng()
    for _ in range(nrot):
        # Random rotation in residual space
        rand_vec = rng.standard_normal(df_residual)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)

        # Rotated statistics
        rot_resid = rand_vec @ U_shrunk  # (G,)
        rot_t = rot_resid / sd_post  # Approximate rotated moderated t

        # Compute ranks
        rot_up_ranks = rankdata(rot_t)
        rot_down_ranks = rankdata(-rot_t)
        rot_mixed_ranks = rankdata(np.abs(rot_t))

        for s in range(nsets):
            idx = set_indices[s]
            if np.mean(rot_up_ranks[idx]) >= obs_up[s]:
                count_up[s] += 1
            if np.mean(rot_down_ranks[idx]) >= obs_down[s]:
                count_down[s] += 1
            if np.mean(rot_mixed_ranks[idx]) >= obs_mixed[s]:
                count_mixed[s] += 1

    # P-values
    p_up = (count_up + 1) / (nrot + 1)
    p_down = (count_down + 1) / (nrot + 1)
    p_mixed = (count_mixed + 1) / (nrot + 1)

    result_df = pd.DataFrame({
        'NGenes': set_sizes,
        'Up': p_up,
        'Down': p_down,
        'Mixed': p_mixed,
    }, index=set_names)

    return result_df


def goana(de, species='Hs', **kwargs):
    """Gene ontology enrichment analysis using g:Profiler.

    Wraps the gprofiler-official Python package for GO enrichment.
    Requires: pip install gprofiler-official

    Parameters
    ----------
    de : dict (DGELRT/DGEExact) or list
        If DGELRT/DGEExact dict (has 'table'), significant genes are extracted.
        If list, used directly as gene identifiers.
    species : str
        Species code. 'Hs' for human, 'Mm' for mouse, etc.
    **kwargs
        Additional arguments passed to GProfiler.profile().

    Returns
    -------
    DataFrame with GO enrichment results.
    """
    try:
        from gprofiler import GProfiler
    except ImportError:
        warnings.warn(
            "goana() requires gprofiler-official. Install with:\n"
            "  pip install gprofiler-official\n"
            "Then:\n"
            "  from gprofiler import GProfiler\n"
            "  gp = GProfiler(return_dataframe=True)\n"
            "  result = gp.profile(organism='hsapiens', query=gene_list)")
        return pd.DataFrame()

    # Map species codes
    species_map = {
        'Hs': 'hsapiens', 'Mm': 'mmusculus', 'Rn': 'rnorvegicus',
        'Dm': 'dmelanogaster', 'Sc': 'scerevisiae', 'Ce': 'celegans',
        'Dr': 'drerio',
    }
    organism = species_map.get(species, species)

    # Extract gene list
    if isinstance(de, dict) and 'table' in de:
        table = de['table']
        if isinstance(table, pd.DataFrame):
            sig = table[table['PValue'] < 0.05] if 'PValue' in table.columns else table
            gene_list = list(sig.index)
        else:
            gene_list = []
    elif isinstance(de, (list, np.ndarray)):
        gene_list = list(de)
    else:
        warnings.warn("goana: cannot extract gene list from input. "
                       "Provide a DGELRT/DGEExact dict or a list of gene IDs.")
        return pd.DataFrame()

    if len(gene_list) == 0:
        warnings.warn("goana: no genes to test")
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    sources = kwargs.pop('sources', ['GO:BP', 'GO:MF', 'GO:CC'])
    result = gp.profile(organism=organism, query=gene_list,
                        sources=sources, **kwargs)
    return result


def kegga(de, species='Hs', **kwargs):
    """KEGG pathway enrichment analysis using g:Profiler.

    Wraps the gprofiler-official Python package for KEGG enrichment.
    Requires: pip install gprofiler-official

    Parameters
    ----------
    de : dict (DGELRT/DGEExact) or list
        If DGELRT/DGEExact dict (has 'table'), significant genes are extracted.
        If list, used directly as gene identifiers.
    species : str
        Species code. 'Hs' for human, 'Mm' for mouse, etc.
    **kwargs
        Additional arguments passed to GProfiler.profile().

    Returns
    -------
    DataFrame with KEGG enrichment results.
    """
    try:
        from gprofiler import GProfiler
    except ImportError:
        warnings.warn(
            "kegga() requires gprofiler-official. Install with:\n"
            "  pip install gprofiler-official\n"
            "Then:\n"
            "  from gprofiler import GProfiler\n"
            "  gp = GProfiler(return_dataframe=True)\n"
            "  result = gp.profile(organism='hsapiens', query=gene_list, "
            "sources=['KEGG'])")
        return pd.DataFrame()

    species_map = {
        'Hs': 'hsapiens', 'Mm': 'mmusculus', 'Rn': 'rnorvegicus',
        'Dm': 'dmelanogaster', 'Sc': 'scerevisiae', 'Ce': 'celegans',
        'Dr': 'drerio',
    }
    organism = species_map.get(species, species)

    # Extract gene list
    if isinstance(de, dict) and 'table' in de:
        table = de['table']
        if isinstance(table, pd.DataFrame):
            sig = table[table['PValue'] < 0.05] if 'PValue' in table.columns else table
            gene_list = list(sig.index)
        else:
            gene_list = []
    elif isinstance(de, (list, np.ndarray)):
        gene_list = list(de)
    else:
        warnings.warn("kegga: cannot extract gene list from input. "
                       "Provide a DGELRT/DGEExact dict or a list of gene IDs.")
        return pd.DataFrame()

    if len(gene_list) == 0:
        warnings.warn("kegga: no genes to test")
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    result = gp.profile(organism=organism, query=gene_list,
                        sources=['KEGG'], **kwargs)
    return result
