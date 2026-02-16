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
                QR_c = np.linalg.qr(contrast_vec.reshape(-1, 1))
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
# camera.default
# -----------------------------------------------------------------------

def _camera_default(y, index, design, contrast, weights=None,
                    use_ranks=False, allow_neg_cor=False, inter_gene_cor=0.01,
                    trend_var=False, sort=True):
    """Standard camera test. Port of limma's camera.default."""
    from .limma_port import squeeze_var

    y = np.asarray(y, dtype=np.float64)
    G, n = y.shape

    if design is None:
        design = np.ones((n, 1))
    design = np.asarray(design, dtype=np.float64)
    p = design.shape[1]
    df_residual = n - p

    fixed_cor = inter_gene_cor is not None and not (
        isinstance(inter_gene_cor, float) and np.isnan(inter_gene_cor))

    if fixed_cor:
        if use_ranks:
            df_camera = np.inf
        else:
            df_camera = G - 2
    else:
        df_camera = min(df_residual, G - 2)

    # Handle contrast: reorder design so contrast column is last
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
                QR_c = np.linalg.qr(contrast_vec.reshape(-1, 1))
                design = (QR_c[0].T @ design.T).T
                if QR_c[1][0, 0] < 0:
                    design[:, 0] = -design[:, 0]
                design = np.column_stack([design[:, 1:], design[:, 0]])

    # QR decomposition of design
    Q_full, R_full = np.linalg.qr(design, mode='complete')
    effects = Q_full.T @ y.T  # n x G

    unscaledt = effects[p - 1, :]
    R_reduced = np.linalg.qr(design, mode='reduced')[1]
    if R_reduced[p - 1, p - 1] < 0:
        unscaledt = -unscaledt

    # Residual effects
    U = effects[p:, :]  # (n-p) x G
    sigma2 = np.mean(U ** 2, axis=0)

    # Normalize residuals for correlation estimation
    U_norm = (U / np.sqrt(np.maximum(sigma2, 1e-8))).T  # G x (n-p)

    # squeezeVar
    A = np.mean(y, axis=1) if trend_var else None
    sv = squeeze_var(sigma2, np.full(G, float(df_residual)), covariate=A)
    var_post = sv['var_post']
    df_prior_val = sv['df_prior']

    modt = unscaledt / np.sqrt(np.maximum(var_post, 1e-15))

    if use_ranks:
        Stat = modt.copy()
    else:
        # zscoreT: convert moderated t to z-scores using Hill's approximation
        # (matches R's limma: zscoreT(modt, df=df.total, approx=TRUE, method="hill"))
        if np.isscalar(df_prior_val) or (hasattr(df_prior_val, 'size') and df_prior_val.size == 1):
            dp = float(np.ravel(df_prior_val)[0])
        else:
            dp = float(np.median(df_prior_val))
        df_total = min(df_residual + dp, G * df_residual)
        Stat = _zscore_t_hill(modt, df_total)
        Stat = np.where(np.isfinite(Stat), Stat, 0.0)

    # Convert index format
    if isinstance(index, dict):
        set_names = list(index.keys())
        set_indices = list(index.values())
    elif isinstance(index, list):
        set_names = [f'Set{i+1}' for i in range(len(index))]
        set_indices = index
    else:
        raise ValueError("index must be a dict or list of lists")

    nsets = len(set_names)

    if not use_ranks:
        meanStat = np.mean(Stat)
        varStat = np.var(Stat, ddof=1)

    results = []
    for s_idx in range(nsets):
        idx = np.asarray(set_indices[s_idx], dtype=int)
        StatInSet = Stat[idx]
        m = len(StatInSet)
        m2 = G - m

        if fixed_cor:
            correlation = inter_gene_cor
            vif = 1 + (m - 1) * correlation
        else:
            if m > 1:
                Uset = U_norm[idx, :]
                vif = m * np.mean(np.mean(Uset, axis=0) ** 2)
                correlation = (vif - 1) / (m - 1)
            else:
                vif = 1
                correlation = np.nan

        if use_ranks:
            if not allow_neg_cor:
                correlation = max(0, correlation)
            p_down, p_up = _rank_sum_test_with_correlation(
                idx, Stat, correlation, df_camera)
        else:
            if not allow_neg_cor:
                vif = max(1.0, vif)
            meanStatInSet = np.mean(StatInSet)
            delta = G / m2 * (meanStatInSet - meanStat)
            varStatPooled = ((G - 1) * varStat - delta ** 2 * m * m2 / G) / (G - 2)
            varStatPooled = max(varStatPooled, 1e-15)
            two_sample_t = delta / np.sqrt(varStatPooled * (vif / m + 1.0 / m2))
            p_down = t_dist.cdf(two_sample_t, df_camera)
            p_up = t_dist.sf(two_sample_t, df_camera)

        p_two = 2 * min(p_down, p_up)
        direction = 'Up' if p_up < p_down else 'Down'

        results.append({
            'NGenes': m,
            'Direction': direction,
            'PValue': p_two
        })

    df = pd.DataFrame(results, index=set_names)
    if nsets > 1:
        _, fdr, _, _ = multipletests(df['PValue'].values, method='fdr_bh')
        df['FDR'] = fdr

    if sort and nsets > 1:
        df = df.sort_values('PValue')

    return df


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
    unique_ranks = np.unique(ranks)
    if len(unique_ranks) < len(ranks):
        nties = np.array([np.sum(ranks == r) for r in unique_ranks])
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
# Public API
# -----------------------------------------------------------------------

def camera(y, index, design=None, contrast=None, weights=None,
           use_ranks=False, allow_neg_cor=False, inter_gene_cor=0.01,
           sort=True):
    """Competitive gene set test accounting for inter-gene correlation.

    Port of edgeR's camera (camera.DGEList + camera.DGEGLM + camera.default).

    Parameters
    ----------
    y : ndarray, DGEList-like dict, or DGEGLM-like dict
        If DGEGLM (has 'coefficients' and 'dispersion'), counts are converted
        to NB z-scores under the null model before testing.
        If DGEList (has 'counts' but no 'coefficients'), counts are converted
        to NB z-scores via _zscore_dge (matching R's camera.DGEList).
        If ndarray, used directly as expression matrix.
    index : dict or list of lists
        Gene set indices. If dict, keys are set names and values are
        lists of gene indices (0-based).
    design : ndarray, optional
        Design matrix.
    contrast : int or ndarray, optional
        Column index (0-based) or contrast vector.
    weights : ndarray, optional
        Gene weights.
    use_ranks : bool
        Use rank-based test.
    allow_neg_cor : bool
        Allow negative inter-gene correlation.
    inter_gene_cor : float
        Fixed inter-gene correlation to use (default 0.01).
    sort : bool
        Sort results by p-value.

    Returns
    -------
    DataFrame with columns NGenes, Direction, PValue, FDR.
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
        return _camera_default(expr, index, design=design, contrast=contrast,
                               weights=weights, use_ranks=use_ranks,
                               allow_neg_cor=allow_neg_cor,
                               inter_gene_cor=inter_gene_cor,
                               trend_var=False, sort=sort)
    elif is_dgelist:
        expr = _zscore_dge(y, design=design, contrast=contrast)
        return _camera_default(expr, index, design=design, contrast=contrast,
                               weights=weights, use_ranks=use_ranks,
                               allow_neg_cor=allow_neg_cor,
                               inter_gene_cor=inter_gene_cor,
                               trend_var=False, sort=sort)
    else:
        expr = np.asarray(y, dtype=np.float64)
        return _camera_default(expr, index, design=design, contrast=contrast,
                               weights=weights, use_ranks=use_ranks,
                               allow_neg_cor=allow_neg_cor,
                               inter_gene_cor=inter_gene_cor,
                               trend_var=False, sort=sort)


def fry(y, index, design=None, contrast=None, sort=True):
    """Fast analytical gene set test (rotation-free).

    Port of edgeR's fry.DGEList → limma's fry.default.

    For DGEList/DGEGLM input, counts are first converted to NB z-scores,
    then fry is applied with standardize="none" (no re-standardization).

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
    sort : bool
        Sort results by p-value.

    Returns
    -------
    DataFrame with columns NGenes, Direction, PValue, FDR, PValue.Mixed, FDR.Mixed.
    """
    expr, design, contrast = _resolve_input(y, design, contrast)
    eff = _extract_effects(expr, design, contrast)

    unscaledt = eff['unscaledt']
    U = eff['U']
    df_residual = eff['df_residual']
    G = len(unscaledt)
    neffects = df_residual + 1  # contrast + residuals

    # For DGEList input (z-scores), standardize="none":
    # Effects matrix is used directly without squeezeVar.
    # This matches R's fry.DGEList → fry(standardize="none")

    # Build the full effects matrix: G × neffects
    # Column 0 = contrast effect, columns 1..df_residual = residual effects
    # In our representation: unscaledt is (G,), U is (df_residual, G)
    # R's .fryEffects works on the effects matrix directly.

    # Convert index format
    if isinstance(index, dict):
        set_names = list(index.keys())
        set_indices = list(index.values())
    elif isinstance(index, list):
        set_names = [f'Set{i+1}' for i in range(len(index))]
        set_indices = index
    else:
        raise ValueError("index must be a dict or list of lists")

    nsets = len(set_names)
    t_stat_arr = np.zeros(nsets)
    p_mixed_arr = np.zeros(nsets)
    ngenes_arr = np.zeros(nsets, dtype=int)

    for s_idx in range(nsets):
        idx = np.asarray(set_indices[s_idx], dtype=int)
        m = len(idx)
        ngenes_arr[s_idx] = m

        # Build EffectsSet: m × neffects (genes × effects)
        # Column 0 = contrast, columns 1: = residuals
        effects_set = np.column_stack([
            unscaledt[idx].reshape(-1, 1),
            U[:, idx].T
        ])  # m × (df_residual + 1)

        # --- Directional test (matching R's .fryEffects) ---
        # Average effects across genes in the set
        mean_effects = np.mean(effects_set, axis=0)  # (neffects,)
        # t-statistic: mean contrast effect / sqrt(mean squared residual effects)
        mean_resid_sq = np.mean(mean_effects[1:] ** 2)
        if mean_resid_sq > 1e-30:
            t_stat_arr[s_idx] = mean_effects[0] / np.sqrt(mean_resid_sq)
        else:
            t_stat_arr[s_idx] = 0.0

        # --- Mixed test (SVD-based, matching R's .fryEffects) ---
        if m > 1:
            svd_vals = np.linalg.svd(effects_set, compute_uv=False)
            A = svd_vals ** 2  # squared singular values
            d1 = len(A)
            d = d1 - 1

            if d > 0 and A[0] > A[-1] + 1e-15:
                beta_mean = 1.0 / d1
                beta_var = d / (d1 * d1 * (d1 / 2.0 + 1.0))

                Fobs = (np.sum(effects_set[:, 0] ** 2) - A[-1]) / (A[0] - A[-1])
                Frb_mean = (np.sum(A) * beta_mean - A[-1]) / (A[0] - A[-1])

                COV = np.full((d1, d1), -beta_var / d)
                np.fill_diagonal(COV, beta_var)
                Frb_var = float(A @ COV @ A) / (A[0] - A[-1]) ** 2

                if Frb_var > 1e-30 and Frb_mean > 0 and Frb_mean < 1:
                    alphaplusbeta = Frb_mean * (1.0 - Frb_mean) / Frb_var - 1.0
                    alpha = alphaplusbeta * Frb_mean
                    beta_param = alphaplusbeta - alpha
                    if alpha > 0 and beta_param > 0:
                        p_mixed_arr[s_idx] = beta_dist.sf(Fobs, alpha, beta_param)
                    else:
                        p_mixed_arr[s_idx] = 1.0
                else:
                    p_mixed_arr[s_idx] = 1.0
            else:
                p_mixed_arr[s_idx] = 1.0
        else:
            p_mixed_arr[s_idx] = 0.0  # will be overwritten below

    # Directional p-values (matching R: 2 * pt(-abs(t.stat), df=df.residual))
    p_dir = 2.0 * t_dist.sf(np.abs(t_stat_arr), df_residual)

    # Direction
    directions = np.where(t_stat_arr >= 0, 'Up', 'Down')

    # For single-gene sets, mixed p-value = directional p-value (matching R)
    p_mixed_arr[ngenes_arr == 1] = p_dir[ngenes_arr == 1]

    results = []
    for s_idx in range(nsets):
        results.append({
            'NGenes': ngenes_arr[s_idx],
            'Direction': directions[s_idx],
            'PValue': p_dir[s_idx],
            'PValue.Mixed': p_mixed_arr[s_idx],
        })

    result_df = pd.DataFrame(results, index=set_names)

    # FDR correction
    if nsets > 1:
        _, fdr, _, _ = multipletests(result_df['PValue'].values, method='fdr_bh')
        result_df['FDR'] = fdr
        _, fdr_mixed, _, _ = multipletests(result_df['PValue.Mixed'].values, method='fdr_bh')
        result_df['FDR.Mixed'] = fdr_mixed
    else:
        result_df['FDR'] = result_df['PValue'].values
        result_df['FDR.Mixed'] = result_df['PValue.Mixed'].values

    # Reorder columns
    result_df = result_df[['NGenes', 'Direction', 'PValue', 'FDR', 'PValue.Mixed', 'FDR.Mixed']]

    if sort and nsets > 1:
        result_df = result_df.sort_values('PValue')

    return result_df


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
