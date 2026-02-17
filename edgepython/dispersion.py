# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Dispersion estimation for edgePython.

Port of edgeR's estimateDisp, estimateCommonDisp, estimateTagwiseDisp,
estimateTrendedDisp, estimateGLMCommonDisp, estimateGLMTrendedDisp,
estimateGLMTagwiseDisp, and WLEB.
"""

import numpy as np
import warnings
from scipy.optimize import minimize_scalar

from .expression import ave_log_cpm
from .utils import (expand_as_matrix, moving_average_by_col, cut_with_min_n,
                    drop_empty_levels, systematic_subset)
from .smoothing import locfit_by_col, loess_by_col
from .limma_port import squeeze_var, choose_lowess_span
from .dispersion_lowlevel import (
    adjusted_profile_lik, adjusted_profile_lik_grid, maximize_interpolant,
    cond_log_lik_der_delta, common_cond_log_lik_der_delta,
    disp_cox_reid, disp_cox_reid_interpolate_tagwise,
    disp_cox_reid_spline_trend, disp_cox_reid_power_trend,
    disp_bin_trend, disp_pearson, disp_deviance
)


def estimate_disp(y, design=None, group=None, lib_size=None, offset=None,
                  prior_df=None, trend_method='locfit', tagwise=True,
                  span=None, legacy_span=False, min_row_sum=5,
                  grid_length=21, grid_range=(-10, 10), robust=False,
                  winsor_tail_p=(0.05, 0.1), tol=1e-6, weights=None):
    """Estimate common, trended and tagwise dispersions.

    Port of edgeR's estimateDisp.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix or DGEList.
    design : ndarray or str, optional
        Design matrix, or an R-style formula string (e.g.
        ``'~ group'``, ``'~ batch + condition'``) evaluated
        against DGEList sample metadata via patsy.
        If None, uses classic edgeR approach.
    group : array-like, optional
        Group factor.
    lib_size : ndarray, optional
        Library sizes.
    offset : ndarray, optional
        Log-scale offsets.
    prior_df : float, optional
        Prior degrees of freedom.
    trend_method : str
        'locfit', 'loess', 'movingave', or 'none'.
    tagwise : bool
        Estimate tagwise dispersions.
    span : float, optional
        Span for smoothing.
    legacy_span : bool
        Use legacy span selection.
    min_row_sum : int
        Minimum row sum for a gene.
    grid_length : int
        Number of grid points.
    grid_range : tuple
        Range for dispersion grid.
    robust : bool
        Robust estimation.
    winsor_tail_p : tuple
        Winsorization tail proportions.
    tol : float
        Tolerance.
    weights : ndarray, optional
        Observation weights.

    Returns
    -------
    DGEList (if input is DGEList) or dict with common.dispersion,
    trended.dispersion, tagwise.dispersion, span, prior.df, prior.n.
    """
    # Resolve formula string to design matrix
    from .utils import _resolve_design
    design = _resolve_design(design, y)

    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist, get_offset
        dge = valid_dgelist(dge)
        group_val = dge['samples']['group'].values
        ls = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values

        if design is None:
            design = dge.get('design')
        else:
            dge['design'] = design

        d = estimate_disp(
            dge['counts'], design=design, group=group_val, lib_size=ls,
            offset=get_offset(dge), prior_df=prior_df,
            trend_method=trend_method, tagwise=tagwise, span=span,
            legacy_span=legacy_span, min_row_sum=min_row_sum,
            grid_length=grid_length, grid_range=grid_range,
            robust=robust, winsor_tail_p=winsor_tail_p, tol=tol,
            weights=dge.get('weights'))

        dge['common.dispersion'] = d['common.dispersion']
        dge['trended.dispersion'] = d['trended.dispersion']
        if tagwise:
            dge['tagwise.dispersion'] = d.get('tagwise.dispersion')
        dge['AveLogCPM'] = ave_log_cpm(dge)
        dge['trend.method'] = trend_method
        dge['prior.df'] = d.get('prior.df')
        dge['prior.n'] = d.get('prior.n')
        dge['span'] = d.get('span')
        return dge

    # Default method
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if ntags == 0:
        return {'span': span, 'prior.df': prior_df, 'prior.n': None}

    # Check trend_method
    valid_methods = ('none', 'loess', 'locfit', 'movingave', 'locfit.mixed')
    if trend_method not in valid_methods:
        raise ValueError(f"trend_method must be one of {valid_methods}")

    # Check group
    if group is None:
        group = np.ones(nlibs, dtype=int)
    group = drop_empty_levels(np.asarray(group))

    # Check lib_size
    if lib_size is None:
        lib_size = y.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    # Build offset
    if offset is None:
        offset = np.log(lib_size)
    offset = np.asarray(offset, dtype=np.float64)
    offset_mat = expand_as_matrix(offset, y.shape)

    if weights is not None:
        w_mat = expand_as_matrix(np.asarray(weights, dtype=np.float64), y.shape)
    else:
        w_mat = np.ones_like(y)

    # Filter genes with small counts
    sel = y.sum(axis=1) >= min_row_sum
    sely = y[sel]
    seloffset = offset_mat[sel]
    selweights = w_mat[sel]

    # Spline points
    spline_pts = np.linspace(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * 2 ** spline_pts
    grid_vals = spline_disp / (1 + spline_disp)
    l0 = np.zeros((np.sum(sel), grid_length))

    if design is None:
        # Classic edgeR approach
        unique_groups = np.unique(group)
        if np.all(np.bincount(group.astype(int) if np.issubdtype(group.dtype, np.integer) else
                              np.searchsorted(unique_groups, group)) <= 1):
            warnings.warn("There is no replication, setting dispersion to NA.")
            return {'common.dispersion': np.nan,
                    'trended.dispersion': np.nan,
                    'tagwise.dispersion': np.nan}

        if len(unique_groups) == 1:
            design_classic = np.ones((nlibs, 1))
        else:
            from .utils import _model_matrix_group
            design_classic = _model_matrix_group(group)

        # Equalize library sizes and estimate common dispersion
        from .exact_test import equalize_lib_sizes, split_into_groups
        eq = equalize_lib_sizes(y, group=group, dispersion=0.01, lib_size=lib_size)
        y_pseudo = eq['pseudo.counts'][sel]
        y_split = split_into_groups(y_pseudo, group=group)

        # Optimize common dispersion
        result = minimize_scalar(
            lambda d: -common_cond_log_lik_der_delta(y_split, d, der=0),
            bounds=(1e-4, 100 / 101), method='bounded')
        delta = result.x
        disp = delta / (1 - delta)

        # Re-equalize
        eq = equalize_lib_sizes(y, group=group, dispersion=disp, lib_size=lib_size)
        y_pseudo = eq['pseudo.counts'][sel]
        y_split = split_into_groups(y_pseudo, group=group)

        # Compute log-likelihoods on grid
        for j in range(grid_length):
            for grp_data in y_split:
                l0[:, j] += cond_log_lik_der_delta(grp_data[sel] if grp_data.shape[0] > np.sum(sel) else grp_data,
                                                    grid_vals[j], der=0)
    else:
        # GLM edgeR approach
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

        if design.shape[1] >= nlibs:
            warnings.warn("No residual df: setting dispersion to NA")
            return {'common.dispersion': np.nan,
                    'trended.dispersion': np.nan,
                    'tagwise.dispersion': np.nan}

        # Compute APL on grid for all genes (fast batch)
        l0 = adjusted_profile_lik_grid(
            spline_disp, sely, design, seloffset, weights=selweights)

    # Calculate common dispersion
    overall = maximize_interpolant(spline_pts, np.sum(l0, axis=0).reshape(1, -1))
    common_dispersion = 0.1 * 2 ** overall[0]

    # Allow dispersion trend
    if trend_method != 'none':
        ave_lcpm = ave_log_cpm(y, lib_size=lib_size, dispersion=common_dispersion,
                               weights=weights)
        out_1 = WLEB(theta=spline_pts, loglik=l0, covariate=ave_lcpm[sel],
                      trend_method=trend_method, span=span, legacy_span=legacy_span,
                      overall=False, individual=False, m0_out=True)
        span = out_1['span']
        m0 = out_1['shared.loglik']
        disp_trend = 0.1 * 2 ** out_1['trend']
        trended_dispersion = np.full(ntags, disp_trend[np.argmin(ave_lcpm[sel])])
        trended_dispersion[sel] = disp_trend
    else:
        ave_lcpm = None
        m0 = np.tile(np.mean(l0, axis=0), (np.sum(sel), 1))
        disp_trend = common_dispersion
        trended_dispersion = None

    # Are tagwise dispersions required?
    if not tagwise:
        return {'common.dispersion': common_dispersion,
                'trended.dispersion': trended_dispersion}

    # Calculate prior_df
    if prior_df is None:
        from .glm_fit import glm_fit
        if design is None:
            design_fit = np.ones((nlibs, 1))
        else:
            design_fit = design
        glmfit = glm_fit(sely, offset=seloffset, weights=selweights,
                         design=design_fit, dispersion=disp_trend, prior_count=0)

        df_residual = glmfit['df.residual'].astype(float)

        # Adjust for zeros
        from .utils import residual_df
        zerofit = (glmfit['counts'] < 1e-4) & (glmfit['fitted.values'] < 1e-4)
        df_residual = residual_df(zerofit, design_fit)

        s2 = glmfit['deviance'] / np.maximum(df_residual, 1e-8)
        s2[df_residual == 0] = 0
        s2 = np.maximum(s2, 0)
        covariate = ave_lcpm[sel] if ave_lcpm is not None else None
        s2_fit = squeeze_var(s2, df=df_residual, covariate=covariate,
                             robust=robust, winsor_tail_p=winsor_tail_p)
        prior_df = s2_fit.get('df.prior', s2_fit.get('df_prior'))

    ncoefs = design.shape[1] if design is not None else 1
    prior_n = prior_df / (nlibs - ncoefs)

    # Initiate tagwise dispersions
    if trend_method != 'none':
        tagwise_dispersion = trended_dispersion.copy()
    else:
        tagwise_dispersion = np.full(ntags, common_dispersion)

    # Estimate tagwise dispersions via WLEB
    too_large = np.atleast_1d(prior_n > 1e6)
    if not np.all(too_large):
        temp_n = np.atleast_1d(prior_n).copy()
        if np.any(too_large):
            temp_n[too_large] = 1e6

        out_2 = WLEB(theta=spline_pts, loglik=l0, prior_n=temp_n,
                      covariate=ave_lcpm[sel] if ave_lcpm is not None else None,
                      trend_method=trend_method, span=span, legacy_span=False,
                      overall=False, trend=False, m0=m0)
        tagwise_dispersion[sel] = 0.1 * 2 ** out_2['individual']

    if robust:
        temp_df = prior_df
        temp_n = prior_n
        prior_df = np.full(ntags, np.inf)
        prior_n = np.full(ntags, np.inf)
        prior_df[sel] = temp_df
        prior_n[sel] = temp_n

    return {
        'common.dispersion': common_dispersion,
        'trended.dispersion': trended_dispersion,
        'tagwise.dispersion': tagwise_dispersion,
        'span': span,
        'prior.df': prior_df,
        'prior.n': prior_n
    }


def WLEB(theta, loglik, prior_n=5, covariate=None, trend_method='locfit',
          span=None, legacy_span=False, overall=True, trend=True,
          individual=True, m0=None, m0_out=False):
    """Weighted likelihood empirical Bayes.

    Port of edgeR's WLEB.

    Parameters
    ----------
    theta : ndarray
        Grid of theta values.
    loglik : ndarray
        Log-likelihood matrix (genes x grid points).
    prior_n : float or ndarray
        Prior sample size.
    covariate : ndarray, optional
        Covariate for trend.
    trend_method : str
        Smoothing method.
    span : float, optional
        Smoothing span.
    legacy_span : bool
        Use legacy span selection.
    overall : bool
        Compute overall estimate.
    trend : bool
        Compute trended estimate.
    individual : bool
        Compute individual estimates.
    m0 : ndarray, optional
        Pre-computed shared loglik.
    m0_out : bool
        Return shared loglik.

    Returns
    -------
    dict with 'overall', 'trend', 'individual', 'span', 'shared.loglik'.
    """
    loglik = np.asarray(loglik, dtype=np.float64)
    if loglik.ndim == 1:
        loglik = loglik.reshape(1, -1)
    ntags = loglik.shape[0]
    theta = np.asarray(theta, dtype=np.float64)

    # Check covariate and trend
    if covariate is None:
        trend_method = 'none'

    # Set span matching R's WLEB formula exactly
    if span is None:
        if ntags <= 50:
            span = 1.0
        else:
            span = 0.25 + 0.75 * (50 / ntags) ** 0.5

    out = {'span': span}

    # Overall prior
    if overall:
        out['overall'] = maximize_interpolant(
            theta, np.sum(loglik, axis=0).reshape(1, -1))[0]

    # Trended prior
    if m0 is None:
        if trend_method == 'movingave':
            o = np.argsort(covariate)
            oo = np.argsort(o)
            width = int(np.floor(span * ntags))
            width = max(width, 1)
            m0 = moving_average_by_col(loglik[o], width=width)[oo]
        elif trend_method == 'loess':
            result = loess_by_col(loglik, x=covariate, span=span)
            m0 = result['fitted_values']
        elif trend_method == 'locfit':
            m0 = locfit_by_col(loglik, x=covariate, span=span, degree=0)
        elif trend_method == 'locfit.mixed':
            deg0 = locfit_by_col(loglik, x=covariate, span=span, degree=0)
            deg1 = locfit_by_col(loglik, x=covariate, span=span, degree=1)
            from scipy.stats import beta as beta_dist
            r = np.array([np.min(covariate), np.max(covariate)])
            if r[1] - r[0] > 0:
                w = beta_dist.cdf((covariate - r[0]) / (r[1] - r[0]), 2, 2)
            else:
                w = np.full(len(covariate), 0.5)
            m0 = w[:, None] * deg0 + (1 - w[:, None]) * deg1
        else:
            # 'none'
            m0 = np.tile(np.mean(loglik, axis=0), (ntags, 1))

    if trend:
        out['trend'] = maximize_interpolant(theta, m0)

    # Weighted empirical Bayes posterior estimates
    if individual:
        prior_n = np.atleast_1d(np.asarray(prior_n, dtype=np.float64))
        if len(prior_n) == 1:
            l0a = loglik + prior_n[0] * m0
        else:
            l0a = loglik + prior_n[:, None] * m0
        out['individual'] = maximize_interpolant(theta, l0a)

    if m0_out:
        out['shared.loglik'] = m0

    return out


def estimate_common_disp(y, group=None, lib_size=None, tol=1e-6,
                         rowsum_filter=5, verbose=False):
    """Estimate common dispersion using exact conditional likelihood.

    Port of edgeR's estimateCommonDisp.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix or DGEList.
    group : array-like, optional
        Group factor.
    lib_size : ndarray, optional
        Library sizes.
    tol : float
        Optimization tolerance.
    rowsum_filter : int
        Minimum row sum.
    verbose : bool
        Print progress.

    Returns
    -------
    DGEList (if input is DGEList) or float.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist
        dge = valid_dgelist(dge)
        group = dge['samples']['group'].values
        ls = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values

        d = estimate_common_disp(dge['counts'], group=group, lib_size=ls,
                                 tol=tol, rowsum_filter=rowsum_filter, verbose=verbose)
        dge['common.dispersion'] = d
        dge['AveLogCPM'] = ave_log_cpm(dge, dispersion=d)
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if group is None:
        group = np.ones(nlibs, dtype=int)
    group = np.asarray(group)

    if lib_size is None:
        lib_size = y.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    # Filter
    keep = y.sum(axis=1) >= rowsum_filter
    y_filt = y[keep]

    if y_filt.shape[0] == 0:
        warnings.warn("No genes pass rowsum filter")
        return 0.1

    # Equalize library sizes and split into groups
    from .exact_test import equalize_lib_sizes, split_into_groups

    # First pass with rough dispersion
    eq = equalize_lib_sizes(y_filt, group=group, dispersion=0.01, lib_size=lib_size)
    y_pseudo = eq['pseudo.counts']
    y_split = split_into_groups(y_pseudo, group=group)

    # Optimize
    result = minimize_scalar(
        lambda d: -common_cond_log_lik_der_delta(y_split, d, der=0),
        bounds=(1e-4, 100 / 101), method='bounded',
        options={'xatol': tol})
    delta = result.x
    disp = delta / (1 - delta)

    if verbose:
        print(f"Disp = {disp:.5f}, BCV = {np.sqrt(disp):.4f}")

    return disp


def estimate_tagwise_disp(y, group=None, lib_size=None, dispersion=None,
                           prior_df=10, trend='movingave', span=None,
                           method='grid', grid_length=11, grid_range=(-6, 6),
                           tol=1e-6, verbose=False):
    """Estimate tagwise dispersions using exact conditional likelihood.

    Port of edgeR's estimateTagwiseDisp.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix or DGEList.
    group : array-like, optional
        Group factor.
    lib_size : ndarray, optional
        Library sizes.
    dispersion : float or ndarray, optional
        Starting dispersion.
    prior_df : float
        Prior degrees of freedom.
    trend : str
        'movingave', 'loess', or 'none'.
    span : float, optional
        Smoothing span.
    method : str
        'grid' or 'optimize'.
    grid_length : int
        Number of grid points.
    grid_range : tuple
        Grid range.
    tol : float
        Tolerance.

    Returns
    -------
    DGEList (if input is DGEList) or ndarray of tagwise dispersions.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist
        dge = valid_dgelist(dge)
        group = dge['samples']['group'].values
        ls = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values

        if dispersion is None:
            dispersion = dge.get('common.dispersion')
            if dispersion is None:
                raise ValueError("No common.dispersion found. Run estimate_common_disp first.")

        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = ave_log_cpm(dge)

        td = estimate_tagwise_disp(
            dge['counts'], group=group, lib_size=ls, dispersion=dispersion,
            prior_df=prior_df, trend=trend, span=span, method=method,
            grid_length=grid_length, grid_range=grid_range, tol=tol)
        dge['tagwise.dispersion'] = td
        dge['prior.df'] = prior_df
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if group is None:
        group = np.ones(nlibs, dtype=int)
    group = np.asarray(group)

    if lib_size is None:
        lib_size = y.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    if dispersion is None:
        dispersion = 0.1

    if span is None:
        span = (10 / ntags) ** 0.23 if ntags > 10 else 1.0

    # Equalize library sizes
    from .exact_test import equalize_lib_sizes, split_into_groups
    eq = equalize_lib_sizes(y, group=group, dispersion=dispersion, lib_size=lib_size)
    y_pseudo = eq['pseudo.counts']
    y_split = split_into_groups(y_pseudo, group=group)

    # Compute log-likelihoods on grid
    spline_pts = np.linspace(grid_range[0], grid_range[1], grid_length)

    if np.isscalar(dispersion):
        disp_base = dispersion
    else:
        disp_base = np.median(dispersion)

    grid_disp = disp_base * 2 ** spline_pts
    grid_delta = grid_disp / (1 + grid_disp)

    l0 = np.zeros((ntags, grid_length))
    for j in range(grid_length):
        for grp_data in y_split:
            l0[:, j] += cond_log_lik_der_delta(grp_data, grid_delta[j], der=0)

    # Compute AveLogCPM for smoothing
    alc = ave_log_cpm(y, lib_size=lib_size)

    # Use WLEB
    prior_n = prior_df / (nlibs - len(np.unique(group)))

    out = WLEB(theta=spline_pts, loglik=l0, prior_n=prior_n,
               covariate=alc, trend_method='movingave' if trend == 'movingave' else
               ('loess' if trend == 'loess' else 'none'),
               span=span)

    tagwise_dispersion = disp_base * 2 ** out['individual']
    return tagwise_dispersion


def estimate_trended_disp(y, group=None, lib_size=None, ave_log_cpm_vals=None,
                           method='bin.spline', df=5, span=2/3):
    """Estimate trended dispersions using exact conditional likelihood.

    Port of edgeR's estimateTrendedDisp.

    Returns
    -------
    DGEList (if input is DGEList) or ndarray of trended dispersions.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist
        dge = valid_dgelist(dge)
        group_val = dge['samples']['group'].values
        ls = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values
        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = ave_log_cpm(dge)
        out = estimate_trended_disp(dge['counts'], group=group_val, lib_size=ls,
                                     ave_log_cpm_vals=dge['AveLogCPM'],
                                     method=method, df=df, span=span)
        dge['trended.dispersion'] = out
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if group is None:
        group = np.ones(nlibs, dtype=int)
    group = drop_empty_levels(np.asarray(group))

    if lib_size is None:
        lib_size = y.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, lib_size=lib_size)

    # Bin genes by abundance and estimate dispersion in each bin
    nbins = 50
    if nbins > ntags:
        nbins = max(1, ntags // 2)

    bins = cut_with_min_n(ave_log_cpm_vals, intervals=nbins,
                          min_n=max(1, ntags // nbins))
    disp_bins = np.zeros(nbins)
    ave_bins = np.zeros(nbins)

    for i in range(1, nbins + 1):
        mask = bins['group'] == i
        if np.sum(mask) == 0:
            continue
        disp_bins[i - 1] = estimate_common_disp(y[mask], group=group,
                                                 lib_size=lib_size,
                                                 rowsum_filter=0)
        ave_bins[i - 1] = np.mean(ave_log_cpm_vals[mask])

    # Fit trend
    if method == 'bin.spline':
        from scipy.interpolate import UnivariateSpline
        order = np.argsort(ave_bins)
        try:
            spl = UnivariateSpline(ave_bins[order],
                                   np.sqrt(np.maximum(disp_bins[order], 0)),
                                   k=min(3, len(ave_bins) - 1),
                                   s=len(ave_bins) * 0.1)
            trended = spl(ave_log_cpm_vals) ** 2
        except Exception:
            trended = np.full(ntags, np.mean(disp_bins))
    else:
        # bin.loess
        from scipy.interpolate import interp1d
        try:
            f = interp1d(ave_bins, np.sqrt(np.maximum(disp_bins, 0)),
                         fill_value='extrapolate')
            trended = f(ave_log_cpm_vals) ** 2
        except Exception:
            trended = np.full(ntags, np.mean(disp_bins))

    return np.maximum(trended, 0)


def estimate_glm_common_disp(y, design=None, offset=None, method='CoxReid',
                              subset=10000, ave_log_cpm_vals=None, verbose=False,
                              weights=None):
    """Estimate common dispersion using GLM approach.

    Port of edgeR's estimateGLMCommonDisp.

    Returns
    -------
    DGEList (if input is DGEList) or float.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        from .dgelist import valid_dgelist, get_offset
        dge = valid_dgelist(dge)
        alc = ave_log_cpm(dge, dispersion=0.05)
        offset_val = get_offset(dge)
        d = estimate_glm_common_disp(
            dge['counts'], design=design, offset=offset_val,
            method=method, subset=subset, ave_log_cpm_vals=alc,
            verbose=verbose, weights=dge.get('weights'))
        dge['common.dispersion'] = d
        dge['AveLogCPM'] = ave_log_cpm(dge, dispersion=d)
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    if design is None:
        design = np.ones((y.shape[1], 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

    if design.shape[1] >= y.shape[1]:
        warnings.warn("No residual df: setting dispersion to NA")
        return np.nan

    if offset is None:
        offset = np.log(y.sum(axis=0))

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset, weights=weights)

    valid_methods = ('CoxReid', 'Pearson', 'deviance')
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    if method != 'CoxReid' and weights is not None:
        warnings.warn("weights only supported by CoxReid method")

    if method == 'CoxReid':
        d = disp_cox_reid(y, design=design, offset=offset, subset=subset,
                          ave_log_cpm_vals=ave_log_cpm_vals, weights=weights)
    elif method == 'Pearson':
        d = disp_pearson(y, design=design, offset=offset, subset=subset,
                         ave_log_cpm_vals=ave_log_cpm_vals)
    else:
        d = disp_deviance(y, design=design, offset=offset, subset=subset,
                          ave_log_cpm_vals=ave_log_cpm_vals)

    if verbose:
        print(f"Disp = {d:.5f}, BCV = {np.sqrt(d):.4f}")

    return d


def estimate_glm_trended_disp(y, design=None, offset=None,
                                ave_log_cpm_vals=None, method='auto',
                                weights=None):
    """Estimate trended dispersion using GLM approach.

    Port of edgeR's estimateGLMTrendedDisp.

    Returns
    -------
    DGEList (if input is DGEList) or ndarray.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = ave_log_cpm(dge)
        from .dgelist import get_offset
        d = estimate_glm_trended_disp(
            dge['counts'], design=design, offset=get_offset(dge),
            ave_log_cpm_vals=dge['AveLogCPM'], method=method,
            weights=dge.get('weights'))
        dge['trended.dispersion'] = d
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags = y.shape[0]
    nlibs = y.shape[1]

    if ntags == 0:
        return np.array([], dtype=np.float64)

    if design is None:
        design = np.ones((nlibs, 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

    if design.shape[1] >= nlibs:
        warnings.warn("No residual df: cannot estimate dispersion")
        return np.full(ntags, np.nan)

    if offset is None:
        offset = np.log(y.sum(axis=0))

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset, weights=weights)

    if method == 'auto':
        method = 'power' if ntags < 200 else 'bin.spline'

    valid_methods = ('bin.spline', 'bin.loess', 'power', 'spline')
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    if method in ('bin.spline', 'bin.loess'):
        mt = 'spline' if method == 'bin.spline' else 'loess'
        result = disp_bin_trend(y, design, offset=offset, method_trend=mt,
                                ave_log_cpm_vals=ave_log_cpm_vals, weights=weights)
    elif method == 'power':
        result = disp_cox_reid_power_trend(y, design, offset=offset,
                                            ave_log_cpm_vals=ave_log_cpm_vals)
    else:
        result = disp_cox_reid_spline_trend(y, design, offset=offset,
                                             ave_log_cpm_vals=ave_log_cpm_vals)

    return result['dispersion']


def estimate_glm_tagwise_disp(y, design=None, offset=None, dispersion=None,
                               prior_df=10, trend=True, span=None,
                               ave_log_cpm_vals=None, weights=None):
    """Estimate tagwise dispersions using GLM approach.

    Port of edgeR's estimateGLMTagwiseDisp.

    Returns
    -------
    DGEList (if input is DGEList) or ndarray.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if trend:
            dispersion = dge.get('trended.dispersion')
            if dispersion is None:
                raise ValueError("No trended.dispersion found. Run estimate_glm_trended_disp first.")
        else:
            if dispersion is None:
                dispersion = dge.get('common.dispersion')
                if dispersion is None:
                    raise ValueError("No common.dispersion found. Run estimate_glm_common_disp first.")

        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = ave_log_cpm(dge)

        ntags = dge['counts'].shape[0]
        if span is None:
            span = (10 / ntags) ** 0.23 if ntags > 10 else 1.0
        dge['span'] = span

        from .dgelist import get_offset
        d = estimate_glm_tagwise_disp(
            dge['counts'], design=design, offset=get_offset(dge),
            dispersion=dispersion, prior_df=prior_df, trend=trend,
            span=span, ave_log_cpm_vals=dge['AveLogCPM'],
            weights=dge.get('weights'))
        dge['prior.df'] = prior_df
        dge['tagwise.dispersion'] = d
        return dge

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntags, nlibs = y.shape

    if ntags == 0:
        return np.array([], dtype=np.float64)

    if design is None:
        design = np.ones((nlibs, 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)

    if design.shape[1] >= nlibs:
        warnings.warn("No residual df: setting dispersion to NA")
        return np.full(ntags, np.nan)

    if offset is None:
        offset = np.log(y.sum(axis=0))

    if span is None:
        span = (10 / ntags) ** 0.23 if ntags > 10 else 1.0

    if ave_log_cpm_vals is None:
        ave_log_cpm_vals = ave_log_cpm(y, offset=offset, weights=weights)

    tagwise = disp_cox_reid_interpolate_tagwise(
        y, design, offset=offset, dispersion=dispersion,
        trend=trend, prior_df=prior_df, span=span,
        ave_log_cpm_vals=ave_log_cpm_vals, weights=weights)

    return tagwise


def _calc_resid(fit, residual_type='pearson'):
    """Compute GLM residual matrix for robust dispersion fitting."""
    residual_type = str(residual_type).lower()
    if residual_type not in ('pearson', 'anscombe', 'deviance'):
        raise ValueError("residual_type must be one of ('pearson', 'anscombe', 'deviance')")

    mu = np.asarray(fit['fitted.values'], dtype=np.float64)
    yi = np.asarray(fit['counts'], dtype=np.float64)
    disp = expand_as_matrix(np.asarray(fit['dispersion'], dtype=np.float64), mu.shape)

    if residual_type == 'pearson':
        res = (yi - mu) / np.sqrt(np.maximum(mu * (1 + disp * mu), 1e-12))
    elif residual_type == 'deviance':
        y_adj = yi + 1e-5
        with np.errstate(divide='ignore', invalid='ignore'):
            r = 2 * (y_adj * np.log(np.maximum(y_adj, 1e-12) / np.maximum(mu, 1e-12)) +
                     (y_adj + 1 / np.maximum(disp, 1e-12)) *
                     np.log((mu + 1 / np.maximum(disp, 1e-12)) /
                            (y_adj + 1 / np.maximum(disp, 1e-12))))
        r = np.maximum(r, 0)
        res = np.sign(yi - mu) * np.sqrt(r)
    else:
        # Numerical approximation to the Anscombe residual integral used by edgeR.
        from scipy.integrate import quad

        def _anscombe_scalar(yv, muv, dv):
            if muv <= 0 or yv <= 0:
                return 0.0

            def ffun(x):
                return (x * (1 + dv * x)) ** (-1.0 / 3.0)

            const = ffun(muv) ** 0.5
            if yv == muv:
                return 0.0
            val, _ = quad(ffun, muv, yv, limit=50)
            return const * val

        res = np.zeros_like(yi, dtype=np.float64)
        for g in range(yi.shape[0]):
            for s in range(yi.shape[1]):
                res[g, s] = _anscombe_scalar(yi[g, s], mu[g, s], disp[g, s])

    res[mu == 0] = 0
    return res


def _psi_huber_matrix(u, k=1.345):
    """Huber psi weights on a residual matrix."""
    u = np.asarray(u, dtype=np.float64)
    out = np.ones_like(u, dtype=np.float64)
    mask = np.abs(u) > k
    out[mask] = k / np.abs(u[mask])
    out[~np.isfinite(out)] = 1.0
    return out


def _record_robust_disp_state(y, i, res=None, weights=None, fit=None):
    """Store per-iteration state for estimate_glm_robust_disp(record=True)."""
    key = f'iteration_{i}'
    rec = y.get('record')
    if rec is None:
        rec = {
            'AveLogCPM': {},
            'trended.dispersion': {},
            'tagwise.dispersion': {},
            'weights': {},
            'res': {},
            'mu': {}
        }

    if y.get('AveLogCPM') is not None:
        rec['AveLogCPM'][key] = np.asarray(y['AveLogCPM']).copy()
    if y.get('trended.dispersion') is not None:
        rec['trended.dispersion'][key] = np.asarray(y['trended.dispersion']).copy()
    if y.get('tagwise.dispersion') is not None:
        rec['tagwise.dispersion'][key] = np.asarray(y['tagwise.dispersion']).copy()
    if weights is not None:
        rec['weights'][key] = np.asarray(weights).copy()
    if res is not None:
        rec['res'][key] = np.asarray(res).copy()
    if fit is not None and fit.get('fitted.values') is not None:
        rec['mu'][key] = np.asarray(fit['fitted.values']).copy()

    y['record'] = rec
    return y


def estimate_glm_robust_disp(y, design=None, prior_df=10, update_trend=True,
                             trend_method='bin.loess', maxit=6, k=1.345,
                             residual_type='pearson', verbose=False,
                             record=False):
    """Robust GLM dispersion estimation via iterative Huber reweighting.

    Port of edgeR's estimateGLMRobustDisp.
    """
    from .utils import _resolve_design
    design = _resolve_design(design, y)

    if not (isinstance(y, dict) and 'counts' in y):
        raise ValueError("Input must be a DGEList-like dict with 'counts'.")

    from .dgelist import valid_dgelist
    y = valid_dgelist(y)

    y['weights'] = np.ones_like(np.asarray(y['counts'], dtype=np.float64), dtype=np.float64)

    if y.get('trended.dispersion') is None:
        y = estimate_glm_trended_disp(y, design=design, method=trend_method,
                                      weights=y['weights'])
    if y.get('tagwise.dispersion') is None:
        y = estimate_glm_tagwise_disp(y, design=design, prior_df=prior_df,
                                      weights=y['weights'])

    if record:
        y = _record_robust_disp_state(y, i=0, weights=y['weights'])

    from .glm_fit import glm_fit

    for i in range(1, int(maxit) + 1):
        if verbose:
            print(f"Iteration {i}: Re-fitting GLM.")

        fit = glm_fit(y, design=design, prior_count=0)
        res = _calc_resid(fit, residual_type=residual_type)

        y['weights'] = _psi_huber_matrix(res, k=k)
        y['AveLogCPM'] = ave_log_cpm(y, dispersion=y.get('trended.dispersion'))

        if update_trend:
            if verbose:
                print("Re-estimating trended dispersion.")
            y = estimate_glm_trended_disp(y, design=design, method=trend_method,
                                          weights=y['weights'])

        if verbose:
            print("Re-estimating tagwise dispersion.")
        y = estimate_glm_tagwise_disp(y, design=design, prior_df=prior_df,
                                      weights=y['weights'])

        if record:
            y = _record_robust_disp_state(y, i=i, res=res,
                                          weights=y['weights'], fit=fit)

    return y


def estimateGLMRobustDisp(*args, **kwargs):
    """Compatibility alias for edgeR-style camelCase naming."""
    return estimate_glm_robust_disp(*args, **kwargs)
