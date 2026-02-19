# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
GLM fitting for edgePython.

Port of edgeR's glmFit, glmQLFit, mglmOneGroup, mglmOneWay.
"""

import numpy as np
import warnings
from .compressed_matrix import (CompressedMatrix, compress_offsets,
                                 compress_weights, compress_dispersions)
from .glm_levenberg import mglm_levenberg, nbinom_deviance
from .utils import (expand_as_matrix, design_as_factor, pred_fc,
                    add_prior_count, residual_df)
from .limma_port import squeeze_var, non_estimable, is_fullrank, choose_lowess_span


def mglm_one_group(y, dispersion=0, offset=0, weights=None,
                   coef_start=None, maxit=50, tol=1e-10):
    """Fit single-group negative-binomial GLM.

    Port of edgeR's mglmOneGroup (C code reimplemented in Python).

    Parameters
    ----------
    y : ndarray
        Count matrix (genes x samples).
    dispersion : float, ndarray, or CompressedMatrix
        NB dispersions.
    offset : float, ndarray, or CompressedMatrix
        Log-scale offsets.
    weights : ndarray or CompressedMatrix, optional
        Observation weights.
    coef_start : ndarray, optional
        Starting coefficient values (one per gene).
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    ndarray of coefficients (one per gene).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ngenes, nlibs = y.shape

    # Expand offset, dispersion, weights
    offset_mat = _expand_to_matrix(offset, y.shape)
    disp_mat = _expand_to_matrix(dispersion, y.shape)
    if weights is not None:
        w_mat = _expand_to_matrix(weights, y.shape)
    else:
        w_mat = np.ones_like(y)

    # Ensure 2D for all
    if disp_mat.ndim == 1:
        disp_mat = np.broadcast_to(disp_mat[:, None] if len(disp_mat) == ngenes
                                    else disp_mat[None, :], y.shape).copy()
    elif disp_mat.ndim == 0:
        disp_mat = np.full_like(y, float(disp_mat))

    # Starting values (vectorized)
    if coef_start is not None:
        b = np.asarray(coef_start, dtype=np.float64).ravel()
        if len(b) == 1:
            b = np.full(ngenes, b[0])
        need_init = np.isnan(b)
    else:
        b = np.full(ngenes, np.nan)
        need_init = np.ones(ngenes, dtype=bool)

    if np.any(need_init):
        lib = np.exp(offset_mat[need_init])
        total_y = np.sum(w_mat[need_init] * y[need_init], axis=1)
        total_lib = np.sum(w_mat[need_init] * lib, axis=1)
        valid = (total_y > 0) & (total_lib > 0)
        b_init = np.full(np.sum(need_init), -20.0)
        b_init[valid] = np.log(total_y[valid] / total_lib[valid])
        b[need_init] = b_init

    # Vectorized Fisher scoring iteration (all genes at once)
    active = np.ones(ngenes, dtype=bool)  # genes still iterating
    for _it in range(maxit):
        if not np.any(active):
            break

        # Compute mu for active genes
        eta = b[active, None] + offset_mat[active]  # (n_active, nlibs)
        mu = np.exp(np.clip(eta, -500, 500))
        mu = np.maximum(mu, 1e-300)

        # Working weights
        denom = 1.0 + disp_mat[active] * mu  # (n_active, nlibs)

        # Score and information
        dl = np.sum(w_mat[active] * (y[active] - mu) / denom, axis=1)  # (n_active,)
        info = np.sum(w_mat[active] * mu / denom, axis=1)  # (n_active,)

        # Guard against zero information
        safe = info > 1e-300
        step = np.zeros_like(dl)
        step[safe] = dl[safe] / info[safe]

        b_new = b[active] + step

        # Check convergence
        converged = np.abs(step) < tol * (np.abs(b[active]) + 0.1)
        converged |= ~safe

        b[active] = b_new

        # Mark converged genes as inactive
        active_indices = np.where(active)[0]
        active[active_indices[converged]] = False

    return b


def mglm_one_way(y, design=None, group=None, dispersion=0, offset=0,
                 weights=None, coef_start=None, maxit=50, tol=1e-10):
    """Fit multiple NB GLMs with a one-way layout.

    Port of edgeR's mglmOneWay.

    Parameters
    ----------
    y : ndarray
        Count matrix (genes x samples).
    design : ndarray, optional
        Design matrix.
    group : ndarray, optional
        Group factor.
    dispersion : float or ndarray
        NB dispersions.
    offset : float or ndarray
        Offsets.
    weights : ndarray, optional
        Observation weights.
    coef_start : ndarray, optional
        Starting coefficients.
    maxit : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict with 'coefficients' and 'fitted.values'.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ngenes, nlibs = y.shape

    offset_mat = _expand_to_matrix(offset, y.shape)
    disp_mat = _expand_to_matrix(dispersion, y.shape)
    if weights is not None:
        w_mat = _expand_to_matrix(weights, y.shape)
    else:
        w_mat = np.ones_like(y)

    # Get group factor
    if group is None:
        if design is None:
            group = np.zeros(nlibs, dtype=int)
        else:
            design = np.asarray(design, dtype=np.float64)
            if design.ndim == 1:
                design = design.reshape(-1, 1)
            group = design_as_factor(design)
    else:
        group = np.asarray(group)

    unique_groups = np.unique(group)
    ngroups = len(unique_groups)

    # Check if design reduces to indicator matrix
    design_unique = None
    if design is not None:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        if design.shape[1] != ngroups:
            raise ValueError("design matrix is not equivalent to a oneway layout")
        # Get representative design rows
        first_of_group = np.array([np.where(group == g)[0][0] for g in unique_groups])
        design_unique = design[first_of_group]
        # Check if it's a simple group indicator
        is_indicator = (np.sum(design_unique == 1) == ngroups and
                        np.sum(design_unique == 0) == (ngroups - 1) * ngroups)
        if is_indicator:
            design_unique = None

    # Convert starting values if needed
    cs = None
    if coef_start is not None:
        coef_start = np.asarray(coef_start, dtype=np.float64)
        if coef_start.ndim == 1:
            coef_start = coef_start.reshape(1, -1)
        if design_unique is not None:
            cs = coef_start @ design_unique.T
        else:
            cs = coef_start

    # Fit each group
    beta = np.zeros((ngenes, ngroups))
    for g_idx, grp in enumerate(unique_groups):
        j = np.where(group == grp)[0]
        cs_g = cs[:, g_idx] if cs is not None else None
        beta[:, g_idx] = mglm_one_group(
            y[:, j], dispersion=disp_mat[:, j] if disp_mat.ndim == 2 else disp_mat,
            offset=offset_mat[:, j] if offset_mat.ndim == 2 else offset_mat,
            weights=w_mat[:, j] if w_mat.ndim == 2 else w_mat,
            coef_start=cs_g, maxit=maxit, tol=tol)

    # Clamp -Inf to large negative
    beta = np.maximum(beta, -1e8)

    # Fitted values from group-wise betas
    mu = np.zeros_like(y)
    for g_idx, grp in enumerate(unique_groups):
        j = np.where(group == grp)[0]
        for jj in j:
            mu[:, jj] = np.exp(np.clip(beta[:, g_idx] + offset_mat[:, jj], -500, 500))

    # If design is not indicator, convert back
    if design_unique is not None:
        beta = np.linalg.solve(design_unique, beta.T).T

    return {
        'coefficients': beta,
        'fitted.values': mu
    }


def glm_fit(y, design=None, dispersion=None, offset=None, lib_size=None,
            weights=None, prior_count=0.125, start=None):
    """Fit negative binomial GLMs for each gene.

    Port of edgeR's glmFit.default.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix (genes x samples), or DGEList.
    design : ndarray or str, optional
        Design matrix, or an R-style formula string (e.g.
        ``'~ group'``, ``'~ batch + condition'``) which is
        evaluated against the DGEList sample metadata via patsy.
    dispersion : float or ndarray
        NB dispersions.
    offset : ndarray, optional
        Log-scale offsets.
    lib_size : ndarray, optional
        Library sizes.
    weights : ndarray, optional
        Observation weights.
    prior_count : float
        Prior count for shrinking log-fold-changes.
    start : ndarray, optional
        Starting coefficient values.

    Returns
    -------
    dict (DGEGLM-like) with coefficients, fitted.values, deviance,
    df.residual, design, offset, dispersion, weights, etc.
    """
    # Resolve formula string to design matrix
    from .utils import _resolve_design
    design = _resolve_design(design, y)

    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if design is None:
            design = dge.get('design')
            if design is None:
                group = dge['samples']['group'].values
                from .utils import drop_empty_levels
                group = drop_empty_levels(group)
                unique_groups = np.unique(group)
                if len(unique_groups) > 1:
                    # model.matrix(~group)
                    from .utils import _model_matrix_group
                    design = _model_matrix_group(group)
        if dispersion is None:
            from .dgelist import get_dispersion
            dispersion = get_dispersion(dge)
            if dispersion is None:
                raise ValueError("No dispersion values found in DGEList object.")
        from .dgelist import get_offset
        offset = get_offset(dge)
        from .expression import ave_log_cpm
        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = ave_log_cpm(dge)

        fit = glm_fit(dge['counts'], design=design, dispersion=dispersion,
                      offset=offset, lib_size=None, weights=dge.get('weights'),
                      prior_count=prior_count, start=start)
        fit['samples'] = dge['samples']
        fit['genes'] = dge.get('genes')
        fit['prior.df'] = dge.get('prior.df')
        fit['AveLogCPM'] = dge.get('AveLogCPM')
        return fit

    # Default method
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    ntag, nlib = y.shape

    # Check design
    if design is None:
        design = np.ones((nlib, 1))
    else:
        design = np.asarray(design, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        if design.shape[0] != nlib:
            raise ValueError("nrow(design) disagrees with ncol(y)")
        ne = non_estimable(design)
        if ne is not None:
            raise ValueError(f"Design matrix not of full rank. Non-estimable: {ne}")

    # Check dispersion
    if dispersion is None:
        raise ValueError("No dispersion values provided.")
    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if np.any(np.isnan(dispersion)):
        raise ValueError("NA dispersions not allowed")
    if np.any(dispersion < 0):
        raise ValueError("Negative dispersions not allowed")

    # Build offset from lib_size and offset
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    elif lib_size is not None:
        lib_size = np.asarray(lib_size, dtype=np.float64)
        offset = np.log(lib_size)
    else:
        offset = np.log(y.sum(axis=0))

    offset_mat = expand_as_matrix(offset, (ntag, nlib))
    disp_mat = expand_as_matrix(dispersion, (ntag, nlib))

    if weights is not None:
        w_mat = expand_as_matrix(np.asarray(weights, dtype=np.float64), (ntag, nlib))
    else:
        w_mat = None

    # Fit: use one-way shortcut if design is equivalent to one-way layout
    group = design_as_factor(design)
    unique_groups = np.unique(group)

    if len(unique_groups) == design.shape[1]:
        fit = mglm_one_way(y, design=design, group=group,
                           dispersion=disp_mat, offset=offset_mat,
                           weights=w_mat, coef_start=start)
        fit['deviance'] = nbinom_deviance(y, fit['fitted.values'], dispersion, w_mat)
        fit['method'] = 'oneway'
    else:
        fit = mglm_levenberg(y, design=design, dispersion=disp_mat,
                             offset=offset_mat, weights=w_mat,
                             coef_start=start, maxit=250)
        fit['method'] = 'levenberg'

    # Prepare output
    fit['counts'] = y
    if prior_count > 0:
        fit['unshrunk.coefficients'] = fit['coefficients'].copy()
        fit['coefficients'] = pred_fc(y, design, offset=offset_mat,
                                      dispersion=disp_mat,
                                      prior_count=prior_count,
                                      weights=w_mat) * np.log(2)

    fit['df.residual'] = np.full(ntag, nlib - design.shape[1])
    fit['design'] = design
    fit['offset'] = offset_mat
    fit['dispersion'] = dispersion
    fit['weights'] = weights
    fit['prior.count'] = prior_count

    return fit


def glm_ql_fit(y, design=None, dispersion=None, offset=None, lib_size=None,
               weights=None, abundance_trend=True, ave_log_cpm=None,
               covariate_trend=None, robust=False, winsor_tail_p=(0.05, 0.1),
               legacy=False, top_proportion=None, keep_unit_mat=False):
    """Fit quasi-likelihood negative binomial GLMs.

    Port of edgeR's glmQLFit.default.

    Parameters
    ----------
    y : ndarray or DGEList
        Count matrix or DGEList.
    design : ndarray or str, optional
        Design matrix, or an R-style formula string (e.g.
        ``'~ group'``, ``'~ batch + condition'``) evaluated
        against DGEList sample metadata via patsy.
    dispersion : float or ndarray, optional
        NB dispersions.
    offset : ndarray, optional
        Offsets.
    lib_size : ndarray, optional
        Library sizes.
    weights : ndarray, optional
        Observation weights.
    abundance_trend : bool
        Use abundance trend for QL prior.
    ave_log_cpm : ndarray, optional
        Average log-CPM values.
    covariate_trend : ndarray, optional
        Covariate for trended prior.
    robust : bool
        Robust empirical Bayes.
    winsor_tail_p : tuple
        Winsorization tail proportions.
    legacy : bool
        Use legacy (old-style) QL method.
    top_proportion : float, optional
        Proportion of top-abundance genes for dispersion estimation.
    keep_unit_mat : bool
        Keep unit deviance matrix.

    Returns
    -------
    dict (DGEGLM-like) with added s2.post, df.prior, s2.prior fields.
    """
    from .expression import ave_log_cpm as _ave_log_cpm
    from .utils import _resolve_design
    design = _resolve_design(design, y)

    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if design is None:
            design = dge.get('design')
            if design is None:
                group = dge['samples']['group'].values
                from .utils import drop_empty_levels
                group = drop_empty_levels(group)
                unique_g = np.unique(group)
                if len(unique_g) > 1:
                    from .utils import _model_matrix_group
                    design = _model_matrix_group(group)

        if dge.get('AveLogCPM') is None:
            dge['AveLogCPM'] = _ave_log_cpm(dge)

        if dispersion is None:
            if legacy:
                dispersion = dge.get('trended.dispersion')
                if dispersion is None:
                    dispersion = dge.get('common.dispersion')
                if dispersion is None:
                    raise ValueError("No dispersion values found in DGEList object.")
            else:
                if dge.get('trended.dispersion') is not None:
                    ntop = int(np.ceil(0.1 * dge['counts'].shape[0]))
                    i = np.argsort(dge['AveLogCPM'])[::-1][:ntop]
                    dispersion = np.mean(dge['trended.dispersion'][i])

        from .dgelist import get_offset
        offset = get_offset(dge)

        fit = glm_ql_fit(dge['counts'], design=design, dispersion=dispersion,
                         offset=offset, lib_size=None,
                         abundance_trend=abundance_trend,
                         ave_log_cpm=dge['AveLogCPM'],
                         robust=robust, winsor_tail_p=winsor_tail_p,
                         weights=dge.get('weights'),
                         legacy=legacy, top_proportion=top_proportion,
                         keep_unit_mat=keep_unit_mat)
        fit['samples'] = dge['samples']
        fit['genes'] = dge.get('genes')
        fit['AveLogCPM'] = dge['AveLogCPM']
        return fit

    # Default method
    y_mat = np.asarray(y, dtype=np.float64)
    if y_mat.ndim == 1:
        y_mat = y_mat.reshape(1, -1)
    ngenes = y_mat.shape[0]
    nlibs = y_mat.shape[1]

    # Check design
    if design is None:
        design = np.ones((nlibs, 1))

    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)

    # Check AveLogCPM
    if ave_log_cpm is None:
        ave_log_cpm = _ave_log_cpm(y_mat, offset=offset, lib_size=lib_size,
                                    weights=weights, dispersion=dispersion)

    # Check dispersion
    if dispersion is None:
        if legacy:
            raise ValueError("No dispersion values provided.")
        else:
            if top_proportion is None:
                df_residual = nlibs - design.shape[1]
                top_proportion = choose_lowess_span(
                    ngenes * np.sqrt(df_residual), small_n=20, min_span=0.02)
            else:
                if top_proportion < 0 or top_proportion > 1:
                    raise ValueError("top_proportion should be between 0 and 1.")
            ntop = int(np.ceil(top_proportion * ngenes))
            i = np.argsort(ave_log_cpm)[::-1][:ntop]
            from .dispersion import estimate_glm_common_disp
            if offset is not None:
                off_sub = np.asarray(offset)
                if off_sub.ndim == 2:
                    off_sub = off_sub[i]
            else:
                off_sub = None
            w_sub = None
            if weights is not None:
                w_arr = np.asarray(weights)
                if w_arr.ndim == 2:
                    w_sub = w_arr[i]
            dispersion = estimate_glm_common_disp(
                y_mat[i], design=design, offset=off_sub, weights=w_sub)
    else:
        # Cap dispersion at 4 for non-legacy
        if not legacy:
            dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
            if np.max(dispersion) > 4:
                dispersion = np.minimum(dispersion, 4.0)

    # Fit GLM (prior_count=0.125 matches R's glmFit.default default for logFC shrinkage)
    fit = glm_fit(y_mat, design=design, dispersion=dispersion, offset=offset,
                  lib_size=lib_size, weights=weights, prior_count=0.125)

    # Store AveLogCPM for computation
    ave_log_cpm2 = ave_log_cpm.copy()

    # Covariate for trended prior
    if covariate_trend is None:
        if abundance_trend:
            fit['AveLogCPM'] = ave_log_cpm
        else:
            ave_log_cpm = None
    else:
        ave_log_cpm = covariate_trend

    # Setting residual deviances and df
    if legacy:
        # Old-style: adjust df for fitted values at zero
        zerofit = (fit['fitted.values'] < 1e-4) & (fit['counts'] < 1e-4)
        df_residual = residual_df(zerofit, fit['design'])
        fit['df.residual.zeros'] = df_residual
        s2 = fit['deviance'] / np.maximum(df_residual, 1e-8)
        s2[df_residual == 0] = 0
    else:
        # New-style: adjusted deviance and df using QL weights (matching R's C code)
        from .ql_weights import update_prior, compute_adjust_vec

        # Expand dispersion to matrix form for ql_weights
        disp_arr = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))

        # Compute average quasi-dispersion via iterative lowess + adjusted deviance
        ave_ql_disp = update_prior(y_mat, fit['fitted.values'], design,
                                   disp_arr, weights, ave_log_cpm2)

        # Refit with dispersion scaled by average quasi-dispersion
        fit = glm_fit(y_mat, design=design, dispersion=dispersion / ave_ql_disp,
                      offset=offset, lib_size=lib_size, weights=weights,
                      prior_count=0.125)
        fit['dispersion'] = dispersion

        # Compute adjusted deviance, df, and s2 using QL weights
        out = compute_adjust_vec(y_mat, fit['fitted.values'], design,
                                 disp_arr, ave_ql_disp, weights)
        s2 = out['s2']
        df_residual = out['df']
        fit['df.residual.adj'] = df_residual
        fit['deviance.adj'] = out['deviance']
        fit['average.ql.dispersion'] = ave_ql_disp

    # Empirical Bayes moderation
    s2 = np.maximum(s2, 0)
    s2_fit = squeeze_var(s2, df=df_residual, covariate=ave_log_cpm,
                         robust=robust, winsor_tail_p=winsor_tail_p)

    fit['df.prior'] = s2_fit['df_prior']
    fit['s2.post'] = s2_fit['var_post']
    fit['s2.prior'] = s2_fit['var_prior']
    if not legacy:
        fit['top.proportion'] = top_proportion

    return fit


def _compute_ave_ql_disp(s2, df, ave_log_cpm):
    """Compute average quasi-likelihood dispersion.

    Matches R's update_prior in ql_glm.c: iteratively fits a lowess trend
    of s2^(1/4) vs AveLogCPM, takes the 90th percentile of the trend,
    and raises to 4th power.
    """
    from scipy.interpolate import UnivariateSpline
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    threshold = 1e-8

    # Filter genes with sufficient df
    mask = df > threshold
    x = ave_log_cpm[mask]
    y_vals = np.power(np.maximum(s2[mask], 0), 0.25)  # s2^(1/4)

    if len(x) < 10:
        return 1.0

    # Two iterations of lowess + 90th percentile (matches R's update_prior)
    prior = 1.0
    for _ in range(2):
        # Fit lowess trend (f=0.5, iter=3 matches R defaults)
        fitted = sm_lowess(y_vals, x, frac=0.5, it=3, return_sorted=False)

        # 90th percentile of fitted values (R type=7 quantile)
        p90 = np.percentile(fitted, 90, interpolation='linear')

        # Cap at minimum of 1.0 (on the ^(1/4) scale)
        if p90 < 1.0:
            p90 = 1.0

        prior = p90 ** 4

    return max(prior, 1.0)


def _expand_to_matrix(x, shape):
    """Expand scalar, vector, or CompressedMatrix to full matrix."""
    if isinstance(x, CompressedMatrix):
        return x.as_matrix()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.size == 1:
        return np.full(shape, x.ravel()[0])
    if x.ndim == 1:
        if len(x) == shape[1]:
            return np.tile(x, (shape[0], 1))
        elif len(x) == shape[0]:
            return np.tile(x.reshape(-1, 1), (1, shape[1]))
    if x.shape == shape:
        return x
    return np.broadcast_to(x, shape).copy()
