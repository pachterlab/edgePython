# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
GLM-based tests for differential expression in edgePython.

Port of edgeR's glmLRT, glmQLFTest, glmTreat.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist
from scipy.special import betainccinv, betaincinv, gammaln

from .expression import ave_log_cpm
from .glm_levenberg import _expand_compressed, _expand_dispersion
from .limma_port import contrast_as_coef


def glm_lrt(glmfit, coef=None, contrast=None):
    """Likelihood ratio test for GLM coefficients.

    Port of edgeR's glmLRT.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted GLM object from glm_fit().
    coef : int, list of int, or str, optional
        Coefficient(s) to test. Default is last column.
    contrast : ndarray, optional
        Contrast vector or matrix.

    Returns
    -------
    dict (DGELRT-like) with 'table', 'comparison', 'df.test'.
    """
    if glmfit.get('AveLogCPM') is None:
        glmfit['AveLogCPM'] = ave_log_cpm(glmfit)

    design = np.asarray(glmfit['design'], dtype=np.float64)
    nbeta = design.shape[1]
    nlibs = design.shape[0]

    if nbeta < 2:
        raise ValueError("Need at least two columns for design")

    coef_names = [f'coef{i}' for i in range(nbeta)]

    # Determine coefficients to test
    if contrast is None:
        if coef is None:
            coef = nbeta - 1  # last column (0-indexed)

        if isinstance(coef, (int, np.integer)):
            coef = [coef]
        coef = list(set(coef))
        coef_name = [coef_names[c] for c in coef]

        logFC = glmfit['coefficients'][:, coef] / np.log(2)
    else:
        contrast = np.asarray(contrast, dtype=np.float64)
        if contrast.ndim == 1:
            contrast = contrast.reshape(-1, 1)

        ncontrasts = np.linalg.matrix_rank(contrast)
        if ncontrasts == 0:
            raise ValueError("contrasts are all zero")

        coef = list(range(ncontrasts))
        logFC = (glmfit['coefficients'] @ contrast) / np.log(2)

        # Reform design
        Q, R = np.linalg.qr(contrast, mode='complete')
        design = design @ Q

        if ncontrasts > 1:
            coef_name = f"LR test on {ncontrasts} degrees of freedom"
        else:
            coef_name = "contrast"

    if len(coef) == 1 and logFC.ndim == 2:
        logFC = logFC.ravel()

    # Null design matrix
    keep_cols = [i for i in range(design.shape[1]) if i not in coef]
    design0 = design[:, keep_cols]

    # Null fit
    from .glm_fit import glm_fit
    dispersion = glmfit.get('dispersion')
    if glmfit.get('average.ql.dispersion') is not None:
        dispersion = np.asarray(dispersion, dtype=np.float64) / glmfit['average.ql.dispersion']

    # As in edgeR, the null model is fitted from edgeR's null starting values.
    # Starting from the full model's coefficients is not safe: coefficients of
    # groups that are entirely zero are -1e8.
    fit_null = glm_fit(glmfit['counts'], design=design0,
                       offset=glmfit.get('offset'),
                       weights=glmfit.get('weights'),
                       dispersion=dispersion, prior_count=0)

    # Likelihood ratio statistic
    LR = fit_null['deviance'] - glmfit['deviance']
    df_test = np.asarray(fit_null['df.residual']) - np.asarray(glmfit['df.residual'])
    df_test_val = df_test[0] if np.all(df_test == df_test[0]) else df_test

    LRT_pvalue = chi2.sf(np.maximum(LR, 0), df=df_test_val)

    # Build output table
    table = pd.DataFrame({
        'logFC': logFC if logFC.ndim == 1 else logFC[:, 0],
        'logCPM': glmfit['AveLogCPM'],
        'LR': LR,
        'PValue': LRT_pvalue
    })

    result = dict(glmfit)
    result.pop('counts', None)
    result['table'] = table
    result['comparison'] = coef_name
    result['df.test'] = df_test_val

    return result


def glm_ql_ftest(glmfit, coef=None, contrast=None, fc=1.0, lfc=None,
                 poisson_bound=True, upshot=True):
    """Quasi-likelihood F-test for GLM coefficients.

    Port of edgeR 5.0's glmQLFTest. With a fold-change threshold (fc > 1 or
    lfc > 0) the test is a TREAT test of H0: |logFC| <= lfc; with upshot it
    is UPSHOT, which averages the TREAT p-value over thresholds between 0
    and lfc by Gauss quadrature.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted QL GLM from glm_ql_fit().
    coef : int or list, optional
        Coefficient(s) to test (0-based). Defaults to the last.
    contrast : ndarray, optional
        Contrast vector, or matrix with one column per contrast.
    fc : float
        Fold-change threshold; ignored if lfc is given.
    lfc : float or array, optional
        log2 fold-change threshold(s), one per tested coefficient or a
        single value for all. Defaults to log2(fc).
    poisson_bound : bool
        For legacy fits, raise p-values to those of a Poisson LRT for
        genes whose quasi-dispersion is below the Poisson bound.
    upshot : bool
        With a threshold, use UPSHOT rather than plain TREAT.

    Returns
    -------
    dict (DGELRT-like) with F-statistics and p-values.
    """
    if glmfit.get('s2.post') is None:
        raise ValueError("need to run glm_ql_fit before glm_ql_ftest")

    design = np.asarray(glmfit['design'], dtype=np.float64)
    nbeta = design.shape[1]
    if nbeta < 2:
        raise ValueError("Need at least two columns for design, usually the first is the intercept column")
    coef_names = [f'coef{i}' for i in range(nbeta)]
    unshrunk = glmfit.get('unshrunk.coefficients')

    if contrast is None:
        if coef is None:
            coef = nbeta - 1
        coef = list(dict.fromkeys(np.atleast_1d(coef).tolist()))
        coef_name = [coef_names[c] for c in coef]
        logFC = glmfit['coefficients'][:, coef] / np.log(2)
        logFCt = logFC if unshrunk is None else np.asarray(unshrunk)[:, coef] / np.log(2)
        test_coef = coef
        logfc_names = coef_name
    else:
        contrast = np.asarray(contrast, dtype=np.float64)
        contrast_mat = contrast.reshape(-1, 1) if contrast.ndim == 1 else contrast
        reform = contrast_as_coef(design, contrast_mat, first=True)
        test_coef = reform['coef']
        design = reform['design']
        if len(test_coef) > 1:
            coef_name = f"LR test on {len(test_coef)} degrees of freedom"
        else:
            c = contrast_mat[:, 0]
            coef_name = " ".join(f"{c[j]:g}*{coef_names[j]}" for j in range(nbeta) if c[j] != 0)
        logFC = (glmfit['coefficients'] @ contrast_mat) / np.log(2)
        logFCt = logFC if unshrunk is None else (np.asarray(unshrunk) @ contrast_mat) / np.log(2)
        logfc_names = [str(j + 1) for j in range(contrast_mat.shape[1])]
    ncoef = len(test_coef)

    if lfc is None:
        lfc = np.log2(fc)
    lfc = np.atleast_1d(np.asarray(lfc, dtype=np.float64))
    if lfc.size == 1:
        lfc = np.full(ncoef, lfc[0])
    if np.any(lfc < 0):
        raise ValueError("lfc should be non-negative")
    if lfc.size != ncoef:
        raise ValueError("fc or lfc vector of wrong length")

    # Dispersion used for the fit, scaled by the average quasi-dispersion
    # for the default (legacy = False) method
    dispersion = glmfit.get('dispersion')
    if glmfit.get('average.ql.dispersion') is not None:
        dispersion = np.asarray(dispersion, dtype=np.float64) / glmfit['average.ql.dispersion']

    if glmfit.get('df.residual.zeros') is None:
        df_residual = glmfit.get('df.residual.adj', glmfit['df.residual'])
        poisson_bound = False
    else:
        df_residual = glmfit['df.residual.zeros']
    df_residual = np.asarray(df_residual, dtype=np.float64)
    df_prior = np.atleast_1d(np.asarray(glmfit['df.prior'], dtype=np.float64))
    df_total = np.minimum(df_prior + df_residual, np.sum(glmfit['df.residual']))

    counts = np.asarray(glmfit['counts'], dtype=np.float64)
    F_stat, F_pvalue = _ql_ftest(
        counts, glmfit['offset'], dispersion, glmfit.get('weights'), design, test_coef,
        np.asarray(glmfit['deviance'], dtype=np.float64),
        np.broadcast_to(np.asarray(glmfit['s2.post'], dtype=np.float64), counts.shape[:1]),
        np.broadcast_to(df_total, counts.shape[:1]), lfc,
        np.asarray(logFCt, dtype=np.float64).reshape(counts.shape[0], ncoef), upshot)

    # Poisson bound, as in edgeR's glmQLFTest: for genes where
    # s2.post * (1 + dispersion * fitted) < 1 in any library, the p-value is
    # the larger of the F-test's and that of a Poisson LRT, refitted at
    # dispersion 0 from the unshrunk coefficients.
    if poisson_bound:
        below = _is_below_poisson_bound(glmfit)
        if np.any(below):
            from .glm_fit import glm_fit
            shape = np.shape(glmfit['counts'])
            weights = glmfit.get('weights')
            if weights is not None:
                weights = _expand_compressed(weights, shape)[below]
            pois_fit = glm_fit(
                np.asarray(glmfit['counts'], dtype=np.float64)[below],
                design=glmfit['design'],
                offset=_expand_compressed(glmfit['offset'], shape)[below],
                weights=weights, dispersion=0.0,
                start=np.asarray(glmfit['unshrunk.coefficients'])[below])
            pois_fit['AveLogCPM'] = np.asarray(glmfit['AveLogCPM'])[below]
            pois_res = glm_lrt(pois_fit, coef=coef, contrast=contrast)
            pois_pvalue = pois_res['table']['PValue'].to_numpy(dtype=np.float64)
            F_pvalue[below] = np.maximum(F_pvalue[below], pois_pvalue)

    if ncoef == 1:
        logfc_cols = {'logFC': np.asarray(logFC).reshape(-1)}
    else:
        logfc_cols = {f'logFC.{n}': logFC[:, k] for k, n in enumerate(logfc_names)}
    table = pd.DataFrame({**logfc_cols, 'logCPM': glmfit['AveLogCPM'],
                          'F': F_stat, 'PValue': F_pvalue})

    out = dict(glmfit)
    out.pop('counts', None)
    if np.any(lfc > 0):
        out['lfc'] = lfc
    out['table'] = table
    out['comparison'] = coef_name
    out['df.test'] = ncoef
    out['df.total'] = df_total
    return out


# UPSHOT Gauss quadrature rule: 8 folded nodes on [0, 1] and the centre
# weight, as in edgeR's ftest.c
_UPSHOT_NODES = np.array([0.1784842, 0.3512318, 0.5126905, 0.6576712,
                          0.7815140, 0.8802392, 0.9506755, 0.9905755])
_UPSHOT_WEIGHTS = np.array([0.1765627, 0.1680041, 0.1540458, 0.1351364,
                            0.1118839, 0.0850362, 0.0554595, 0.0241483])
_UPSHOT_CENTRE = 0.08972310


def _ql_ftest(counts, offset, dispersion, weights, design, test_coef, deviance,
              s2_post, df_total, lfc, logFCt, upshot):
    """QL F-statistics and p-values, with TREAT or UPSHOT thresholds.

    Port of edgeR 5.0's ftest_engine. The null model drops the tested
    columns. With a threshold, a gene whose |logFC| exceeds lfc * s for a
    quadrature node s is refitted with the tested coefficients pinned at
    each corner of the box +/- min(lfc * s, |logFC|), and the smallest of
    those deviances gives the node's F-test.
    """
    from .glm_fit import glm_fit
    shape = counts.shape
    ncoef = len(test_coef)
    keep = [j for j in range(design.shape[1]) if j not in test_coef]
    design0 = design[:, keep]
    design1 = design[:, test_coef]
    offset = _expand_compressed(offset, shape)
    disp = _expand_dispersion(dispersion, shape)
    w = None if weights is None else _expand_compressed(weights, shape)

    def null_deviance(rows, off):
        return glm_fit(counts[rows], design=design0, dispersion=disp[rows], offset=off,
                       weights=None if w is None else w[rows], prior_count=0)['deviance']

    def ftest(dev0, rows):
        F = np.maximum(dev0 - deviance[rows], 0) / ncoef / s2_post[rows]
        return F, f_dist.sf(F, ncoef, df_total[rows])

    all_rows = np.arange(shape[0])
    F_stat, p_base = ftest(null_deviance(all_rows, offset), all_rows)
    if not np.any(lfc > 0):
        return F_stat, p_base

    nodes = _UPSHOT_NODES if upshot else np.array([1.0])
    pvalue = _UPSHOT_CENTRE * p_base if upshot else None
    abs_lfc = np.abs(logFCt) * np.log(2)
    for i, s in enumerate(nodes):
        node_p = np.ones(shape[0])
        thr = lfc * s * np.log(2)
        rows = np.flatnonzero(np.any(abs_lfc > thr, axis=1))
        if rows.size:
            pinned = np.minimum(thr, abs_lfc[rows])
            min_dev = np.full(rows.size, np.inf)
            for b in range(2 ** ncoef):
                sign = np.array([2.0 * ((b >> c) & 1) - 1.0 for c in range(ncoef)])
                off = offset[rows] + (pinned * sign) @ design1.T
                min_dev = np.minimum(min_dev, null_deviance(rows, off))
            node_p[rows] = ftest(min_dev, rows)[1]
        if upshot:
            pvalue = pvalue + _UPSHOT_WEIGHTS[i] * node_p
        else:
            pvalue = (p_base + node_p) / 2
    return _f_isf(pvalue, ncoef, df_total), pvalue


def _f_isf(p, dfn, dfd):
    """Upper-tail F quantile, qf(p, dfn, dfd, lower.tail = FALSE).

    Inverts the regularized incomplete beta function directly. scipy's
    f.isf works from 1 - p and returns inf for p below about 1e-17.
    """
    p, dfn, dfd = np.broadcast_arrays(np.asarray(p, dtype=np.float64),
                                      np.asarray(dfn, dtype=np.float64),
                                      np.asarray(dfd, dtype=np.float64))
    out = np.empty(p.shape)
    small = p < 0.5
    # P(F > x) = I_w(dfd/2, dfn/2) with w = dfd / (dfd + dfn x)
    w = betaincinv(dfd[small] / 2, dfn[small] / 2, p[small])
    out[small] = dfd[small] * (1 - w) / (dfn[small] * w)
    # P(F > x) = 1 - I_u(dfn/2, dfd/2) with u = dfn x / (dfn x + dfd)
    u = betainccinv(dfn[~small] / 2, dfd[~small] / 2, p[~small])
    out[~small] = dfd[~small] * u / (dfn[~small] * (1 - u))
    return out


def _is_below_poisson_bound(glmfit):
    """Port of edgeR's .isBelowPoissonBound: one flag per gene."""
    fitted = np.asarray(glmfit['fitted.values'], dtype=np.float64)
    ngenes = fitted.shape[0]
    disp = np.asarray(glmfit['dispersion'], dtype=np.float64)
    if disp.ndim < 2:
        disp = np.broadcast_to(disp.reshape(-1), ngenes)[:, None]
    s2 = np.broadcast_to(
        np.asarray(glmfit['s2.post'], dtype=np.float64).reshape(-1), ngenes)
    return np.any(s2[:, None] * (1 + disp * fitted) < 1, axis=1)


def glm_treat(glmfit, coef=None, contrast=None, lfc=np.log2(1.2),
              null='interval'):
    """Likelihood ratio or quasi-likelihood test with a log-FC threshold.

    Port of edgeR's glmTreat.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted GLM from glm_fit() or glm_ql_fit().
    coef : int, optional
        Coefficient to test.
    contrast : ndarray, optional
        Contrast vector.
    lfc : float
        Log2-fold-change threshold.
    null : str
        'interval' or 'worst.case'.

    Returns
    -------
    dict (DGELRT-like) with table including p-values.
    """
    from scipy.stats import norm as norm_dist, t as t_dist
    from .glm_fit import glm_fit
    from .compressed_matrix import CompressedMatrix

    if lfc < 0:
        raise ValueError("lfc has to be non-negative")

    is_lrt = glmfit.get('df.prior') is None

    # If lfc is zero, fall back to standard test
    if lfc == 0:
        if is_lrt:
            return glm_lrt(glmfit, coef=coef, contrast=contrast)
        else:
            return glm_ql_ftest(glmfit, coef=coef, contrast=contrast)

    if glmfit.get('AveLogCPM') is None:
        glmfit['AveLogCPM'] = ave_log_cpm(glmfit)
    ngenes = glmfit['counts'].shape[0]

    design = np.asarray(glmfit['design'], dtype=np.float64)
    nbeta = design.shape[1]

    if nbeta < 2:
        raise ValueError("Need at least two columns for design")

    # Determine coefficient to test
    if coef is None:
        coef = nbeta - 1

    shrunk = glmfit.get('prior.count', 0) != 0

    if contrast is None:
        if isinstance(coef, (int, np.integer)):
            coef_idx = coef
        else:
            coef_idx = coef[0]
        # R: logFC uses shrunk coefficients for display, unshrunk for test
        logFC = glmfit['coefficients'][:, coef_idx] / np.log(2)
        unshrunk_logFC = logFC.copy()
        if shrunk and glmfit.get('unshrunk.coefficients') is not None:
            unshrunk_logFC = glmfit['unshrunk.coefficients'][:, coef_idx] / np.log(2)
    else:
        contrast = np.asarray(contrast, dtype=np.float64).ravel()
        reform = contrast_as_coef(design, contrast, first=True)
        coef_idx = 0
        logFC = (glmfit['coefficients'] @ contrast) / np.log(2)
        unshrunk_logFC = logFC.copy()
        if shrunk and glmfit.get('unshrunk.coefficients') is not None:
            unshrunk_logFC = (glmfit['unshrunk.coefficients'] @ contrast) / np.log(2)
        design = reform['design']

    # Null design matrix
    keep_cols = [i for i in range(design.shape[1]) if i != coef_idx]
    design0 = design[:, keep_cols]

    # Get dispersion
    dispersion = glmfit.get('dispersion')
    if glmfit.get('average.ql.dispersion') is not None:
        dispersion = np.asarray(dispersion, dtype=np.float64) / glmfit['average.ql.dispersion']

    # Offset adjustment
    offset = np.asarray(glmfit.get('offset', np.zeros((ngenes, design.shape[0]))),
                        dtype=np.float64)
    if offset.ndim == 1:
        offset = np.tile(offset, (ngenes, 1))

    offset_adj = lfc * np.log(2) * design[:, coef_idx]
    offset_adj_mat = np.tile(offset_adj, (ngenes, 1))

    # Test at beta_0 = +tau
    offset_new = offset + offset_adj_mat
    fit0 = glm_fit(glmfit['counts'], design=design0, offset=offset_new,
                   weights=glmfit.get('weights'), dispersion=dispersion,
                   prior_count=0)
    fit1 = glm_fit(glmfit['counts'], design=design, offset=offset_new,
                   weights=glmfit.get('weights'), dispersion=dispersion,
                   prior_count=0)
    z_left = np.sqrt(np.maximum(0, fit0['deviance'] - fit1['deviance']))

    # Test at beta_0 = -tau
    offset_new = offset - offset_adj_mat
    fit0 = glm_fit(glmfit['counts'], design=design0, offset=offset_new,
                   weights=glmfit.get('weights'), dispersion=dispersion,
                   prior_count=0)
    fit1 = glm_fit(glmfit['counts'], design=design, offset=offset_new,
                   weights=glmfit.get('weights'), dispersion=dispersion,
                   prior_count=0)
    z_right = np.sqrt(np.maximum(0, fit0['deviance'] - fit1['deviance']))

    # Make sure z_left <= z_right
    swap = z_left > z_right
    z_left_tmp = z_left.copy()
    z_left[swap] = z_right[swap]
    z_right[swap] = z_left_tmp[swap]

    # Convert t to z under QL pipeline
    if not is_lrt:
        if glmfit.get('df.residual.zeros') is None:
            df_residual = glmfit.get('df.residual.adj', glmfit['df.residual'])
        else:
            df_residual = glmfit['df.residual.zeros']

        df_total = np.asarray(glmfit['df.prior']) + np.asarray(df_residual)
        s2_post = np.asarray(glmfit['s2.post'])
        z_left = _zscore_t(z_left / np.sqrt(s2_post), df_total)
        z_right = _zscore_t(z_right / np.sqrt(s2_post), df_total)

    # Apply sign based on whether |logFC| <= lfc
    within = np.abs(unshrunk_logFC) <= lfc
    sgn = 2 * within.astype(float) - 1
    z_left = z_left * sgn

    # Compute p-values
    if null == 'interval':
        c = 1.470402
        j = (z_right + z_left) > c
        p_value = np.ones(ngenes)
        p_value[j] = (_integrate_pnorm(-z_right[j], -z_right[j] + c) +
                       _integrate_pnorm(z_left[j] - c, z_left[j]))
        p_value[~j] = 2 * _integrate_pnorm(-z_right[~j], z_left[~j])
    else:
        p_value = norm_dist.cdf(-z_right) + norm_dist.cdf(z_left)

    # Build table — use shrunk logFC for display (matching R)
    table = pd.DataFrame({
        'logFC': logFC,
        'logCPM': glmfit['AveLogCPM'],
        'PValue': p_value
    })

    result = dict(glmfit)
    result.pop('counts', None)
    result['lfc'] = lfc
    result['table'] = table
    result['comparison'] = f'coef{coef_idx}'

    return result


def _zscore_t(x, df):
    """Convert t-statistics to z-scores."""
    from scipy.stats import t as t_dist, norm as norm_dist
    df = np.asarray(df, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    # Use log p-value for precision
    log_p = t_dist.logsf(np.abs(x), df)
    z = norm_dist.isf(np.exp(log_p))
    z = np.where(x < 0, -z, z)
    return z


def _integrate_pnorm(a, b):
    """Integrate the standard normal CDF from a to b.

    Port of edgeR's .integratepnorm.
    """
    from scipy.stats import norm as norm_dist
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    equal = np.abs(a - b) < 1e-15
    result = np.where(
        equal,
        norm_dist.cdf(a),
        (b * norm_dist.cdf(b) + norm_dist.pdf(b) -
         (a * norm_dist.cdf(a) + norm_dist.pdf(a))) / np.maximum(b - a, 1e-15)
    )
    return result
