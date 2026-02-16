# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
GLM-based tests for differential expression in edgePython.

Port of edgeR's glmLRT, glmQLFTest, glmTreat.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist
from scipy.special import gammaln

from .expression import ave_log_cpm
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


def glm_ql_ftest(glmfit, coef=None, contrast=None, poisson_bound=True):
    """Quasi-likelihood F-test for GLM coefficients.

    Port of edgeR's glmQLFTest.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted QL GLM from glm_ql_fit().
    coef : int or list, optional
        Coefficient(s) to test.
    contrast : ndarray, optional
        Contrast vector.
    poisson_bound : bool
        Apply Poisson bound.

    Returns
    -------
    dict (DGELRT-like) with F-statistics and p-values.
    """
    if glmfit.get('s2.post') is None:
        raise ValueError("need to run glm_ql_fit before glm_ql_ftest")

    # Run glmLRT to get the LR statistics
    out = glm_lrt(glmfit, coef=coef, contrast=contrast)

    # Get adjusted df
    if glmfit.get('df.residual.zeros') is None:
        df_residual = glmfit.get('df.residual.adj', glmfit['df.residual'])
        poisson_bound = False
    else:
        df_residual = glmfit['df.residual.zeros']

    df_residual = np.asarray(df_residual, dtype=np.float64)

    # Compute F-statistic
    df_test = out['df.test']
    if np.isscalar(df_test):
        df_test_val = float(df_test)
    else:
        df_test_val = np.asarray(df_test, dtype=np.float64)

    F_stat = out['table']['LR'].values / df_test_val / glmfit['s2.post']

    df_prior = np.atleast_1d(np.asarray(glmfit['df.prior'], dtype=np.float64))
    df_total = df_prior + df_residual

    # Cap df.total
    df_residual_total = np.sum(glmfit['df.residual'])
    df_total = np.minimum(df_total, df_residual_total)

    # P-values from F-distribution
    F_pvalue = f_dist.sf(np.maximum(F_stat, 0), dfn=df_test_val, dfd=df_total)

    # Update output
    out['table'].drop(columns=['LR'], inplace=True, errors='ignore')
    out['table']['F'] = F_stat
    out['table']['PValue'] = F_pvalue
    out['df.total'] = df_total

    return out


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
