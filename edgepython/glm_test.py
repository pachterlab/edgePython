# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
GLM-based tests for differential expression in edgePython.

Port of edgeR's glmLRT, glmQLFTest, glmTreat.
"""

import warnings

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
    keep = [j for j in range(design.shape[1]) if j not in test_coef]
    design0 = design[:, keep]
    design1 = design[:, test_coef]
    offset = _expand_compressed(offset, shape)
    disp = _expand_dispersion(dispersion, shape)
    w = None if weights is None else _expand_compressed(weights, shape)

    def null_deviance(rows, off):
        return glm_fit(counts[rows], design=design0, dispersion=disp[rows], offset=off,
                       weights=None if w is None else w[rows], prior_count=0)['deviance']

    return _ftest_engine(null_deviance, offset, design1, deviance, s2_post, df_total, lfc,
                         logFCt, upshot)


def _ftest_engine(null_deviance, offset, design1, deviance, s2_post, df_total, lfc,
                  logFCt, upshot):
    """edgeR 5.0's ftest_engine, shared by the NB and binomial QL F-tests.

    null_deviance(rows, off) refits the null model for those genes with
    offsets off (None for the base fit when there is no base offset).
    offset is the base offset matrix, or None (binomial).
    """
    ngenes = deviance.shape[0]
    ncoef = design1.shape[1]

    def ftest(dev0, rows):
        F = np.maximum(dev0 - deviance[rows], 0) / ncoef / s2_post[rows]
        return F, f_dist.sf(F, ncoef, df_total[rows])

    all_rows = np.arange(ngenes)
    F_stat, p_base = ftest(null_deviance(all_rows, offset), all_rows)
    if not np.any(lfc > 0):
        return F_stat, p_base

    nodes = _UPSHOT_NODES if upshot else np.array([1.0])
    pvalue = _UPSHOT_CENTRE * p_base if upshot else None
    abs_lfc = np.abs(logFCt) * np.log(2)
    for i, s in enumerate(nodes):
        node_p = np.ones(ngenes)
        thr = lfc * s * np.log(2)
        rows = np.flatnonzero(np.any(abs_lfc > thr, axis=1))
        if rows.size:
            pinned = np.minimum(thr, abs_lfc[rows])
            min_dev = np.full(rows.size, np.inf)
            for b in range(2 ** ncoef):
                sign = np.array([2.0 * ((b >> c) & 1) - 1.0 for c in range(ncoef)])
                off = (pinned * sign) @ design1.T
                if offset is not None:
                    off = offset[rows] + off
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
              null='interval', legacy=False):
    """Likelihood ratio or quasi-likelihood test with a log-FC threshold.

    Port of edgeR 5.0's glmTreat. The null model is refitted with the
    tested coefficient fixed at +lfc and at -lfc, giving a deviance test at
    each boundary. By default (legacy=False, new in edgeR 5.0) each
    boundary's F-test (QL fits) or LR test is converted to a signed
    z-score; legacy=True gives edgeR 4's statistics.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted GLM from glm_fit() or glm_ql_fit().
    coef : int, optional
        Coefficient to test (0-based). Defaults to the last.
    contrast : ndarray, optional
        Contrast vector; only the first column of a matrix is used.
    lfc : float
        Log2-fold-change threshold.
    null : str
        'interval' or 'worst.case'.
    legacy : bool
        Use edgeR 4's statistics.

    Returns
    -------
    dict (DGELRT-like) with table including p-values.
    """
    from scipy.stats import norm as norm_dist
    from .glm_fit import glm_fit

    if lfc < 0:
        raise ValueError("lfc has to be non-negative")
    is_lrt = glmfit.get('df.prior') is None
    if lfc == 0:
        if is_lrt:
            return glm_lrt(glmfit, coef=coef, contrast=contrast)
        return glm_ql_ftest(glmfit, coef=coef, contrast=contrast)
    if null not in ('interval', 'worst.case'):
        raise ValueError("null should be 'interval' or 'worst.case'")

    shrunk = glmfit.get('prior.count', 0) != 0
    if glmfit.get('AveLogCPM') is None:
        glmfit['AveLogCPM'] = ave_log_cpm(glmfit)
    design = np.asarray(glmfit['design'], dtype=np.float64)
    nbeta = design.shape[1]
    if nbeta < 2:
        raise ValueError("Need at least two columns for design, usually the first is the intercept column")
    coef_names = [f'coef{i}' for i in range(nbeta)]
    unshrunk = glmfit.get('unshrunk.coefficients')

    if contrast is None:
        if coef is None:
            coef = nbeta - 1
        coef_idx = int(np.atleast_1d(coef)[0])
        coef_name = coef_names[coef_idx]
        logFC = glmfit['coefficients'][:, coef_idx] / np.log(2)
        unshrunk_logFC = logFC
        if shrunk:
            unshrunk_logFC = np.asarray(unshrunk)[:, coef_idx] / np.log(2)
    else:
        contrast = np.asarray(contrast, dtype=np.float64)
        if contrast.ndim == 2:
            if contrast.shape[1] > 1:
                warnings.warn("Found more than one contrast vector in `contrast`. Using first column only.")
            contrast = contrast[:, 0]
        reform = contrast_as_coef(design, contrast, first=True)
        coef_idx = 0
        logFC = (glmfit['coefficients'] @ contrast) / np.log(2)
        unshrunk_logFC = logFC
        if shrunk:
            unshrunk_logFC = (np.asarray(unshrunk) @ contrast) / np.log(2)
        coef_name = " ".join(f"{contrast[j]:g}*{coef_names[j]}"
                             for j in range(nbeta) if contrast[j] != 0)
        design = reform['design']
    unshrunk_logFC = np.asarray(unshrunk_logFC, dtype=np.float64)

    # Null model refitted with the tested coefficient fixed at +lfc and -lfc
    design0 = np.delete(design, coef_idx, axis=1)
    shape = np.shape(glmfit['counts'])
    offset = _expand_compressed(glmfit['offset'], shape)
    offset_adj = lfc * np.log(2) * design[:, coef_idx]
    dispersion = glmfit.get('dispersion')
    if glmfit.get('average.ql.dispersion') is not None:
        dispersion = np.asarray(dispersion, dtype=np.float64) / glmfit['average.ql.dispersion']
    fit_args = dict(design=design0, weights=glmfit.get('weights'), dispersion=dispersion,
                    prior_count=0)
    dev_left = glm_fit(glmfit['counts'], offset=offset + offset_adj, **fit_args)['deviance']
    dev_right = glm_fit(glmfit['counts'], offset=offset - offset_adj, **fit_args)['deviance']
    lr_left = np.maximum(0, dev_left - glmfit['deviance'])
    lr_right = np.maximum(0, dev_right - glmfit['deviance'])

    poisson_bound = False
    if legacy:
        # edgeR 4: order the two statistics, then sign them
        swap = lr_left > lr_right
        lr_left, lr_right = np.where(swap, lr_right, lr_left), np.where(swap, lr_left, lr_right)
    if is_lrt:
        if legacy:
            z_left, z_right = np.sqrt(lr_left), np.sqrt(lr_right)
        else:
            ind = np.sign(unshrunk_logFC)
            z_left = ind * np.sign(unshrunk_logFC - lfc) * (-np.sqrt(lr_left))
            z_right = ind * np.sign(unshrunk_logFC + lfc) * (-np.sqrt(lr_right))
    else:
        if glmfit.get('df.residual.zeros') is None:
            df_residual = glmfit['df.residual.adj']
        else:
            df_residual = glmfit['df.residual.zeros']
            # edgeR 4's value; edgeR 4.99.8 leaves it unset and fails here
            poisson_bound = True
        df_total = np.minimum(np.asarray(glmfit['df.prior']) + np.asarray(df_residual),
                              np.sum(glmfit['df.residual']))
        s2_post = np.asarray(glmfit['s2.post'], dtype=np.float64)
        if legacy:
            z_left = _zscore_t(np.sqrt(lr_left) / np.sqrt(s2_post), df_total)
            z_right = _zscore_t(np.sqrt(lr_right) / np.sqrt(s2_post), df_total)
        else:
            p_left = f_dist.sf(lr_left / s2_post, 1, df_total)
            p_right = f_dist.sf(lr_right / s2_post, 1, df_total)
            ind = np.sign(unshrunk_logFC)
            z_left = ind * np.sign(unshrunk_logFC - lfc) * norm_dist.ppf(p_left / 2)
            z_right = ind * np.sign(unshrunk_logFC + lfc) * norm_dist.ppf(p_right / 2)
    if legacy:
        # NA logFC (all-zero genes) gives NA, as in R
        within = np.where(np.isnan(unshrunk_logFC), np.nan, np.abs(unshrunk_logFC) <= lfc)
        z_left = z_left * (2.0 * within - 1)
        z_right = -z_right

    if null == 'interval':
        p_value = 2 * _integrate_pnorm(z_right, z_left)
        if legacy:
            # Genewise. (edgeR drops NAs from this logical index, which
            # misaligns later genes when some genes are NA.)
            c = 1.470402
            with np.errstate(invalid='ignore'):
                j = (-z_right + z_left) > c
            p_value[j] = (_integrate_pnorm(z_right[j], z_right[j] + c) +
                          _integrate_pnorm(z_left[j] - c, z_left[j]))
    else:
        p_value = norm_dist.cdf(z_right) + norm_dist.cdf(z_left)

    if poisson_bound:
        below = _is_below_poisson_bound(glmfit)
        if np.any(below):
            weights = glmfit.get('weights')
            if weights is not None:
                weights = _expand_compressed(weights, shape)[below]
            pois_fit = glm_fit(
                np.asarray(glmfit['counts'], dtype=np.float64)[below],
                design=glmfit['design'], offset=offset[below], weights=weights,
                dispersion=0.0, start=np.asarray(glmfit['unshrunk.coefficients'])[below])
            pois_fit['AveLogCPM'] = np.asarray(glmfit['AveLogCPM'])[below]
            # As in edgeR, legacy and null are not passed on
            pois_res = glm_treat(pois_fit, coef=coef_idx if contrast is None else None,
                                 contrast=contrast, lfc=lfc)
            p_value[below] = np.maximum(p_value[below], pois_res['table']['PValue'].to_numpy())

    columns = {'logFC': logFC}
    if shrunk:
        columns['unshrunk.logFC'] = unshrunk_logFC
    columns.update({'logCPM': glmfit['AveLogCPM'], 'PValue': p_value})
    result = dict(glmfit)
    result.pop('counts', None)
    result['lfc'] = lfc
    result['table'] = pd.DataFrame(columns)
    result['comparison'] = coef_name
    return result


def _zscore_t(x, df):
    """z-scores with the same tail probabilities as t-statistics.

    Port of limma's zscoreT(approx = FALSE), on the log scale throughout.
    """
    from scipy.stats import t as t_dist
    x = np.asarray(x, dtype=np.float64)
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), x.shape)
    log_p = t_dist.logsf(np.abs(x), df)
    try:
        from scipy.special import ndtri_exp
        z = -ndtri_exp(log_p)
    except ImportError:
        from scipy.stats import norm as norm_dist
        z = norm_dist.isf(np.exp(log_p))
    return np.where(x > 0, z, -z)


def _integrate_pnorm(a, b):
    """Average of the standard normal CDF over [a, b].

    Port of edgeR's .integratepnorm; a and b may be in either order.
    """
    from scipy.stats import norm as norm_dist
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = (b * norm_dist.cdf(b) + norm_dist.pdf(b)
               - (a * norm_dist.cdf(a) + norm_dist.pdf(a))) / (b - a)
    return np.where(a == b, norm_dist.cdf(a), avg)
