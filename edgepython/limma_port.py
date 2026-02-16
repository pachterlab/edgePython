# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Essential limma functions ported for edgePython.

Port of limma's squeezeVar, contrastAsCoef, nonEstimable, is.fullrank,
chooseLowessSpan, and related utility functions.
"""

import numpy as np
from scipy import stats, interpolate
import warnings


def squeeze_var(var, df, covariate=None, span=None, robust=False, winsor_tail_p=(0.05, 0.1), legacy=None):
    """Empirical Bayes moderation of genewise variances.

    Port of limma's squeezeVar().

    Parameters
    ----------
    var : array-like
        Genewise variances.
    df : array-like
        Residual degrees of freedom.
    covariate : array-like, optional
        Covariate for trended prior.
    span : float, optional
        Loess span. If provided, forces legacy=False.
    robust : bool
        Use robust estimation.
    winsor_tail_p : tuple
        Tail proportions for Winsorization when robust=True (legacy only).
    legacy : bool or None
        If True, use original limma algorithm (fitFDist).
        If False, use fitFDistUnequalDF1.
        If None (default), auto-detect based on whether df values are equal.

    Returns
    -------
    dict with keys: var_post, var_prior, df_prior
    """
    var = np.asarray(var, dtype=np.float64)
    n = len(var)

    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return {'var_post': var.copy(), 'var_prior': var.copy(), 'df_prior': 0.0}

    df = np.atleast_1d(np.asarray(df, dtype=np.float64))
    if len(df) == 1:
        df = np.full(n, df[0])

    # When df==0, guard against missing or infinite values in var
    var = var.copy()
    var[df == 0] = 0

    # Auto-detect legacy mode
    if span is not None:
        legacy = False
    if legacy is None:
        dfp = df[df > 0]
        if len(dfp) > 0:
            legacy = (np.min(dfp) == np.max(dfp))
        else:
            legacy = True

    if legacy:
        # Original limma algorithm (fitFDist / fitFDistRobustly)
        ok = np.isfinite(var) & np.isfinite(df) & (df > 0)
        if not np.any(ok):
            return {'var_post': var, 'var_prior': np.nan, 'df_prior': 0.0}

        if covariate is not None:
            covariate = np.asarray(covariate, dtype=np.float64)

        if robust:
            cov_arg = covariate
            if cov_arg is not None and len(np.unique(cov_arg[ok])) < 2:
                cov_arg = None
            fit = _fit_f_dist_robustly(var, df, covariate=cov_arg,
                                        winsor_tail_p=winsor_tail_p)
            var_prior = fit['scale']
            df_prior = fit['df2_shrunk']
            var_post = _posterior_var(var, df, var_prior, df_prior)
            return {'var_post': var_post, 'var_prior': var_prior, 'df_prior': df_prior}

        # Estimate prior (non-robust)
        if covariate is None or len(np.unique(covariate[ok])) < 2:
            # No trend
            result = _fit_f_dist(var[ok], df[ok])
            df_prior = result['df2']
            var_prior = result['s2']
            var_post = _posterior_var(var, df, var_prior, df_prior)
            return {'var_post': var_post, 'var_prior': var_prior, 'df_prior': df_prior}
        else:
            # Trended prior
            result = _fit_f_dist_trend(var[ok], df[ok], covariate[ok])
            var_prior_full = np.full(n, np.nan)
            var_prior_full[ok] = result['var_prior']
            if not np.all(ok):
                from scipy.interpolate import interp1d
                f = interp1d(covariate[ok], result['var_prior'], kind='linear',
                             bounds_error=False, fill_value='extrapolate')
                var_prior_full[~ok] = f(covariate[~ok])
            df_prior = result['df_prior']
            var_post = _posterior_var(var, df, var_prior_full, df_prior)
            return {'var_post': var_post, 'var_prior': var_prior_full, 'df_prior': df_prior}
    else:
        # New method: fitFDistUnequalDF1
        fit = _fit_f_dist_unequal_df1(var, df, covariate=covariate, span=span, robust=robust)
        df_prior = fit.get('df2_shrunk')
        if df_prior is None:
            df_prior = fit['df2']
        scale = fit['scale']
        var_post = _posterior_var(var, df, scale, df_prior)
        return {'var_post': var_post, 'var_prior': scale, 'df_prior': df_prior}


def _posterior_var(var, df, var_prior, df_prior):
    """Compute posterior variance: (df*var + df_prior*var_prior) / (df + df_prior)."""
    var = np.asarray(var, dtype=np.float64)
    df = np.atleast_1d(np.asarray(df, dtype=np.float64))
    var_prior = np.atleast_1d(np.asarray(var_prior, dtype=np.float64))
    if len(df) == 1:
        df = np.full(len(var), df[0])
    df_prior_val = np.atleast_1d(np.asarray(df_prior, dtype=np.float64))
    if len(df_prior_val) == 1:
        df_prior_val = np.full(len(var), df_prior_val[0])
    if len(var_prior) == 1:
        var_prior = np.full(len(var), var_prior[0])
    total_df = df + df_prior_val
    # Handle infinite df_prior: var_post = var_prior when df_prior is infinite
    inf_mask = np.isinf(df_prior_val)
    with np.errstate(invalid='ignore', divide='ignore'):
        var_post = np.where(inf_mask, var_prior,
                            (df * var + df_prior_val * var_prior) / np.where(total_df == 0, 1, total_df))
    var_post[total_df <= 0] = var[total_df <= 0]
    return var_post


def _fit_f_dist(x, df1):
    """Fit a scaled F-distribution to data.

    Moment matching to estimate s2 (scale) and df2 (prior df).
    Faithful port of limma's fitFDist() (no-covariate case).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    df1 = np.atleast_1d(np.asarray(df1, dtype=np.float64))

    if n == 0:
        return {'s2': np.nan, 'df2': np.nan}
    if n == 1:
        return {'s2': float(x[0]), 'df2': 0.0}

    # Filter ok values: R uses df1 > 1e-15 and x > -1e-15
    ok_df1 = np.isfinite(df1) & (df1 > 1e-15)
    if len(df1) == 1:
        if not ok_df1[0]:
            return {'s2': np.nan, 'df2': np.nan}
        ok = np.full(n, True)
    else:
        ok = ok_df1
    ok = ok & np.isfinite(x) & (x > -1e-15)

    nok = int(np.sum(ok))
    if nok <= 1:
        if nok == 1:
            return {'s2': float(x[ok][0]), 'df2': 0.0}
        return {'s2': np.nan, 'df2': np.nan}

    x_ok = x[ok].copy()
    df1_ok = df1[ok] if len(df1) > 1 else df1

    # Clamp x: match R's pmax(x, 0), handle zeros, pmax(x, 1e-5 * median)
    x_ok = np.maximum(x_ok, 0.0)
    m = np.median(x_ok)
    if m == 0:
        m = 1.0
    x_ok = np.maximum(x_ok, 1e-5 * m)

    # Compute e = log(x) + logmdigamma(df1/2), matching R exactly
    z = np.log(x_ok)
    e = z + logmdigamma(df1_ok / 2)
    emean = np.mean(e)
    evar = np.sum((e - emean) ** 2) / (nok - 1)  # R uses /(nok - 1L)

    # Subtract trigamma(df1/2) contribution
    evar = evar - np.mean(_trigamma_safe(df1_ok / 2))

    if evar > 0:
        df2 = 2.0 * _trigamma_inverse(evar)
        df2 = max(df2, 1e-6)
        if df2 > 1e15:
            df2 = np.inf
        s2 = float(np.exp(emean - logmdigamma(df2 / 2)))
    else:
        df2 = np.inf
        s2 = float(np.mean(x_ok))  # R: mean(x) for no-covariate case

    s2 = max(s2, 1e-15)

    return {'s2': s2, 'df2': df2}


def _fit_f_dist_trend(var, df, covariate):
    """Fit an F-distribution with trended prior variance.

    Faithful port of R's fitFDist() with covariate parameter.
    Uses natural spline basis + OLS regression, matching R's approach of
    fitting e = log(x) + logmdigamma(df1/2) on ns(covariate, df=splinedf).
    """
    n = len(var)
    var = np.asarray(var, dtype=np.float64).copy()
    df_arr = np.atleast_1d(np.asarray(df, dtype=np.float64))
    if len(df_arr) == 1:
        df_arr = np.full(n, df_arr[0])
    covariate = np.asarray(covariate, dtype=np.float64)

    # Handle infinite covariate values (matching R)
    isfin = np.isfinite(covariate)
    if not np.all(isfin):
        if np.any(isfin):
            r = (np.min(covariate[isfin]), np.max(covariate[isfin]))
            covariate = covariate.copy()
            covariate[covariate == -np.inf] = r[0] - 1
            covariate[covariate == np.inf] = r[1] + 1
        else:
            covariate = np.sign(covariate)

    # Adaptive spline df (matching R: 1 + (nok>=3) + (nok>=6) + (nok>=30))
    splinedf = 1 + int(n >= 3) + int(n >= 6) + int(n >= 30)
    splinedf = min(splinedf, len(np.unique(covariate)))

    if splinedf < 2:
        # Fall back to scalar (no-covariate) fit, matching R's Recall()
        result = _fit_f_dist(var, df_arr)
        return {'var_prior': np.full(n, result['s2']), 'df_prior': result['df2']}

    # Clamp var: match R's pmax(x, 0), handle zeros, pmax(x, 1e-5 * median)
    var = np.maximum(var, 0.0)
    m = np.median(var)
    if m == 0:
        m = 1.0
    var = np.maximum(var, 1e-5 * m)

    # Compute e = log(x) + logmdigamma(df1/2), matching R exactly
    z = np.log(var)
    e = z + logmdigamma(df_arr / 2)

    # Fit natural spline basis + OLS (matching R's lm.fit(ns(...), e))
    basis = _natural_spline_basis(covariate, df=splinedf)
    coeffs, _, _, _ = np.linalg.lstsq(basis, e, rcond=None)
    emean = basis @ coeffs

    # Residual variance: R uses mean(fit$effects[-(1:rank)]^2) = RSS/(n-rank)
    resid = e - emean
    actual_rank = np.linalg.matrix_rank(basis)
    if n > actual_rank:
        evar = np.sum(resid ** 2) / (n - actual_rank)
    else:
        evar = 0.0

    # Subtract trigamma(df1/2) contribution
    evar = evar - np.mean(_trigamma_safe(df_arr / 2))

    # Estimate df2 and s20
    if evar > 0:
        df2 = 2.0 * _trigamma_inverse(evar)
        if df2 > 1e15:
            df2 = np.inf
        s20 = np.exp(emean - logmdigamma(df2 / 2))
    else:
        df2 = np.inf
        s20 = np.exp(emean)

    return {'var_prior': s20, 'df_prior': df2}


def _natural_spline_basis(x, df):
    """Create natural cubic spline basis matrix matching R's ns(x, df=df, intercept=TRUE).

    Uses the truncated power basis representation from Hastie, Tibshirani &
    Friedman (Elements of Statistical Learning, eq 5.4-5.5).

    Parameters
    ----------
    x : array-like
        Covariate values.
    df : int
        Number of basis functions (columns in the returned matrix).

    Returns
    -------
    ndarray of shape (n, df)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Number of internal knots: R uses df - 1 - intercept = df - 2
    n_internal = df - 2

    # Boundary knots at range
    a = np.min(x)
    b = np.max(x)

    if n_internal <= 0 or a == b:
        # Linear basis only: [1, x]
        basis = np.column_stack([np.ones(n), x])
        return basis[:, :df]

    # Internal knots at quantiles (matching R's quantile placement)
    probs = np.linspace(0, 1, n_internal + 2)[1:-1]
    internal_knots = np.quantile(x, probs)

    # All knots sorted: [boundary_left, internal_1, ..., internal_K, boundary_right]
    all_knots = np.sort(np.concatenate([[a], internal_knots, [b]]))
    K = len(all_knots)  # Total knots = df

    # Build basis: [1, x, d_1-d_{K-1}, d_2-d_{K-1}, ..., d_{K-2}-d_{K-1}]
    # where d_k(x) = [(x - xi_k)_+^3 - (x - xi_K)_+^3] / (xi_K - xi_k)
    basis = np.zeros((n, df))
    basis[:, 0] = 1.0
    basis[:, 1] = x

    if K > 2:
        xi_K = all_knots[-1]   # rightmost boundary knot
        xi_Km1 = all_knots[-2]  # second-to-last knot (K-1 in 1-indexed)

        def d_func(xi_j):
            return (np.maximum(x - xi_j, 0) ** 3 - np.maximum(x - xi_K, 0) ** 3) / (xi_K - xi_j)

        d_Km1 = d_func(xi_Km1)

        for j in range(K - 2):
            d_j = d_func(all_knots[j])
            basis[:, 2 + j] = d_j - d_Km1

    return basis


def _fit_f_dist_robustly(x, df1, covariate=None, winsor_tail_p=(0.05, 0.1)):
    """Fit F-distribution with robust outlier detection.

    Port of limma's fitFDistRobustly().
    Returns dict with 'scale', 'df2', and 'df2_shrunk' (per-gene).

    Parameters
    ----------
    x : array-like
        Genewise variances.
    df1 : array-like or scalar
        Residual degrees of freedom.
    covariate : array-like, optional
        Covariate for trended prior.
    winsor_tail_p : tuple
        Tail proportions for Winsorization (lower, upper).

    Returns
    -------
    dict with keys: scale, df2, df2_shrunk
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = len(x)

    if n < 2:
        return {'scale': np.nan, 'df2': np.nan, 'df2_shrunk': np.full(max(n, 1), np.nan)}

    df1 = np.atleast_1d(np.asarray(df1, dtype=np.float64)).copy()
    if len(df1) == 1:
        df1 = np.full(n, df1[0])

    if n == 2:
        if covariate is None:
            result = _fit_f_dist(x, df1)
            return {'scale': result['s2'], 'df2': result['df2'],
                    'df2_shrunk': np.full(n, result['df2'])}
        else:
            result = _fit_f_dist_trend(x, df1, covariate)
            return {'scale': result['var_prior'], 'df2': result['df_prior'],
                    'df2_shrunk': np.full(n, result['df_prior'])}

    # Filter ok values
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)

    if not np.all(ok):
        # Recursive call on ok subset
        df2_shrunk_full = np.empty(n)
        x_ok = x[ok]
        df1_ok = df1[ok]
        cov_ok = covariate[ok] if covariate is not None else None

        fit = _fit_f_dist_robustly(x_ok, df1_ok, covariate=cov_ok,
                                    winsor_tail_p=winsor_tail_p)

        df2_shrunk_full[ok] = fit['df2_shrunk']
        df2_shrunk_full[~ok] = fit['df2']

        if covariate is None:
            scale = fit['scale']
        else:
            scale_ok = np.atleast_1d(fit['scale'])
            scale = np.empty(n)
            scale[ok] = scale_ok
            from scipy.interpolate import interp1d
            f_interp = interp1d(covariate[ok], np.log(scale_ok), kind='linear',
                                bounds_error=False, fill_value='extrapolate')
            scale[~ok] = np.exp(f_interp(covariate[~ok]))

        return {'scale': scale, 'df2': fit['df2'], 'df2_shrunk': df2_shrunk_full}

    # All values ok from here
    m = np.median(x)
    if m <= 0:
        return {'scale': np.nan, 'df2': np.nan, 'df2_shrunk': np.full(n, np.nan)}

    small = x < m * 1e-12
    if np.any(small):
        x[small] = m * 1e-12

    # Non-robust initial fit
    if covariate is None:
        non_robust = _fit_f_dist(x, df1)
        nr_s20 = non_robust['s2']
        nr_df2 = non_robust['df2']
    else:
        non_robust = _fit_f_dist_trend(x, df1, covariate)
        nr_s20 = non_robust['var_prior']
        nr_df2 = non_robust['df_prior']

    if not np.isfinite(nr_df2) and nr_df2 != np.inf:
        return {'scale': nr_s20, 'df2': nr_df2, 'df2_shrunk': np.full(n, 0.0)}

    # Winsor tail probabilities
    wtp = [float(winsor_tail_p[0]), float(winsor_tail_p[1])]
    prob = [wtp[0], 1.0 - wtp[1]]

    # Check if winsor_tail_p is too small for this sample size
    if all(p < 1.0 / n for p in wtp):
        return {'scale': nr_s20, 'df2': nr_df2, 'df2_shrunk': np.full(n, nr_df2)}

    # Unify df1 if vector with different values
    if np.min(df1) < np.max(df1) - 1e-14:
        df1max = np.max(df1)
        i = df1 < (df1max - 1e-14)
        if np.any(i):
            if covariate is None:
                s = nr_s20
            else:
                s = nr_s20[i]
            f_vals = x[i] / s
            d2 = nr_df2
            pupper = stats.f.logsf(f_vals, df1[i], d2)
            plower = stats.f.logcdf(f_vals, df1[i], d2)
            up = pupper < plower
            f_new = f_vals.copy()
            if np.any(up):
                f_new[up] = stats.f.isf(np.exp(np.clip(pupper[up], -500, 0)), df1max, d2)
            if np.any(~up):
                f_new[~up] = stats.f.ppf(np.exp(np.clip(plower[~up], -500, 0)), df1max, d2)
            x[i] = f_new * s
            df1_val = df1max
        else:
            df1_val = df1[0]
    else:
        df1_val = df1[0]

    z = np.log(x)

    if covariate is None:
        # Trimmed mean matching R's mean(z, trim=winsor.tail.p[2])
        ztrend = float(stats.trim_mean(z, proportiontocut=wtp[1]))
        zresid = z - ztrend
    else:
        from .weighted_lowess import weighted_lowess as _wlowess
        lo = _wlowess(covariate, z, span=0.4, iterations=4, npts=200)
        ztrend = lo['fitted']
        zresid = z - ztrend

    # Winsorize z-residuals
    zrq = np.quantile(zresid, prob)
    zwins = np.clip(zresid, zrq[0], zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.mean((zwins - zwmean) ** 2) * n / (n - 1))

    # Gauss-Legendre quadrature on [0,1] (128 nodes)
    gl_nodes_raw, gl_weights_raw = np.polynomial.legendre.leggauss(128)
    gl_nodes_01 = (gl_nodes_raw + 1.0) / 2.0
    gl_weights_01 = gl_weights_raw / 2.0

    def linkfun(v):
        return v / (1.0 + v)

    def linkinv(v):
        return v / (1.0 - v)

    def winsorized_moments(d1, d2, wtp_arg):
        """Compute Winsorized mean and variance of log(F(d1, d2))."""
        fq = stats.f.ppf([wtp_arg[0], 1.0 - wtp_arg[1]], d1, d2)
        zq = np.log(fq)
        q = linkfun(fq)
        nodes = q[0] + (q[1] - q[0]) * gl_nodes_01
        fnodes = linkinv(nodes)
        znodes = np.log(fnodes)
        f_dens = stats.f.pdf(fnodes, d1, d2) / (1.0 - nodes) ** 2
        q21 = q[1] - q[0]
        wtp_arr = np.array(wtp_arg)
        m_val = q21 * np.sum(gl_weights_01 * f_dens * znodes) + np.sum(zq * wtp_arr)
        v_val = (q21 * np.sum(gl_weights_01 * f_dens * (znodes - m_val) ** 2)
                 + np.sum((zq - m_val) ** 2 * wtp_arr))
        return {'mean': m_val, 'var': v_val}

    # Check df2=Inf case
    mom_inf = winsorized_moments(df1_val, np.inf, wtp)

    if mom_inf['var'] <= 0 or zwvar <= 0:
        return {'scale': nr_s20, 'df2': nr_df2, 'df2_shrunk': np.full(n, nr_df2)}

    funval_inf = np.log(zwvar / mom_inf['var'])

    if funval_inf <= 0:
        # df2 = Inf: observed variance <= theoretical at df2=Inf
        df2 = np.inf
        ztrendcorrected = ztrend + zwmean - mom_inf['mean']
        s20 = np.exp(ztrendcorrected)
        Fstat = np.exp(z - ztrendcorrected)
        TailP = stats.chi2.sf(Fstat * df1_val, df1_val)
        r = stats.rankdata(Fstat)
        EmpiricalTailProb = (n - r + 0.5) / n
        ProbNotOutlier = np.minimum(TailP / EmpiricalTailProb, 1.0)
        df_pooled = n * df1_val
        df2_shrunk = np.full(n, float(df2))
        O = ProbNotOutlier < 1
        if np.any(O):
            df2_shrunk[O] = ProbNotOutlier[O] * df_pooled
            o = np.argsort(TailP)
            df2_shrunk[o] = np.maximum.accumulate(df2_shrunk[o])
        return {'scale': s20, 'df2': df2, 'df2_shrunk': df2_shrunk}

    # Check if non-robust already gives Inf
    if nr_df2 == np.inf:
        return {'scale': nr_s20, 'df2': nr_df2, 'df2_shrunk': np.full(n, nr_df2)}

    # Root-finding for df2
    rbx = linkfun(nr_df2)

    def fun_root(par):
        d2 = linkinv(par)
        mom = winsorized_moments(df1_val, d2, wtp)
        if mom['var'] <= 0:
            return funval_inf
        return np.log(zwvar / mom['var'])

    funval_low = fun_root(rbx)

    if funval_low >= 0:
        df2 = nr_df2
    else:
        from scipy.optimize import brentq
        root = brentq(fun_root, rbx, 1.0 - 1e-10, xtol=1e-8)
        df2 = linkinv(root)

    mom = winsorized_moments(df1_val, df2, wtp)
    ztrendcorrected = ztrend + zwmean - mom['mean']
    s20 = np.exp(ztrendcorrected)
    Fstat = np.exp(z - ztrendcorrected)

    LogTailP = stats.f.logsf(Fstat, df1_val, df2)
    TailP = np.exp(LogTailP)
    r = stats.rankdata(Fstat)
    LogEmpiricalTailProb = np.log(n - r + 0.5) - np.log(n)
    LogProbNotOutlier = np.minimum(LogTailP - LogEmpiricalTailProb, 0.0)
    ProbNotOutlier = np.exp(LogProbNotOutlier)
    ProbOutlier = -np.expm1(LogProbNotOutlier)

    if np.any(LogProbNotOutlier < 0):
        minLogTailP = np.min(LogTailP)
        if minLogTailP == -np.inf:
            df2_outlier = 0.0
            df2_shrunk = ProbNotOutlier * df2
        else:
            df2_outlier = np.log(0.5) / minLogTailP * df2
            NewLogTailP = stats.f.logsf(np.max(Fstat), df1_val, df2_outlier)
            df2_outlier = np.log(0.5) / NewLogTailP * df2_outlier
            df2_shrunk = ProbNotOutlier * df2 + ProbOutlier * df2_outlier

        # Monotonize via cummax on ordered tail p-values
        o = np.argsort(LogTailP)
        df2_ordered = df2_shrunk[o].copy()
        m_arr = np.cumsum(df2_ordered) / np.arange(1, n + 1, dtype=np.float64)
        imin = int(np.argmin(m_arr))
        df2_ordered[:imin + 1] = m_arr[imin]
        df2_shrunk_final = np.empty(n)
        df2_shrunk_final[o] = np.maximum.accumulate(df2_ordered)
        df2_shrunk = df2_shrunk_final
    else:
        df2_shrunk = np.full(n, df2)

    return {'scale': s20, 'df2': df2, 'df2_shrunk': df2_shrunk}


def _digamma_safe(x):
    """Safe digamma that handles arrays."""
    from scipy.special import digamma
    return digamma(np.asarray(x, dtype=np.float64))


def _trigamma_safe(x):
    """Safe trigamma (polygamma of order 1)."""
    from scipy.special import polygamma
    return polygamma(1, np.asarray(x, dtype=np.float64))


def _trigamma_inverse(x):
    """Inverse of the trigamma function.

    Port of limma's trigammaInverse().
    Uses Newton's method.
    """
    from scipy.special import polygamma

    x = float(x)
    if x > 1e7:
        return 1.0 / x
    if x < 1e-6:
        return 1.0 / x

    # Starting value
    if x > 0.5:
        y = 1.0 / x
    else:
        y = 1.0 / (x * (1 + x))

    # Newton iterations
    for _ in range(50):
        tri = float(polygamma(1, y))
        dif = tri * (1 - tri / x) / float(polygamma(2, y))
        y = y + dif
        if y <= 0:
            y = x  # reset
        if abs(dif / y) < 1e-10:
            break

    return y


def logmdigamma(x):
    """Compute log(x) - digamma(x) avoiding subtractive cancellation.

    Port of statmod's logmdigamma().
    Uses recursive shift for small values and asymptotic expansion for large.
    """
    x = np.asarray(x, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.full_like(x, np.nan)

    valid = x > 0
    if not np.any(valid):
        return float(result[0]) if scalar_input else result

    xv = x[valid]
    rv = np.empty_like(xv)
    large = xv >= 5
    small = ~large

    # Large values: asymptotic expansion
    if np.any(large):
        z = xv[large]
        inv_z2 = 1.0 / (z * z)
        tail = inv_z2 * (-1.0/12 + inv_z2 * (1.0/120 + inv_z2 * (-1.0/252 + inv_z2 * (
            1.0/240 + inv_z2 * (-1.0/132 + inv_z2 * (691.0/32760 + inv_z2 * (
            -1.0/12 + 3617.0/8160 * inv_z2)))))))
        rv[large] = 1.0 / (2.0 * z) - tail

    # Small values: recursive shift by 5, then use asymptotic on z+5
    if np.any(small):
        z = xv[small]
        z5 = z + 5.0
        inv_z5_2 = 1.0 / (z5 * z5)
        tail5 = inv_z5_2 * (-1.0/12 + inv_z5_2 * (1.0/120 + inv_z5_2 * (-1.0/252 + inv_z5_2 * (
            1.0/240 + inv_z5_2 * (-1.0/132 + inv_z5_2 * (691.0/32760 + inv_z5_2 * (
            -1.0/12 + 3617.0/8160 * inv_z5_2)))))))
        lmd_z5 = 1.0 / (2.0 * z5) - tail5
        rv[small] = (np.log(z / z5) + lmd_z5
                     + 1.0/z + 1.0/(z+1) + 1.0/(z+2) + 1.0/(z+3) + 1.0/(z+4))

    result[valid] = rv
    return float(result[0]) if scalar_input else result


def _p_adjust_bh(p):
    """Benjamini-Hochberg p-value adjustment.

    Port of R's p.adjust(method="BH").
    """
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    o = np.argsort(p)[::-1]
    ro = np.argsort(o)
    i_vals = np.arange(n, 0, -1, dtype=np.float64)
    adjusted = np.minimum.accumulate(n / i_vals * p[o])
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted[ro]


def _fit_f_dist_unequal_df1(x, df1, covariate=None, span=None, robust=True, prior_weights=None):
    """Fit a scaled F-distribution with unequal df1 values.

    Port of limma's fitFDistUnequalDF1().
    Uses MLE to estimate scale and df2 (prior df).

    Parameters
    ----------
    x : array-like
        Genewise variances (s2 values).
    df1 : array-like
        Residual degrees of freedom per gene.
    covariate : array-like, optional
        Covariate for trended prior (e.g. AveLogCPM).
    span : float, optional
        Loess span.
    robust : bool
        Robust estimation with outlier handling.
    prior_weights : array-like, optional
        Prior weights for each observation.

    Returns
    -------
    dict with keys: scale, df2, and optionally df2_shrunk, df2_outlier.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    x = np.asarray(x, dtype=np.float64).copy()
    df1 = np.atleast_1d(np.asarray(df1, dtype=np.float64)).copy()
    n = len(x)

    if len(df1) == 1:
        df1 = np.full(n, df1[0])

    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=np.float64).copy()

    # Handle NA values
    na_mask = np.isnan(x)
    if np.any(na_mask):
        if prior_weights is None:
            prior_weights = (~na_mask).astype(np.float64)
        else:
            prior_weights[na_mask] = 0
        x[na_mask] = 0

    # Handle small df1
    small_df1 = df1 < 0.01
    if np.any(small_df1):
        if prior_weights is None:
            prior_weights = (~small_df1).astype(np.float64)
        else:
            prior_weights[small_df1] = 0
        df1[small_df1] = 1

    has_pw = prior_weights is not None

    # Identify informative observations
    informative = x > 0
    if has_pw:
        informative = informative & (prior_weights > 0)
    n_informative = int(np.sum(informative))

    if n_informative < 2:
        return {'scale': np.nan, 'df2': np.nan}

    if n_informative == 2:
        covariate = None
        robust = False
        prior_weights = None
        has_pw = False

    m = np.median(x[informative])
    xpos = np.maximum(x, 1e-12 * m)
    z = np.log(xpos)
    d1 = df1 / 2.0
    e = z + logmdigamma(d1)
    w = 1.0 / _trigamma_safe(d1)
    if len(w) < n:
        w = np.full(n, w[0])
    if has_pw:
        w = w * prior_weights

    if covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        covariate = np.asarray(covariate, dtype=np.float64)
        if span is None:
            span = choose_lowess_span(n, small_n=500)
        # Normalize weights: w / quantile(w, 0.75), clipped to [1e-8, 100]
        w_q75 = np.quantile(w, 0.75)
        loess_w = w / w_q75 if w_q75 > 0 else w.copy()
        loess_w = np.clip(loess_w, 1e-08, 100)

        from .weighted_lowess import weighted_lowess as _wlowess
        wl_result = _wlowess(covariate, e, weights=loess_w, span=span,
                             iterations=1, npts=200)
        emean = wl_result['fitted']

    d1x = d1 * xpos

    # MLE optimization for d2 = par/(1-par) over par in [0.5, 0.9998]
    def minus_twice_loglik(par):
        d2 = par / (1 - par)
        lmd2 = logmdigamma(d2)
        d2s20 = d2 * np.exp(emean - lmd2)
        ll = (-(d1 + d2) * np.log1p(d1x / d2s20)
              - d1 * np.log(d2s20)
              + gammaln(d1 + d2) - gammaln(d2))
        if has_pw:
            return -2 * np.sum(prior_weights * ll)
        return -2 * np.sum(ll)

    opt = minimize_scalar(minus_twice_loglik, bounds=(0.5, 0.9998), method='bounded')
    d2 = opt.x / (1 - opt.x)
    s20 = np.exp(emean - logmdigamma(d2))

    if not robust:
        return {'scale': s20, 'df2': 2 * d2}

    # Robust estimation: detect and down-weight outliers
    df2 = 2 * d2
    f_stat = x / s20

    right_p = stats.f.sf(f_stat, df1, df2)
    left_p = 1 - right_p

    # Better computation for very small left p-values
    small_left = left_p < 0.001
    if np.any(small_left):
        df1_sub = df1[small_left] if len(df1) > 1 else df1
        left_p[small_left] = stats.f.cdf(f_stat[small_left], df1_sub, df2)

    two_sided_p = 2 * np.minimum(left_p, right_p)

    fdr = _p_adjust_bh(two_sided_p)
    fdr[fdr > 0.3] = 1

    if np.min(fdr) == 1:
        return {'scale': s20, 'df2': df2}

    # Re-fit with FDR as prior weights
    outpw = _fit_f_dist_unequal_df1(x, df1, covariate=covariate, span=span,
                                     robust=False, prior_weights=fdr)
    s20 = outpw['scale']
    df2 = outpw['df2']

    r = stats.rankdata(f_stat)
    uniform_p = (n - r + 0.5) / n
    prob_not_outlier = np.minimum(right_p / uniform_p, 1)

    if np.min(prob_not_outlier) == 1:
        return outpw

    i_min = int(np.argmin(right_p))
    min_right_p = right_p[i_min]

    if min_right_p == 0:
        df2_outlier = 0.0
        df2_shrunk = prob_not_outlier * df2
    else:
        df2_outlier = np.log(0.5) / np.log(min_right_p) * df2
        df1_i = df1[i_min] if len(df1) > 1 else df1[0]
        new_log_right_p = stats.f.logsf(f_stat[i_min], df1_i, df2_outlier)
        df2_outlier = np.log(0.5) / new_log_right_p * df2_outlier
        df2_shrunk = prob_not_outlier * df2 + (1 - prob_not_outlier) * df2_outlier

    # Monotonize df2_shrunk
    o = np.argsort(right_p)
    df2_ordered = df2_shrunk[o].copy()
    m_arr = np.cumsum(df2_ordered) / np.arange(1, n + 1, dtype=np.float64)
    imin = int(np.argmin(m_arr))
    df2_ordered[:imin + 1] = m_arr[imin]
    df2_shrunk_final = np.empty(n)
    df2_shrunk_final[o] = np.maximum.accumulate(df2_ordered)

    return {'scale': s20, 'df2': df2, 'df2_outlier': df2_outlier, 'df2_shrunk': df2_shrunk_final}


def non_estimable(x):
    """Identify non-estimable coefficients in a design matrix.

    Port of limma's nonEstimable().
    """
    x = np.asarray(x, dtype=np.float64)
    p = x.shape[1]
    if p == 0:
        return None
    _, R = np.linalg.qr(x)
    d = np.abs(np.diag(R))
    if len(d) == 0:
        return np.arange(p)
    tol = np.max(d) * max(x.shape) * np.finfo(np.float64).eps
    non_est = np.where(d < tol)[0]
    if len(non_est) == 0:
        return None
    # Return coefficient names if available
    return non_est


def is_fullrank(x):
    """Check if a matrix is full column rank.

    Port of limma's is.fullrank().
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.linalg.matrix_rank(x) == x.shape[1]


def choose_lowess_span(n, small_n=25, min_span=0.2, power=1/3):
    """Choose lowess span based on number of observations.

    Port of limma's chooseLowessSpan().
    Formula: min(min_span + (1 - min_span) * (small_n/n)^power, 1)
    """
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


def contrast_as_coef(design, contrast, first=False):
    """Reform a design matrix so that a contrast becomes a coefficient.

    Port of limma's contrastAsCoef().

    Parameters
    ----------
    design : ndarray
        Design matrix.
    contrast : array-like
        Contrast vector.
    first : bool
        If True, put contrast as first column.

    Returns
    -------
    dict with 'design' (reformed design) and 'coef' (column index of contrast).
    """
    design = np.asarray(design, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64).ravel()
    p = design.shape[1]

    if len(contrast) != p:
        raise ValueError("Length of contrast must equal number of columns in design")

    # Port of limma's contrastAsCoef: QR decompose contrast, apply Q^T
    # rotation, then backsolve the contrast coefficient row by R so that
    # the coefficient directly represents the contrast effect.
    contrast_mat = contrast.reshape(-1, 1)
    Q, R_mat = np.linalg.qr(contrast_mat, mode='complete')
    r_val = R_mat[0, 0]  # scalar R factor (= ±||contrast||)

    # design_rotated = design @ Q (apply QR rotation)
    design_rotated = design @ Q

    # Backsolve: divide contrast coefficient column by R to normalize
    # This makes the coefficient directly represent the logFC
    ncontrasts = 1
    design_rotated[:, 0] = design_rotated[:, 0] / r_val

    if first:
        new_design = design_rotated
        coef = 0
    else:
        # Move contrast column (first) to last
        cols = list(range(1, p)) + [0]
        new_design = design_rotated[:, cols]
        coef = p - 1

    return {'design': new_design, 'coef': coef}


def logsumexp(x, y):
    """Compute log(exp(x) + exp(y)) avoiding overflow.

    Helper used in zscoreNBinom.
    """
    m = np.maximum(x, y)
    return m + np.log(np.exp(x - m) + np.exp(y - m))
