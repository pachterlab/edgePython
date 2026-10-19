# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Binomial generalized linear models for paired counts.

Ports of edgeR 5.0's PCList, readBismark2PC, binFit, mbinOneWay,
mbinIWLS, binQLFit, binQLFTest and the binomial sampleWeights, with the
fitters of binomial.c (one-way shortcut and IWLS) and the quasi-likelihood
adjustment of ql_glm.c / ql_weights.c for the binomial family.

Counts are paired: `counts` are successes (e.g. methylated reads) and
`counts2` failures (unmethylated reads). The response is the proportion
counts / (counts + counts2), with the coverage counts + counts2 as weight.
Coefficients are on the natural log-odds scale; tests report log2 odds
ratios (logOR).
"""

import math

import numpy as np
import pandas as pd
from numba import njit, prange

from .classes import PCList, DGEBIN, DGELRT
from .ql_weights import pois_alpha, pois_kappa, _hat_values

_THRESH = 30.0
_MTHRESH = -30.0
_DOUEPS = 1e-16
_INVEPS = 1e16
_THRESHOLD_ZERO = 1e-4
_LOW_BOUND_BIN = 1e-10

# binFit's convergence settings: one-way Newton (50, 1e-8), IWLS (100, 1e-6)
_MAXIT_OW, _TOL_OW, _MAXIT_IW, _TOL_IW = 50, 1e-8, 100, 1e-6


# =====================================================================
# Kernels (port of binomial.c)
# =====================================================================
#
# The kernels use IEEE arithmetic (error_model="numpy"), as edgeR's C does:
# a division by zero gives inf or NaN rather than an exception, which in a
# parallel loop would leave the row unset.

@njit(cache=True, error_model="numpy")
def _logit(mu):
    return math.log(mu / (1 - mu))


@njit(cache=True, error_model="numpy")
def _logit2(mu):
    if abs(mu) < _DOUEPS:
        return _MTHRESH
    if abs(1 - mu) < _DOUEPS:
        return _THRESH
    return math.log(mu / (1 - mu))


@njit(cache=True, error_model="numpy")
def _sigmod(eta):
    if eta < _MTHRESH:
        t = _DOUEPS
    elif eta > _THRESH:
        t = _INVEPS
    else:
        t = math.exp(eta)
    return t / (1 + t)


@njit(cache=True, error_model="numpy")
def _dsigmod(eta):
    if eta < _MTHRESH or eta > _THRESH:
        return _DOUEPS
    t = math.exp(eta)
    return t / ((1 + t) * (1 + t))


@njit(cache=True, error_model="numpy")
def _y_log_y(y, mu):
    if abs(y) > _DOUEPS and abs(mu) > _DOUEPS:
        return y * math.log(y / mu)
    return 0.0


@njit(cache=True, error_model="numpy")
def _unit_dev(y, mu):
    return 2 * (_y_log_y(y, mu) + _y_log_y(1 - y, 1 - mu))


@njit(cache=True, error_model="numpy")
def _group_fit(y, o, w, offzero, maxit, tol, mu_out):
    """One group of libraries (bin_group_fit): returns logit, converged, deviance.

    With zero offsets, or a proportion of 0 or 1, the fit is the weighted
    mean proportion; otherwise Newton-Raphson on the logit with offsets.
    """
    n = y.shape[0]
    sum_counts = 0.0
    sum_weights = 0.0
    for i in range(n):
        sum_counts += y[i] * w[i]
        sum_weights += w[i]
    p = 0.5 if abs(sum_weights) < _DOUEPS else sum_counts / sum_weights
    dev = 0.0
    conv = True
    if offzero or abs(p) < _DOUEPS or abs(1 - p) < _DOUEPS or abs(sum_weights) < _DOUEPS:
        for i in range(n):
            mu_out[i] = p
            dev += w[i] * _unit_dev(y[i], p)
        return _logit2(p), conv, dev
    it = 1
    z = _logit(p)
    while True:
        it += 1
        dl = 0.0
        info = 0.0
        for i in range(n):
            eta = z + o[i]
            dl += w[i] * (y[i] - _sigmod(eta))
            info += w[i] * _dsigmod(eta)
        step = dl / info
        z += step
        if abs(step / z) < tol:
            break
        if it == maxit:
            conv = False
            break
    for i in range(n):
        mu_out[i] = _sigmod(z + o[i])
        dev += w[i] * _unit_dev(y[i], mu_out[i])
    return z, conv, dev


@njit(cache=True, error_model="numpy")
def _one_way_gene(y, o, w, group, ngroups, offzero, maxit, tol, mu, beta, conv):
    n = y.shape[0]
    dev = 0.0
    for g in range(ngroups):
        m = 0
        for i in range(n):
            if group[i] == g:
                m += 1
        ys = np.empty(m)
        os_ = np.empty(m)
        ws = np.empty(m)
        idx = np.empty(m, dtype=np.int64)
        k = 0
        for i in range(n):
            if group[i] == g:
                ys[k] = y[i]
                os_[k] = o[i]
                ws[k] = w[i]
                idx[k] = i
                k += 1
        mus = np.empty(m)
        b, c, d = _group_fit(ys, os_, ws, offzero, maxit, tol, mus)
        beta[g] = b
        conv[g] = c
        for k in range(m):
            mu[idx[k]] = mus[k]
        dev += max(d, 0.0)
    return dev


@njit(parallel=True, cache=True, error_model="numpy")
def _one_way_mat(prop, offset, w0, group, ngroups, offzero, maxit, tol):
    """bin_one_way_mat: group logits, fitted values, deviance and convergence."""
    ntag, nlib = prop.shape
    mu = np.empty((ntag, nlib))
    beta = np.empty((ntag, ngroups))
    conv = np.empty((ntag, ngroups), dtype=np.bool_)
    dev = np.empty(ntag)
    for t in prange(ntag):
        dev[t] = _one_way_gene(prop[t], offset[t], w0[t], group, ngroups, offzero,
                               maxit, tol, mu[t], beta[t], conv[t])
    return mu, beta, dev, conv


@njit(cache=True, error_model="numpy")
def _dnrm2(x):
    s = 0.0
    for v in x:
        s += v * v
    return math.sqrt(s)


@njit(cache=True, error_model="numpy")
def _dqrls(x, y, tol):
    """LINPACK dqrls: least squares by dqrdc2 (Householder QR with limited
    pivoting of near-collinear columns to the end) and dqrsl.

    Returns coefficients in the original column order (columns set aside
    get 0), the residuals and the rank. x and y are not modified.
    """
    n, p = x.shape
    a = x.copy()
    qraux = np.empty(p)
    work1 = np.empty(p)
    work2 = np.empty(p)
    jpvt = np.arange(p)
    for j in range(p):
        qraux[j] = _dnrm2(a[:, j])
        work1[j] = qraux[j]
        work2[j] = qraux[j] if qraux[j] != 0.0 else 1.0
    lup = min(n, p)
    k = p + 1
    for l in range(lup):
        # Move negligible columns to the end
        while not (l + 1 >= k or qraux[l] >= work2[l] * tol):
            col = a[:, l].copy()
            jl, ql, w1l, w2l = jpvt[l], qraux[l], work1[l], work2[l]
            for j in range(l + 1, p):
                a[:, j - 1] = a[:, j]
                jpvt[j - 1] = jpvt[j]
                qraux[j - 1] = qraux[j]
                work1[j - 1] = work1[j]
                work2[j - 1] = work2[j]
            a[:, p - 1] = col
            jpvt[p - 1], qraux[p - 1], work1[p - 1], work2[p - 1] = jl, ql, w1l, w2l
            k -= 1
        if l == n - 1:
            break
        nrmxl = _dnrm2(a[l:, l])
        if nrmxl == 0.0:
            continue
        if a[l, l] != 0.0:
            nrmxl = math.copysign(nrmxl, a[l, l])
        for i in range(l, n):
            a[i, l] /= nrmxl
        a[l, l] += 1.0
        for j in range(l + 1, p):
            t = 0.0
            for i in range(l, n):
                t -= a[i, l] * a[i, j]
            t /= a[l, l]
            for i in range(l, n):
                a[i, j] += t * a[i, l]
            if qraux[j] != 0.0:
                tt = 1.0 - (abs(a[l, j]) / qraux[j]) ** 2
                tt = max(tt, 0.0)
                if abs(tt) < 1e-6:
                    qraux[j] = _dnrm2(a[l + 1:, j])
                    work1[j] = qraux[j]
                else:
                    qraux[j] = qraux[j] * math.sqrt(tt)
        qraux[l] = a[l, l]
        a[l, l] = -nrmxl
    rank = min(k - 1, n)

    # dqrsl: qty = Q'y, b from R b = qty[:rank], rsd = Q (0, qty[rank:])
    qty = y.copy()
    ju = min(rank, n - 1)
    for j in range(ju):
        if qraux[j] == 0.0:
            continue
        temp = a[j, j]
        a[j, j] = qraux[j]
        t = 0.0
        for i in range(j, n):
            t -= a[i, j] * qty[i]
        t /= a[j, j]
        for i in range(j, n):
            qty[i] += t * a[i, j]
        a[j, j] = temp
    bk = qty[:rank].copy()
    for j in range(rank - 1, -1, -1):
        bk[j] /= a[j, j]
        for i in range(j):
            bk[i] -= bk[j] * a[i, j]
    rsd = np.zeros(n)
    for i in range(rank, n):
        rsd[i] = qty[i]
    for j in range(ju - 1, -1, -1):
        if qraux[j] == 0.0:
            continue
        temp = a[j, j]
        a[j, j] = qraux[j]
        t = 0.0
        for i in range(j, n):
            t -= a[i, j] * rsd[i]
        t /= a[j, j]
        for i in range(j, n):
            rsd[i] += t * a[i, j]
        a[j, j] = temp
    b = np.zeros(p)
    for j in range(rank):
        b[jpvt[j]] = bk[j]
    return b, rsd, rank


@njit(cache=True, error_model="numpy")
def _iwls_gene(y, offset, w, design, maxit, tol, beta, mu):
    """bin_iwls_vec: IWLS for one gene. Returns deviance, iterations, failed."""
    n, p = design.shape
    lp = np.empty(n)
    for i in range(n):
        mu[i] = (y[i] * w[i] + 0.5) / (w[i] + 1)
        lp[i] = _logit(mu[i]) - offset[i]
    b, rsd, _ = _dqrls(design, lp, tol)
    dev = 0.0
    for i in range(n):
        lp[i] = lp[i] - rsd[i]
        mu[i] = _sigmod(lp[i] + offset[i])
        dev += w[i] * _unit_dev(y[i], mu[i])
    it = 1
    failed = False
    sw = np.empty(n)
    xw = np.empty((n, p))
    while True:
        it += 1
        for i in range(n):
            if abs(mu[i]) < _DOUEPS or abs(1 - mu[i]) < _DOUEPS:
                varp = _DOUEPS
            else:
                varp = mu[i] * (1 - mu[i])
            sw[i] = math.sqrt(w[i] * varp)
            if abs(w[i]) < _DOUEPS:
                lp[i] = _DOUEPS
            else:
                lp[i] = sw[i] * ((y[i] - mu[i]) / varp + lp[i])
            for j in range(p):
                xw[i, j] = design[i, j] * sw[i]
        b, rsd, _ = _dqrls(xw, lp, 1e-12)
        ndev = 0.0
        for i in range(n):
            if abs(w[i]) < _DOUEPS:
                lp[i] = _DOUEPS
            else:
                lp[i] = (lp[i] - rsd[i]) / sw[i]
            mu[i] = _sigmod(lp[i] + offset[i])
            ndev += w[i] * _unit_dev(y[i], mu[i])
        if abs(dev - ndev) / (abs(dev) + 1) < tol:
            break
        dev = ndev
        if it > maxit:
            failed = True
            break
    for j in range(p):
        beta[j] = b[j]
    return dev, it, failed


@njit(parallel=True, cache=True, error_model="numpy")
def _iwls_mat(prop, offset, w0, design, maxit, tol):
    ntag, nlib = prop.shape
    p = design.shape[1]
    beta = np.empty((ntag, p))
    mu = np.empty((ntag, nlib))
    dev = np.empty(ntag)
    iters = np.empty(ntag, dtype=np.int64)
    failed = np.empty(ntag, dtype=np.bool_)
    for t in prange(ntag):
        d, it, f = _iwls_gene(prop[t], offset[t], w0[t], design, maxit, tol, beta[t], mu[t])
        dev[t] = max(d, 0.0)
        iters[t] = it
        failed[t] = f
    return beta, mu, dev, iters, failed


def _fit_mat(prop, offset, w0, design, offzero, maxit_ow=_MAXIT_OW, tol_ow=_TOL_OW,
             maxit_iw=_MAXIT_IW, tol_iw=_TOL_IW):
    """bin_fit_mat: one-way shortcut when the design is a one-way layout,
    otherwise IWLS. Returns coefficients, fitted values, deviance, iter, failed."""
    from .glm_kernels import groups_from_design
    group, ngroups = groups_from_design(design)
    ntag = prop.shape[0]
    p = design.shape[1]
    if ngroups == p:
        mu, gbeta, dev, conv = _one_way_mat(prop, offset, w0, group, ngroups, offzero,
                                            maxit_ow, tol_ow)
        first = np.array([np.flatnonzero(group == g)[0] for g in range(ngroups)])
        coef = np.linalg.solve(design[first], gbeta.T).T
        return coef, mu, dev, np.zeros(ntag, dtype=np.int64), ~np.all(conv, axis=1)
    return _iwls_mat(prop, offset, w0, design, maxit_iw, tol_iw)


# =====================================================================
# Quasi-likelihood adjustment (ql_glm.c, ql_weights.c, binomial family)
# =====================================================================

@njit(cache=True, error_model="numpy")
def _lchoose(n, k):
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


@njit(cache=True, error_model="numpy")
def _weight_binomial(y, mm):
    """compute_weight_binomial: the QL weights (alpha, kappa) for a fitted
    proportion y over mm trials, exact for small means, Poisson otherwise."""
    pp = y if y < 0.5 else 1 - y
    mu = pp * mm
    if mu < _LOW_BOUND_BIN or mm < 2:
        return 0.0, 0.0
    if mm < 20 and mu < 3.0:
        pr = np.empty(mm + 1)
        de = np.empty(mm + 1)
        for k in range(mm + 1):
            pr[k] = math.exp(_lchoose(mm, k) + k * math.log(pp) + (mm - k) * math.log1p(-pp))
        de[0] = 2 * mm * math.log(1 / (1 - pp))
        de[mm] = 2 * mm * math.log(1 / pp)
        for k in range(1, mm):
            de[k] = 2 * (k * math.log(k / mu) + (mm - k) * math.log((mm - k) / (mm - mu)))
        dd = pr[0] * de[0] + pr[mm] * de[mm]
        for k in range(1, mm):
            dd += pr[k] * de[k]
        out = pr[0] * (de[0] - dd) ** 2 + pr[mm] * (de[mm] - dd) ** 2
        for k in range(1, mm):
            out += pr[k] * (de[k] - dd) ** 2
        return 2 * dd / out, 2 * dd * dd / out
    return pois_alpha(mu), pois_kappa(mu)


@njit(cache=True, error_model="numpy")
def _adjust_gene(prop, mu, design, cover, w0, dev_unit, df_unit):
    n = prop.shape[0]
    zw = np.empty(n)
    for i in range(n):
        zw[i] = math.sqrt(cover[i] * mu[i] * (1 - mu[i]))
    h = _hat_values(design, zw)
    for i in range(n):
        alpha, kappa = _weight_binomial(mu[i], int(cover[i]))
        udp = _unit_dev(prop[i], mu[i])
        hdp = 1.0 - h[i]
        if hdp < _THRESHOLD_ZERO:
            udp = 0.0
            hdp = 0.0
        dev_unit[i] = udp * alpha
        df_unit[i] = hdp * kappa * (1.0 if cover[i] > 0.5 else 0.0)


@njit(parallel=True, cache=True, error_model="numpy")
def _adjust_mat(prop, mu, design, cover, w0):
    """ql_adjust_mat for the binomial: unit deviances and unit df."""
    ntag, nlib = prop.shape
    dev = np.empty((ntag, nlib))
    df = np.empty((ntag, nlib))
    for t in prange(ntag):
        _adjust_gene(prop[t], mu[t], design, cover[t], w0[t], dev[t], df[t])
    return dev, df


def _adjust_vec(prop, mu, design, cover, w0):
    """ql_adjust_vec for the binomial: adjusted deviance, df and s2."""
    dev_unit, df_unit = _adjust_mat(prop, mu, design, cover, w0)
    dev = np.maximum((dev_unit * w0).sum(axis=1), 0.0)
    df = df_unit.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(df < _THRESHOLD_ZERO, 0.0, dev / df)
    return df, dev, s2


# =====================================================================
# Inputs
# =====================================================================

def _as_matrix(x, shape, name):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != shape:
        try:
            x = np.broadcast_to(x, shape)
        except ValueError:
            raise ValueError(f"{name} doesn't conform with the counts")
    return np.ascontiguousarray(x)


def _prepare(y, z, weights, offset):
    """Proportions, coverage and coverage-scaled weights, as binFit.default."""
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if y.ndim == 1:
        y, z = y[None, :], z[None, :]
    if y.shape != z.shape:
        raise ValueError("The dimensions of 'y' and 'z' are not matched")
    cover = y + z
    w0 = cover if weights is None else _as_matrix(weights, y.shape, "weights") * cover
    prop = y / np.maximum(cover, 1.0)
    if offset is None:
        off = np.zeros(y.shape)
        offzero = True
    else:
        off = np.asarray(offset, dtype=np.float64)
        if off.ndim == 1 and off.size == y.shape[1]:
            off = np.broadcast_to(off, y.shape)
        off = _as_matrix(off, y.shape, "offset")
        offzero = not np.any(off)
    return y, z, np.ascontiguousarray(prop), np.ascontiguousarray(cover), np.ascontiguousarray(w0), off, offzero


def _check_design(design, nlib):
    from .limma_port import non_estimable
    if design is None:
        return np.ones((nlib, 1))
    design = np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != nlib:
        raise ValueError("nrow(design) disagrees with ncol(y)")
    ne = non_estimable(design)
    if ne is not None and len(ne):
        raise ValueError("Design matrix not of full rank. The following coefficients not estimable: "
                         + " ".join(map(str, ne)))
    return np.ascontiguousarray(design)


def _pclist_design(y, design):
    """design, else y['design'], else a design for y's groups (if more than one)."""
    if design is None:
        design = y.get("design")
    if design is None:
        group = pd.Series(y["samples"]["group"]).astype("category").cat.remove_unused_categories()
        if len(group.cat.categories) > 1:
            from .utils import _model_matrix_group
            design = _model_matrix_group(np.asarray(group))
    return design


def _with_weights(y, weights):
    """Weights given explicitly act as the PCList's own weights."""
    if weights is None:
        return y
    y = type(y)(y)
    y["weights"] = weights
    return y


def _pclist_ave_log_cpm(y):
    """y['AveLogCPM'], or aveLogCPM of the PCList (without storing it in y)."""
    if y.get("AveLogCPM") is not None:
        return y["AveLogCPM"]
    from .expression import ave_log_cpm
    return ave_log_cpm(y)


def _names(y):
    """Row and column names of a counts matrix, if it has them."""
    if isinstance(y, pd.DataFrame):
        return list(y.index), list(y.columns)
    return None, None


# =====================================================================
# PCList
# =====================================================================

def make_pclist(counts, counts2, samples=None, group=None, genes=None):
    """Create a PCList: paired counts for the same features and samples.

    Port of edgeR's PCList(), for example methylated (counts) and
    unmethylated (counts2) reads.

    Parameters
    ----------
    counts, counts2 : array-like or DataFrame
        Genes (or loci) x samples matrices of paired counts.
    samples : DataFrame, optional
        Sample annotation; a 'group' column is used as group.
    group : array-like, optional
        Group factor.
    genes : DataFrame, optional
        Feature annotation.

    Returns
    -------
    PCList with counts, counts2, samples (group) and genes.
    """
    rn, cn = _names(counts)
    counts = np.asarray(counts, dtype=np.float64)
    counts2 = np.asarray(counts2, dtype=np.float64)
    if counts.ndim == 1:
        counts, counts2 = counts.reshape(-1, 1), counts2.reshape(-1, 1)
    if counts.size == 0:
        raise ValueError("'counts' must contain at least one value")
    if counts2.size == 0:
        raise ValueError("'counts2' must contain at least one value")
    if counts.shape != counts2.shape:
        raise ValueError("The dimensions of 'counts' and 'counts2' are not matched")
    if np.isnan(counts).any() or np.isnan(counts2).any():
        raise ValueError("NA counts not allowed")
    if counts.min() < 0 or counts2.min() < 0:
        raise ValueError("Negative counts not allowed")
    if not (np.isfinite(counts).all() and np.isfinite(counts2).all()):
        raise ValueError("Infinite counts not allowed")
    ntags, nlib = counts.shape
    if cn is None:
        cn = [f"Sample{j + 1}" for j in range(nlib)]
    if rn is None:
        rn = [str(i + 1) for i in range(ntags)]

    if samples is not None:
        samples = pd.DataFrame(samples).reset_index(drop=True)
        if samples.shape[0] != nlib:
            raise ValueError("Number of rows in 'samples' must equal number of columns in 'counts'")
        if group is None and "group" in samples.columns:
            group = samples["group"].to_numpy()
            samples = samples.drop(columns="group")
    if group is None:
        group = pd.Categorical(["1"] * nlib)
    else:
        if len(group) != nlib:
            raise ValueError("Length of 'group' must equal number of columns in 'counts'")
        group = pd.Categorical(group).remove_unused_categories()
    sam = pd.DataFrame({"group": group})
    if samples is not None:
        sam = pd.concat([sam, samples], axis=1)
    sam.index = pd.Index([str(c) for c in cn]) if len(set(cn)) == nlib else pd.RangeIndex(1, nlib + 1)

    x = PCList({"counts": counts, "counts2": counts2, "samples": sam})
    if genes is not None:
        genes = pd.DataFrame(genes).copy()
        if genes.shape[0] != ntags:
            raise ValueError("Counts and genes have different numbers of rows")
        genes.index = pd.Index([str(r) for r in rn])
        x["genes"] = genes
    else:
        x["genes"] = pd.DataFrame(index=pd.Index([str(r) for r in rn]))
    return x


def _remove_ext(x, sep="."):
    """limma's removeExt: drop the extension shared by all strings."""
    x = [str(s) for s in x]
    if not all(sep in s for s in x):
        return x
    ext = [s.split(sep)[-1] for s in x]
    if all(e == ext[0] for e in ext):
        return [s[:len(s) - len(ext[0]) - 1] for s in x]
    return x


def read_bismark2pc(files, sample_names=None, verbose=True):
    """Read Bismark coverage files into a PCList.

    Port of edgeR's readBismark2PC(). Each file has columns chromosome,
    start, end, methylation percentage, methylated count and unmethylated
    count. Loci are merged across samples (in order of first appearance)
    and the methylated and unmethylated counts become counts and counts2.

    Parameters
    ----------
    files : list of str
        Bismark coverage files (.cov or .cov.gz).
    sample_names : list of str, optional
        Defaults to the file names without their extensions.
    verbose : bool
        Print progress.

    Returns
    -------
    PCList with genes columns Chr (categorical) and Locus, and row names
    'Chr-Locus'.
    """
    files = [str(f) for f in np.atleast_1d(files)]
    if sample_names is None:
        sample_names = _remove_ext(_remove_ext(_remove_ext(files)))
    tables, chr_names = [], []
    for f in files:
        if verbose:
            print("Reading", f)
        x = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 4, 5],
                        dtype={0: str, 1: np.int64, 4: np.int64, 5: np.int64})
        tables.append(x)
        for c in pd.unique(x[0]):
            if c not in chr_names:
                chr_names.append(c)
    if verbose:
        print("Hashing ...")
    chr_index = {c: i for i, c in enumerate(chr_names)}
    base = len(chr_names) + 1
    keys = [x[1].to_numpy() * base + x[0].map(chr_index).to_numpy() for x in tables]
    unique_keys = pd.unique(np.concatenate(keys))
    if verbose:
        print("Collating counts ...")
    pos = pd.Index(unique_keys)
    counts1 = np.zeros((len(unique_keys), len(files)))
    counts2 = np.zeros((len(unique_keys), len(files)))
    for i, (x, k) in enumerate(zip(tables, keys)):
        m = pos.get_indexer(k)
        counts1[m, i] = x[4].to_numpy()
        counts2[m, i] = x[5].to_numpy()
    locus = unique_keys // base
    chr_ = pd.Categorical.from_codes(unique_keys % base, categories=chr_names)
    rownames = [f"{c}-{l}" for c, l in zip(chr_, locus)]
    return make_pclist(pd.DataFrame(counts1, index=rownames, columns=sample_names),
                       pd.DataFrame(counts2, index=rownames, columns=sample_names),
                       genes=pd.DataFrame({"Chr": chr_, "Locus": locus}))


def rowsum_pclist(x, group):
    """Sum a PCList's counts over groups of rows.

    Port of edgeR's rowsum.PCList(). Groups are in order of first
    appearance; genes columns that are constant within every group are
    kept.
    """
    group = np.asarray(group)
    codes, uniq = pd.factorize(group, sort=False)
    first = np.unique(codes, return_index=True)[1]
    out = x[first, :]
    n = len(uniq)
    for k in ("counts", "counts2"):
        s = np.zeros((n, x[k].shape[1]))
        np.add.at(s, codes, np.asarray(x[k], dtype=np.float64))
        out[k] = s
    names = pd.Index([str(u) for u in uniq])
    genes = x.get("genes")
    if genes is not None and genes.shape[1]:
        keep = [c for c in genes.columns
                if genes[c].groupby(codes).nunique(dropna=False).le(1).all()]
        out["genes"] = genes.iloc[first][keep]
    out["genes"] = (out.get("genes") if out.get("genes") is not None else pd.DataFrame(index=range(n)))
    out["genes"].index = names
    out.pop("AveLogCPM", None)
    return out


# =====================================================================
# Fitting
# =====================================================================

def mbin_one_way(y, z, design=None, group=None, offset=None, weights=None, maxit=50, tol=1e-8):
    """Fit one-way layout binomial models by groupwise weighted means.

    Port of edgeR's mbinOneWay(). Returns dict with coefficients (group
    logits, or design coefficients when a design is given), fitted.values
    and deviance.
    """
    y, z, prop, cover, w0, off, offzero = _prepare(y, z, weights, offset)
    nlib = y.shape[1]
    if group is None:
        if design is None:
            group_codes, levels = np.zeros(nlib, dtype=np.int64), ["1"]
        else:
            design = np.asarray(design, dtype=np.float64)
            if design.shape[0] != nlib:
                raise ValueError("nrow(design) disagrees with ncol(y)")
            from .glm_kernels import groups_from_design
            group_codes, ng = groups_from_design(design)
            levels = [str(g + 1) for g in range(ng)]
    else:
        cat = pd.Categorical(group)
        group_codes, levels = cat.codes.astype(np.int64), [str(c) for c in cat.categories]
    ngroups = len(levels)
    mu, beta, dev, _ = _one_way_mat(prop, off, w0, group_codes, ngroups, offzero, maxit, tol)
    if design is not None:
        design = np.asarray(design, dtype=np.float64)
        if design.shape[1] != ngroups:
            raise ValueError("design matrix is not equivalent to a oneway layout")
        first = np.array([np.flatnonzero(group_codes == g)[0] for g in range(ngroups)])
        beta = np.linalg.solve(design[first], beta.T).T
    return {"coefficients": beta, "fitted.values": mu, "deviance": dev}


def mbin_iwls(y, z, design, offset=None, weights=None, maxit=100, tol=1e-6):
    """Fit binomial logistic models by iteratively reweighted least squares.

    Port of edgeR's mbinIWLS(). Returns dict with coefficients,
    fitted.values, deviance, iter and failed.
    """
    y, z, prop, cover, w0, off, _ = _prepare(y, z, weights, offset)
    design = np.ascontiguousarray(np.asarray(design, dtype=np.float64))
    if design.shape[0] != y.shape[1]:
        raise ValueError("nrow(design) disagrees with ncol(y)")
    if not np.all(np.isfinite(design)):
        raise ValueError("all entries of design matrix must be finite and non-missing")
    beta, mu, dev, iters, failed = _iwls_mat(prop, off, w0, design, maxit, tol)
    return {"coefficients": beta, "fitted.values": mu, "deviance": dev, "iter": iters,
            "failed": failed}


def _bin_fit_default(y, z, design=None, weights=None, offset=None, prior_count=1):
    rn, _ = _names(y)
    y, z, prop, cover, w0, off, offzero = _prepare(y, z, weights, offset)
    ntag, nlib = y.shape
    design = _check_design(design, nlib)
    coef, mu, dev, iters, failed = _fit_mat(prop, off, w0, design, offzero)
    fit = DGEBIN({"coefficients": coef, "fitted.values": mu, "deviance": dev,
                  "iter": iters, "failed": failed, "counts": y, "counts2": z})
    if prior_count > 0:
        fit["unshrunk.coefficients"] = coef
        add = math.ceil(prior_count) / 2
        fit["coefficients"] = _bin_fit_default(y + add, z + add, design, weights, offset, 0)["coefficients"]
    fit["df.residual"] = np.full(ntag, nlib - design.shape[1])
    fit["design"] = design
    fit["weights"] = None if weights is None else _as_matrix(weights, y.shape, "weights")
    return fit


def bin_fit(y, counts2=None, design=None, weights=None, offset=None, prior_count=1):
    """Fit a binomial generalized linear model for each feature.

    Port of edgeR's binFit(). Uses the one-way shortcut when the design is
    a one-way layout, otherwise IWLS.

    Parameters
    ----------
    y : PCList or array-like
        A PCList, or the success counts (then counts2 are the failures).
    counts2 : array-like, optional
        Failure counts, when y is a matrix.
    design : ndarray, optional
        Design matrix. For a PCList, defaults to y['design'] or a design
        for the sample groups; otherwise an intercept.
    weights : ndarray, optional
        Observation weights (multiplied by the coverage).
    offset : ndarray, optional
        Offsets on the logit scale.
    prior_count : float
        Coefficients are from counts with prior_count/2 added to both
        counts and counts2; unshrunk.coefficients hold the unmodified fit.

    Returns
    -------
    DGEBIN with coefficients (log-odds scale), fitted.values (proportions),
    deviance, df.residual and the inputs.
    """
    if isinstance(y, dict) and "counts2" in y:
        y = _with_weights(y, weights)
        design = _pclist_design(y, design)
        fit = _bin_fit_default(y["counts"], y["counts2"], design=design, weights=y.get("weights"),
                               offset=offset, prior_count=prior_count)
        fit["samples"] = y["samples"]
        fit["genes"] = y.get("genes")
        fit["AveLogCPM"] = _pclist_ave_log_cpm(y)
        return fit
    if counts2 is None:
        raise ValueError("counts2 must be given when y is not a PCList")
    return _bin_fit_default(y, counts2, design=design, weights=weights, offset=offset,
                            prior_count=prior_count)


def _bin_ql_fit_default(y, z, design=None, weights=None, offset=None, ave_log_cpm=None,
                        abundance_trend=True, covariate_trend=None, robust=True, prior_count=1):
    from .limma_port import squeeze_var
    from .expression import _ave_log_cpm_default
    y, z, prop, cover, w0, off, offzero = _prepare(y, z, weights, offset)
    ntag, nlib = y.shape
    design = _check_design(design, nlib)
    coef, mu, dev, iters, failed = _fit_mat(prop, off, w0, design, offzero)
    df_adj, dev_adj, s2 = _adjust_vec(prop, np.ascontiguousarray(mu), design, cover, w0)
    fit = DGEBIN({"coefficients": coef, "fitted.values": mu, "deviance": dev, "iter": iters,
                  "failed": failed, "df.residual.adj": df_adj, "deviance.adj": dev_adj})
    if ave_log_cpm is None:
        ave_log_cpm = _ave_log_cpm_default(y + z, lib_size=y.sum(axis=0) + z.sum(axis=0),
                                           prior_count=2, dispersion=0, weights=weights)
    fit["AveLogCPM"] = ave_log_cpm
    covariate = covariate_trend if covariate_trend is not None else (
        ave_log_cpm if abundance_trend else None)
    sv = squeeze_var(s2, df_adj, covariate=covariate, robust=robust, legacy=False)
    fit["df.prior"] = sv["df_prior"]
    fit["s2.post"] = sv["var_post"]
    fit["s2.prior"] = sv["var_prior"]
    fit["average.ql.dispersion"] = float(np.median(sv["var_prior"]))
    fit["counts"] = y
    fit["counts2"] = z
    if prior_count > 0:
        fit["unshrunk.coefficients"] = coef
        add = math.ceil(prior_count) / 2
        fit["coefficients"] = _bin_fit_default(y + add, z + add, design, weights, offset, 0)["coefficients"]
    fit["df.residual"] = np.full(ntag, nlib - design.shape[1])
    fit["design"] = design
    fit["weights"] = None if weights is None else _as_matrix(weights, y.shape, "weights")
    return fit


def bin_ql_fit(y, counts2=None, design=None, weights=None, offset=None, ave_log_cpm=None,
               abundance_trend=True, covariate_trend=None, robust=True, prior_count=1):
    """Fit quasi-binomial generalized linear models.

    Port of edgeR 5.0's binQLFit(): binomial GLMs as bin_fit, with
    leverage- and bias-adjusted deviances and residual df, and quasi-
    dispersions squeezed by empirical Bayes against average log-CPM of the
    total coverage.

    Parameters
    ----------
    y : PCList or array-like
        A PCList, or the success counts (then counts2 are the failures).
    counts2, design, weights, offset, prior_count :
        As for bin_fit.
    ave_log_cpm : ndarray, optional
        Average log2-CPM of the coverage; computed if not given.
    abundance_trend : bool
        Trend the prior quasi-dispersion on ave_log_cpm.
    covariate_trend : ndarray, optional
        Covariate for the trend, instead of ave_log_cpm.
    robust : bool
        Robust empirical Bayes.

    Returns
    -------
    DGEBIN, as bin_fit, plus df.residual.adj, deviance.adj, df.prior,
    s2.post, s2.prior and average.ql.dispersion.
    """
    if isinstance(y, dict) and "counts2" in y:
        y = _with_weights(y, weights)
        design = _pclist_design(y, design)
        fit = _bin_ql_fit_default(y["counts"], y["counts2"], design=design, weights=y.get("weights"),
                                  offset=offset, ave_log_cpm=_pclist_ave_log_cpm(y),
                                  abundance_trend=abundance_trend, covariate_trend=covariate_trend,
                                  robust=robust, prior_count=prior_count)
        fit["samples"] = y["samples"]
        fit["genes"] = y.get("genes")
        return fit
    if counts2 is None:
        raise ValueError("counts2 must be given when y is not a PCList")
    return _bin_ql_fit_default(y, counts2, design=design, weights=weights, offset=offset,
                               ave_log_cpm=ave_log_cpm, abundance_trend=abundance_trend,
                               covariate_trend=covariate_trend, robust=robust,
                               prior_count=prior_count)


def bin_ql_ftest(glmfit, coef=None, contrast=None, or_=1.0, lor=None, upshot=True):
    """Quasi-likelihood F-tests for binomial GLMs.

    Port of edgeR 5.0's binQLFTest(). With an odds-ratio threshold (or_ > 1
    or lor > 0) the test is TREAT of H0: |logOR| <= lor; with upshot it is
    UPSHOT, as for glm_ql_ftest.

    Parameters
    ----------
    glmfit : DGEBIN
        Fit from bin_ql_fit().
    coef : int or list, optional
        Coefficient(s) to test (0-based). Defaults to the last.
    contrast : ndarray, optional
        Contrast vector, or matrix with one column per contrast.
    or_ : float
        Odds-ratio threshold; ignored if lor is given.
    lor : float or array, optional
        log2 odds-ratio threshold(s).
    upshot : bool
        With a threshold, use UPSHOT rather than plain TREAT.

    Returns
    -------
    DGELRT whose table has logOR (log2 odds ratio), logCPM, F and PValue.
    """
    from .glm_test import _ftest_engine
    from .limma_port import contrast_as_coef
    if glmfit.get("s2.post") is None:
        raise ValueError("need to run bin_ql_fit before bin_ql_ftest")
    design = np.asarray(glmfit["design"], dtype=np.float64)
    nbeta = design.shape[1]
    if nbeta < 2:
        raise ValueError("Need at least two columns for design, usually the first is the intercept column")
    coef_names = [f"coef{i}" for i in range(nbeta)]
    unshrunk = glmfit.get("unshrunk.coefficients")
    coefs = np.asarray(glmfit["coefficients"], dtype=np.float64)
    if contrast is None:
        if coef is None:
            coef = nbeta - 1
        coef = list(dict.fromkeys(np.atleast_1d(coef).tolist()))
        coef_name = [coef_names[c] for c in coef]
        logOR = coefs[:, coef] / np.log(2)
        logORt = logOR if unshrunk is None else np.asarray(unshrunk)[:, coef] / np.log(2)
        test_coef = coef
        names = coef_name
    else:
        contrast = np.asarray(contrast, dtype=np.float64)
        cmat = contrast.reshape(-1, 1) if contrast.ndim == 1 else contrast
        reform = contrast_as_coef(design, cmat, first=True)
        test_coef = reform["coef"]
        design = reform["design"]
        if len(test_coef) > 1:
            coef_name = f"LR test on {len(test_coef)} degrees of freedom"
        else:
            c = cmat[:, 0]
            coef_name = " ".join(f"{c[j]:g}*{coef_names[j]}" for j in range(nbeta) if c[j] != 0)
        logOR = (coefs @ cmat) / np.log(2)
        logORt = logOR if unshrunk is None else (np.asarray(unshrunk) @ cmat) / np.log(2)
        names = [str(j + 1) for j in range(cmat.shape[1])]
    ncoef = len(test_coef)
    if ncoef >= nbeta:
        raise ValueError("The null model must keep at least one design column")

    if lor is None:
        lor = np.log2(or_)
    lor = np.atleast_1d(np.asarray(lor, dtype=np.float64))
    if lor.size == 1:
        lor = np.full(ncoef, lor[0])
    if np.any(lor < 0):
        raise ValueError("lor should be non-negative")
    if lor.size != ncoef:
        raise ValueError("'or' or 'lor' vector of wrong length")

    ntag = coefs.shape[0]
    df_total = np.minimum(np.asarray(glmfit["df.prior"], dtype=np.float64)
                          + np.asarray(glmfit["df.residual.adj"], dtype=np.float64),
                          np.sum(glmfit["df.residual"]))
    df_total = np.broadcast_to(df_total, (ntag,))

    _, _, prop, cover, w0, _, _ = _prepare(glmfit["counts"], glmfit["counts2"], glmfit.get("weights"), None)
    keep = [j for j in range(nbeta) if j not in test_coef]
    design0 = np.ascontiguousarray(design[:, keep])

    def null_deviance(rows, offset):
        if offset is None:
            off, offzero = np.zeros((len(rows), prop.shape[1])), True
        else:
            off, offzero = np.ascontiguousarray(offset), False
        return _fit_mat(np.ascontiguousarray(prop[rows]), off, np.ascontiguousarray(w0[rows]),
                        design0, offzero)[2]

    F_stat, pvalue = _ftest_engine(
        null_deviance, None, design[:, test_coef], np.asarray(glmfit["deviance"], dtype=np.float64),
        np.broadcast_to(np.asarray(glmfit["s2.post"], dtype=np.float64), (ntag,)), df_total, lor,
        np.asarray(logORt, dtype=np.float64).reshape(ntag, ncoef), upshot)

    if ncoef == 1:
        cols = {"logOR": np.asarray(logOR).reshape(-1)}
    else:
        cols = {f"logOR.{n}": logOR[:, k] for k, n in enumerate(names)}
    table = pd.DataFrame({**cols, "logCPM": glmfit["AveLogCPM"], "F": F_stat, "PValue": pvalue})
    genes = glmfit.get("genes")
    if genes is not None and len(genes.index) == ntag:
        table.index = genes.index
    out = DGELRT({k: v for k, v in glmfit.items() if k not in ("counts", "counts2")})
    if np.any(lor > 0):
        out["lor"] = lor
    out["table"] = table
    out["comparison"] = coef_name
    out["df.test"] = ncoef
    out["df.total"] = df_total
    return out


def _sample_weights_bin(glmfit):
    """sampleWeights for a DGEBIN (sample_weights_binom in sample_weights.c)."""
    y = np.asarray(glmfit["counts"], dtype=np.float64)
    weights = glmfit.get("weights")
    _, _, prop, cover, w0, _, _ = _prepare(y, glmfit["counts2"], weights, None)
    design = np.ascontiguousarray(np.asarray(glmfit["design"], dtype=np.float64))
    dev, df = _adjust_mat(prop, np.ascontiguousarray(glmfit["fitted.values"], dtype=np.float64),
                          design, cover, w0)
    w = np.ones(y.shape) if weights is None else _as_matrix(weights, y.shape, "weights")
    v = np.log((dev * w).sum(axis=0) / df.sum(axis=0))
    v = np.exp(np.mean(v) - v)
    return np.tile(v, (y.shape[0], 1))
