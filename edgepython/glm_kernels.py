# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Numba ports of edgeR's genewise NB GLM fitters.

Ports the C code of edgeR 5.0 (glm.c and compute_nbdev.c): the one-group
Newton-Raphson fit (glm_one_group_vec), the Levenberg-Marquardt fit
(fit_leven_vec), its null-model starting values (get_leven_start) and the
unit NB deviance (compute_unit_nb_deviance). Each kernel follows the C loop
for loop, including its constants and stopping rules, so that fits agree with
edgeR to rounding error. This matters beyond the coefficients: fitted values
that are zero to working precision decide which genes pass thresholds such as
the 1e-8 df cutoff in the quasi-likelihood prior.
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def unit_nb_deviance(y, mu, phi):
    """Unit NB deviance (compute_nbdev.c compute_unit_nb_deviance).

    A small value is added to y and mu to protect logs and divisions, a
    Taylor expansion is used for phi < 1e-4 (including the Poisson case
    phi = 0), and a gamma approximation is used when mu * phi > 1e6.
    """
    y = y + 1e-8
    mu = mu + 1e-8
    if phi < 1e-4:
        resid = y - mu
        out = 2.0 * (y * math.log(y / mu) - resid
                     - 0.5 * resid * resid * phi * (1.0 + phi * (2.0 / 3.0 * resid - y)))
    else:
        product = mu * phi
        if product > 1e6:
            out = 2.0 * ((y - mu) / mu - math.log(y / mu)) * mu / (1.0 + product)
        else:
            invphi = 1.0 / phi
            out = 2.0 * (y * math.log(y / mu) + (y + invphi) * math.log((mu + invphi) / (y + invphi)))
    if out < 0.0:
        return 0.0
    return out


@njit(cache=True, parallel=True)
def unit_nb_deviance_mat(y, mu, disp):
    """Elementwise unit NB deviance for (ngenes, nlibs) matrices."""
    ngenes, nlibs = y.shape
    out = np.empty((ngenes, nlibs))
    for g in prange(ngenes):
        for j in range(nlibs):
            out[g, j] = unit_nb_deviance(y[g, j], mu[g, j], disp[g, j])
    return out


@njit(cache=True, parallel=True)
def nb_deviance_sum_mat(y, mu, disp, w):
    """Genewise weighted sum of unit NB deviances (compute_nbdev_sum)."""
    ngenes, nlibs = y.shape
    out = np.empty(ngenes)
    for g in prange(ngenes):
        s = 0.0
        for j in range(nlibs):
            s += w[g, j] * unit_nb_deviance(y[g, j], mu[g, j], disp[g, j])
        out[g] = s
    return out


@njit(cache=True)
def _one_group_vec(y, offset, disp, w, maxit, tol, cur_beta):
    """One-group NB fit by Newton-Raphson (glm_one_group_vec).

    Returns (beta, converged). Rows of zeros return -1e8 rather than -Inf,
    as edgeR does, so that exp(beta + offset) is exactly zero.
    """
    nlib = y.shape[0]
    allzero = True
    if math.isnan(cur_beta):
        cur_beta = 0.0
        totweight = 0.0
        for j in range(nlib):
            if y[j] > 1e-10:
                cur_beta += y[j] / math.exp(offset[j]) * w[j]
                allzero = False
            totweight += w[j]
        cur_beta = math.log(cur_beta / totweight) if not allzero else 0.0
    else:
        for j in range(nlib):
            if y[j] > 1e-10:
                allzero = False
                break
    if allzero:
        return -1e8, True
    for _ in range(maxit):
        dl = 0.0
        info = 0.0
        for j in range(nlib):
            mu = math.exp(cur_beta + offset[j])
            denom = 1.0 + mu * disp[j]
            dl += (y[j] - mu) / denom * w[j]
            info += mu / denom * w[j]
        step = dl / info
        cur_beta += step
        if abs(step) < tol:
            return cur_beta, True
    return cur_beta, False


@njit(cache=True, parallel=True)
def fit_one_group_mat(y, offset, disp, w, maxit, tol, beta_start):
    """One-group fits for every gene; beta_start NaN means no start value."""
    ngenes = y.shape[0]
    beta = np.empty(ngenes)
    conv = np.empty(ngenes, dtype=np.bool_)
    for g in prange(ngenes):
        beta[g], conv[g] = _one_group_vec(y[g], offset[g], disp[g], w[g],
                                          maxit, tol, beta_start[g])
    return beta, conv


@njit(cache=True, parallel=True)
def fit_one_way_mat(y, offset, disp, w, group, ngroups, maxit, tol, beta_start):
    """Groupwise one-group fits (fit_one_way_mat); beta has one column per group."""
    ngenes, nlibs = y.shape
    beta = np.empty((ngenes, ngroups))
    conv = np.empty((ngenes, ngroups), dtype=np.bool_)
    for g in prange(ngenes):
        for k in range(ngroups):
            n_k = 0
            for j in range(nlibs):
                if group[j] == k:
                    n_k += 1
            ys = np.empty(n_k)
            os_ = np.empty(n_k)
            ds = np.empty(n_k)
            ws = np.empty(n_k)
            i = 0
            for j in range(nlibs):
                if group[j] == k:
                    ys[i] = y[g, j]
                    os_[i] = offset[g, j]
                    ds[i] = disp[g, j]
                    ws[i] = w[g, j]
                    i += 1
            beta[g, k], conv[g, k] = _one_group_vec(ys, os_, ds, ws, maxit, tol,
                                                    beta_start[g, k])
    return beta, conv


@njit(cache=True)
def _autofill(beta, offset, design, mu):
    """mu = exp(design @ beta + offset), summed in dgemv's column order."""
    nlib, ncoef = design.shape
    for j in range(nlib):
        mu[j] = offset[j]
    for c in range(ncoef):
        b = beta[c]
        for j in range(nlib):
            mu[j] += design[j, c] * b
    for j in range(nlib):
        mu[j] = math.exp(mu[j])


@njit(cache=True)
def _nb_dev_sum(y, mu, disp, w):
    s = 0.0
    for j in range(y.shape[0]):
        s += w[j] * unit_nb_deviance(y[j], mu[j], disp[j])
    return s


@njit(cache=True)
def _cholesky_upper(a, u):
    """Upper Cholesky factor U with U'U = a (LAPACK dpotf2, uplo = 'U').

    Returns False if a is not positive definite.
    """
    n = a.shape[0]
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s -= u[k, j] * u[k, j]
        if not s > 0.0:
            return False
        ujj = math.sqrt(s)
        u[j, j] = ujj
        for i in range(j + 1, n):
            t = a[j, i]
            for k in range(j):
                t -= u[k, j] * u[k, i]
            u[j, i] = t / ujj
    return True


@njit(cache=True)
def _cholesky_solve_upper(u, b, x):
    """Solve U'U x = b given the upper factor U (LAPACK dpotrs, uplo = 'U')."""
    n = u.shape[0]
    for i in range(n):
        t = b[i]
        for k in range(i):
            t -= u[k, i] * x[k]
        x[i] = t / u[i, i]
    for i in range(n - 1, -1, -1):
        t = x[i]
        for k in range(i + 1, n):
            t -= u[i, k] * x[k]
        x[i] = t / u[i, i]


@njit(cache=True)
def _leven_vec(y, offset, disp, w, design, beta, mu, maxit, tol):
    """Levenberg-Marquardt NB fit for one gene (fit_leven_vec).

    beta holds the starting values on entry and the estimates on exit; mu
    receives the fitted values. Returns (deviance, iterations, failed).
    """
    nlib, ncoef = design.shape
    ymax = 0.0
    for j in range(nlib):
        if y[j] > ymax:
            ymax = y[j]
    if ymax < 1e-10:
        for c in range(ncoef):
            beta[c] = np.nan
        for j in range(nlib):
            mu[j] = 0.0
        return 0.0, 0, False

    zwpt = np.empty(nlib)
    drvt = np.empty(nlib)
    dl = np.empty(ncoef)
    db = np.zeros(ncoef)
    xtwx = np.empty((ncoef, ncoef))
    xtwc = np.empty((ncoef, ncoef))
    chol = np.zeros((ncoef, ncoef))
    nbeta = np.empty(ncoef)
    nmu = np.empty(nlib)

    _autofill(beta, offset, design, mu)
    dev = _nb_dev_sum(y, mu, disp, w)

    max_info = -1.0
    lam = 0.0
    failed = False
    it = 0
    while True:
        it += 1
        if it > maxit:
            break
        for j in range(nlib):
            denom = 1.0 + mu[j] * disp[j]
            zwpt[j] = mu[j] / denom * w[j]
            drvt[j] = (y[j] - mu[j]) / denom * w[j]
        for c1 in range(ncoef):
            for c2 in range(c1 + 1):
                s = 0.0
                for j in range(nlib):
                    s += design[j, c2] * zwpt[j] * design[j, c1]
                xtwx[c2, c1] = s
                xtwx[c1, c2] = s
        for c in range(ncoef):
            s = 0.0
            for j in range(nlib):
                s += drvt[j] * design[j, c]
            dl[c] = s
            if xtwx[c, c] > max_info:
                max_info = xtwx[c, c]
        if it == 1:
            lam = max_info * 1e-6
            if lam < 1e-13:
                lam = 1e-13

        lev = 0
        low_dev = False
        while True:
            lev += 1
            # Increase the damping until the damped information matrix factorizes
            while True:
                for c1 in range(ncoef):
                    for c2 in range(c1 + 1):
                        xtwc[c2, c1] = xtwx[c2, c1]
                    xtwc[c1, c1] += lam
                if _cholesky_upper(xtwc, chol):
                    break
                lam *= 10.0
                if lam <= 0.0:
                    lam = 1e-100
            _cholesky_solve_upper(chol, dl, db)
            for c in range(ncoef):
                nbeta[c] = beta[c] + db[c]
            _autofill(nbeta, offset, design, nmu)
            ndev = _nb_dev_sum(y, nmu, disp, w)
            if ndev / ymax < 1e-13:
                low_dev = True
            if ndev <= dev or low_dev:
                for c in range(ncoef):
                    beta[c] = nbeta[c]
                for j in range(nlib):
                    mu[j] = nmu[j]
                dev = ndev
                break
            lam *= 2.0
            if lam <= 0.0:
                lam = 1e-100
            if lam / max_info > 1e13:
                failed = True
                break

        divergence = 0.0
        for c in range(ncoef):
            divergence += dl[c] * db[c]
        if failed or low_dev or divergence < tol:
            break
        if lev == 1:
            lam /= 10.0
    return dev, it, failed


@njit(cache=True, parallel=True)
def fit_leven_mat(y, offset, disp, w, design, beta_start, maxit, tol):
    """Levenberg-Marquardt fits for every gene (fit_leven)."""
    ngenes, nlibs = y.shape
    ncoef = design.shape[1]
    beta = beta_start.copy()
    mu = np.empty((ngenes, nlibs))
    dev = np.empty(ngenes)
    iters = np.empty(ngenes, dtype=np.int64)
    failed = np.empty(ngenes, dtype=np.bool_)
    for g in prange(ngenes):
        b = np.empty(ncoef)
        for c in range(ncoef):
            b[c] = beta[g, c]
        m = np.empty(nlibs)
        d, i, f = _leven_vec(y[g], offset[g], disp[g], w[g], design, b, m, maxit, tol)
        for c in range(ncoef):
            beta[g, c] = b[c]
        for j in range(nlibs):
            mu[g, j] = m[j]
        dev[g] = d
        iters[g] = i
        failed[g] = f
    return beta, mu, dev, iters, failed


def groups_from_design(design, tol=1e-12):
    """Groups of libraries with matching design rows (get_groups_from_design).

    Two libraries share a group when every design entry matches to within
    tol. Groups are numbered from 0 in order of first appearance. Returns
    (group, ngroups).
    """
    design = np.asarray(design, dtype=np.float64)
    nlibs = design.shape[0]
    group = np.full(nlibs, -1, dtype=np.int64)
    ngroups = 0
    for i in range(nlibs):
        if group[i] != -1:
            continue
        match = (group == -1) & np.all(np.abs(design - design[i]) <= tol, axis=1)
        group[match] = ngroups
        ngroups += 1
    return group, ngroups


def leven_start(y, offset, disp, w, design, use_null=True):
    """Starting values for the Levenberg fit (get_leven_start).

    With use_null, each gene starts from the least-squares fit of the
    constant log(weighted average count / library size) on the design.
    Otherwise it starts from a weighted least-squares fit of the log counts,
    floored at min(max count, 1/6), minus the offsets.
    """
    ngenes, nlibs = y.shape
    if use_null:
        lib = np.exp(offset)
        curweight = w * lib / (1.0 + disp * lib)
        with np.errstate(divide='ignore', invalid='ignore'):
            level = np.log(np.sum(y * curweight / lib, axis=1) / np.sum(curweight, axis=1))
        q, r = np.linalg.qr(design)
        unit = np.linalg.solve(r, q.T @ np.ones(nlibs))
        return level[:, None] * unit[None, :]

    delta = min(float(np.max(y)), 1.0 / 6.0)
    z = np.log(np.maximum(delta, y)) - offset
    sw = np.sqrt(w)
    if np.all(w == w[0]):
        q, r = np.linalg.qr(design * sw[0][:, None])
        return np.linalg.solve(r, q.T @ (z * sw[0]).T).T
    beta = np.empty((ngenes, design.shape[1]))
    for g in range(ngenes):
        q, r = np.linalg.qr(design * sw[g][:, None])
        beta[g] = np.linalg.solve(r, q.T @ (z[g] * sw[g]))
    return beta
