# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Single-cell NB mixed model fitting (NEBULA-LN port).

Implements ``glm_sc_fit()`` and ``glm_sc_test()`` for cell-level
negative binomial gamma mixed model (NBGMM) analysis of multi-subject
single-cell RNA-seq data.

Reference
---------
He L, Davila-Velderrain J, Sumida TS, Hafler DA, Bhatt DL et al.
NEBULA is a fast negative binomial mixed model for differential or
co-expression analysis of large-scale multi-subject single-cell data.
*Communications Biology*, 4:629, 2021.
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import lgamma as _lgamma
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize as _minimize
from scipy.special import digamma as _digamma, gammaln as _gammaln
from scipy.stats import chi2 as _chi2

from .normalization import calc_norm_factors
from .dgelist import make_dgelist


# ---------------------------------------------------------------------------
# Numba-accelerated core functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _digamma_nb(x):
    """Digamma (psi) function for x > 0. Accurate to ~15 digits."""
    result = 0.0
    while x < 7.0:
        result -= 1.0 / x
        x += 1.0
    r = 1.0 / (x * x)
    result += math.log(x) - 0.5 / x
    result -= r * (1.0/12.0 - r * (1.0/120.0 - r * (1.0/252.0
              - r * (1.0/240.0 - r * (5.0/660.0 - r * 691.0/32760.0)))))
    return result


@njit(cache=True)
def _ptmg_negll_and_grad_nb(para, X, offset, Y, n_one, n_two, ytwo,
                             fid, cumsumy, posind, posindy, nb, nind, k):
    """Numba-compiled NBGMM negative log-likelihood + gradient."""
    beta = para[:nb]
    sigma_param = para[nb]
    phi = para[nb + 1]

    exps = math.exp(sigma_param)
    exps_m1 = exps - 1.0
    if exps_m1 <= 0:
        return 1e30, np.zeros(nb + 2)
    alpha = 1.0 / exps_m1
    exps_s = math.sqrt(exps)
    lam = alpha / exps_s
    gamma = phi

    exps_m1_sq = exps_m1 * exps_m1
    alpha_pr = -exps / exps_m1_sq
    lambda_pr = (1.0 - 3.0 * exps) / (2.0 * exps_s * exps_m1_sq)

    log_lambda = math.log(lam)
    log_gamma = math.log(gamma) if gamma > 0 else -1e30

    nelem = len(posindy)

    # Linear predictor: xtb = offset + X @ beta
    xtb = np.empty(nind)
    for i in range(nind):
        s = offset[i]
        for j in range(nb):
            s += X[i, j] * beta[j]
        xtb[i] = s

    # term1 = sum y_j * eta_j (only non-zero)
    term1 = 0.0
    for i in range(nelem):
        term1 += xtb[posindy[i]] * Y[i]

    # exp(xtb) with overflow protection
    extb = np.empty(nind)
    for i in range(nind):
        v = xtb[i]
        extb[i] = math.exp(min(v, 500.0))

    # Per-sample sums
    cumsumxtb = np.empty(k)
    for s in range(k):
        start = fid[s]
        end = fid[s + 1]
        acc = 0.0
        for i in range(start, end):
            acc += extb[i]
        cumsumxtb[s] = acc

    ystar = np.empty(k)
    mustar = np.empty(k)
    mustar_log = np.empty(k)
    ymustar = np.empty(k)
    ymumustar = np.empty(k)
    for s in range(k):
        ystar[s] = cumsumy[s] + alpha
        mustar[s] = cumsumxtb[s] + lam
        mustar_log[s] = math.log(mustar[s])
        ymustar[s] = ystar[s] / mustar[s]
        ymumustar[s] = ymustar[s] / mustar[s]

    for s in range(k):
        term1 -= ystar[s] * mustar_log[s]
    term1 += k * alpha * log_lambda
    term1 += nind * gamma * log_gamma

    # gstar = gamma + y_j
    gstar_vec = np.full(nind, gamma)
    for i in range(nelem):
        gstar_vec[posindy[i]] += Y[i]

    # sum_elgcp[j] = ymustar[s(j)] * extb[j]
    sum_elgcp = np.empty(nind)
    for s in range(k):
        start = fid[s]
        end = fid[s + 1]
        val = ymustar[s]
        for i in range(start, end):
            sum_elgcp[i] = val * extb[i]

    for i in range(nind):
        term1 += sum_elgcp[i]

    sum_elgcp_pg = np.empty(nind)
    gstar_phiymustar = np.empty(nind)
    log_sum_elgcp_pg = np.empty(nind)
    slpey = 0.0
    for i in range(nind):
        sum_elgcp_pg[i] = sum_elgcp[i] + gamma
        gstar_phiymustar[i] = gstar_vec[i] / sum_elgcp_pg[i]
        log_sum_elgcp_pg[i] = math.log(sum_elgcp_pg[i])
        slpey += log_sum_elgcp_pg[i]

    term1 -= gamma * slpey
    for i in range(nelem):
        term1 -= Y[i] * log_sum_elgcp_pg[posindy[i]]

    fn_cpp = -term1

    # --- Gradient ---
    dbeta_42 = np.zeros(k)
    xexb_f = np.zeros((nb, k))
    dbeta_41 = np.zeros((nb, k))

    for s in range(k):
        start = fid[s]
        end = fid[s + 1]
        for i in range(start, end):
            gp = gstar_phiymustar[i]
            ext_i = extb[i]
            dbeta_42[s] += gp * ext_i
            for j in range(nb):
                xexb = X[i, j] * ext_i
                xexb_f[j, s] += xexb
                dbeta_41[j, s] += gp * xexb

    db = np.zeros(nb)
    for i in range(nelem):
        for j in range(nb):
            db[j] += X[posindy[i], j] * Y[i]

    for s in range(k):
        val = ymumustar[s] * (dbeta_42[s] - cumsumxtb[s])
        for j in range(nb):
            db[j] += xexb_f[j, s] * val - dbeta_41[j, s] * ymustar[s]

    ldm = log_lambda * k
    for s in range(k):
        ldm -= mustar_log[s]
    adlmy = exps_s * k
    for s in range(k):
        adlmy -= ymustar[s]

    dtau = 0.0
    dtau_lp = 0.0
    for s in range(k):
        dtau += alpha_pr * (cumsumxtb[s] - dbeta_42[s]) / mustar[s]
        dtau_lp += ymumustar[s] * (dbeta_42[s] - cumsumxtb[s])
    dtau += lambda_pr * dtau_lp
    dtau += alpha_pr * ldm + lambda_pr * adlmy

    dtau2 = log_gamma * nind + nind - slpey
    for i in range(nind):
        dtau2 -= gstar_phiymustar[i]

    gr = np.zeros(nb + 2)
    for j in range(nb):
        gr[j] = -db[j]
    gr[nb] = -dtau
    gr[nb + 1] = -dtau2

    # --- R-level lgamma corrections ---
    n_one_plus_two = n_one + n_two

    lgamma_fn = 0.0
    for s_idx in range(len(posind)):
        lgamma_fn += math.lgamma(cumsumy[posind[s_idx]] + alpha)
    lgamma_fn -= len(posind) * math.lgamma(alpha)
    for v_idx in range(len(ytwo)):
        lgamma_fn += math.lgamma(ytwo[v_idx] + gamma)
    lgamma_fn -= (nelem - n_one_plus_two) * math.lgamma(gamma)
    if n_one_plus_two > 0:
        lgamma_fn += n_one_plus_two * math.log(gamma)
    if n_two > 0:
        lgamma_fn += n_two * math.log(gamma + 1.0)

    fn = fn_cpp - lgamma_fn

    # --- Digamma corrections ---
    dig_alpha_sum = 0.0
    for s_idx in range(len(posind)):
        dig_alpha_sum += _digamma_nb(cumsumy[posind[s_idx]] + alpha)
    dig_alpha_sum -= len(posind) * _digamma_nb(alpha)

    dig_gamma_sum = 0.0
    for v_idx in range(len(ytwo)):
        dig_gamma_sum += _digamma_nb(ytwo[v_idx] + gamma)
    dig_gamma_sum -= (nelem - n_one_plus_two) * _digamma_nb(gamma)
    if n_one_plus_two > 0:
        dig_gamma_sum += n_one_plus_two / gamma
    if n_two > 0:
        dig_gamma_sum += n_two / (gamma + 1.0)

    gr[nb] -= alpha_pr * dig_alpha_sum
    gr[nb + 1] -= dig_gamma_sum

    return fn, gr


@njit(cache=True)
def _compute_pml_loglik_nb(offset, X, beta, logw, fid, k, posindy, Y,
                            cumsumy, gamma, alpha, lam, nind, nb):
    """Numba-compiled PML log-likelihood evaluation."""
    nelem = len(posindy)

    # extb_lin = offset + X @ beta
    extb_lin = np.empty(nind)
    for i in range(nind):
        s = offset[i]
        for j in range(nb):
            s += X[i, j] * beta[j]
        extb_lin[i] = s

    loglik = 0.0
    for i in range(nelem):
        loglik += extb_lin[posindy[i]] * Y[i]

    # logw @ cumsumy
    for s in range(k):
        loglik += logw[s] * cumsumy[s]

    # Add logw to linear predictor per sample
    for s in range(k):
        start = fid[s]
        end = fid[s + 1]
        for i in range(start, end):
            extb_lin[i] += logw[s]

    # exp(extb_lin)
    extb = np.empty(nind)
    for i in range(nind):
        extb[i] = math.exp(min(extb_lin[i], 500.0))

    for i in range(nind):
        extbphil = math.log(extb[i] + gamma)
        loglik -= gamma * extbphil

    for i in range(nelem):
        loglik -= Y[i] * math.log(extb[posindy[i]] + gamma)

    for s in range(k):
        loglik += alpha * logw[s] - lam * math.exp(logw[s])

    return loglik, extb


@njit(cache=True)
def _opt_pml_nb(X, offset, Y_vals, fid, cumsumy, posindy, nb, nind, k,
                beta_init, sigma0, sigma1, eps, ord_):
    """Numba-compiled PML optimizer.

    Returns (beta, logw, vb2, loglik, loglikp, logdet, step, stepd, sec_ord).
    """
    exps = math.exp(sigma0)
    alpha = 1.0 / (exps - 1.0)
    lam = 1.0 / (math.sqrt(exps) * (exps - 1.0))
    gamma = sigma1

    logw = np.zeros(k)
    beta = beta_init.copy()

    # gstar: gamma + y for non-zero cells
    gstar = np.full(nind, gamma)
    nelem = len(posindy)
    for i in range(nelem):
        gstar[posindy[i]] += Y_vals[i]

    # Precompute yx = X^T @ y (only non-zero entries)
    yx = np.zeros(nb)
    for i in range(nelem):
        for j in range(nb):
            yx[j] += X[posindy[i], j] * Y_vals[i]

    # Initial log-likelihood
    loglik, extb = _compute_pml_loglik_nb(
        offset, X, beta, logw, fid, k, posindy, Y_vals,
        cumsumy, gamma, alpha, lam, nind, nb
    )

    loglikp = 0.0
    step = 0
    maxstep = 50
    maxstd = 10
    convd = 0.01
    stepd = 0

    vb = np.zeros((nb, nb))
    vb2 = np.zeros((nb, nb))
    vw = np.zeros(k)
    vwb = np.zeros((k, nb))

    while step == 0 or (loglik - loglikp > eps and step < maxstep):
        step += 1

        damp = np.ones(nb)
        damp_w = np.ones(k)

        # gstar_extb_phi = gstar / (1 + gamma/extb)
        gstar_extb_phi = np.empty(nind)
        for i in range(nind):
            if extb[i] < 1e-300:
                gstar_extb_phi[i] = 0.0
            else:
                gstar_extb_phi[i] = gstar[i] / (1.0 + gamma / extb[i])

        # Gradient w.r.t. beta: db = yx - X^T @ gstar_extb_phi
        db = np.empty(nb)
        for j in range(nb):
            s = yx[j]
            for i in range(nind):
                s -= X[i, j] * gstar_extb_phi[i]
            db[j] = s

        # Gradient w.r.t. logw
        dw = np.empty(k)
        w = np.empty(k)
        for s in range(k):
            start = fid[s]
            end = fid[s + 1]
            acc = 0.0
            for i in range(start, end):
                acc += gstar_extb_phi[i]
            w[s] = math.exp(logw[s])
            dw[s] = cumsumy[s] - acc - lam * w[s] + alpha

        # Hessian diagonal w.r.t. logw
        gstar_extb_phi2 = np.empty(nind)
        for i in range(nind):
            denom = extb[i] + gamma
            if denom < 1e-300:
                gstar_extb_phi2[i] = 0.0
            else:
                gstar_extb_phi2[i] = gstar_extb_phi[i] / denom

        for s in range(k):
            start = fid[s]
            end = fid[s + 1]
            acc = 0.0
            for i in range(start, end):
                acc += gstar_extb_phi2[i]
            vw[s] = gamma * acc + lam * w[s]

        # Cross-term Hessian vwb (k × nb)
        for s in range(k):
            start = fid[s]
            end = fid[s + 1]
            for j in range(nb):
                acc = 0.0
                for i in range(start, end):
                    acc += X[i, j] * gstar_extb_phi2[i]
                vwb[s, j] = gamma * acc

        # Hessian w.r.t. beta (nb × nb)
        for ii in range(nb):
            for jj in range(ii, nb):
                acc = 0.0
                for i in range(nind):
                    acc += X[i, ii] * gstar_extb_phi2[i] * X[i, jj]
                vb[ii, jj] = gamma * acc
                if ii != jj:
                    vb[jj, ii] = vb[ii, jj]

        # Floor vw to avoid division by zero
        for s in range(k):
            if vw[s] < 1e-15:
                vw[s] = 1e-15

        # Schur complement: vb2 = vb - vwb^T @ diag(1/vw) @ vwb
        for ii in range(nb):
            for jj in range(nb):
                acc = 0.0
                for s in range(k):
                    acc += vwb[s, ii] * vwb[s, jj] / vw[s]
                vb2[ii, jj] = vb[ii, jj] - acc

        # Newton step
        dwvw = np.empty(k)
        for s in range(k):
            dwvw[s] = dw[s] / vw[s]

        # rhs = db - vwb^T @ dwvw
        rhs = np.empty(nb)
        for j in range(nb):
            acc = 0.0
            for s in range(k):
                acc += vwb[s, j] * dwvw[s]
            rhs[j] = db[j] - acc

        # Regularize if needed
        for ii in range(nb):
            if abs(vb2[ii, ii]) < 1e-10:
                vb2[ii, ii] += 1e-8

        stepbeta = np.linalg.solve(vb2, rhs)

        # steplogw = dwvw - (vwb @ stepbeta) / vw  (vw already floored above)
        steplogw = np.empty(k)
        for s in range(k):
            acc = 0.0
            for j in range(nb):
                acc += vwb[s, j] * stepbeta[j]
            steplogw[s] = dwvw[s] - acc / vw[s]

        new_b = beta + stepbeta
        new_w = logw + steplogw

        loglikp = loglik
        loglik, extb = _compute_pml_loglik_nb(
            offset, X, new_b, new_w, fid, k, posindy, Y_vals,
            cumsumy, gamma, alpha, lam, nind, nb
        )

        likdif = loglik - loglikp
        stepd = 0
        minstep = 40.0

        while likdif < 0 or math.isinf(loglik):
            stepd += 1
            minstep /= 2.0

            if stepd > maxstd:
                likdif = 0.0
                loglik = loglikp
                mabsdb = 0.0
                mabsdw = 0.0
                for j in range(nb):
                    if abs(db[j]) > mabsdb:
                        mabsdb = abs(db[j])
                for s in range(k):
                    if abs(dw[s]) > mabsdw:
                        mabsdw = abs(dw[s])
                if mabsdb > convd or mabsdw > convd:
                    stepd += 1
                break

            for i in range(nb):
                if -40 < stepbeta[i] < 40:
                    damp[i] /= 2.0
                    new_b[i] = beta[i] + stepbeta[i] * damp[i]
                else:
                    new_b[i] = beta[i] + (minstep if stepbeta[i] > 0 else -minstep)

            for s in range(k):
                if -40 < steplogw[s] < 40:
                    damp_w[s] /= 2.0
                    new_w[s] = logw[s] + steplogw[s] * damp_w[s]
                else:
                    new_w[s] = logw[s] + (minstep if steplogw[s] > 0 else -minstep)

            loglik, extb = _compute_pml_loglik_nb(
                offset, X, new_b, new_w, fid, k, posindy, Y_vals,
                cumsumy, gamma, alpha, lam, nind, nb
            )
            likdif = loglik - loglikp

        beta = new_b
        logw = new_w

    # Log-determinant
    logdet = 0.0
    for s in range(k):
        logdet += math.log(max(abs(vw[s]), 1e-300))

    # Second-order correction
    sec_ord = 0.0
    if ord_ > 1:
        for i in range(nind):
            if extb[i] < 1e-300:
                gstar_extb_phi[i] = 0.0
            else:
                gstar_extb_phi[i] = gstar[i] / (1.0 + gamma / extb[i])
        extbg = np.empty(nind)
        for i in range(nind):
            extbg[i] = extb[i] + gamma
            if extbg[i] < 1e-300:
                gstar_extb_phi[i] = 0.0
            else:
                gstar_extb_phi[i] /= extbg[i]
        for s in range(k):
            start = fid[s]
            end = fid[s + 1]
            acc = 0.0
            for i in range(start, end):
                acc += gstar_extb_phi[i]
            vw[s] = gamma * acc + lam * math.exp(logw[s])
            if vw[s] < 1e-15:
                vw[s] = 1e-15
        vws = np.empty(k)
        for s in range(k):
            vws[s] = vw[s] * vw[s]

        for i in range(nind):
            if extbg[i] < 1e-300:
                gstar_extb_phi[i] = 0.0
            else:
                gstar_extb_phi[i] /= extbg[i]
        third_der = np.empty(k)
        for s in range(k):
            start = fid[s]
            end = fid[s + 1]
            acc = 0.0
            for i in range(start, end):
                acc += gstar_extb_phi[i] * (gamma - extb[i])
            third_der[s] = gamma * acc + lam * math.exp(logw[s])
        acc = 0.0
        for s in range(k):
            acc += third_der[s] * third_der[s] / (vws[s] * vw[s])
        sec_ord += 5.0 / 24.0 * acc

        if ord_ > 2:
            for i in range(nind):
                if extbg[i] < 1e-300:
                    gstar_extb_phi[i] = 0.0
                else:
                    gstar_extb_phi[i] /= extbg[i]
            four_der = np.empty(k)
            for s in range(k):
                start = fid[s]
                end = fid[s + 1]
                acc = 0.0
                for i in range(start, end):
                    extbp = extb[i] * extb[i]
                    acc += gstar_extb_phi[i] * (gamma*gamma + extbp - 4*gamma*extb[i])
                four_der[s] = gamma * acc + lam * math.exp(logw[s])
            acc = 0.0
            for s in range(k):
                acc += four_der[s] / vws[s]
            sec_ord -= acc / 8.0

            for i in range(nind):
                if extbg[i] < 1e-300:
                    gstar_extb_phi[i] = 0.0
                else:
                    gstar_extb_phi[i] /= extbg[i]
            for s in range(k):
                start = fid[s]
                end = fid[s + 1]
                acc2 = 0.0
                for i in range(start, end):
                    extbp = extb[i] * extb[i]
                    acc2 += gstar_extb_phi[i] * (
                        gamma**3 - 11*gamma*gamma*extb[i]
                        + 11*gamma*extbp - extbp*extb[i]
                    )
                four_der[s] = gamma * acc2 + lam * math.exp(logw[s])
            acc = 0.0
            for s in range(k):
                acc += four_der[s] * third_der[s] / (vws[s] * vws[s])
            sec_ord += 7.0 / 48.0 * acc

    return beta, logw, vb2, loglik, loglikp, logdet, step, stepd, sec_ord


@njit(cache=True)
def _get_cell_nb(X, fid, nb, k):
    """Numba-compiled cell-level covariate detection."""
    iscell = np.zeros(nb)
    for i in range(nb):
        for j in range(k):
            start = fid[j]
            end = fid[j + 1]
            ref = X[start, i]
            found = False
            for idx in range(start, end):
                if X[idx, i] != ref:
                    found = True
                    break
            if found:
                iscell[i] = 1.0
                break
    return iscell

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _center_design(pred: np.ndarray):
    """Center design columns and scale to unit variance.

    Matches nebula's ``center_m`` C++ function exactly.

    Returns
    -------
    pred_centered : ndarray (n × p)
    sds : ndarray (p,)
        Column standard deviations (population, not sample).
        The intercept column gets sd=0; zero-vector columns get sd=-1.
    int_col : int
        0-based index of the intercept column.
    """
    pred = np.asarray(pred, dtype=np.float64).copy()
    n, p = pred.shape
    means = pred.mean(axis=0)
    cm = pred - means
    sds = np.sqrt((cm * cm).mean(axis=0))

    int_col = None
    for i in range(p):
        if sds[i] > 0:
            cm[:, i] /= sds[i]
        else:
            if pred[0, i] != 0:
                # intercept column: fill with ones
                cm[:, i] = 1.0
                sds[i] = 0.0
                int_col = i
            else:
                sds[i] = -1.0

    if int_col is None:
        raise ValueError("The design matrix must include an intercept term.")
    if (sds == 0).sum() > 1 or (sds < 0).any():
        raise ValueError(
            "Some predictors have zero variation or a zero vector."
        )

    return cm, sds, int_col


def _cv_offset(offset: np.ndarray | None, nind: int):
    """Process offset, matching nebula's ``cv_offset`` C++ function.

    Parameters
    ----------
    offset : array (nind,) of *positive* scaling factors, or None.
    nind : int

    Returns
    -------
    log_offset : ndarray (nind,)  — log of the offset
    moffset : float  — mean of log-offset (0 if offset was None)
    cv2 : float  — squared CV of the raw offset
    """
    if offset is None:
        log_offset = np.zeros(nind)
        return log_offset, 0.0, 0.0

    offset = np.asarray(offset, dtype=np.float64)
    moffset_raw = offset.mean()
    cv = 0.0
    if moffset_raw > 0:
        cv = np.sqrt(((offset - moffset_raw) ** 2).sum() / nind) / moffset_raw
    log_offset = np.log(offset)
    moffset = log_offset.mean()
    return log_offset, moffset, cv * cv


def _call_cumsumy(count, fid, k, ngene):
    """Sum counts per gene per sample, matching nebula's ``call_cumsumy``.

    Parameters
    ----------
    count : sparse or dense, genes × cells
    fid : int array (k+1,) of segment boundaries (0-based)
    k : int, number of samples
    ngene : int

    Returns
    -------
    cumsumy : ndarray (ngene, k)
    """
    cumsumy = np.zeros((ngene, k), dtype=np.float64)
    # Use dense or sparse slicing
    for s in range(k):
        start, end = fid[s], fid[s + 1]
        chunk = count[:, start:end]
        if hasattr(chunk, 'toarray'):
            cumsumy[:, s] = np.asarray(chunk.sum(axis=1)).ravel()
        else:
            cumsumy[:, s] = chunk.sum(axis=1).ravel()
    return cumsumy


def _call_posindy(y_gene: np.ndarray):
    """Extract non-zero positions and counts for a single gene row.

    Matches nebula's ``call_posindy``.

    Parameters
    ----------
    y_gene : 1D array (ncells,)

    Returns
    -------
    dict with keys:
        posindy : int array — 0-based indices of non-zero cells
        Y : float array — corresponding count values
        mct : float — mean count per cell
        n_onetwo : int array (2,) — [n_one, n_two]
        ytwo : float array — counts > 2
    """
    nz = np.nonzero(y_gene)[0]
    posindy = nz.astype(np.int32)
    Y = y_gene[nz].astype(np.float64)
    mct = Y.sum() / len(y_gene)

    n_one = int((Y == 1).sum())
    n_two = int((Y == 2).sum())
    ytwo = Y[Y > 2]

    return {
        'posindy': posindy,
        'Y': Y,
        'mct': mct,
        'n_onetwo': np.array([n_one, n_two], dtype=np.int32),
        'ytwo': ytwo,
    }


def _get_cell(X, fid, nb, k):
    """Identify cell-level covariates (vary within a subject).

    Delegates to numba-compiled ``_get_cell_nb``.
    """
    return _get_cell_nb(X, fid, nb, k)


def _get_cv(offset, X, beta, cell_ind, ncell, nc):
    """Compute squared CV of fitted values at cell-level predictors.

    Matches nebula's ``get_cv``.
    """
    extb = offset.copy()
    for i in range(ncell):
        ind = int(cell_ind[i])
        extb = extb + X[:, ind] * beta[ind]
    with np.errstate(over='ignore'):
        extb = np.exp(extb)
    m = extb.mean()
    if m > 0:
        return ((extb - m) ** 2).sum() / nc / (m * m)
    return 0.0


# ---------------------------------------------------------------------------
# NBGMM log-likelihood + gradient (ptmg_ll_der)
# ---------------------------------------------------------------------------

def _ptmg_negll_and_grad(para, X, offset, Y, n_onetwo, ytwo, fid, cumsumy,
                         posind, posindy, nb, nind, k):
    """Negative log-likelihood and gradient for NBGMM (L-BFGS-B stage).

    Delegates to numba-compiled ``_ptmg_negll_and_grad_nb``.
    """
    return _ptmg_negll_and_grad_nb(
        para, X, offset, Y,
        int(n_onetwo[0]), int(n_onetwo[1]), ytwo,
        fid, cumsumy, posind, posindy, nb, nind, k
    )


# ---------------------------------------------------------------------------
# Penalized ML optimizer (opt_pml for NBGMM)
# ---------------------------------------------------------------------------

def _opt_pml(X, offset, Y_vals, fid, cumsumy, posind, posindy, nb, nind, k,
             beta_init, sigma, reml=0, eps=1e-6, ord_=1):
    """Port of nebula's ``opt_pml`` C++ function.

    Delegates to numba-compiled ``_opt_pml_nb``.
    """
    beta, logw, vb2, loglik, loglikp, logdet, step, stepd, sec_ord = \
        _opt_pml_nb(X, offset, Y_vals, fid, cumsumy, posindy, nb, nind, k,
                    beta_init, sigma[0], sigma[1], eps, ord_)
    return {
        'beta': beta,
        'logw': logw,
        'var': vb2,
        'loglik': loglik,
        'loglikp': loglikp,
        'logdet': logdet,
        'iter': int(step),
        'damp': int(stepd),
        'second': sec_ord,
    }


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def _check_conv(repml, conv, nb, vare, min_bounds, max_bounds, cutoff=1e-8):
    """Port of nebula's ``check_conv``."""
    if conv == 1:
        if vare[0] == max_bounds[0] or vare[1] == min_bounds[1]:
            conv = -60
        elif np.isnan(repml['loglik']):
            conv = -30
        elif repml['iter'] == 50:
            conv = -20
        elif repml['damp'] == 11:
            conv = -10
        elif repml['damp'] == 12:
            conv = -40

    if nb > 1:
        try:
            eigvals = np.linalg.eigvalsh(repml['var'])
            if eigvals.min() < cutoff:
                conv = -25
        except np.linalg.LinAlgError:
            conv = -25

    return conv


# ---------------------------------------------------------------------------
# Per-gene fitting
# ---------------------------------------------------------------------------

def _fit_gene_nebula_ln(gene_idx, y_gene, X, offset, fid, cumsumy_gene,
                        posind, nb, nind, k, sds, int_col, moffset,
                        min_bounds, max_bounds, mfs, cutoff_cell, kappa):
    """Fit NBGMM (NEBULA-LN) for a single gene.

    Returns
    -------
    tuple: (beta_rescaled, se_rescaled, sigma2, inv_phi, conv, logw)
    """
    posv = _call_posindy(y_gene)
    posindy = posv['posindy']
    Y = posv['Y']
    mct = posv['mct']
    n_onetwo = posv['n_onetwo']
    ytwo = posv['ytwo']

    # ord parameter
    if mct * mfs < 3:
        ord_ = 3
    else:
        ord_ = 1

    # Initial beta
    lmct = np.log(max(mct, 1e-300))
    para_init = np.zeros(nb + 2)
    para_init[int_col] = lmct - moffset
    para_init[nb] = 1.0       # sigma_param
    para_init[nb + 1] = 1.0   # phi (cell-level overdispersion)

    lower = np.concatenate([np.full(nb, -100.0), [min_bounds[0], min_bounds[1]]])
    upper = np.concatenate([np.full(nb, 100.0), [max_bounds[0], max_bounds[1]]])
    bounds = list(zip(lower, upper))

    # Stage 1: L-BFGS-B
    try:
        res = _minimize(
            _ptmg_negll_and_grad,
            para_init,
            args=(X, offset, Y, n_onetwo, ytwo, fid, cumsumy_gene,
                  posind, posindy, nb, nind, k),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 200},
        )
        refp = res.x
        is_conv = 1 if res.success else 0
    except Exception:
        # Fallback: use initial values
        refp = para_init.copy()
        is_conv = 0

    conv = is_conv
    vare = np.array([refp[nb], refp[nb + 1]])

    # Determine cell-level predictor CV
    cell_ind_arr = _get_cell(X, fid, nb, k)
    ncell = int(cell_ind_arr.sum())
    cell_ind = np.where(cell_ind_arr > 0)[0]
    if ncell > 0:
        try:
            cv2p = _get_cv(offset, X, refp[:nb], cell_ind, ncell, nind)
        except Exception:
            cv2p = float('nan')
    else:
        cv2p = 0.0

    gni = mfs * vare[1]

    # Determine if we need HL refinement
    fit = 1
    if (gni < cutoff_cell) or (conv == 0) or np.isnan(cv2p):
        # Would need NEBULA-HL refinement — skip for LN-only impl
        # Just note fit=2 for algorithm tracking
        fit = 2
    else:
        kappa_obs = gni / (1.0 + cv2p)
        if (kappa_obs < 20) or (kappa_obs < kappa and vare[0] < 8.0 / kappa_obs):
            fit = 3

    # Beta for PML: start from intercept init, not L-BFGS-B result
    betae = np.zeros(nb)
    betae[int_col] = lmct - moffset

    # Bias correction for intercept
    betae[int_col] -= vare[0] / 2.0

    # Stage 2: Penalized ML
    try:
        repml = _opt_pml(
            X, offset, Y, fid, cumsumy_gene, posind, posindy,
            nb, nind, k, betae, vare, reml=0, eps=1e-6, ord_=ord_,
        )
    except Exception:
        # Numerical failure in PML — mark as non-converged
        return (np.full(nb, np.nan), np.full(nb, np.nan),
                vare[0], 1.0 / vare[1] if vare[1] > 0 else np.inf,
                -30, 0, np.zeros(k))

    conv = _check_conv(repml, conv, nb, vare, min_bounds, max_bounds)

    # Invert Fisher information to get covariance
    beta_pml = repml['beta']
    logw = repml['logw']
    fisher = repml['var']  # This is vb2 (the Schur complement)

    se = np.full(nb, np.nan)
    if conv != -25:
        try:
            cov = np.linalg.inv(fisher)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            conv = -25

    # Rescale by column SDs (undo centering)
    sds_use = sds.copy()
    sds_use[int_col] = 1.0
    beta_rescaled = beta_pml / sds_use
    se_rescaled = se / sds_use

    sigma2 = vare[0]
    inv_phi = 1.0 / vare[1] if vare[1] > 0 else np.inf

    return beta_rescaled, se_rescaled, sigma2, inv_phi, conv, fit, logw


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def glm_sc_fit(y, cell_meta=None, design=None, sample=None,
               offset=None, norm_method='TMM', method='nebula',
               min_bounds=None, max_bounds=None,
               cpc=0.005, mincp=5, cutoff_cell=20, kappa=800,
               ncore=1, verbose=True):
    """Fit a single-cell NB gamma mixed model (NEBULA-LN).

    Parameters
    ----------
    y : AnnData, dict, or ndarray
        Count data. AnnData objects are (cells × genes); raw matrices
        should be (genes × cells).
    cell_meta : DataFrame, optional
        Cell-level metadata. Extracted from ``y.obs`` for AnnData.
    design : ndarray or str, optional
        Design matrix (cells × predictors) with an intercept column.
        If ``None``, an intercept-only model is fitted.
    sample : str or array-like
        Subject/sample identifiers. If a string, it names a column in
        *cell_meta*.
    offset : array-like, optional
        Positive per-cell scaling factors. If provided, ``norm_method``
        is ignored.
    norm_method : str
        ``'TMM'`` (default): compute per-cell offset from per-cell
        library sizes and pseudobulk TMM normalization factors.
        ``'none'``: all-ones offset (original nebula behaviour).
    method : str
        ``'nebula'`` (default): NEBULA-LN algorithm.
    min_bounds, max_bounds : tuple of float, optional
        Bounds for (sigma_param, phi). Defaults (1e-4, 1e-4) and
        (10, 1000).
    cpc : float
        Minimum mean counts per cell for gene filtering.
    mincp : int
        Minimum non-zero cells for gene filtering.
    cutoff_cell : float
        Threshold for NEBULA-HL fallback (cells_per_subject × phi).
    kappa : float
        Accuracy threshold for subject-level overdispersion.
    ncore : int
        Number of parallel workers (1 = sequential).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        DGEGLM-like fit result with keys ``'coefficients'``, ``'se'``,
        ``'dispersion'``, ``'design'``, ``'offset'``, ``'genes'``,
        ``'sigma_sample'``, ``'convergence'``, ``'method'``, etc.
        Pass to ``top_tags(fit, coef=...)`` for Wald testing.
    """
    if min_bounds is None:
        min_bounds = (1e-4, 1e-4)
    if max_bounds is None:
        max_bounds = (10.0, 1000.0)

    # --- Input handling ---
    gene_names = None
    try:
        import anndata
        is_anndata = isinstance(y, anndata.AnnData)
    except ImportError:
        is_anndata = False

    if is_anndata:
        adata = y
        X_raw = adata.X
        if hasattr(X_raw, 'toarray'):
            X_raw = X_raw.toarray()
        counts = np.asarray(X_raw, dtype=np.float64).T  # genes × cells
        if cell_meta is None:
            cell_meta = adata.obs.copy()
        gene_names = np.array(adata.var_names)
    elif isinstance(y, dict) and 'counts' in y:
        counts = np.asarray(y['counts'], dtype=np.float64)
        if cell_meta is None and 'obs' in y:
            cell_meta = y['obs']
        if gene_names is None and 'genes' in y:
            gene_names = np.asarray(y['genes'])
    else:
        if hasattr(y, 'toarray'):
            counts = np.asarray(y.toarray(), dtype=np.float64)
        else:
            counts = np.asarray(y, dtype=np.float64)

    ngene, nind = counts.shape
    if nind < 2:
        raise ValueError("There is no more than one cell in the count matrix.")

    # --- Resolve sample IDs ---
    if sample is None:
        raise ValueError(
            "The 'sample' argument is required. Provide per-cell sample IDs."
        )
    if isinstance(sample, str):
        if cell_meta is None:
            raise ValueError(
                f"sample='{sample}' requires cell_meta with that column."
            )
        sample_ids = np.asarray(cell_meta[sample])
    else:
        sample_ids = np.asarray(sample)
    if len(sample_ids) != nind:
        raise ValueError(
            "Length of sample IDs should equal the number of cells."
        )

    # --- Save design column names before sort (which may convert DataFrame → numpy) ---
    _design_colnames = None
    if design is not None and hasattr(design, 'columns'):
        _design_colnames = list(design.columns)

    # --- Sort cells by sample (group_cell) ---
    sample_ids_str = np.array([str(s) for s in sample_ids])
    levels = list(dict.fromkeys(sample_ids_str))  # unique, order-preserving
    sample_numeric = np.array(
        [levels.index(s) + 1 for s in sample_ids_str], dtype=np.int32
    )
    if not np.all(sample_numeric[:-1] <= sample_numeric[1:]):
        # Need to sort
        order = np.argsort(sample_numeric, kind='stable')
        counts = counts[:, order]
        sample_numeric = sample_numeric[order]
        sample_ids_str = sample_ids_str[order]
        if cell_meta is not None:
            if isinstance(cell_meta, pd.DataFrame):
                cell_meta = cell_meta.iloc[order].reset_index(drop=True)
            else:
                cell_meta = cell_meta[order]
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)[order]
        if design is not None and not isinstance(design, str):
            design = np.asarray(design, dtype=np.float64)[order]

    k = len(levels)
    # Build fid: 0-based start index of each sample's cells + sentinel
    diffs = np.where(np.concatenate([[1], np.diff(sample_numeric)]))[0]
    fid = np.concatenate([diffs, [nind]]).astype(np.int32)

    # --- Design matrix ---
    if design is None:
        pred = np.ones((nind, 1), dtype=np.float64)
        predn = None
        sds = np.array([0.0])
        int_col = 0
    else:
        if isinstance(design, str):
            # Formula — resolve against cell_meta
            from .utils import model_matrix
            pred = np.asarray(
                model_matrix(design, cell_meta), dtype=np.float64
            )
        else:
            pred = np.asarray(design, dtype=np.float64)
        if pred.shape[0] != nind:
            raise ValueError(
                "Design matrix rows must equal number of cells."
            )
        predn = None
        if hasattr(design, 'columns'):
            predn = list(design.columns)
        elif isinstance(design, pd.DataFrame):
            predn = list(design.columns)
        pred, sds, int_col = _center_design(pred)

    nb = pred.shape[1]

    # --- Offset ---
    if offset is not None:
        # User-provided offset (positive scaling factors)
        log_offset, moffset, cv2 = _cv_offset(offset, nind)
    elif norm_method.upper() == 'TMM':
        # Pseudobulk TMM normalization → per-cell offset
        lib_size = counts.sum(axis=0).astype(np.float64)
        pb = np.zeros((ngene, k), dtype=np.float64)
        for s in range(k):
            start, end = fid[s], fid[s + 1]
            chunk = counts[:, start:end]
            if hasattr(chunk, 'toarray'):
                pb[:, s] = np.asarray(chunk.sum(axis=1)).ravel()
            else:
                pb[:, s] = chunk.sum(axis=1).ravel()
        pb_dge = make_dgelist(pb)
        pb_dge = calc_norm_factors(pb_dge)
        norm_factors = pb_dge['samples']['norm.factors'].values
        # Expand sample-level norm factors to per-cell
        cell_nf = np.empty(nind, dtype=np.float64)
        for s in range(k):
            start, end = fid[s], fid[s + 1]
            cell_nf[start:end] = norm_factors[s]
        # Floor at 0.5 to avoid log(0) for zero-count cells
        offset_raw = np.maximum(lib_size * cell_nf, 0.5)
        log_offset, moffset, cv2 = _cv_offset(offset_raw, nind)
    else:
        # No normalization (original nebula behaviour)
        log_offset, moffset, cv2 = _cv_offset(None, nind)

    # --- CPS check ---
    mfs = nind / k
    if mfs < 30 and verbose:
        warnings.warn(
            f"The average number of cells per subject ({mfs:.1f}) is less "
            f"than 30. NEBULA-LN may be inaccurate for small cell counts."
        )

    # --- Cumsumy ---
    cumsumy = _call_cumsumy(counts, fid, k, ngene)

    # --- Gene filtering ---
    # Non-zero cell counts per gene
    if hasattr(counts, 'nnz'):
        # sparse
        from scipy.sparse import issparse
        nz_per_gene = np.diff(counts.indptr) if hasattr(counts, 'indptr') else \
            np.array([(counts[g, :] != 0).sum() for g in range(ngene)])
    else:
        nz_per_gene = (counts != 0).sum(axis=1)

    mean_cpc = cumsumy.sum(axis=1) / nind
    mask_cpc = mean_cpc > cpc
    mask_mincp = nz_per_gene >= mincp
    gene_mask = mask_cpc & mask_mincp
    gid = np.where(gene_mask)[0]
    lgid = len(gid)

    if verbose:
        print(f"Remove {ngene - lgid} genes having low expression.")
    if lgid == 0:
        raise ValueError("No gene passed the filtering.")
    if verbose:
        print(f"Analyzing {lgid} genes with {k} subjects and {nind} cells.")

    # posind per gene: which samples have non-zero counts
    posind_per_gene = [np.where(cumsumy[g, :] > 0)[0] for g in gid]

    # --- Per-gene fitting ---
    def _fit_one(idx):
        g = gid[idx]
        if hasattr(counts, 'toarray'):
            y_gene = np.asarray(counts[g, :].toarray()).ravel()
        else:
            y_gene = counts[g, :]
        return _fit_gene_nebula_ln(
            g, y_gene, pred, log_offset, fid, cumsumy[g, :],
            posind_per_gene[idx], nb, nind, k, sds, int_col, moffset,
            min_bounds, max_bounds, mfs, cutoff_cell, kappa,
        )

    if ncore > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=ncore) as executor:
            results = list(executor.map(_fit_one, range(lgid)))
    else:
        results = []
        for idx in range(lgid):
            if verbose and lgid > 100 and idx % max(1, lgid // 10) == 0:
                print(f"  Gene {idx + 1}/{lgid}...")
            results.append(_fit_one(idx))

    # --- Collect results ---
    coefficients = np.zeros((lgid, nb))
    se_arr = np.zeros((lgid, nb))
    sigma_sample = np.zeros(lgid)
    cell_disp = np.zeros(lgid)  # 1/phi
    convergence = np.zeros(lgid, dtype=np.int32)
    algorithm_codes = np.zeros(lgid, dtype=np.int32)

    for idx, res in enumerate(results):
        beta_r, se_r, sigma2, inv_phi, conv, fit, logw = res
        coefficients[idx, :] = beta_r
        se_arr[idx, :] = se_r
        sigma_sample[idx] = sigma2
        cell_disp[idx] = inv_phi
        convergence[idx] = conv
        algorithm_codes[idx] = fit

    # --- Resolve predictor names ---
    if predn is None:
        predn = _design_colnames
    if predn is None:
        if design is not None and hasattr(design, 'columns'):
            predn = list(design.columns)
    if predn is None:
        predn = [f"V{i+1}" for i in range(nb)]

    # --- Gene annotation DataFrame ---
    if gene_names is not None:
        genes_df = pd.DataFrame({'gene': gene_names[gid]})
    else:
        genes_df = None

    # --- Average log abundance for filtered genes ---
    ave_log_abund = np.log2(mean_cpc[gid] + 0.5)

    # --- DGEGLM-like return ---
    return {
        'coefficients': coefficients,
        'se': se_arr,
        'dispersion': cell_disp,
        'sigma_sample': sigma_sample,
        'convergence': convergence,
        'design': pred,
        'offset': log_offset,
        'genes': genes_df,
        'gene_mask': gene_mask,
        'method': 'nebula_ln',
        'ncells': nind,
        'nsamples': k,
        'predictor_names': predn,
        'sample_map': sample_ids_str,
        'samples_unique': np.array(levels),
        'ave_log_abundance': ave_log_abund,
    }


def shrink_sc_disp(fit, counts=None, covariate=None, robust=True):
    """Empirical Bayes shrinkage of cell-level NB dispersion.

    Shrinks the per-gene NB overdispersion parameter phi toward a
    (possibly trended) prior using limma's squeezeVar framework.

    Parameters
    ----------
    fit : dict
        Output from ``glm_sc_fit()``.
    counts : ndarray or sparse matrix, optional
        Gene-by-cell count matrix (same genes/ordering as used in
        ``glm_sc_fit``).  Used to compute log-mean abundance as
        covariate for the trended prior.  If *None* and
        ``fit['ave_log_abundance']`` exists, that is used instead.
    covariate : array-like, optional
        Custom covariate for the trended prior.  Overrides the
        abundance covariate derived from *counts*.
    robust : bool
        Use robust estimation (default True).  Protects against
        outlier genes with extremely high or low dispersion.

    Returns
    -------
    dict
        The input *fit* dict, updated in-place with new keys:

        - ``phi_raw`` : raw per-gene phi (= 1/dispersion)
        - ``phi_post`` : posterior (shrunk) phi
        - ``phi_prior`` : prior phi (scalar or trended)
        - ``df_residual`` : residual degrees of freedom
        - ``df_prior_phi`` : prior df from empirical Bayes
        - ``dispersion_shrunk`` : 1/phi_post (shrunk dispersion)
    """
    import warnings
    from .limma_port import squeeze_var

    dispersion = fit['dispersion']
    ngenes = len(dispersion)

    # Convert to phi = 1/dispersion
    with np.errstate(divide='ignore'):
        phi_raw = np.where(dispersion > 0, 1.0 / dispersion, np.inf)

    # Convergence mask: only use converged genes for prior estimation
    conv_mask = fit['convergence'] == 1

    # Floor phi at a small positive value; mark inf as NaN
    phi_floor = 1e-8
    phi_use = np.maximum(phi_raw.copy(), phi_floor)
    phi_use[~np.isfinite(phi_use)] = np.nan

    # Residual degrees of freedom: N - p - (K - 1)
    n_cells = fit['ncells']
    n_predictors = fit['design'].shape[1]
    n_samples = fit['nsamples']
    df_residual = n_cells - n_predictors - (n_samples - 1)
    df_residual = max(df_residual, 1)

    # Determine covariate for trended prior
    if covariate is not None:
        cov = np.asarray(covariate, dtype=np.float64)
    elif counts is not None:
        if hasattr(counts, 'toarray'):
            mean_cpc = np.asarray(counts.mean(axis=1)).ravel()
        else:
            mean_cpc = counts.mean(axis=1).ravel()
        gene_mask = fit['gene_mask']
        cov = np.log2(mean_cpc[gene_mask] + 0.5)
    elif 'ave_log_abundance' in fit:
        cov = fit['ave_log_abundance']
    else:
        cov = None

    # Filter to converged genes with finite phi
    ok_mask = conv_mask & np.isfinite(phi_use)
    idx_ok = np.where(ok_mask)[0]

    if len(idx_ok) < 3:
        warnings.warn("Fewer than 3 converged genes; skipping shrinkage.")
        fit['phi_raw'] = phi_raw
        fit['phi_post'] = phi_raw.copy()
        fit['phi_prior'] = np.nan
        fit['df_residual'] = df_residual
        fit['df_prior_phi'] = 0.0
        fit['dispersion_shrunk'] = fit['dispersion'].copy()
        return fit

    phi_ok = phi_use[idx_ok]
    cov_ok = cov[idx_ok] if cov is not None else None

    # Call squeeze_var with scalar df (same for all genes).
    # Fall back gracefully: trended → untrended → no shrinkage.
    sv = None
    for cov_attempt in ([cov_ok, None] if cov_ok is not None else [None]):
        try:
            sv = squeeze_var(phi_ok, df=float(df_residual),
                             covariate=cov_attempt, robust=robust)
            break
        except (ValueError, RuntimeError):
            continue
    if sv is None:
        try:
            sv = squeeze_var(phi_ok, df=float(df_residual),
                             covariate=None, robust=False)
        except (ValueError, RuntimeError):
            warnings.warn("squeeze_var failed; returning unshrunk estimates.")
            fit['phi_raw'] = phi_raw
            fit['phi_post'] = phi_raw.copy()
            fit['phi_prior'] = np.nanmedian(phi_ok)
            fit['df_residual'] = df_residual
            fit['df_prior_phi'] = 0.0
            fit['dispersion_shrunk'] = fit['dispersion'].copy()
            return fit

    # Map results back to full gene array
    phi_post = np.full(ngenes, np.nan)
    phi_post[idx_ok] = sv['var_post']

    phi_prior_full = np.full(ngenes, np.nan)
    if isinstance(sv['var_prior'], np.ndarray):
        phi_prior_full[idx_ok] = sv['var_prior']
        median_prior = np.nanmedian(sv['var_prior'])
    else:
        phi_prior_full[:] = sv['var_prior']
        median_prior = sv['var_prior']

    # Non-converged genes get the prior value
    phi_post[~ok_mask] = median_prior
    phi_prior_full[~ok_mask] = median_prior

    # Store results
    fit['phi_raw'] = phi_raw
    fit['phi_post'] = phi_post
    fit['phi_prior'] = phi_prior_full
    fit['df_residual'] = df_residual
    fit['df_prior_phi'] = sv['df_prior']
    with np.errstate(divide='ignore'):
        fit['dispersion_shrunk'] = np.where(
            phi_post > 0, 1.0 / phi_post, np.inf
        )

    return fit


def glm_sc_test(fit, coef=None, contrast=None):
    """Wald test on a ``glm_sc_fit`` result.

    Parameters
    ----------
    fit : dict
        Output from ``glm_sc_fit()``.
    coef : int, optional
        0-based column index of the coefficient to test. Default: last
        column.
    contrast : ndarray, optional
        Custom contrast vector (length p). If given, *coef* is ignored.

    Returns
    -------
    dict with key ``'table'`` containing a DataFrame with columns:
    logFC, SE, z, PValue, FDR, sigma_sample, dispersion, converged.
    """
    coefficients = fit['coefficients']
    se_arr = fit['se']
    ngenes, nb = coefficients.shape

    if contrast is not None:
        contrast = np.asarray(contrast, dtype=np.float64)
        logFC = coefficients @ contrast
        se = np.sqrt(np.maximum(
            np.sum((se_arr ** 2) * (contrast ** 2), axis=1), 0
        ))
    else:
        if coef is None:
            coef = nb - 1
        logFC = coefficients[:, coef]
        se = se_arr[:, coef]

    z = logFC / se
    pvalue = _chi2.sf(z ** 2, 1)

    # FDR (Benjamini-Hochberg)
    n = len(pvalue)
    valid = ~np.isnan(pvalue)
    fdr = np.full(n, np.nan)
    if valid.any():
        from statsmodels.stats.multitest import multipletests
        _, fdr_vals, _, _ = multipletests(pvalue[valid], method='fdr_bh')
        fdr[valid] = fdr_vals

    table = pd.DataFrame({
        'logFC': logFC,
        'SE': se,
        'z': z,
        'PValue': pvalue,
        'FDR': fdr,
        'sigma_sample': fit['sigma_sample'],
        'dispersion': fit['dispersion'],
        'converged': fit['convergence'],
    })

    if fit.get('genes') is not None:
        table.index = fit['genes']

    return {'table': table}
