# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""voom and voomLmFit-style functionality for edgepython.

Ports of limma 3.99's voom and voomLmFit and the functions they use:
lmFit (least squares and GLS with a block correlation), arrayWeights
(gene-by-gene and REML, with or without prior weights) and
duplicateCorrelation (REML).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numba import njit, prange
from scipy import linalg as sla

from .glm_fit import glm_fit
from .limma_port import choose_lowess_span
from .weighted_lowess import weighted_lowess


_EPS = 1e-8


@dataclass
class _FitResult:
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    sigma: np.ndarray
    df_residual: np.ndarray
    rank: int
    pivot: np.ndarray
    design: np.ndarray
    stdev_unscaled: np.ndarray = None


def _as_array(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype)


def _validate_counts(counts):
    counts = _as_array(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D array shaped (genes, samples)")
    if counts.shape[0] < 2:
        raise ValueError("Need at least two genes to fit a mean-variance trend")
    if np.any(~np.isfinite(counts)):
        raise ValueError("NA/non-finite counts not allowed")
    if np.min(counts) < 0:
        raise ValueError("Negative counts not allowed")
    return counts


def _validate_design(design, n_samples):
    if design is None:
        out = np.ones((n_samples, 1), dtype=np.float64)
        return out
    design = _as_array(design)
    if design.ndim != 2:
        raise ValueError("design must be a 2D array")
    if design.shape[0] != n_samples:
        raise ValueError("nrow(design) disagrees with ncol(counts)")
    if np.any(~np.isfinite(design)):
        raise ValueError("NAs not allowed in design")
    return design


def _as_matrix_weights(weights, shape):
    if weights is None:
        return None
    w = _as_array(weights)
    if w.ndim == 0:
        return np.full(shape, float(w), dtype=np.float64)
    if w.ndim == 1:
        if w.shape[0] == shape[0]:
            return np.repeat(w[:, None], shape[1], axis=1)
        if w.shape[0] == shape[1]:
            return np.repeat(w[None, :], shape[0], axis=0)
        raise ValueError("1D weights must have length n_genes or n_samples")
    if w.ndim == 2 and w.shape == shape:
        return w.astype(np.float64, copy=False)
    raise ValueError("weights has incompatible dimensions")


def _normalize_between_arrays(y, method="none", cyclic_span=0.7, cyclic_iter=3):
    method = (method or "none").lower()
    if method == "none":
        return y

    if method == "scale":
        med = np.nanmedian(y, axis=0)
        return y - med[None, :] + np.nanmedian(med)

    if method == "quantile":
        n_genes, n_samples = y.shape
        order = np.argsort(y, axis=0)
        y_sorted = np.take_along_axis(y, order, axis=0)
        ref = np.nanmean(y_sorted, axis=1)
        out = np.empty_like(y)
        for j in range(n_samples):
            out[order[:, j], j] = ref
        return out

    if method == "cyclicloess":
        out = np.asarray(y, dtype=np.float64).copy()
        n_samples = out.shape[1]
        for _ in range(int(cyclic_iter)):
            for i in range(n_samples - 1):
                for j in range(i + 1, n_samples):
                    a = out[:, i] - out[:, j]
                    m = 0.5 * (out[:, i] + out[:, j])
                    ok = np.isfinite(a) & np.isfinite(m)
                    if np.sum(ok) < 10:
                        continue
                    fit = weighted_lowess(
                        m[ok],
                        a[ok],
                        span=float(cyclic_span),
                        iterations=2,
                        npts=200,
                    )
                    adj = np.zeros_like(a)
                    adj[ok] = fit["fitted"]
                    out[:, i] -= 0.5 * adj
                    out[:, j] += 0.5 * adj
        return out

    raise ValueError(
        "normalize_method must be one of: none, scale, quantile, cyclicloess"
    )


def _hat_diag(design):
    x = np.asarray(design, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        xtx_inv = np.linalg.pinv(x.T @ x, rcond=1e-12)
        h = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    h = np.where(np.isfinite(h), h, 1.0)
    return np.clip(h, 1e-12, None)


def _row_lm_fit_with_missing(
    y_row,
    design,
    w_row=None,
    block=None,
    correlation=None,
    design_rank=None,
    r_inv_full=None,
):
    obs = np.isfinite(y_row)
    if np.sum(obs) == 0:
        p = design.shape[1]
        return (
            np.full(p, np.nan),
            np.full(y_row.shape, np.nan),
            np.full(y_row.shape, np.nan),
            np.nan,
            0,
            np.full(p, np.nan),
        )

    y_obs = y_row[obs]
    x_obs = design[obs, :]

    if w_row is None:
        w_obs = np.ones_like(y_obs)
    else:
        w_obs = np.clip(w_row[obs], _EPS, None)

    n_obs = x_obs.shape[0]
    p = x_obs.shape[1]

    all_obs = bool(np.all(obs))

    # Build GLS precision matrix P.
    if block is None or correlation is None or abs(float(correlation)) < 1e-12:
        pmat = np.diag(w_obs)
    else:
        dhalf = np.sqrt(w_obs)
        if all_obs and (r_inv_full is not None):
            R_inv = r_inv_full
        else:
            b_obs = block[obs]
            n = n_obs
            r = float(correlation)
            r = float(np.clip(r, -0.95, 0.95))
            R = np.eye(n, dtype=np.float64)
            for b in np.unique(b_obs):
                idx = np.where(b_obs == b)[0]
                if idx.size > 1:
                    for i in idx:
                        for j in idx:
                            if i != j:
                                R[i, j] = r
            R_inv = np.linalg.pinv(R, rcond=1e-10)
        pmat = (dhalf[:, None] * R_inv) * dhalf[None, :]

    xtp = x_obs.T @ pmat
    xtpx = xtp @ x_obs
    xtpy = xtp @ y_obs

    if p == 1:
        d = float(xtpx[0, 0])
        if abs(d) > _EPS:
            beta = np.array([xtpy[0] / d], dtype=np.float64)
        else:
            beta = np.linalg.pinv(xtpx, rcond=1e-12) @ xtpy
    elif p == 2:
        a = float(xtpx[0, 0])
        b = float(xtpx[0, 1])
        c = float(xtpx[1, 1])
        det = a * c - b * b
        if abs(det) > _EPS:
            beta = np.array(
                [
                    (xtpy[0] * c - xtpy[1] * b) / det,
                    (xtpy[1] * a - xtpy[0] * b) / det,
                ],
                dtype=np.float64,
            )
        else:
            beta = np.linalg.pinv(xtpx, rcond=1e-12) @ xtpy
    else:
        beta = np.linalg.pinv(xtpx, rcond=1e-12) @ xtpy
    fitted_obs = x_obs @ beta
    resid_obs = y_obs - fitted_obs

    if all_obs and (design_rank is not None):
        rank = int(design_rank)
    else:
        rank = int(np.linalg.matrix_rank(x_obs))
    df = max(int(n_obs - rank), 0)

    if df > 0:
        rss = float(resid_obs.T @ pmat @ resid_obs)
        sigma = np.sqrt(max(rss / df, 0.0))
    else:
        sigma = np.nan

    fitted = np.full_like(y_row, np.nan, dtype=np.float64)
    resid = np.full_like(y_row, np.nan, dtype=np.float64)
    fitted[obs] = fitted_obs
    resid[obs] = resid_obs
    stdev_unscaled = np.sqrt(np.diag(np.linalg.pinv(xtpx, rcond=1e-12)))

    return beta, fitted, resid, sigma, df, stdev_unscaled


def _lm_fit(y, design, weights=None, block=None, correlation=None):
    y = _as_array(y)
    n_genes, n_samples = y.shape
    p = design.shape[1]

    # Fast OLS path (matches lmFit baseline path more closely when no weights/correlation/missing).
    if (
        weights is None
        and (block is None or correlation is None or abs(float(correlation)) < 1e-12)
        and np.all(np.isfinite(y))
    ):
        x = design
        yt = y.T  # samples x genes
        beta = np.linalg.lstsq(x, yt, rcond=None)[0]  # p x genes
        fitted = (x @ beta).T
        resid = y - fitted
        rank = int(np.linalg.matrix_rank(x))
        df = int(max(n_samples - rank, 0))
        if df > 0:
            sigma = np.sqrt(np.sum(resid * resid, axis=1) / df)
        else:
            sigma = np.full(n_genes, np.nan, dtype=np.float64)
        su = np.sqrt(np.diag(np.linalg.pinv(x.T @ x)))
        return _FitResult(
            coefficients=beta.T,
            fitted_values=fitted,
            residuals=resid,
            sigma=sigma,
            df_residual=np.full(n_genes, df, dtype=np.int64),
            rank=rank,
            pivot=np.arange(p, dtype=np.int64),
            design=design,
            stdev_unscaled=np.tile(su, (n_genes, 1)),
        )

    w_mat = _as_matrix_weights(weights, y.shape) if weights is not None else None
    design_rank = int(np.linalg.matrix_rank(design))

    r_inv_full = None
    if block is not None and correlation is not None and abs(float(correlation)) > 1e-12:
        b_full = np.asarray(block)
        n = design.shape[0]
        r = float(np.clip(float(correlation), -0.95, 0.95))
        R = np.eye(n, dtype=np.float64)
        for b in np.unique(b_full):
            idx = np.where(b_full == b)[0]
            if idx.size > 1:
                for i in idx:
                    for j in idx:
                        if i != j:
                            R[i, j] = r
        r_inv_full = np.linalg.pinv(R, rcond=1e-10)

    # Batched normal equations when every gene has all observations and the
    # design is of full rank; the per-gene loop below handles the rest.
    if design_rank == p and np.all(np.isfinite(y)):
        x = np.asarray(design, dtype=np.float64)
        w = np.ones_like(y) if w_mat is None else np.clip(w_mat, _EPS, None)
        if r_inv_full is None:
            xtpx = np.einsum("gi,ij,ik->gjk", w, x, x)
            xtpy = np.einsum("gi,ij,gi->gj", w, x, y)
        else:
            xs = np.sqrt(w)[:, :, None] * x[None, :, :]
            ys = np.sqrt(w) * y
            xr = np.einsum("gip,ij->gpj", xs, r_inv_full)
            xtpx = np.einsum("gpj,gjq->gpq", xr, xs)
            xtpy = np.einsum("gpj,gj->gp", xr, ys)
        xtpx_inv = np.linalg.inv(xtpx)
        coef = np.einsum("gjk,gk->gj", xtpx_inv, xtpy)
        fitted = coef @ x.T
        resid = y - fitted
        df = n_samples - p
        if df > 0:
            if r_inv_full is None:
                rss = np.sum(w * resid * resid, axis=1)
            else:
                rs = np.sqrt(w) * resid
                rss = np.einsum("gi,ij,gj->g", rs, r_inv_full, rs)
            sigma = np.sqrt(np.maximum(rss / df, 0.0))
        else:
            sigma = np.full(n_genes, np.nan, dtype=np.float64)
        return _FitResult(
            coefficients=coef,
            fitted_values=fitted,
            residuals=resid,
            sigma=sigma,
            df_residual=np.full(n_genes, df, dtype=np.int64),
            rank=p,
            pivot=np.arange(p, dtype=np.int64),
            design=design,
            stdev_unscaled=np.sqrt(np.einsum("gjj->gj", xtpx_inv)),
        )

    coef = np.empty((n_genes, p), dtype=np.float64)
    fitted = np.empty((n_genes, n_samples), dtype=np.float64)
    resid = np.empty((n_genes, n_samples), dtype=np.float64)
    sigma = np.empty(n_genes, dtype=np.float64)
    df_resid = np.empty(n_genes, dtype=np.int64)
    su = np.full((n_genes, p), np.nan, dtype=np.float64)

    for g in range(n_genes):
        w_row = None if w_mat is None else w_mat[g, :]
        b, f, r, s, d, su[g, :] = _row_lm_fit_with_missing(
            y[g, :],
            design,
            w_row=w_row,
            block=block,
            correlation=correlation,
            design_rank=design_rank,
            r_inv_full=r_inv_full,
        )
        coef[g, :] = b
        fitted[g, :] = f
        resid[g, :] = r
        sigma[g] = s
        df_resid[g] = d

    rank = design_rank
    pivot = np.arange(design.shape[1], dtype=np.int64)

    return _FitResult(
        coefficients=coef,
        fitted_values=fitted,
        residuals=resid,
        sigma=sigma,
        df_residual=df_resid,
        rank=rank,
        pivot=pivot,
        design=design,
        stdev_unscaled=su,
    )


def _trimmed_mean(x, trim=0.15):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if trim >= 0.5:
        return float(np.median(x))
    x = np.sort(x)
    k = int(np.floor(trim * x.size))
    return float(np.mean(x[k : x.size - k]))


def _contr_sum(n):
    if n < 2:
        return np.zeros((n, 0), dtype=np.float64)
    z = np.zeros((n, n - 1), dtype=np.float64)
    z[: n - 1, :] = np.eye(n - 1)
    z[n - 1, :] = -1.0
    return z


def _prepare_var_design(n_samples, var_design=None, var_group=None):
    if var_group is not None:
        vg = np.asarray(var_group)
        if vg.shape[0] != n_samples:
            raise ValueError("var_group has wrong length")
        levels, inv = np.unique(vg, return_inverse=True)
        if levels.size < 2:
            raise ValueError("Need at least two variance groups")
        csum = _contr_sum(levels.size)
        z2 = csum[inv, :]
    elif var_design is None:
        z2 = _contr_sum(n_samples)
    else:
        z2 = _as_array(var_design)
        if z2.ndim != 2 or z2.shape[0] != n_samples:
            raise ValueError("var_design must be a 2D array with one row per sample")
        z2 = z2 - np.mean(z2, axis=0, keepdims=True)
        # Drop near-zero columns before QR (e.g. centered intercept column).
        keep = np.where(np.nanstd(z2, axis=0) > 1e-12)[0]
        if keep.size == 0:
            z2 = np.zeros((n_samples, 0), dtype=np.float64)
        else:
            z2 = z2[:, keep]
            q, r, piv = sla.qr(z2, mode="economic", pivoting=True)
            diag = np.abs(np.diag(r))
            if diag.size == 0:
                z2 = np.zeros((n_samples, 0), dtype=np.float64)
            else:
                tol = np.max(diag) * max(z2.shape) * np.finfo(np.float64).eps
                rank = int(np.sum(diag > tol))
                if rank == 0:
                    z2 = np.zeros((n_samples, 0), dtype=np.float64)
                else:
                    z2 = z2[:, piv[:rank]]
    return np.asarray(z2, dtype=np.float64)


def _weighted_lm_single(y, x, w):
    w = np.asarray(w, dtype=np.float64)
    good = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(good) <= x.shape[1]:
        return None
    y = y[good]
    xg = x[good, :]
    wg = w[good]
    sw = np.sqrt(wg)
    xw = xg * sw[:, None]
    yw = y * sw
    p = xg.shape[1]
    if p == 1:
        x0 = xg[:, 0]
        a00 = float(np.sum(wg * x0 * x0))
        if abs(a00) <= _EPS:
            return None
        b0 = float(np.sum(wg * x0 * y))
        coef = np.array([b0 / a00], dtype=np.float64)
        xtwx_inv = np.array([[1.0 / a00]], dtype=np.float64)
        rank = 1
    elif p == 2:
        x0 = xg[:, 0]
        x1 = xg[:, 1]
        a00 = float(np.sum(wg * x0 * x0))
        a01 = float(np.sum(wg * x0 * x1))
        a11 = float(np.sum(wg * x1 * x1))
        det = a00 * a11 - a01 * a01
        if abs(det) <= _EPS:
            coef = np.linalg.lstsq(xw, yw, rcond=None)[0]
            xtwx_inv = np.linalg.pinv(xg.T @ (wg[:, None] * xg), rcond=1e-12)
            rank = int(np.linalg.matrix_rank(xw))
        else:
            b0 = float(np.sum(wg * x0 * y))
            b1 = float(np.sum(wg * x1 * y))
            coef = np.array(
                [
                    (b0 * a11 - b1 * a01) / det,
                    (b1 * a00 - b0 * a01) / det,
                ],
                dtype=np.float64,
            )
            xtwx_inv = np.array(
                [
                    [a11 / det, -a01 / det],
                    [-a01 / det, a00 / det],
                ],
                dtype=np.float64,
            )
            rank = 2
    else:
        coef = np.linalg.lstsq(xw, yw, rcond=None)[0]
        xtwx_inv = np.linalg.pinv(xg.T @ (wg[:, None] * xg), rcond=1e-12)
        rank = int(np.linalg.matrix_rank(xw))

    resid = y - xg @ coef
    h = np.sum((xg @ xtwx_inv) * xg, axis=1) * wg
    df = max(len(y) - rank, 0)
    if df > 0:
        rssw = float(np.sum((resid * resid) * wg))
        s2 = rssw / df
    else:
        s2 = np.nan
    return {
        "coef": coef,
        "resid": resid,
        "h": np.clip(h, 0.0, 1.0),
        "rank": rank,
        "s2": s2,
        "obs": good,
    }


def _array_weights_genebygene(E, design, weights, z2, prior_n=10.0):
    ngenes, narrays = E.shape
    if z2.shape[1] == 0:
        return np.ones(narrays, dtype=np.float64)
    z = np.column_stack([np.ones(narrays), z2])
    info2 = float(prior_n) * (z2.T @ z2)
    gam = np.zeros(z2.shape[1], dtype=np.float64)
    aw = np.ones(narrays, dtype=np.float64)
    p = design.shape[1]
    ngam = z2.shape[1]

    no_missing = np.all(np.isfinite(E))
    fast_small_design = no_missing and p in (1, 2) and int(np.linalg.matrix_rank(design)) == p

    if fast_small_design:
        x = design
        x0 = x[:, 0]
        x1 = x[:, 1] if p == 2 else None
        df = narrays - p
        if df < 1:
            return aw

        for i in range(ngenes):
            yi = E[i, :]
            w = aw if weights is None else aw * weights[i, :]
            w = np.clip(w, _EPS, None)

            if p == 1:
                a00 = float(np.sum(w * x0 * x0))
                if abs(a00) <= _EPS:
                    continue
                b0 = float(np.sum(w * x0 * yi))
                beta0 = b0 / a00
                resid = yi - beta0 * x0
                h = w * (x0 * x0) / a00
            else:
                a00 = float(np.sum(w * x0 * x0))
                a01 = float(np.sum(w * x0 * x1))
                a11 = float(np.sum(w * x1 * x1))
                det = a00 * a11 - a01 * a01
                if abs(det) <= _EPS:
                    # Rare numerically-degenerate case.
                    fit = _weighted_lm_single(yi, design, w)
                    if fit is None:
                        continue
                    resid = fit["resid"]
                    h = fit["h"]
                    s2 = float(fit["s2"])
                    if s2 < 1e-15:
                        continue
                    h1 = 1.0 - h
                    info = z.T @ (h1[:, None] * z)
                    if info[0, 0] <= 0:
                        continue
                    info2 = info2 + info[1:, 1:] - np.outer(info[1:, 0] / info[0, 0], info[0, 1:])
                    d = w * resid * resid
                    zvec = d / s2 - h1
                    dl = z2.T @ zvec
                    gam_step = np.linalg.pinv(info2, rcond=1e-12) @ dl
                    gam = gam + gam_step
                    aw = np.exp(-(z2 @ gam))
                    continue

                b0 = float(np.sum(w * x0 * yi))
                b1 = float(np.sum(w * x1 * yi))
                beta0 = (b0 * a11 - b1 * a01) / det
                beta1 = (b1 * a00 - b0 * a01) / det
                resid = yi - (beta0 * x0 + beta1 * x1)
                h = w * (a11 * x0 * x0 - 2.0 * a01 * x0 * x1 + a00 * x1 * x1) / det

            s2 = float(np.sum(w * resid * resid) / df)
            if s2 < 1e-15:
                continue
            h = np.clip(h, 0.0, 1.0)
            h1 = 1.0 - h
            d = w * resid * resid
            zvec = d / s2 - h1

            if ngam == 1:
                z1 = z2[:, 0]
                info00 = float(np.sum(h1))
                if info00 <= 0:
                    continue
                info01 = float(np.sum(h1 * z1))
                info11 = float(np.sum(h1 * z1 * z1))
                info2_scalar = float(info2[0, 0]) + (info11 - (info01 * info01 / info00))
                if abs(info2_scalar) <= _EPS:
                    continue
                dl = float(np.sum(z1 * zvec))
                gam_step = dl / info2_scalar
                gam[0] += gam_step
                info2[0, 0] = info2_scalar
                aw = np.exp(-(z1 * gam[0]))
            else:
                info = z.T @ (h1[:, None] * z)
                if info[0, 0] <= 0:
                    continue
                info2 = info2 + info[1:, 1:] - np.outer(info[1:, 0] / info[0, 0], info[0, 1:])
                dl = z2.T @ zvec
                gam_step = np.linalg.pinv(info2, rcond=1e-12) @ dl
                gam = gam + gam_step
                aw = np.exp(-(z2 @ gam))
        return aw

    for i in range(ngenes):
        y = E[i, :]
        w = aw if weights is None else aw * weights[i, :]
        fit = _weighted_lm_single(y, design, w)
        if fit is None:
            continue

        obs = fit["obs"]
        resid = fit["resid"]
        h_obs = fit["h"]
        s2 = float(fit["s2"])
        if s2 < 1e-15:
            continue

        d = np.zeros(narrays, dtype=np.float64)
        h1 = np.zeros(narrays, dtype=np.float64)
        d[obs] = w[obs] * resid * resid
        h1[obs] = 1.0 - h_obs

        info = z.T @ (h1[:, None] * z)
        if info[0, 0] <= 0:
            continue
        info2 = info2 + info[1:, 1:] - np.outer(info[1:, 0] / info[0, 0], info[0, 1:])

        zvec = d / s2 - h1
        dl = z2.T @ zvec
        gam_step = np.linalg.pinv(info2, rcond=1e-12) @ dl
        gam = gam + gam_step
        aw = np.exp(-(z2 @ gam))

    return aw


def _array_weights_reml(E, design, z2, prior_n=10.0, maxiter=50, tol=1e-6):
    ngenes, narrays = E.shape
    if z2.shape[1] == 0:
        return np.ones(narrays, dtype=np.float64)

    p = design.shape[1]
    p2 = p * (p + 1) // 2
    z = np.column_stack([np.ones(narrays), z2])
    ngam = z2.shape[1]

    gam = np.zeros(ngam, dtype=np.float64)
    w = np.ones(narrays, dtype=np.float64)
    conv_last = np.inf

    y_t = E.T  # narrays x ngenes
    fit0 = np.linalg.lstsq(design, y_t, rcond=None)[0]
    resid0 = y_t - design @ fit0
    q0 = np.linalg.qr(design, mode="complete")[0]
    eff0 = q0.T @ y_t
    rank0 = int(np.linalg.matrix_rank(design))
    effects = eff0[rank0:, :]
    s2 = np.mean(effects * effects, axis=0)
    ok = s2 >= 1e-15
    if np.sum(ok) < 2:
        return np.ones(narrays, dtype=np.float64)
    E = E[ok, :]
    ngenes = E.shape[0]
    y_t = E.T

    for _ in range(int(maxiter)):
        sw = np.sqrt(w)
        xw = design * sw[:, None]
        p = design.shape[1]
        n = design.shape[0]
        rank = p

        if p == 1:
            x0 = xw[:, 0]
            a00 = float(np.sum(x0 * x0))
            if abs(a00) <= _EPS:
                break
            b0 = np.sum(x0[:, None] * (y_t * sw[:, None]), axis=0)
            beta = np.array([b0 / a00], dtype=np.float64)
        elif p == 2:
            x0 = xw[:, 0]
            x1 = xw[:, 1]
            a00 = float(np.sum(x0 * x0))
            a01 = float(np.sum(x0 * x1))
            a11 = float(np.sum(x1 * x1))
            det = a00 * a11 - a01 * a01
            if abs(det) <= _EPS:
                beta = np.linalg.lstsq(xw, y_t * sw[:, None], rcond=None)[0]
                rank = int(np.linalg.matrix_rank(xw))
            else:
                b0 = np.sum(x0[:, None] * (y_t * sw[:, None]), axis=0)
                b1 = np.sum(x1[:, None] * (y_t * sw[:, None]), axis=0)
                beta0 = (b0 * a11 - b1 * a01) / det
                beta1 = (b1 * a00 - b0 * a01) / det
                beta = np.vstack([beta0, beta1])
        else:
            beta = np.linalg.lstsq(xw, y_t * sw[:, None], rcond=None)[0]
            rank = int(np.linalg.matrix_rank(xw))

        resid = y_t - design @ beta
        rssw = np.sum((resid * resid) * w[:, None], axis=0)
        df = max(n - rank, 1)
        s2 = rssw / df
        s2 = np.where(s2 > 1e-15, s2, np.nan)

        q_use = np.linalg.qr(xw, mode="reduced")[0]
        q2 = np.zeros((narrays, p2), dtype=np.float64)
        j0 = 0
        for k in range(p):
            cols = p - k
            q2[:, j0 : j0 + cols] = q_use[:, :cols] * q_use[:, k:p]
            j0 += cols
        if p > 1:
            q2[:, p:] *= np.sqrt(2.0)
        h = np.sum(q2[:, :p], axis=1)

        info = z.T @ ((1.0 - 2.0 * h)[:, None] * z) + (q2.T @ z).T @ (q2.T @ z)
        if info[0, 0] <= 0:
            break
        info2 = info[1:, 1:] - np.outer(info[1:, 0] / info[0, 0], info[0, 1:])

        zmat = (w[:, None] * resid * resid).T / s2[:, None]
        zvec = np.nanmean(zmat, axis=0) - (1.0 - h)
        info2 = ngenes * info2 + float(prior_n) * (z2.T @ z2)
        zvec = ngenes * zvec + float(prior_n) * (w - 1.0)
        dl = z2.T @ zvec
        gam_step = np.linalg.pinv(info2, rcond=1e-12) @ dl
        conv = float(dl.T @ gam_step) / ngam / (ngenes + float(prior_n))
        if not np.isfinite(conv) or conv >= conv_last:
            break
        conv_last = conv
        gam = gam + gam_step
        w = np.exp(-(z2 @ gam))
        if conv < tol:
            break

    return w


def array_weights(
    y,
    design,
    weights=None,
    var_design=None,
    var_group=None,
    prior_n=10.0,
    method="auto",
    max_iter=50,
    tol=1e-5,
):
    """Estimate sample-level precision weights (limma arrayWeights-style)."""
    y = _as_array(y)
    n_genes, n_samples = y.shape
    z2 = _prepare_var_design(n_samples, var_design=var_design, var_group=var_group)

    if n_genes < 2:
        return np.ones(n_samples, dtype=np.float64)

    if design is None:
        design = np.ones((n_samples, 1), dtype=np.float64)
    else:
        design = _as_array(design)
        if design.ndim != 2 or design.shape[0] != n_samples:
            raise ValueError("design must have one row per sample")
        # Columns that R's qr() keeps
        from .dispersion import _independent_columns
        keep = _independent_columns(design)
        if len(keep) < design.shape[1]:
            design = design[:, keep]

    if n_samples - design.shape[1] < 2:
        return np.ones(n_samples, dtype=np.float64)

    wmat = None
    has_na = np.any(~np.isfinite(y))
    if weights is not None:
        wmat = _as_matrix_weights(weights, y.shape)
        if np.any(~np.isfinite(wmat)):
            raise ValueError("NA/inf weights not allowed")
        if np.any(wmat < 0):
            raise ValueError("Negative weights not allowed")
        if np.any(wmat == 0):
            y = y.copy()
            y[wmat == 0] = np.nan
            wmat = wmat.copy()
            wmat[wmat == 0] = 1.0
            has_na = True

    method = (method or "auto").lower()
    if method not in {"auto", "genebygene", "reml"}:
        raise ValueError("method must be one of: auto, genebygene, reml")
    if method == "auto":
        method = "genebygene" if (has_na or wmat is not None) else "reml"

    if method == "genebygene":
        return _array_weights_genebygene(y, design, wmat, z2, prior_n=float(prior_n))
    if has_na:
        # As limma: REML uses only the rows without missing values
        ok = np.all(np.isfinite(y), axis=1)
        y = y[ok]
        if wmat is not None:
            wmat = wmat[ok]
        if y.shape[0] < 2:
            return np.ones(n_samples, dtype=np.float64)
    if wmat is None:
        return _array_weights_reml(
            y, design, z2, prior_n=float(prior_n), maxiter=int(max_iter), tol=float(tol)
        )
    return _array_weights_prwts_reml(y, design, wmat, z2, prior_n=float(prior_n),
                                     maxiter=int(max_iter), tol=float(tol))


@njit(cache=True)
def _awreml_gene(yg, design, wg, Z, info2, z):
    """Add one gene's REML information and score to info2 and z."""
    narrays, p = design.shape
    ngam = Z.shape[1] - 1
    sw = np.sqrt(wg)
    Q = np.zeros((narrays, p))
    for k in range(p):
        v = design[:, k] * sw
        for _ in range(2):
            for q in range(k):
                v = v - np.sum(Q[:, q] * v) * Q[:, q]
        Q[:, k] = v / np.sqrt(np.sum(v * v))
    yw = yg * sw
    rw = yw.copy()
    for k in range(p):
        rw = rw - np.sum(Q[:, k] * yw) * Q[:, k]
    s2 = np.sum(rw * rw) / (narrays - p)
    # Pairwise products of the columns of Q, off-diagonal ones times sqrt(2)
    p2 = p * (p + 1) // 2
    Q2 = np.empty((narrays, p2))
    j0 = 0
    for k in range(p):
        for i in range(p - k):
            Q2[:, j0 + i] = Q[:, i] * Q[:, i + k] * (np.sqrt(2.0) if k > 0 else 1.0)
        j0 += p - k
    h = np.zeros(narrays)
    for i in range(p):
        h = h + Q2[:, i]
    info = (Z * (1.0 - 2.0 * h).reshape(-1, 1)).T @ Z
    QZ = Q2.T @ Z
    info = info + QZ.T @ QZ
    for a in range(ngam):
        for b in range(ngam):
            info2[a, b] += info[a + 1, b + 1] - info[a + 1, 0] / info[0, 0] * info[0, b + 1]
    if s2 > 1e-15:
        for j in range(narrays):
            z[j] += rw[j] * rw[j] / s2 - (1.0 - h[j])


@njit(cache=True, parallel=True)
def _awreml_genes(y, design, weights, w, Z):
    """Per-gene REML information and score sums for array weights with
    prior weights (limma's awremlfit loop over genes)."""
    ngenes, narrays = y.shape
    ngam = Z.shape[1] - 1
    info2_g = np.zeros((ngenes, ngam, ngam))
    z_g = np.zeros((ngenes, narrays))
    for g in prange(ngenes):
        _awreml_gene(y[g], design, w * weights[g], Z, info2_g[g], z_g[g])
    return info2_g.sum(axis=0), z_g.sum(axis=0)


def _array_weights_prwts_reml(y, design, weights, z2, prior_n=10.0, maxiter=50, tol=1e-5):
    """REML array weights given prior observation weights.

    Port of limma's .arrayWeightsPrWtsREML: Fisher scoring for log-linear
    array variances, w = exp(-Z2 gam), with a prior of prior_n arrays.
    """
    ngenes, narrays = y.shape
    Z = np.column_stack([np.ones(narrays), z2])
    ngam = z2.shape[1]
    gam = np.zeros(ngam)
    w = np.ones(narrays)
    for it in range(1, int(maxiter) + 1):
        info2, z = _awreml_genes(np.ascontiguousarray(y), np.ascontiguousarray(design),
                                 np.ascontiguousarray(weights), w, np.ascontiguousarray(Z))
        info2 = (prior_n * z2.T @ z2 + info2) / (ngenes + prior_n)
        z = (prior_n * (w - 1) + z) / (ngenes + prior_n)
        dl = z2.T @ z
        gamstep = np.linalg.solve(info2, dl)
        gam = gam + gamstep
        w = np.exp(z2 @ (-gam))
        convcrit = float(dl @ gamstep) / (ngenes + prior_n) / ngam
        if not np.isfinite(convcrit):
            warnings.warn("convergence tolerance not achievable, stopping prematurely")
            break
        if convcrit < tol:
            break
        if it == maxiter:
            warnings.warn("iteration limit reached")
    return w


@njit(cache=True)
def _gammadev(y, mu):
    dev = 0.0
    for i in range(y.shape[0]):
        if mu[i] < 0.0 or not np.isfinite(mu[i]):
            return np.inf
        if y[i] < 1e-15 and mu[i] < 1e-15:
            continue
        if mu[i] <= 0.0:
            return np.inf
        dev += (y[i] - mu[i]) / mu[i] - np.log(y[i] / mu[i])
    return 2.0 * dev


@njit(cache=True)
def _gammafit(d, y, beta0, beta1):
    """Identity-link gamma GLM, mean beta0 + beta1 d, by damped IRLS (limma's
    gammafit in dupcor.c). Returns (beta0, beta1, ok)."""
    n = d.shape[0]
    if np.max(y) == 0.0:
        return 0.0, 0.0, True
    mu = beta0 + beta1 * d
    dev = _gammadev(y, mu)
    lam = 0.0
    for it in range(1, 22):
        b0_old, b1_old, dev_old = beta0, beta1, dev
        maxmu = np.max(mu)
        maxv = np.max(mu * mu)
        floor_v = maxv / 1000.0
        a = b = c = s0 = s1 = 0.0
        for i in range(n):
            v = max(mu[i] * mu[i], floor_v)
            if v <= 0.0 or not np.isfinite(v):
                return beta0, beta1, False
            iv = 1.0 / v
            r = y[i] - mu[i]
            a += iv
            b += d[i] * iv
            c += d[i] * d[i] * iv
            s0 += r * iv
            s1 += d[i] * r * iv
        maxinfo = max(a, c)
        if it == 1:
            lam = abs((a + c) / 2.0) / 2.0
        lev = 0
        db0 = db1 = 0.0
        while True:
            lev += 1
            aa = a + lam
            cc = c + lam
            det = aa * cc - b * b
            if det <= 0.0 or not np.isfinite(det):
                return beta0, beta1, False
            db0 = (cc * s0 - b * s1) / det
            db1 = (aa * s1 - b * s0) / det
            beta0 = b0_old + db0
            beta1 = b1_old + db1
            mu = beta0 + beta1 * d
            dev = _gammadev(y, mu)
            if dev <= dev_old or dev / maxmu < 1e-15:
                break
            if lam / maxinfo > 1e15:
                return b0_old, b1_old, True
            lam *= 2.0
        if lam / maxinfo > 1e15:
            break
        if lev == 1:
            lam /= 10.0
        if s0 * db0 + s1 * db1 < 1e-6 or dev / maxmu < 1e-15:
            break
    return beta0, beta1, np.isfinite(beta0) and np.isfinite(beta1)


@njit(cache=True)
def _dupcor_gene(yv, wv, design, block, nlev):
    """REML intra-block correlation for one gene (limma's dupcorgene, which
    reproduces statmod::mixedModel2Fit). NaN if not estimable."""
    narrays, nbeta = design.shape
    bmap = np.full(nlev, -1)
    obs = np.empty(narrays, dtype=np.int64)
    nobs = 0
    nblocks = 0
    for j in range(narrays):
        if np.isfinite(yv[j]) and np.isfinite(wv[j]):
            if bmap[block[j]] < 0:
                bmap[block[j]] = nblocks
                nblocks += 1
            obs[nobs] = j
            nobs += 1
    if nobs <= nbeta + 2 or nblocks <= 1 or nblocks >= nobs - 1:
        return np.nan
    sc = np.empty(nobs)
    X = np.empty((nobs, nbeta))
    Z = np.zeros((nobs, nblocks))
    yw = np.empty(nobs)
    for i in range(nobs):
        j = obs[i]
        sc[i] = np.sqrt(wv[j])
        for k in range(nbeta):
            X[i, k] = design[j, k] * sc[i]
        Z[i, bmap[block[j]]] = 1.0
        yw[i] = yv[j] * sc[i]
    # Rank of X by qr()'s rule (a column is dropped if its residual norm is
    # below 1e-7 of its norm), then an orthonormal basis of its complement
    basis = np.zeros((nobs, nbeta))
    rank = 0
    for k in range(nbeta):
        v = X[:, k].copy()
        nv0 = np.sqrt(np.sum(v * v))
        for _ in range(2):
            for q in range(rank):
                v -= np.sum(basis[:, q] * v) * basis[:, q]
        nv = np.sqrt(np.sum(v * v))
        if nv0 > 0.0 and nv >= 1e-7 * nv0:
            basis[:, rank] = v / nv
            rank += 1
    mq = nobs - rank
    if mq == 0:
        return np.nan
    if rank > 0:
        u, sv, vt = np.linalg.svd(basis[:, :rank].copy(), full_matrices=True)
        Q2 = u[:, rank:].copy()
    else:
        Q2 = np.eye(nobs)
    A = Q2.T @ Z
    g = Q2.T @ yw
    ua, sa, vta = np.linalg.svd(A, full_matrices=True)
    s = min(mq, nblocks)
    r = ua.T @ g
    dd = np.zeros(mq)
    for i in range(s):
        dd[i] = sa[i] * sa[i]
    yy = r * r
    # Moment start (fitlm2)
    sd = np.sum(dd)
    sy = np.sum(yy)
    sdd = np.sum(dd * dd)
    sdy = np.sum(dd * yy)
    det = mq * sdd - sd * sd
    if det <= 2.220446049250313e-16 * max(1.0, mq * sdd):
        return np.nan
    beta0 = (sy * sdd - sd * sdy) / det
    beta1 = (mq * sdy - sd * sy) / det
    nonzero_d = 0
    for i in range(mq):
        if abs(dd[i]) > 1e-15:
            nonzero_d += 1
    if mq > 2 and nonzero_d > 1 and (sdd - sd * sd / mq) / (mq - 1) > 1e-15:
        if np.min(beta0 + beta1 * dd) < 0.0:
            beta0 = sy / mq
            beta1 = 0.0
        beta0, beta1, ok = _gammafit(dd, yy, beta0, beta1)
        if not ok:
            return np.nan
    return beta1 / (beta0 + beta1)


@njit(cache=True, parallel=True)
def _dupcor_all(M, W, design, block, nlev):
    ngenes = M.shape[0]
    rho = np.empty(ngenes)
    for g in prange(ngenes):
        rho[g] = _dupcor_gene(M[g], W[g], design, block, nlev)
    return rho


def duplicate_correlation(y, design, block, weights=None, trim=0.15):
    """Consensus intra-block correlation, as limma's duplicateCorrelation.

    Each gene's correlation is the REML estimate of a block random-effect
    model (statmod::mixedModel2Fit, as limma's C code computes it). They are
    limited to [1 / (1 - largest block size) + 0.01, 0.99] and averaged on
    the atanh scale with a trimmed mean. Returns 'consensus_correlation'
    (also 'cor'), 'atanh_correlations' and the genewise 'correlation'.
    """
    if block is None:
        return {"consensus_correlation": np.nan, "correlation": np.array([])}
    M = _as_array(y).copy()
    if M.ndim == 1:
        M = M.reshape(1, -1)
    ngenes, narrays = M.shape
    design = np.ones((narrays, 1)) if design is None else np.asarray(design, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.shape[0] != narrays:
        raise ValueError("Number of rows of design matrix does not match number of arrays")
    block = np.asarray(block)
    if block.shape[0] != narrays:
        raise ValueError("block must have one entry per sample")
    zero = {"consensus_correlation": 0.0, "cor": 0.0, "atanh_correlations": np.zeros(ngenes),
            "correlation": np.zeros(ngenes)}

    levels, block_idx = np.unique(block, return_inverse=True)
    max_block_size = np.max(np.bincount(block_idx))
    if max_block_size == 1:
        warnings.warn("Blocks all of size 1: setting intrablock correlation to zero.")
        return zero
    # Block factor already encoded in the design?
    zb = (block_idx[:, None] == np.arange(1, len(levels))[None, :]).astype(np.float64)
    q, r = np.linalg.qr(design, mode='complete')
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-7 * np.max(np.abs(np.diag(r))))) if design.size else 0
    if zb.shape[1] and np.max(np.abs((q.T @ zb)[rank:])) < 1e-8:
        warnings.warn("Block factor already encoded in the design matrix: setting intrablock correlation to zero.")
        return zero

    if weights is None:
        W = np.ones_like(M)
    else:
        W = _as_matrix_weights(weights, M.shape).astype(np.float64).copy()
        W[W <= 0] = np.nan
        M[~np.isfinite(W)] = np.nan

    rho = _dupcor_all(np.ascontiguousarray(M), np.ascontiguousarray(W), np.ascontiguousarray(design),
                      block_idx.astype(np.int64), len(levels))
    rhomin = 1.0 / (1.0 - max_block_size) + 0.01
    with np.errstate(invalid='ignore'):
        rho = np.where(rho < rhomin, rhomin, rho)
        rho = np.where(rho > 0.99, 0.99, rho)
    arho = np.arctanh(rho)
    mrho = float(np.tanh(_trimmed_mean(arho, trim=trim)))
    return {"consensus_correlation": mrho, "cor": mrho, "atanh_correlations": arho,
            "correlation": rho}


def _voom_inputs(counts, design, lib_size, offset, offset_prior):
    """Counts, design, library sizes, offset and offset.prior from a DGEList
    or a matrix, as voom and voomLmFit take them."""
    if isinstance(counts, dict) and 'counts' in counts:
        dge = counts
        samples = dge['samples']
        if design is None and 'group' in samples.columns and len(np.unique(samples['group'])) > 1:
            from .utils import _model_matrix_group
            design = _model_matrix_group(np.asarray(samples['group']))
        if lib_size is None:
            nf = samples['norm.factors'].to_numpy() if 'norm.factors' in samples.columns else 1.0
            lib_size = samples['lib.size'].to_numpy() * nf
        if offset is None:
            offset = dge.get('offset')
        if offset_prior is None:
            offset_prior = dge.get('offset.prior')
        counts = dge['counts']
    counts = _validate_counts(counts)
    design = _validate_design(design, counts.shape[1])
    lib_size = counts.sum(axis=0) if lib_size is None else _as_array(lib_size)
    if offset is not None:
        offset = offset.as_matrix() if hasattr(offset, 'as_matrix') else _as_array(offset)
    if offset_prior is not None:
        offset_prior = _as_array(offset_prior)
        if offset_prior.shape != counts.shape:
            raise ValueError("counts and offset_prior must have equal dimensions.")
    return counts, design, lib_size, offset, offset_prior


def _voom_line(sx, sy, span, weights=None):
    """The mean-variance trend as voom computes it, sorted by x: R's
    lowess(sx, sy, f = span), or limma's weightedLowess with weights."""
    if weights is None:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        line = lowess(sy, sx, frac=span, it=3, delta=0.01 * np.ptp(sx), is_sorted=False,
                      return_sorted=True)
        return line[:, 0], line[:, 1]
    fitted = weighted_lowess(sx, sy, weights=weights, span=span)["fitted"]
    o = np.argsort(sx, kind="stable")
    return sx[o], fitted[o]


def _voom_weights(fit, design, lib_matrix, line):
    """1 / trend(fitted log-count)^4, the trend interpolated by
    approxfun(rule = 2, ties = mean)."""
    from .dispersion_lowlevel import _approx_rule2
    cols = fit.pivot[: fit.rank]
    fitted = fit.coefficients[:, cols] @ design[:, cols].T
    fitted_count = 1e-6 * np.power(2.0, fitted) * (lib_matrix + 1.0)
    return 1.0 / _approx_rule2(line[0], line[1], np.log2(fitted_count)) ** 4


def voom(
    counts,
    design=None,
    lib_size=None,
    normalize_method="none",
    span=0.5,
    adaptive_span=True,
    block=None,
    correlation=None,
    prior_weights=None,
    sample_weights=False,
    var_design=None,
    var_group=None,
    prior_n=10.0,
    structural_zeros=False,
    save_plot=False,
    keep_elist=True,
    offset=None,
    offset_prior=None,
    weights=None,
):
    """Transform counts to log2-CPM with precision weights (limma's voom).

    Port of limma 3.99's voom. The mean-variance trend is R's lowess of
    sqrt(residual SD) against average log count over genes that are not all
    zero; each observation's weight is 1 / trend(fitted log count)^4.

    Parameters follow limma's voom; prior_weights is a synonym for weights,
    which are used in the linear model fit only. The library sizes are
    exp(log(lib_size) + offset_prior); an offset is converted to an
    offset_prior by centring each gene. With sample_weights, var_design,
    var_group or structural_zeros, voom_lmfit's algorithm is used instead.

    Returns a dict with E (log2-CPM), weights, design, lib_size, span,
    Amean, the initial linear model fit (coefficients, sigma, df_residual,
    ...) and the trend (trend_x, trend_y).
    """
    if weights is None:
        weights = prior_weights
    if sample_weights or var_design is not None or var_group is not None or structural_zeros:
        return voom_lmfit(counts, design=design, block=block, prior_weights=weights,
                          sample_weights=sample_weights, var_design=var_design,
                          var_group=var_group, prior_n=prior_n, lib_size=lib_size,
                          offset=offset, offset_prior=offset_prior,
                          normalize_method=normalize_method, span=span,
                          adaptive_span=adaptive_span, save_plot=save_plot,
                          keep_elist=keep_elist)

    counts, design, lib_size, offset, offset_prior = _voom_inputs(
        counts, design, lib_size, offset, offset_prior)
    n_genes, n_samples = counts.shape
    lib_matrix = np.tile(lib_size, (n_genes, 1))
    if offset is not None:
        if offset_prior is None:
            if offset.shape != counts.shape:
                raise ValueError("counts and offset must have equal dimensions.")
            offset_prior = offset - offset.mean(axis=1, keepdims=True)
        else:
            warnings.warn("Ignoring offset in favor of offset_prior. Should not set both.")
    if offset_prior is not None:
        lib_matrix = np.exp(np.log(lib_matrix) + offset_prior)

    if adaptive_span:
        span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3)

    y = np.log2((counts + 0.5) / (lib_matrix + 1.0) * 1e6)
    y = _normalize_between_arrays(y, method=normalize_method)
    prior_w = _as_matrix_weights(weights, counts.shape) if weights is not None else None
    fit = _lm_fit(y, design, weights=prior_w, block=block, correlation=correlation)
    amean = np.mean(y, axis=1)
    out = {
        "E": y, "design": design, "lib_size": lib_size, "span": float(span), "Amean": amean,
        "coefficients": fit.coefficients, "stdev_unscaled": fit.stdev_unscaled,
        "fitted": fit.fitted_values, "residuals": fit.residuals, "sigma": fit.sigma,
        "df_residual": fit.df_residual, "offset_prior": offset_prior,
        # Kept from earlier edgePython versions; voom_lmfit sets sample_weights
        "correlation": correlation, "sample_weights": None,
    }

    n_with_reps = int(np.sum(fit.df_residual > 0))
    if n_with_reps < 2:
        warnings.warn("The experimental design has no replication. Setting weights to 1."
                      if n_with_reps == 0 else
                      "Only one gene with any replication. Setting weights to 1.")
        out["weights"] = np.ones_like(y)
        return out

    sx = amean + np.mean(np.log2(lib_size + 1.0)) - np.log2(1e6)
    sy = np.sqrt(fit.sigma)
    keep = counts.sum(axis=1) != 0
    line = _voom_line(sx[keep], sy[keep], span)
    out["weights"] = _voom_weights(fit, design, lib_matrix, line)
    out["trend_x"], out["trend_y"] = line
    if save_plot:
        out["voom_xy"] = {"x": sx[keep], "y": sy[keep], "xlab": "log2( count size + 0.5 )",
                          "ylab": "Sqrt( standard deviation )"}
        out["voom_line"] = {"x": line[0], "y": line[1]}
    if keep_elist:
        out["EList"] = {"E": y, "weights": out["weights"], "genes": None}
    return out


def _structural_zeros(counts, design, lib_size, eps=1e-4):
    """Rows with exact zeros that the design fits as zero, as voomLmFit
    finds them: rows with more zeros than the smallest group size (at least
    2) and a Poisson fit below eps where the count is below eps.
    Returns (row indices, is_zero matrix for those rows)."""
    min_group_size = 1.0 / np.max(_hat_diag(design))
    rows = np.flatnonzero(np.sum(counts < eps, axis=1) > (max(2.0, min_group_size) - eps))
    if rows.size == 0:
        return rows, None
    counts_zero = counts[rows]
    fitted = glm_fit(counts_zero, design=design, offset=np.log(lib_size), dispersion=0.0,
                     prior_count=0)["fitted.values"]
    is_zero = (fitted < eps) & (counts_zero < eps)
    exact = np.flatnonzero(np.sum(is_zero, axis=1) > eps)
    return rows[exact], is_zero[exact]


def _fit_with_zeros(y, design, weights, block, correlation, rows, y_na_short):
    """lmFit on y, with df.residual and sigma of the structural-zero rows
    from a fit that treats their zeros as missing."""
    fit = _lm_fit(y, design, weights=weights, block=block, correlation=correlation)
    if rows.size:
        fit_na = _lm_fit(y_na_short, design, weights=None if weights is None else weights[rows],
                         block=block, correlation=correlation)
        fit.df_residual[rows] = fit_na.df_residual
        fit.sigma[rows] = fit_na.sigma
    return fit


def voom_lmfit(
    counts,
    design=None,
    block=None,
    prior_weights=None,
    sample_weights=False,
    var_design=None,
    var_group=None,
    prior_n=10.0,
    lib_size=None,
    normalize_method="none",
    span=0.5,
    adaptive_span=True,
    save_plot=False,
    keep_elist=True,
    offset=None,
    offset_prior=None,
):
    """voom followed by a linear model fit (limma's voomLmFit).

    Port of limma 3.99's voomLmFit. Exact zeros that the design fits as
    zero are treated as missing when estimating each gene's residual SD
    (and the trend then uses weightedLowess with the residual df as
    weights). With sample_weights (or var_design / var_group) array weights
    are estimated, and with block an intra-block correlation; both are
    estimated twice, with the voom weights updated in between.

    Returns a dict with the final fit (coefficients, stdev_unscaled, sigma,
    df_residual, Amean), the weights, E, sample_weights, correlation and
    the trend.
    """
    counts, design, lib_size, offset, offset_prior = _voom_inputs(
        counts, design, lib_size, offset, offset_prior)
    G, n = counts.shape
    use_block = block is not None
    use_sw = bool(sample_weights) or var_design is not None or var_group is not None
    if prior_weights is not None and use_sw:
        raise ValueError("Can't specify prior_weights and estimate sample weights")
    if offset is None:
        lib_matrix = np.tile(lib_size, (G, 1))
        if offset_prior is not None:
            lib_matrix = np.exp(np.log(lib_matrix) + offset_prior)
    else:
        if offset.ndim == 2 and offset.shape != counts.shape:
            raise ValueError("counts and offset must have equal dimensions")
        if offset.ndim == 1 and offset.shape[0] != n:
            raise ValueError("if offset is a vector, its length must be the number of samples")
        lib_matrix = np.exp(np.broadcast_to(offset, counts.shape))
    prior_w = _as_matrix_weights(prior_weights, counts.shape) if prior_weights is not None else None
    if adaptive_span:
        span = choose_lowess_span(G, small_n=50, min_span=0.3, power=1 / 3)

    y = np.log2((counts + 0.5) / (lib_matrix + 1.0) * 1e6)
    y = _normalize_between_arrays(y, method=normalize_method)

    rows, is_zero = _structural_zeros(counts, design, lib_size)
    y_na_short = y_na_full = None
    if rows.size:
        y_na_short = y[rows].copy()
        y_na_short[is_zero] = np.nan
        y_na_full = y.copy()
        y_na_full[rows] = y_na_short
    any_zero_rows = rows.size > 0
    fit = _fit_with_zeros(y, design, prior_w, None, None, rows, y_na_short)

    has_rep = fit.df_residual > 0
    if int(np.sum(has_rep)) < 2:
        warnings.warn("The experimental design has no replication. Setting weights to 1."
                      if not np.any(has_rep) else
                      "Only one gene with any replication. Setting weights to 1.")
        return {"coefficients": fit.coefficients, "stdev_unscaled": fit.stdev_unscaled,
                "sigma": fit.sigma, "df_residual": fit.df_residual, "design": design,
                "weights": np.ones_like(y), "E": y, "Amean": np.mean(y, axis=1),
                "lib_size": lib_size, "span": float(span), "correlation": None,
                "sample_weights": None}

    amean = np.mean(y, axis=1)
    amean2 = amean.copy()
    if any_zero_rows:
        amean2[rows] = np.nanmean(y_na_short, axis=1)
    sx = amean2[has_rep] + np.mean(np.log2(lib_size + 1.0)) - np.log2(1e6)

    def trend_weights(fit):
        sy = np.sqrt(fit.sigma[has_rep])
        line = _voom_line(sx, sy, span,
                          weights=fit.df_residual[has_rep].astype(np.float64) if any_zero_rows else None)
        w = _voom_weights(fit, design, lib_matrix, line)
        return (w * prior_w if prior_w is not None else w), line, sy

    weights, line, sy = trend_weights(fit)
    y_for_estimates = y_na_full if any_zero_rows else y
    aw_method = "genebygene" if any_zero_rows else "reml"
    sw = None
    correlation = None
    if use_sw:
        sw = array_weights(y_for_estimates, design, weights=weights, var_design=var_design,
                           var_group=var_group, prior_n=prior_n, method=aw_method)
        if use_block:
            weights = weights * sw[None, :]
    if use_block:
        correlation = duplicate_correlation(y_for_estimates, design, block,
                                            weights=weights)["consensus_correlation"]
        if not np.isfinite(correlation):
            warnings.warn("Intra-block correlation not estimable, setting to zero.")
            correlation = 0.0
        if correlation == 0.0:
            use_block = False

    if use_block or use_sw:
        base = np.tile(sw, (G, 1)) if use_sw else prior_w
        fit = _fit_with_zeros(y, design, base, block if use_block else None, correlation,
                              rows, y_na_short)
        weights, line, sy = trend_weights(fit)
        if use_sw:
            sw = array_weights(y_for_estimates, design, weights=weights, var_design=var_design,
                               var_group=var_group, prior_n=prior_n, method=aw_method)
            weights = weights * sw[None, :]
        if use_block:
            correlation = duplicate_correlation(y_for_estimates, design, block,
                                                weights=weights)["consensus_correlation"]
            if not np.isfinite(correlation):
                warnings.warn("Intra-block correlation not estimable, setting to zero.")
                correlation = 0.0

    final_block = block if use_block else None
    fit = _fit_with_zeros(y, design, weights, final_block, correlation, rows, y_na_short)
    out = {
        "coefficients": fit.coefficients, "stdev_unscaled": fit.stdev_unscaled,
        "fitted": fit.fitted_values, "residuals": fit.residuals, "sigma": fit.sigma,
        "df_residual": fit.df_residual, "design": design, "weights": weights, "E": y,
        "Amean": amean, "lib_size": lib_size, "span": float(span),
        "correlation": correlation, "sample_weights": sw,
        "trend_x": line[0], "trend_y": line[1],
    }
    if save_plot:
        out["voom_xy"] = {"x": sx, "y": sy, "xlab": "log2( count size + 0.5 )",
                          "ylab": "Sqrt( standard deviation )"}
        out["voom_line"] = {"x": line[0], "y": line[1]}
    if keep_elist:
        out["EList"] = {"E": y, "weights": weights, "genes": None}
    return out


def voom_basic(
    counts,
    design=None,
    lib_size=None,
    norm_factors=None,
    prior_count=0.5,
    span=None,
    lowess_iterations=4,
    lowess_npts=200,
):
    """voom with a fixed span (0.5 by default) and optional norm factors.

    Backward-compatible wrapper around `voom`; prior_count,
    lowess_iterations and lowess_npts are accepted for compatibility and
    ignored (voom uses 0.5 and R's lowess).
    """
    counts = _validate_counts(counts)
    if norm_factors is not None:
        norm_factors = _as_array(norm_factors)
        if norm_factors.shape != (counts.shape[1],):
            raise ValueError("norm_factors must have one entry per sample")
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    if norm_factors is not None:
        lib_size = lib_size * norm_factors
    if prior_count <= 0:
        raise ValueError("prior_count must be positive")
    out = voom(counts, design=design, lib_size=lib_size, span=0.5 if span is None else float(span),
               adaptive_span=False, keep_elist=False)
    out["trend_fitted"] = out["trend_y"]
    return out
