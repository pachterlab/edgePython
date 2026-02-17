# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""voom and voomLmFit-style functionality for edgepython.

This module ports the core limma/edgeR voom pipeline to Python:
- log-CPM transform
- optional between-array normalization
- iterative mean-variance trend fitting via weighted lowess
- precision-weighted linear model fitting
- optional sample-weight estimation
- optional block-correlation estimation and GLS fitting
- structural-zero handling for sparse rows

This is a pragmatic port intended for end-to-end usability in edgepython,
including APIs aligned with both `voom` and `voomLmFit` workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
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


def _lib_sizes(counts, lib_size=None):
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    else:
        lib_size = _as_array(lib_size)
        if lib_size.shape != (counts.shape[1],):
            raise ValueError("lib_size must have one entry per sample")
    lib_size = np.where(lib_size > 0, lib_size, 1.0)
    return lib_size


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

    return beta, fitted, resid, sigma, df


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
        return _FitResult(
            coefficients=beta.T,
            fitted_values=fitted,
            residuals=resid,
            sigma=sigma,
            df_residual=np.full(n_genes, df, dtype=np.int64),
            rank=rank,
            pivot=np.arange(p, dtype=np.int64),
            design=design,
        )

    # Fast weighted LS path for common small designs (p=1/2) without correlation/missing.
    if (
        weights is not None
        and (block is None or correlation is None or abs(float(correlation)) < 1e-12)
        and np.all(np.isfinite(y))
    ):
        x = np.asarray(design, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(x))
        if rank == x.shape[1] and x.shape[1] in (1, 2):
            w_mat = _as_matrix_weights(weights, y.shape)
            w_mat = np.clip(w_mat, _EPS, None)
            n = y.shape[1]
            p = x.shape[1]
            df = int(max(n - rank, 0))

            if p == 1:
                x0 = x[:, 0]
                a00 = np.sum(w_mat * (x0[None, :] * x0[None, :]), axis=1)
                b0 = np.sum(w_mat * (x0[None, :] * y), axis=1)
                ok = np.abs(a00) > _EPS
                beta0 = np.full(y.shape[0], np.nan, dtype=np.float64)
                beta0[ok] = b0[ok] / a00[ok]
                fitted = beta0[:, None] * x0[None, :]
                resid = y - fitted
                if df > 0:
                    sigma = np.sqrt(np.sum(w_mat * resid * resid, axis=1) / df)
                else:
                    sigma = np.full(y.shape[0], np.nan, dtype=np.float64)
                coef = beta0[:, None]
            else:
                x0 = x[:, 0]
                x1 = x[:, 1]
                a00 = np.sum(w_mat * (x0[None, :] * x0[None, :]), axis=1)
                a01 = np.sum(w_mat * (x0[None, :] * x1[None, :]), axis=1)
                a11 = np.sum(w_mat * (x1[None, :] * x1[None, :]), axis=1)
                b0 = np.sum(w_mat * (x0[None, :] * y), axis=1)
                b1 = np.sum(w_mat * (x1[None, :] * y), axis=1)
                det = a00 * a11 - a01 * a01
                ok = np.abs(det) > _EPS
                beta0 = np.full(y.shape[0], np.nan, dtype=np.float64)
                beta1 = np.full(y.shape[0], np.nan, dtype=np.float64)
                beta0[ok] = (b0[ok] * a11[ok] - b1[ok] * a01[ok]) / det[ok]
                beta1[ok] = (b1[ok] * a00[ok] - b0[ok] * a01[ok]) / det[ok]
                fitted = beta0[:, None] * x0[None, :] + beta1[:, None] * x1[None, :]
                resid = y - fitted
                if df > 0:
                    sigma = np.sqrt(np.sum(w_mat * resid * resid, axis=1) / df)
                else:
                    sigma = np.full(y.shape[0], np.nan, dtype=np.float64)
                coef = np.column_stack([beta0, beta1])

            return _FitResult(
                coefficients=coef,
                fitted_values=fitted,
                residuals=resid,
                sigma=sigma,
                df_residual=np.full(y.shape[0], df, dtype=np.int64),
                rank=rank,
                pivot=np.arange(p, dtype=np.int64),
                design=design,
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

    coef = np.empty((n_genes, p), dtype=np.float64)
    fitted = np.empty((n_genes, n_samples), dtype=np.float64)
    resid = np.empty((n_genes, n_samples), dtype=np.float64)
    sigma = np.empty(n_genes, dtype=np.float64)
    df_resid = np.empty(n_genes, dtype=np.int64)

    for g in range(n_genes):
        w_row = None if w_mat is None else w_mat[g, :]
        b, f, r, s, d = _row_lm_fit_with_missing(
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
    )


def _weighted_lowess_trend(sx, sy, span, w, weighted=False):
    ok = np.isfinite(sx) & np.isfinite(sy) & (sy > 0)
    if np.sum(ok) < 2:
        x = np.asarray([0.0, 1.0])
        y = np.asarray([1.0, 1.0])
        return x, y

    x = np.asarray(sx[ok], dtype=np.float64)
    y0 = np.asarray(sy[ok], dtype=np.float64)

    # Use the fast weighted_lowess implementation in both paths.
    # For the unweighted case, prior weights are set to 1.
    prior_w = np.clip(w[ok], _EPS, None) if weighted else np.ones_like(x)
    fit = weighted_lowess(
        x,
        y0,
        weights=prior_w,
        span=float(span),
        iterations=4 if weighted else 3,
        # Fewer seed points materially speeds voom while keeping close agreement.
        npts=200 if weighted else 120,
    )
    y = np.asarray(fit["fitted"], dtype=np.float64)
    o = np.argsort(x)
    x = x[o]
    y = y[o]

    y = np.clip(y, _EPS, None)

    # Ensure strictly non-decreasing x for interpolation.
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if xu.size < 2:
        xu = np.array([x[0] - 1.0, x[0] + 1.0], dtype=np.float64)
        yu = np.array([y[0], y[0]], dtype=np.float64)
    return xu, yu


def _interp_extrap(x, xp, fp):
    y = np.interp(x, xp, fp)
    y = np.where(x < xp[0], fp[0], y)
    y = np.where(x > xp[-1], fp[-1], y)
    return y


def _trimmed_mean(x, trim=0.15):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    k = int(np.floor(trim * x.size))
    if 2 * k >= x.size:
        return float(np.mean(x))
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
        q, r, piv = sla.qr(design, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        if diag.size:
            tol_rank = np.max(diag) * max(design.shape) * np.finfo(np.float64).eps
            rank = int(np.sum(diag > tol_rank))
            if rank < design.shape[1]:
                design = design[:, piv[:rank]]

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
    return _array_weights_reml(
        y, design, z2, prior_n=float(prior_n), maxiter=int(max_iter), tol=float(tol)
    )


def duplicate_correlation(y, design, block, weights=None, trim=0.15):
    """Estimate consensus intra-block correlation (duplicateCorrelation-style)."""
    if block is None:
        return {"consensus_correlation": np.nan, "correlation": np.array([])}

    y = _as_array(y)
    block = np.asarray(block)
    if block.shape[0] != y.shape[1]:
        raise ValueError("block must have one entry per sample")

    fit = _lm_fit(y, design, weights=weights)
    r = fit.residuals
    n_genes, n_samples = r.shape
    levels = np.unique(block)

    # Fast vectorized path when residual matrix has no missing values.
    if np.all(np.isfinite(r)):
        group_idx = [np.where(block == lv)[0] for lv in levels]
        m = np.array([idx.size for idx in group_idx], dtype=np.float64)
        present = m > 0
        m = m[present]
        group_idx = [group_idx[i] for i in range(len(group_idx)) if present[i]]
        g_used = len(group_idx)
        if g_used < 2:
            return {"consensus_correlation": np.nan, "correlation": np.full(n_genes, np.nan)}

        means = np.column_stack([np.mean(r[:, idx], axis=1) for idx in group_idx])  # genes x groups
        grand = np.mean(r, axis=1, keepdims=True)
        ss_between = np.sum(m[None, :] * (means - grand) ** 2, axis=1)

        ss_within = np.zeros(n_genes, dtype=np.float64)
        for j, idx in enumerate(group_idx):
            d = r[:, idx] - means[:, j][:, None]
            ss_within += np.sum(d * d, axis=1)

        n = float(np.sum(m))
        ms_between = ss_between / (g_used - 1.0)
        ms_within = ss_within / max(n - g_used, 1.0)

        m_sum = float(np.sum(m))
        m_sq_sum = float(np.sum(m * m))
        n0 = (m_sum - (m_sq_sum / m_sum)) / (g_used - 1.0)
        denom = ms_between + (n0 - 1.0) * ms_within

        gene_corr = np.full(n_genes, np.nan, dtype=np.float64)
        ok = denom > 0
        rho = np.empty(n_genes, dtype=np.float64)
        rho[:] = np.nan
        rho[ok] = (ms_between[ok] - ms_within[ok]) / denom[ok]
        gene_corr[ok] = np.clip(rho[ok], -0.99, 0.99)
    else:
        gene_corr = np.full(n_genes, np.nan, dtype=np.float64)
        for g in range(n_genes):
            rg = r[g, :]
            ok = np.isfinite(rg)
            if np.sum(ok) < 3:
                continue
            b = block[ok]
            v = rg[ok]
            present = np.unique(b)
            if present.size < 2:
                continue

            n = 0
            grand = np.mean(v)
            ss_within = 0.0
            ss_between = 0.0
            m_sum = 0.0
            m_sq_sum = 0.0
            groups_used = 0
            for lv in present:
                idx = np.where(b == lv)[0]
                m = idx.size
                if m == 0:
                    continue
                x = v[idx]
                mu = float(np.mean(x))
                ss_within += float(np.sum((x - mu) ** 2))
                ss_between += float(m * (mu - grand) ** 2)
                n += m
                m_sum += m
                m_sq_sum += m * m
                groups_used += 1
            if groups_used < 2 or n <= groups_used:
                continue
            ms_between = ss_between / (groups_used - 1)
            ms_within = ss_within / (n - groups_used)
            n0 = (m_sum - (m_sq_sum / m_sum)) / (groups_used - 1)
            denom = ms_between + (n0 - 1.0) * ms_within
            if denom <= 0:
                continue
            rho = (ms_between - ms_within) / denom
            gene_corr[g] = float(np.clip(rho, -0.99, 0.99))

    z = np.arctanh(np.clip(gene_corr[np.isfinite(gene_corr)], -0.99, 0.99))
    if z.size == 0:
        consensus = np.nan
    else:
        consensus = float(np.tanh(_trimmed_mean(z, trim=trim)))
    return {"consensus_correlation": consensus, "correlation": gene_corr}


def _detect_structural_zeros(counts, design, lib_size, eps=1e-4):
    h = _hat_diag(design)
    min_group_size = 1.0 / np.max(h)

    row_has_zero = np.where(np.sum(counts < eps, axis=1) > (max(2.0, min_group_size) - eps))[0]
    if row_has_zero.size == 0:
        return row_has_zero, None

    counts_zero = counts[row_has_zero, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            pois = glm_fit(
                counts_zero,
                design=design,
                lib_size=lib_size,
                dispersion=np.zeros(counts_zero.shape[0], dtype=np.float64),
                prior_count=0.0,
            )
    fitted = pois["fitted.values"]
    is_zero = (fitted < eps) & (counts_zero < eps)
    keep = np.where(np.sum(is_zero, axis=1) > eps)[0]

    if keep.size == 0:
        return np.array([], dtype=np.int64), None

    row_has_exact = row_has_zero[keep]
    is_zero = is_zero[keep, :]
    return row_has_exact, is_zero


def _compute_voom_weights_from_fit(fit, design, lib_size, trend_x, trend_y):
    fitted_values = fit.fitted_values
    fitted_cpm = np.power(2.0, fitted_values)
    fitted_count = 1e-6 * fitted_cpm * (lib_size[None, :] + 1.0)
    fitted_logcount = np.log2(np.clip(fitted_count, _EPS, None))

    trend = _interp_extrap(fitted_logcount, trend_x, trend_y)
    trend = np.clip(trend, _EPS, None)
    w = 1.0 / np.power(trend, 4.0)
    return w


def voom(
    counts,
    design=None,
    lib_size=None,
    normalize_method="none",
    span=0.5,
    adaptive_span=False,
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
):
    """Python port of limma/edgeR voomLmFit pipeline.

    Parameters broadly follow edgeR::voomLmFit.
    """
    counts = _validate_counts(counts)
    n_genes, n_samples = counts.shape

    design = _validate_design(design, n_samples)
    lib_size = _lib_sizes(counts, lib_size=lib_size)

    prior_w = _as_matrix_weights(prior_weights, counts.shape) if prior_weights is not None else None

    sample_weights_flag = bool(sample_weights) or (var_design is not None) or (var_group is not None)
    if (prior_w is not None) and sample_weights_flag:
        raise ValueError("Can't specify prior_weights and estimate sample weights")

    if adaptive_span:
        span = choose_lowess_span(n_genes, small_n=50, min_span=0.3, power=1 / 3)

    y = np.log2((counts + 0.5) / (lib_size[None, :] + 1.0) * 1e6)
    y = _normalize_between_arrays(y, method=normalize_method)

    fit = _lm_fit(y, design, weights=prior_w, block=block, correlation=correlation)

    # Structural zeros: mask exact zeros and refit those rows.
    if structural_zeros:
        row_zero, is_zero = _detect_structural_zeros(counts, design, lib_size)
    else:
        row_zero, is_zero = np.array([], dtype=np.int64), None
    y_na_short = None
    if row_zero.size > 0:
        y_na_short = y[row_zero, :].copy()
        y_na_short[is_zero] = np.nan
        w_short = None if prior_w is None else prior_w[row_zero, :]
        fit_na = _lm_fit(
            y_na_short,
            design,
            weights=w_short,
            block=block,
            correlation=correlation,
        )
        fit.df_residual[row_zero] = fit_na.df_residual
        fit.sigma[row_zero] = fit_na.sigma

    has_rep = fit.df_residual > 0
    n_rep = int(np.sum(has_rep))
    if n_rep < 2:
        w = np.ones_like(y)
        out = {
            "E": y,
            "weights": w,
            "design": design,
            "coefficients": fit.coefficients,
            "fitted": fit.fitted_values,
            "residuals": fit.residuals,
            "sigma": fit.sigma,
            "df_residual": fit.df_residual,
            "Amean": np.nanmean(y, axis=1),
            "lib_size": lib_size,
            "span": float(span),
        }
        return out

    amean = np.nanmean(y, axis=1)
    amean2 = amean.copy()
    if y_na_short is not None:
        amean2[row_zero] = np.nanmean(y_na_short, axis=1)

    sx = amean2[has_rep] + np.mean(np.log2(lib_size + 1.0)) - np.log2(1e6)
    sy = np.sqrt(np.clip(fit.sigma[has_rep], _EPS, None))

    any_zero_rows = y_na_short is not None
    trend_x, trend_y = _weighted_lowess_trend(
        sx,
        sy,
        span=float(span),
        w=np.clip(fit.df_residual[has_rep], 1, None),
        weighted=any_zero_rows,
    )

    w_voom = _compute_voom_weights_from_fit(fit, design, lib_size, trend_x, trend_y)

    if prior_w is not None:
        weights = w_voom * prior_w
    else:
        weights = w_voom

    # First sample weight / correlation pass.
    sw = None
    if sample_weights_flag:
        y_for_sw = y.copy()
        if y_na_short is not None:
            y_for_sw[row_zero, :] = y_na_short
        sw = array_weights(
            y_for_sw,
            design,
            weights,
            var_design=var_design,
            var_group=var_group,
            prior_n=float(prior_n),
        )
        if block is not None:
            weights = weights * sw[None, :]

    corr = correlation
    if block is not None and corr is None:
        y_for_corr = y.copy()
        if y_na_short is not None:
            y_for_corr[row_zero, :] = y_na_short
        dc = duplicate_correlation(y_for_corr, design, block, weights=weights)
        corr = dc["consensus_correlation"]
        if not np.isfinite(corr):
            corr = 0.0

    # Second iteration if sample/block modeling requested.
    if (block is not None) or sample_weights_flag:
        if sample_weights_flag and sw is not None:
            base_weights = np.repeat(sw[None, :], n_genes, axis=0)
        else:
            base_weights = prior_w

        fit2 = _lm_fit(y, design, weights=base_weights, block=block, correlation=corr)
        if y_na_short is not None:
            w_short = None if base_weights is None else base_weights[row_zero, :]
            fit2_na = _lm_fit(
                y_na_short,
                design,
                weights=w_short,
                block=block,
                correlation=corr,
            )
            fit2.df_residual[row_zero] = fit2_na.df_residual
            fit2.sigma[row_zero] = fit2_na.sigma

        sy2 = np.sqrt(np.clip(fit2.sigma[has_rep], _EPS, None))
        trend_x, trend_y = _weighted_lowess_trend(
            sx,
            sy2,
            span=float(span),
            w=np.clip(fit2.df_residual[has_rep], 1, None),
            weighted=any_zero_rows,
        )

        w_voom = _compute_voom_weights_from_fit(fit2, design, lib_size, trend_x, trend_y)
        if prior_w is not None:
            weights = w_voom * prior_w
        else:
            weights = w_voom

        if sample_weights_flag:
            y_for_sw = y.copy()
            if y_na_short is not None:
                y_for_sw[row_zero, :] = y_na_short
            sw = array_weights(
                y_for_sw,
                design,
                weights,
                var_design=var_design,
                var_group=var_group,
                prior_n=float(prior_n),
            )
            weights = weights * sw[None, :]

        if block is not None:
            y_for_corr = y.copy()
            if y_na_short is not None:
                y_for_corr[row_zero, :] = y_na_short
            dc2 = duplicate_correlation(y_for_corr, design, block, weights=weights)
            corr_new = dc2["consensus_correlation"]
            if np.isfinite(corr_new):
                corr = corr_new
            else:
                corr = 0.0

    # Final fit with final voom weights.
    fit_final = _lm_fit(y, design, weights=weights, block=block, correlation=corr)
    if y_na_short is not None:
        fit_final_na = _lm_fit(
            y_na_short,
            design,
            weights=weights[row_zero, :],
            block=block,
            correlation=corr,
        )
        fit_final.df_residual[row_zero] = fit_final_na.df_residual
        fit_final.sigma[row_zero] = fit_final_na.sigma

    out = {
        "coefficients": fit_final.coefficients,
        "fitted": fit_final.fitted_values,
        "residuals": fit_final.residuals,
        "sigma": fit_final.sigma,
        "df_residual": fit_final.df_residual,
        "design": design,
        "weights": weights,
        "E": y,
        "Amean": amean,
        "lib_size": lib_size,
        "span": float(span),
        "correlation": corr,
        "sample_weights": sw,
        "trend_x": trend_x,
        "trend_y": trend_y,
    }

    if save_plot:
        out["voom_xy"] = {
            "x": sx,
            "y": sy,
            "xlab": "log2( count size + 0.5 )",
            "ylab": "Sqrt( standard deviation )",
        }
        out["voom_line"] = {"x": trend_x, "y": trend_y}

    if keep_elist:
        out["EList"] = {
            "E": y,
            "weights": weights,
            "genes": None,
        }

    return out


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
    adaptive_span=False,
    save_plot=False,
    keep_elist=True,
):
    """Alias-style API matching edgeR::voomLmFit argument names."""
    return voom(
        counts=counts,
        design=design,
        lib_size=lib_size,
        normalize_method=normalize_method,
        span=span,
        adaptive_span=adaptive_span,
        block=block,
        correlation=None,
        prior_weights=prior_weights,
        sample_weights=sample_weights,
        var_design=var_design,
        var_group=var_group,
        prior_n=prior_n,
        structural_zeros=True,
        save_plot=save_plot,
        keep_elist=keep_elist,
    )


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
    """Backward-compatible wrapper around `voom`.

    This keeps the previous lightweight API while using the same core engine.
    `norm_factors`, `prior_count`, `lowess_iterations`, and `lowess_npts` are
    accepted for compatibility; the current implementation applies voom's
    canonical prior_count=0.5 and weighted_lowess defaults.
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

    out = voom(
        counts=counts,
        design=design,
        lib_size=lib_size,
        normalize_method="none",
        span=0.5 if span is None else float(span),
        adaptive_span=False,
        prior_weights=None,
        sample_weights=False,
        structural_zeros=False,
        save_plot=False,
        keep_elist=False,
    )

    # Preserve old return keys.
    out["trend_fitted"] = _interp_extrap(out["trend_x"], out["trend_x"], out["trend_y"])
    return out
