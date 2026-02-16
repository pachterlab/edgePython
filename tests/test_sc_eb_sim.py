# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Simulation-based validation for single-cell EB shrinkage."""

import numpy as np

import edgepython as ep


def _simulate_phi(ngenes, df_resid, trend, rng):
    x = rng.uniform(0.0, 1.0, size=ngenes)
    if trend:
        log_phi = np.log(0.2) + 1.1 * (x - 0.5)
        phi_true = np.exp(log_phi)
    else:
        phi_true = np.full(ngenes, 0.2)

    chi2 = rng.chisquare(df_resid, size=ngenes)
    chi2 = np.maximum(chi2, 1e-12)
    phi_hat = phi_true * df_resid / chi2
    return phi_true, phi_hat, x


def _fit_from_phi_hat(phi_hat, n_cells, n_predictors, n_samples):
    dispersion = np.where(phi_hat > 0, 1.0 / phi_hat, np.inf)
    return {
        'dispersion': dispersion,
        'convergence': np.ones_like(phi_hat, dtype=np.int32),
        'ncells': int(n_cells),
        'nsamples': int(n_samples),
        'design': np.zeros((int(n_cells), int(n_predictors)), dtype=np.float64),
    }


def test_shrink_sc_disp_improves_mse():
    rng = np.random.RandomState(7)
    ngenes = 400
    df_resid = 50
    n_predictors = 2
    n_samples = 5
    n_cells = df_resid + n_predictors + (n_samples - 1)

    phi_true, phi_hat, cov = _simulate_phi(ngenes, df_resid, trend=True, rng=rng)
    fit = _fit_from_phi_hat(phi_hat, n_cells, n_predictors, n_samples)
    fit = ep.shrink_sc_disp(fit, covariate=cov, robust=True)

    mse_raw = np.mean((fit['phi_raw'] - phi_true) ** 2)
    mse_post = np.mean((fit['phi_post'] - phi_true) ** 2)

    assert mse_post < mse_raw * 0.9, (
        f"Shrinkage failed to improve MSE: post={mse_post:.6g}, raw={mse_raw:.6g}"
    )
