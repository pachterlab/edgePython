import numpy as np

from edgepython.voom_lmfit import (
    voom_basic,
    voom_lmfit,
    array_weights,
    duplicate_correlation,
)


def test_voom_basic_returns_positive_weights_with_expected_trend():
    rng = np.random.default_rng(7)

    n_genes = 250
    n_samples = 6

    # Two-group design
    group = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    design = np.column_stack([np.ones(n_samples), group])

    # Simulate genes spanning low-to-high expression means.
    base_mu = np.exp(rng.uniform(np.log(2), np.log(800), size=n_genes))
    fc = rng.lognormal(mean=0.0, sigma=0.15, size=n_genes)

    mu = np.repeat(base_mu[:, None], n_samples, axis=1)
    mu[:, group == 1] *= fc[:, None]

    counts = rng.poisson(mu).astype(float)

    out = voom_basic(counts, design=design)

    w = out["weights"]
    assert w.shape == counts.shape
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)

    # Higher-expression genes should generally receive higher precision.
    gene_mean = counts.mean(axis=1)
    low = w[gene_mean <= np.quantile(gene_mean, 0.25), 0]
    high = w[gene_mean >= np.quantile(gene_mean, 0.75), 0]
    assert np.median(high) > np.median(low)


def test_voom_basic_default_design_intercept_only():
    rng = np.random.default_rng(11)
    counts = rng.poisson(lam=20, size=(40, 5)).astype(float)

    out = voom_basic(counts)
    assert out["design"].shape == (5, 1)
    assert out["coefficients"].shape == (40, 1)


def test_voom_lmfit_runs_with_block_and_sample_weights():
    rng = np.random.default_rng(1234)
    n_genes = 180
    n_samples = 8

    group = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    block = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    design = np.column_stack([np.ones(n_samples), group])

    base_mu = np.exp(rng.uniform(np.log(2), np.log(500), size=n_genes))
    fc = rng.lognormal(mean=0.0, sigma=0.2, size=n_genes)

    mu = np.repeat(base_mu[:, None], n_samples, axis=1)
    mu[:, group == 1] *= fc[:, None]
    counts = rng.poisson(mu).astype(float)

    out = voom_lmfit(
        counts,
        design=design,
        block=block,
        sample_weights=True,
        keep_elist=True,
    )

    assert out["weights"].shape == counts.shape
    assert out["coefficients"].shape == (n_genes, design.shape[1])
    assert np.all(np.isfinite(out["weights"]))
    assert np.all(out["weights"] > 0)
    assert out["sample_weights"] is not None
    assert out["sample_weights"].shape == (n_samples,)
    assert "EList" in out


def test_voom_lmfit_structural_zero_branch():
    rng = np.random.default_rng(909)
    n_genes = 80
    n_samples = 6
    group = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    design = np.column_stack([np.ones(n_samples), group])

    counts = rng.poisson(lam=25, size=(n_genes, n_samples)).astype(float)
    counts[:10, group == 1] = 0.0

    out = voom_lmfit(counts, design=design)

    assert out["df_residual"].shape == (n_genes,)
    assert out["sigma"].shape == (n_genes,)
    assert np.sum(np.isfinite(out["sigma"])) >= n_genes - 2


def test_array_weights_respects_var_design():
    rng = np.random.default_rng(321)
    n_genes = 120
    n_samples = 8

    group = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    batch = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    design = np.column_stack([np.ones(n_samples), group])
    var_design = np.column_stack([np.ones(n_samples), batch])

    base_mu = np.exp(rng.uniform(np.log(3), np.log(400), size=n_genes))
    mu = np.repeat(base_mu[:, None], n_samples, axis=1)
    mu[:, group == 1] *= 1.15
    counts = rng.poisson(mu).astype(float)
    y = np.log2((counts + 0.5) / (counts.sum(axis=0, keepdims=True) + 1.0) * 1e6)

    sw = array_weights(y, design, var_design=var_design, prior_n=10)
    assert sw.shape == (n_samples,)
    assert np.all(np.isfinite(sw))
    assert np.all(sw > 0)
    assert np.std(sw) > 0


def test_duplicate_correlation_detects_positive_block_signal():
    rng = np.random.default_rng(99)
    n_genes = 150
    n_samples = 8
    block = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    design = np.ones((n_samples, 1))

    u = rng.normal(0, 1.0, size=(n_genes, len(np.unique(block))))
    eps = rng.normal(0, 0.6, size=(n_genes, n_samples))
    y = np.zeros((n_genes, n_samples), dtype=float)
    for j in range(n_samples):
        y[:, j] = u[:, block[j]] + eps[:, j]

    dc = duplicate_correlation(y, design, block=block)
    assert np.isfinite(dc["consensus_correlation"])
    assert dc["consensus_correlation"] > 0.2


def test_voom_lmfit_extended_normalization_modes():
    rng = np.random.default_rng(88)
    counts = rng.poisson(lam=np.array([5, 20, 80, 200])[None, :] * rng.uniform(0.6, 1.6, size=(100, 4))).astype(float)
    design = np.ones((4, 1))

    out_q = voom_lmfit(counts, design=design, normalize_method="quantile", keep_elist=False)
    out_c = voom_lmfit(counts, design=design, normalize_method="cyclicloess", keep_elist=False)

    assert out_q["E"].shape == counts.shape
    assert out_c["E"].shape == counts.shape
    assert np.all(np.isfinite(out_q["E"]))
    assert np.all(np.isfinite(out_c["E"]))

    # Quantile normalization should align ordered distributions closely.
    sorted_cols = np.sort(out_q["E"], axis=0)
    max_pair_diff = np.max(np.abs(sorted_cols[:, 0] - sorted_cols[:, 1]))
    assert max_pair_diff < 1e-6
