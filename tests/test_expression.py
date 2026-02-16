# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for expression measures: cpm, rpkm, tpm, aveLogCPM, cpmByGroup, rpkmByGroup."""

import os

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def d1_expr():
    """Part 1 data with calcNormFactors."""
    y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
    group = np.array([1, 1, 2, 2])
    d = ep.make_dgelist(counts=y, group=group)
    d = ep.calc_norm_factors(d)
    return d, y, group


# ── cpm, rpkm, aveLogCPM, cpmByGroup ────────────────────────────────

class TestExpressionMeasures:
    """cpm, rpkm, aveLogCPM, cpmByGroup."""

    def test_cpm_row3(self, d1_expr):
        d, y, group = d1_expr
        cpm_vals = ep.cpm(d)
        # R row 3: [37372.51, 108936.21, 43996.10, 28033.97]
        expected = [37372.51, 108936.21, 43996.10, 28033.97]
        assert np.allclose(cpm_vals[2], expected, rtol=0.001)

    def test_cpm_log_row3(self, d1_expr):
        d, y, group = d1_expr
        cpm_log = ep.cpm(d, log=True)
        # R row 3: [15.507, 16.833, 15.695, 15.191]
        expected = [15.507, 16.833, 15.695, 15.191]
        assert np.allclose(cpm_log[2], expected, atol=0.01)

    def test_ave_log_cpm(self, d1_expr):
        d, y, group = d1_expr
        alc = ep.ave_log_cpm(d)
        # R first 3: [13.279, 13.861, 15.895]
        assert np.allclose(alc[:3], [13.279, 13.861, 15.895], atol=0.01)

    def test_rpkm(self, d1_expr):
        d, y, group = d1_expr
        gene_lengths = np.linspace(500, 11000, 22)
        rpkm_vals = ep.rpkm(d, gene_length=gene_lengths)
        # R row 3: [24915.01, 72624.14, 29330.73, 18689.32]
        expected = [24915.01, 72624.14, 29330.73, 18689.32]
        assert np.allclose(rpkm_vals[2], expected, rtol=0.001)

    def test_cpm_by_group_row3(self, d1_expr):
        d, y, group = d1_expr
        cpmbg = ep.cpm_by_group(d)
        # R row 3: [69712.22, 35883.86]
        expected = [69712.22, 35883.86]
        assert np.allclose(cpmbg[2], expected, rtol=0.001)


# ── rpkmByGroup ──────────────────────────────────────────────────────

class TestRpkmByGroup:
    """rpkmByGroup."""

    def test_rpkm_by_group(self, d1_expr):
        d, y, group = d1_expr
        gene_lengths = np.linspace(500, 11000, 22)
        rpkmbg = ep.rpkm_by_group(d, gene_length=gene_lengths)
        # R row 3: [46474.81, 23922.57]
        assert np.allclose(rpkmbg[2], [46474.81, 23922.57], rtol=0.001)

    def test_rpkm_by_group_log(self, d1_expr):
        d, y, group = d1_expr
        gene_lengths = np.linspace(500, 11000, 22)
        rpkmbg_log = ep.rpkm_by_group(d, gene_length=gene_lengths,
                                        log=True)
        # R row 3: [15.6745, 14.8766]
        assert np.allclose(rpkmbg_log[2], [15.6745, 14.8766], atol=0.01)


# ── Prior count and logFC shrinkage ──────────────────────────────────

class TestPriorCount:
    """Prior count and logFC shrinkage."""

    @pytest.fixture(scope="class")
    def d_p37(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        return d, y, design

    def test_logfc_shrinkage_increases_with_prior(self, d_p37):
        d, y, design = d_p37
        logfc_maxabs = []
        for pc in [0.125, 0.5, 1, 5]:
            fit = ep.glm_fit(d, design=design, prior_count=pc)
            lrt = ep.glm_lrt(fit, coef=1)
            logfc_maxabs.append(
                np.max(np.abs(lrt['table']['logFC'].values)))
        # Shrinkage monotonically increasing
        assert all(logfc_maxabs[i] >= logfc_maxabs[i + 1]
                   for i in range(len(logfc_maxabs) - 1))

    def test_pvalues_unchanged_across_prior(self, d_p37):
        d, y, design = d_p37
        pvals = {}
        for pc in [0.125, 0.5, 1, 5]:
            fit = ep.glm_fit(d, design=design, prior_count=pc)
            lrt = ep.glm_lrt(fit, coef=1)
            pvals[pc] = lrt['table']['PValue'].values[:5]
        # PValues should be approximately the same
        for pc in [0.125, 0.5, 1]:
            assert np.allclose(pvals[pc], pvals[0.125], atol=0.02)

    def test_predfc_match(self, d_p37):
        d, y, design = d_p37
        norm_lib = ep.get_norm_lib_sizes(d)
        # R: predFC col2 gene2 pc=0.125: 4.006801
        pfc = ep.pred_fc(d['counts'], design=design, prior_count=0.125,
                         offset=np.log(norm_lib))
        assert abs(pfc[1, 1] - 4.006801) < 0.01


# ── TPM ─────────────────────────────────────────────────────────────

class TestTPM:
    """TPM calculation."""

    def test_tpm_basic(self):
        y = np.array([[100, 200, 150],
                       [50, 100, 75],
                       [0, 0, 0],
                       [500, 600, 550],
                       [25, 30, 28]], dtype=float)
        gene_lengths = np.array([1000, 2000, 500, 3000, 1500], dtype=float)
        tpm_vals = ep.tpm(y, effective_tx_length=gene_lengths)
        # Geometric mean of column sums should be ~1e6
        col_sums = tpm_vals.sum(axis=0)
        geo_mean = np.exp(np.mean(np.log(col_sums)))
        assert abs(geo_mean - 1e6) / 1e6 < 0.01
        # Zero gene should have zero TPM
        assert np.allclose(tpm_vals[2, :], [0, 0, 0])
        # Non-zero genes should have positive TPM
        assert np.all(tpm_vals[np.array([0, 1, 3, 4])] > 0)

    def test_tpm_dgelist(self):
        y = np.array([[100, 200, 150],
                       [50, 100, 75],
                       [500, 600, 550]], dtype=float)
        gene_lengths = np.array([1000, 2000, 3000], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2]))
        tpm_dge = ep.tpm(d, effective_tx_length=gene_lengths)
        tpm_mat = ep.tpm(y, effective_tx_length=gene_lengths)
        assert np.allclose(tpm_dge, tpm_mat, atol=1e-6)
