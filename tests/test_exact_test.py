# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for exact tests: exactTest, exactTestDoubleTail, exactTestBetaApprox, edge cases."""

import os

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Basic exact test ─────────────────────────────────────────────────

class TestExactTestBasic:
    """Basic exact test on 22-gene data."""

    def test_exact_test_top_gene(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        group = np.array([1, 1, 2, 2])
        lib_size = np.array([1001, 1002, 1003, 1004])
        d = ep.make_dgelist(counts=y, group=group, lib_size=lib_size)
        d = ep.estimate_common_disp(d)
        de = ep.exact_test(d)
        tt = ep.top_tags(de)['table']
        # R: Tag.17 logFC=2.045096, PValue=0.01976
        top = tt.iloc[0]
        assert abs(top['logFC'] - 2.045096) < 0.001
        assert abs(top['PValue'] - 0.01976) < 0.001


# ── Low-level exact tests ───────────────────────────────────────────

class TestExactTestsLowLevel:
    """exactTestDoubleTail, exactTestBetaApprox."""

    def test_double_tail_4samples(self):
        from edgepython.exact_test import split_into_groups_pseudo
        y = pd.read_csv(f"{CSV_DIR}/test_data_part3a.csv").values
        group = np.array([1, 1, 2, 2])
        ys = split_into_groups_pseudo(y, group, pair=[1, 2])
        pvals = ep.exact_test_double_tail(ys['y1'], ys['y2'],
                                           dispersion=2/3)
        # R: [0.1334, 0.6344, 0.7280, 0.7125, 0.3919]
        expected = [0.1334, 0.6344, 0.7280, 0.7125, 0.3919]
        assert np.allclose(pvals, expected, atol=0.001)

    def test_double_tail_7samples(self):
        from edgepython.exact_test import split_into_groups_pseudo
        y = pd.read_csv(f"{CSV_DIR}/test_data_part3b.csv").values
        group = np.array([1, 1, 2, 2, 3, 3, 3])
        ys = split_into_groups_pseudo(y, group, pair=[1, 3])
        pvals = ep.exact_test_double_tail(ys['y1'], ys['y2'],
                                           dispersion=2/3)
        # R: [1.0000, 0.4486, 1.0000, 0.9390, 0.4591]
        expected = [1.0000, 0.4486, 1.0000, 0.9390, 0.4591]
        assert np.allclose(pvals, expected, atol=0.001)

    def test_beta_approx(self):
        from edgepython.exact_test import (split_into_groups_pseudo,
                                           _exact_test_beta_approx)
        y = pd.read_csv(f"{CSV_DIR}/test_data_part3b.csv").values
        group = np.array([1, 1, 2, 2, 3, 3, 3])
        ys = split_into_groups_pseudo(y, group, pair=[1, 3])
        pvals = _exact_test_beta_approx(ys['y1'], ys['y2'],
                                         dispersion=2/3)
        # R: [1.0000, 0.4493, 1.0000, 0.9422, 0.4589]
        expected = [1.0000, 0.4493, 1.0000, 0.9422, 0.4589]
        assert np.allclose(pvals, expected, atol=0.001)


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: all-zero genes, tiny datasets, large counts, single replicate."""

    def test_all_zero_gene(self):
        y = np.array([[0, 0, 0, 0],
                       [10, 20, 30, 40],
                       [5, 5, 5, 5]], dtype=float)
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        fit = ep.glm_fit(d, design=design, dispersion=0.1)
        lrt = ep.glm_lrt(fit, coef=1)
        # R: zero_gene deviance=0, LR=0, PValue=1
        assert lrt['table']['LR'].iloc[0] == 0
        assert lrt['table']['PValue'].iloc[0] == 1.0

    def test_tiny_dataset(self):
        y = np.array([[10, 12, 20, 22],
                       [5, 7, 15, 13],
                       [3, 4, 8, 9]], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        d = ep.estimate_common_disp(d)
        # R: common.dispersion=0.000101
        assert abs(d['common.dispersion'] - 0.000101) < 0.01
        et = ep.exact_test(d)
        pv = et['table']['PValue'].values
        # R: PValues=[0.699, 1.000, 0.832]
        assert np.allclose(pv, [0.699, 1.000, 0.832], atol=0.05)

    def test_large_counts(self):
        y = np.array([[1e6, 1.2e6, 0.9e6, 1.1e6],
                       [500, 600, 400, 550]], dtype=float)
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        fit = ep.glm_fit(d, design=design, dispersion=0.1)
        lrt = ep.glm_lrt(fit, coef=1)
        # R: big_gene PValue=1.000, deviance=0
        assert lrt['table']['PValue'].iloc[0] >= 0.99
        assert fit['deviance'][0] < 0.01

    def test_single_replicate_exact(self):
        y = np.array([[10, 15], [20, 3], [5, 8]], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        et = ep.exact_test(d, dispersion=0.4)
        # R: logFC=[1.003, -2.268, 1.085], PValues=[0.594, 0.239, 0.572]
        logfc = et['table']['logFC'].values
        pv = et['table']['PValue'].values
        assert np.allclose(logfc, [1.003, -2.268, 1.085], atol=0.01)
        assert np.allclose(pv, [0.594, 0.239, 0.572], atol=0.01)

    def test_single_replicate_glm(self):
        y = np.array([[10, 15], [20, 3], [5, 8]], dtype=float)
        design = np.column_stack([np.ones(2), np.array([0, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        fit = ep.glm_fit(d, design=design, dispersion=0.1)
        lrt = ep.glm_lrt(fit, coef=1)
        # R: logFC=[1.003, -2.268, 1.085], PValues=[0.245, 0.025, 0.286]
        logfc = lrt['table']['logFC'].values
        pv = lrt['table']['PValue'].values
        assert np.allclose(logfc, [1.000, -2.272, 1.082], atol=0.05)
        assert np.allclose(pv, [0.245, 0.025, 0.286], atol=0.01)


# ── Single-replicate groups ─────────────────────────────────────────

class TestSingleReplicate:
    """Single-replicate groups."""

    def test_exact_test_single_replicate(self):
        y = np.array([[10, 20], [50, 50], [0, 5],
                       [100, 200], [30, 10]], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        d['common.dispersion'] = 0.1
        de = ep.exact_test(d, dispersion=0.1)
        pvals = de['table']['PValue'].values
        assert not np.any(np.isnan(pvals))
        assert not np.any(np.isinf(pvals))

    def test_glm_single_replicate(self):
        y = np.array([[10, 20], [50, 50], [0, 5],
                       [100, 200], [30, 10]], dtype=float)
        design = np.column_stack([np.ones(2), np.array([0, 1], dtype=float)])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        fit = ep.glm_fit(d, design=design, dispersion=0.1)
        lrt = ep.glm_lrt(fit, coef=1)
        pvals = lrt['table']['PValue'].values
        assert not np.any(np.isnan(pvals))
        assert not np.any(np.isinf(pvals))


# ── Stress test ─────────────────────────────────────────────────────

class TestStressTest:
    """All-zero and near-zero genes stress test."""

    def test_full_pipeline_extreme(self):
        y = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
            [1e6, 1e6, 1e6, 100, 100, 100],
            [10, 20, 15, 12, 18, 14],
            [10, 12, 11, 50, 55, 48],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [100, 200, 150, 100, 200, 150],
        ], dtype=float)
        design = np.column_stack([np.ones(6),
                                   np.array([0, 0, 0, 1, 1, 1], dtype=float)])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 1, 2, 2, 2]))
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        lrt = ep.glm_lrt(fit, coef=1)
        pvals = lrt['table']['PValue'].values

        # All-zero gene (gene1) PValue = 1.0
        assert pvals[0] == 1.0
        # No NaN in non-zero gene PValues
        assert not np.any(np.isnan(pvals[1:]))
        # No Inf in any PValues
        assert not np.any(np.isinf(pvals))
        # Large vs small gene should be DE
        assert pvals[4] < 0.05
