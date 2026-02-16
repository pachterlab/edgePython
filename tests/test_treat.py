# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for glmTreat function with various thresholds and options."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Fixtures (from conftest) ─────────────────────────────────────────
# small_counts, group6, design6, dgelist, dgelist_disp are from conftest.py


@pytest.fixture
def ql_fit(dgelist_disp, design6):
    """Quasi-likelihood GLM fit."""
    return ep.glm_ql_fit(dgelist_disp, design6)


@pytest.fixture
def lrt_fit(dgelist_disp, design6):
    """Standard GLM fit (for LRT-based treat)."""
    return ep.glm_fit(dgelist_disp, design=design6)


# ── QL-based Treat (Part 15 data) ───────────────────────────────────

class TestGlmTreatPart15:
    """glmTreat on 100-gene x 3-sample data (Part 15)."""

    @pytest.fixture(scope="class")
    def d4_treat(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        group = np.array([1, 2, 2])
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=group)
        d = ep.estimate_disp(d, design)
        return d, y, design

    def test_ql_treat_lfc12_top_genes(self, d4_treat):
        d, y, design = d4_treat
        fit = ep.glm_ql_fit(d, design)
        tr = ep.glm_treat(fit, coef=1, lfc=np.log2(1.2))
        tt = ep.top_tags(tr)['table']
        # R: gene5 logFC=-7.394, PValue=0.00270
        top = tt.iloc[0]
        assert abs(top['logFC'] - (-7.394)) < 0.1
        assert abs(top['PValue'] - 0.00270) < 0.005

    def test_ql_treat_decide_tests_lfc12(self, d4_treat):
        d, y, design = d4_treat
        fit = ep.glm_ql_fit(d, design)
        tr = ep.glm_treat(fit, coef=1, lfc=np.log2(1.2))
        dt = ep.decide_tests(tr)
        # R: Down=0, NotSig=100, Up=0
        assert np.sum(dt == -1) == 0
        assert np.sum(dt == 0) == 100
        assert np.sum(dt == 1) == 0

    def test_ql_treat_decide_tests_lfc15(self, d4_treat):
        d, y, design = d4_treat
        fit = ep.glm_ql_fit(d, design)
        tr = ep.glm_treat(fit, coef=1, lfc=np.log2(1.5))
        dt = ep.decide_tests(tr)
        assert np.sum(dt == -1) == 0
        assert np.sum(dt == 0) == 100
        assert np.sum(dt == 1) == 0

    def test_lrt_treat_lfc12(self, d4_treat):
        d, y, design = d4_treat
        fit = ep.glm_fit(d, design=design)
        tr = ep.glm_treat(fit, coef=1, lfc=np.log2(1.2))
        tt = ep.top_tags(tr)['table']
        top = tt.iloc[0]
        assert top['PValue'] < 1e-4


# ── Treat in multi-factor design (Part 29) ──────────────────────────

class TestGlmTreatMultiFactor:
    """glmTreat with multi-factor design (Part 29)."""

    def test_treat_lfc15(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part29.csv").values
        treatment = np.array(['Control'] * 3 + ['Treated'] * 3)
        batch = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        design = np.column_stack([
            np.ones(6),
            (batch == 'B').astype(float),
            (batch == 'C').astype(float),
            (treatment == 'Treated').astype(float)
        ])
        d = ep.make_dgelist(counts=y, group=treatment)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        treat = ep.glm_treat(fit_ql, coef=3, lfc=np.log2(1.5))
        tt = ep.top_tags(treat, n=10)['table']
        # R: gene1 PValue=0.02335
        top = tt.iloc[0]
        assert abs(top['PValue'] - 0.02335) < 0.005


# ── Treat thresholds (Part 45) ──────────────────────────────────────

class TestGlmTreatThresholds:
    """glmTreat with different lfc thresholds (Part 45)."""

    @pytest.fixture(scope="class")
    def fits45(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        design = np.column_stack([np.ones(4),
                                   np.array([0, 0, 1, 1], dtype=float)])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        d = ep.estimate_disp(d, design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        fit_lrt = ep.glm_fit(d, design=design)
        return fit_ql, fit_lrt

    def test_ql_treat_lfc15_decide(self, fits45):
        fit_ql, fit_lrt = fits45
        tr = ep.glm_treat(fit_ql, coef=1, lfc=np.log2(1.5))
        dt = ep.decide_tests(tr)
        # R: 21 NotSig, 1 Up
        assert np.sum(dt == 1) <= 2
        assert np.sum(dt == 0) >= 20

    def test_ql_treat_lfc10_decide(self, fits45):
        fit_ql, fit_lrt = fits45
        tr = ep.glm_treat(fit_ql, coef=1, lfc=1.0)
        dt = ep.decide_tests(tr)
        assert np.sum(dt == 1) <= 2
        assert np.sum(dt == 0) >= 20


# ── QL-based Treat (conftest fixtures) ──────────────────────────────

class TestGlmTreatQL:
    """Tests for glmTreat with QL fits (conftest fixtures)."""

    def test_treat_returns_dict(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        assert isinstance(result, dict)

    def test_treat_has_table(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        assert 'table' in result
        assert isinstance(result['table'], pd.DataFrame)

    def test_treat_table_columns(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        table = result['table']
        assert 'logFC' in table.columns
        assert 'logCPM' in table.columns
        assert 'PValue' in table.columns

    def test_treat_pvalues_valid(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        pv = result['table']['PValue'].values
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_treat_ngenes_correct(self, ql_fit, small_counts):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        assert len(result['table']) == small_counts.shape[0]

    def test_treat_different_lfc_thresholds(self, ql_fit):
        r1 = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        r2 = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.5))
        r3 = ep.glm_treat(ql_fit, coef=1, lfc=1.0)
        assert r2['table']['PValue'].mean() >= r1['table']['PValue'].mean()

    def test_treat_worst_case_null(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.5),
                              null='worst.case')
        assert isinstance(result, dict)
        pv = result['table']['PValue'].values
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_treat_contrast_vector(self, ql_fit):
        result = ep.glm_treat(ql_fit, contrast=np.array([0.0, 1.0]),
                              lfc=np.log2(1.5))
        assert isinstance(result, dict)
        assert len(result['table']) > 0

    def test_treat_top_tags(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        tt = ep.top_tags(result, n=10, sort_by='PValue')
        assert isinstance(tt, dict)
        assert 'table' in tt
        assert len(tt['table']) == 10

    def test_treat_decide_tests(self, ql_fit):
        result = ep.glm_treat(ql_fit, coef=1, lfc=np.log2(1.2))
        dt = ep.decide_tests(result)
        assert dt.shape[0] == len(result['table'])
        assert set(np.unique(dt)) <= {-1, 0, 1}


# ── LRT-based Treat (conftest fixtures) ─────────────────────────────

class TestGlmTreatLRT:
    """Tests for glmTreat with LRT fits (conftest fixtures)."""

    def test_treat_lrt_returns_dict(self, lrt_fit):
        result = ep.glm_treat(lrt_fit, coef=1, lfc=np.log2(1.2))
        assert isinstance(result, dict)

    def test_treat_lrt_pvalues_valid(self, lrt_fit):
        result = ep.glm_treat(lrt_fit, coef=1, lfc=np.log2(1.2))
        pv = result['table']['PValue'].values
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_treat_lrt_different_lfc(self, lrt_fit):
        r1 = ep.glm_treat(lrt_fit, coef=1, lfc=np.log2(1.2))
        r2 = ep.glm_treat(lrt_fit, coef=1, lfc=1.0)
        np.testing.assert_array_equal(
            r1['table']['logFC'].values,
            r2['table']['logFC'].values
        )

    def test_treat_lrt_worst_case(self, lrt_fit):
        result = ep.glm_treat(lrt_fit, coef=1, lfc=np.log2(1.5),
                              null='worst.case')
        pv = result['table']['PValue'].values
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_treat_lrt_contrast(self, lrt_fit):
        result = ep.glm_treat(lrt_fit, contrast=np.array([0.0, 1.0]),
                              lfc=np.log2(1.5))
        assert len(result['table']) > 0
