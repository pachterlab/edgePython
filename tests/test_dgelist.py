# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for DGEList construction, filtering, normalization, cbind/rbind, accessors."""

import os

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def d1_expr():
    """Part 1 data with calcNormFactors (no explicit lib_size)."""
    y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
    group = np.array([1, 1, 2, 2])
    d = ep.make_dgelist(counts=y, group=group)
    d = ep.calc_norm_factors(d)
    return d, y, group


# ── FilterByExpr ─────────────────────────────────────────────────────

class TestFilterByExpr:
    """filterByExpr on 22-gene data."""

    def test_filter_by_expr_keeps_11(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        group = np.array([1, 1, 2, 2])
        lib_size = np.array([1001, 1002, 1003, 1004])
        d = ep.make_dgelist(counts=y, group=group, lib_size=lib_size)
        keep = ep.filter_by_expr(d)
        assert np.sum(keep) == 11


# ── Normalization Methods ────────────────────────────────────────────

class TestNormalization:
    """Normalization methods: TMM, RLE, upperquartile, none."""

    @pytest.fixture(scope="class")
    def d_fresh(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        group = np.array([1, 1, 2, 2])
        return ep.make_dgelist(counts=y, group=group)

    def test_upperquartile(self, d_fresh):
        d = ep.calc_norm_factors(d_fresh, method='upperquartile')
        nf = d['samples']['norm.factors'].values
        # R: [0.917796, 1.201101, 0.913137, 0.993434]
        assert np.allclose(nf, [0.917796, 1.201101, 0.913137, 0.993434],
                           atol=1e-4)

    def test_rle(self, d_fresh):
        d = ep.calc_norm_factors(d_fresh, method='RLE')
        nf = d['samples']['norm.factors'].values
        # R: [1.085513, 0.956217, 0.917485, 1.050049]
        assert np.allclose(nf, [1.085513, 0.956217, 0.917485, 1.050049],
                           atol=1e-4)

    def test_none(self, d_fresh):
        d = ep.calc_norm_factors(
            ep.make_dgelist(counts=d_fresh['counts'],
                            group=np.array([1, 1, 2, 2])),
            method='none')
        nf = d['samples']['norm.factors'].values
        assert np.allclose(nf, [1, 1, 1, 1])

    def test_cpm_with_rle(self, d_fresh):
        d = ep.calc_norm_factors(d_fresh, method='RLE')
        cpm_rle = ep.cpm(d)
        # R row 3: [37600.96, 100442.9, 49794.01, 26701.04]
        expected = [37600.96, 100442.9, 49794.01, 26701.04]
        assert np.allclose(cpm_rle[2], expected, rtol=0.001)


# ── cbind / rbind ────────────────────────────────────────────────────

class TestCbindRbind:
    """cbind/rbind DGEList."""

    def test_cbind(self, d1_expr):
        d, y, group = d1_expr
        d1a = ep.make_dgelist(counts=d['counts'][:, :2],
                              group=np.array([1, 1]),
                              lib_size=d['samples']['lib.size'].values[:2])
        d1a['samples']['norm.factors'] = d['samples']['norm.factors'].values[:2]
        d1b = ep.make_dgelist(counts=d['counts'][:, 2:],
                              group=np.array([2, 2]),
                              lib_size=d['samples']['lib.size'].values[2:])
        d1b['samples']['norm.factors'] = d['samples']['norm.factors'].values[2:]
        d_cb = ep.cbind_dgelist(d1a, d1b)
        # R: counts dim=22x4
        assert d_cb['counts'].shape == (22, 4)
        assert np.allclose(d_cb['counts'], d['counts'])

    def test_rbind(self, d1_expr):
        d, y, group = d1_expr
        lib_size = np.array([1001, 1002, 1003, 1004])
        d1c = ep.make_dgelist(counts=d['counts'][:10, :],
                              group=group, lib_size=lib_size)
        d1d = ep.make_dgelist(counts=d['counts'][10:, :],
                              group=group, lib_size=lib_size)
        d_rb = ep.rbind_dgelist(d1c, d1d)
        assert d_rb['counts'].shape == (22, 4)
        assert np.allclose(d_rb['counts'], d['counts'])


# ── Accessors and Validation ─────────────────────────────────────────

class TestAccessors:
    """DGEList accessors and validation."""

    def test_get_counts(self):
        y = np.array([[10, 20], [30, 40]], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        assert np.array_equal(ep.get_counts(d), y)

    def test_get_dispersion_before_after(self):
        y = np.array([[10, 20], [30, 40]], dtype=float)
        d = ep.make_dgelist(counts=y, group=np.array([1, 2]))
        disp_before = ep.get_dispersion(d)
        assert disp_before is None

    def test_get_offset(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        offset = ep.get_offset(d)
        assert offset is not None

    def test_valid_dgelist(self):
        minimal = {'counts': np.array([[5, 10], [15, 20]], dtype=float)}
        filled = ep.valid_dgelist(minimal)
        assert 'samples' in filled
        assert np.allclose(filled['samples']['lib.size'].values, [20, 30])
        assert np.allclose(filled['samples']['norm.factors'].values, [1, 1])
