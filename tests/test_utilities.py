# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for utility functions: Good-Turing, squeezeVar, decideTests, addPriorCount,
predFC, sumTechReps, thinCounts, gini, zscore, q2q, equalizeLibSizes, gof,
scaleOffset, splitIntoGroups, estimateTrendedDisp.
"""

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


# ── Good-Turing ──────────────────────────────────────────────────────

class TestGoodTuring:
    """Good-Turing frequency estimation."""

    def test_large_frequency_table(self):
        test1 = np.arange(1, 10)
        freq1 = np.array([2018046, 449721, 188933, 105668, 68379,
                          48190, 35709, 37710, 22280])
        gt = ep.good_turing(np.repeat(test1, freq1))
        # R: P0=0.3814719
        assert abs(gt['P0'] - 0.3814719) < 1e-5
        # R: proportion[1]=8.035e-08, proportion[9]=1.461e-06
        assert abs(gt['proportion'][0] - 8.035e-08) < 1e-10
        assert abs(gt['proportion'][-1] - 1.461e-06) < 1e-8

    def test_small_counts(self):
        test2 = np.array([312, 14491, 16401, 65124, 129797,
                          323321, 366051, 368599, 405261, 604962])
        gt = ep.good_turing(test2)
        # R: P0=0
        assert gt['P0'] == 0
        assert abs(gt['proportion'][0] - 0.000136) < 0.001


# ── addPriorCount, predFC, sumTechReps ───────────────────────────────

class TestAddPriorCountPredFC:
    """addPriorCount, predFC, sumTechReps."""

    def test_add_prior_count(self, d1_expr):
        d, y, group = d1_expr
        raw_lib = d['counts'].sum(axis=0).astype(float)
        apc = ep.add_prior_count(d['counts'], lib_size=raw_lib,
                                  prior_count=1)
        # R adj counts[1]: [1.000, 0.903, 1.005, 1.092]
        expected = [1.000, 0.903, 1.005, 1.092]
        assert np.allclose(apc['y'][0], expected, atol=0.01)
        # R offset: [5.288, 5.186, 5.293, 5.376]
        offset = apc['offset'].ravel()[:4]
        assert np.allclose(offset, [5.288, 5.186, 5.293, 5.376], atol=0.01)

    def test_pred_fc(self, d1_expr):
        d, y, group = d1_expr
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        norm_lib = ep.get_norm_lib_sizes(d)
        pfc = ep.pred_fc(d['counts'], design=design, prior_count=0.125,
                         offset=np.log(norm_lib))
        # R row 1: Intercept=-10.625, group2=0.000
        assert abs(pfc[0, 0] - (-10.625)) < 0.1
        assert abs(pfc[0, 1] - 0.000) < 0.01
        # R row 10: Intercept=-4.195, group2=0.199
        assert abs(pfc[9, 0] - (-4.195)) < 0.01
        assert abs(pfc[9, 1] - 0.199) < 0.01

    def test_sum_tech_reps_numeric(self, d1_expr):
        d, y, group = d1_expr
        result = ep.sum_tech_reps(y, ID=np.array([1, 1, 2, 2]))
        # R row1=[0,0], row2=[0,4], row3=[25,15]
        assert result[0, 0] == 0 and result[0, 1] == 0
        assert result[2, 0] + result[2, 1] > 0

    def test_sum_tech_reps_string(self, d1_expr):
        d, y, group = d1_expr
        result = ep.sum_tech_reps(y, ID=np.array(['A', 'B', 'A', 'B']))
        assert result[0, 0] == 0 and result[0, 1] == 0


# ── squeezeVar ──────────────────────────────────────────────────────

class TestSqueezeVar:
    """squeezeVar (basic, covariate, robust, unequal df)."""

    @pytest.fixture(scope="class")
    def s2_data(self):
        df = pd.read_csv(f"{CSV_DIR}/test_data_part17_s2.csv")
        return df['s2'].values

    def test_basic(self, s2_data):
        df_vec = np.full(100, 3.0)
        sv = ep.squeeze_var(s2_data, df_vec)
        # R: var.prior=0.531550, df.prior=8.0569
        assert abs(sv['var_prior'] - 0.531550) < 0.001
        dp = np.atleast_1d(sv['df_prior'])[0]
        assert abs(dp - 8.0569) < 0.1
        # R: var.post[1:3]=[0.767197, 0.579916, 0.513519]
        assert np.allclose(sv['var_post'][:3],
                           [0.767197, 0.579916, 0.513519], atol=0.01)

    def test_with_covariate(self, s2_data):
        df_vec = np.full(100, 3.0)
        covariate = np.arange(1, 101, dtype=float)
        sv = ep.squeeze_var(s2_data, df_vec, covariate=covariate)
        # R: df.prior=8.8731
        dp = np.atleast_1d(sv['df_prior'])[0]
        assert abs(dp - 8.8731) < 0.1
        # R: var.post[1:3]=[1.076943, 0.886807, 0.809594]
        assert np.allclose(sv['var_post'][:3],
                           [1.076943, 0.886807, 0.809594], atol=0.01)

    def test_robust(self, s2_data):
        df_vec = np.full(100, 3.0)
        sv = ep.squeeze_var(s2_data, df_vec, robust=True)
        # R: var.prior=0.530424
        vp = np.atleast_1d(sv['var_prior'])[0]
        assert abs(vp - 0.531) < 0.01

    def test_unequal_df(self, s2_data):
        df_unequal = np.tile([2.0, 3.0, 4.0, 5.0], 25)
        sv = ep.squeeze_var(s2_data, df_unequal)
        # R: var.prior=0.508725, df.prior=7.5627
        assert abs(sv['var_prior'] - 0.508725) < 0.001
        dp = np.atleast_1d(sv['df_prior'])[0]
        assert abs(dp - 7.5627) < 0.1
        # R: var.post[1:3]=[0.695144, 0.565837, 0.493631]
        assert np.allclose(sv['var_post'][:3],
                           [0.695144, 0.565837, 0.493631], atol=0.01)


# ── decideTests ─────────────────────────────────────────────────────

class TestDecideTests:
    """decideTests with various thresholds."""

    @pytest.fixture(scope="class")
    def qlf_dt(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        fit = ep.glm_ql_fit(d, design)
        qlf = ep.glm_ql_ftest(fit, coef=1)
        return qlf

    def test_p005(self, qlf_dt):
        dt = ep.decide_tests(qlf_dt, p_value=0.05)
        # R: Down=0, NotSig=100, Up=0
        assert np.sum(dt == -1) == 0
        assert np.sum(dt == 0) == 100
        assert np.sum(dt == 1) == 0

    def test_p01(self, qlf_dt):
        dt = ep.decide_tests(qlf_dt, p_value=0.1)
        assert np.sum(dt == -1) == 0
        assert np.sum(dt == 0) == 100

    def test_p005_lfc1(self, qlf_dt):
        dt = ep.decide_tests(qlf_dt, p_value=0.05, lfc=1)
        assert np.sum(dt == 0) == 100

    def test_unadjusted(self, qlf_dt):
        dt = ep.decide_tests(qlf_dt, adjust_method='none', p_value=0.05)
        # R: Down=5, NotSig=93, Up=2
        assert np.sum(dt == -1) == 5
        assert np.sum(dt == 0) == 93
        assert np.sum(dt == 1) == 2


# ── thinCounts, gini ────────────────────────────────────────────────

class TestThinCountsGini:
    """thinCounts, gini."""

    def test_gini(self, d1_expr):
        d, y, group = d1_expr
        gi = ep.gini(y)
        # R: [0.309833, 0.371597, 0.429395, 0.370858]
        expected = [0.309833, 0.371597, 0.429395, 0.370858]
        assert np.allclose(gi, expected, atol=1e-4)

    def test_thin_counts_target_size(self, d1_expr):
        d, y, group = d1_expr
        np.random.seed(456)
        tc = ep.thin_counts(y.copy(), target_size=150)
        # R colSums: [150, 150, 150, 150]
        assert np.allclose(tc.sum(axis=0), [150, 150, 150, 150])

    def test_thin_counts_prob(self, d1_expr):
        d, y, group = d1_expr
        np.random.seed(123)
        tc = ep.thin_counts(y[:10].copy(), prob=0.5)
        assert np.all(tc <= y[:10])
        ratio = tc.sum() / y[:10].sum()
        assert 0.3 < ratio < 0.7


# ── zscoreNBinom, q2qNBinom, equalizeLibSizes ──────────────────────

class TestZscoreQ2Q:
    """zscoreNBinom, q2qNBinom, equalizeLibSizes."""

    def test_zscore_nbinom_1(self):
        z = ep.zscore_nbinom(np.arange(0, 11, dtype=float), size=5, mu=5)
        # R: [-2.153875, ..., 1.443011]
        assert abs(z[0] - (-2.153875)) < 1e-4
        assert abs(z[-1] - 1.443011) < 1e-4

    def test_zscore_nbinom_2(self):
        z = ep.zscore_nbinom(np.array([0, 1, 5, 10, 20], dtype=float),
                              size=10, mu=8)
        # R: [-2.9888, -2.364605, -0.750342, 0.605684, 2.545517]
        expected = [-2.9888, -2.364605, -0.750342, 0.605684, 2.545517]
        assert np.allclose(z, expected, atol=1e-4)

    def test_q2q_nbinom_disp01(self):
        qq = ep.q2q_nbinom(np.array([0, 5, 10, 20], dtype=float),
                            input_mean=np.full(4, 10.0),
                            output_mean=np.full(4, 20.0),
                            dispersion=0.1)
        # R: [1.339746, 11.263600, 20.078711, 37.161500]
        expected = [1.339746, 11.263600, 20.078711, 37.161500]
        assert np.allclose(qq, expected, atol=0.001)

    def test_q2q_nbinom_disp0(self):
        qq = ep.q2q_nbinom(np.array([0, 5, 10, 20], dtype=float),
                            input_mean=np.full(4, 10.0),
                            output_mean=np.full(4, 20.0),
                            dispersion=0)
        # R: [2.928932, 12.743062, 20.069637, 33.763551]
        expected = [2.928932, 12.743062, 20.069637, 33.763551]
        assert np.allclose(qq, expected, atol=0.001)

    def test_equalize_lib_sizes(self, d1_expr):
        d, y, group = d1_expr
        eq = ep.equalize_lib_sizes(d)
        pseudo_counts = eq.get('pseudo.counts', eq.get('pseudo_counts'))
        pseudo_lib_size = eq.get('pseudo.lib.size', eq.get('pseudo_lib_size'))
        # R: pseudo.lib.size=195.5588
        assert abs(pseudo_lib_size - 195.5588) < 0.01
        # R pseudo.counts[3,:]: [7.1521, 20.7734, 8.6271, 5.4265]
        expected = [7.1521, 20.7734, 8.6271, 5.4265]
        assert np.allclose(pseudo_counts[2], expected, atol=0.01)


# ── gof ─────────────────────────────────────────────────────────────

class TestGof:
    """Goodness of fit."""

    def test_gof(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        g = ep.gof(fit, plot=False)
        # R: df=1, outlier sum=0
        df_val = g['df']
        if hasattr(df_val, '__len__'):
            df_val = df_val[0]
        assert df_val == 1
        assert np.sum(g['outlier']) == 0
        # R summary: Median=0.6054, Mean=0.8485
        stats = g['gof.statistics']
        assert abs(np.median(stats) - 0.6054) < 0.05
        assert abs(np.mean(stats) - 0.8485) < 0.02


# ── scaleOffset ─────────────────────────────────────────────────────

class TestScaleOffset:
    """scaleOffset function."""

    def test_scale_offset(self):
        from edgepython.utils import scale_offset
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        lib_size = y.sum(axis=0).astype(float)
        offset_mat = np.log(lib_size)[None, :] * np.ones((y.shape[0], 1))
        offset_mat[:11, :] += 0.1
        offset_mat[11:, :] -= 0.1
        scaled = scale_offset(lib_size, offset_mat)
        # R row 0: [5.278115, 5.176150, 5.283204, 5.365976]
        expected = [5.278115, 5.176150, 5.283204, 5.365976]
        assert np.allclose(scaled[0], expected, atol=0.01)


# ── splitIntoGroups ─────────────────────────────────────────────────

class TestSplitIntoGroups:
    """splitIntoGroups function."""

    def test_split_into_groups(self):
        y = np.array([[10, 20, 30, 40],
                       [5, 15, 25, 35]], dtype=float)
        groups = ep.split_into_groups(y, group=np.array([1, 1, 2, 2]))
        assert len(groups) == 2
        assert np.allclose(groups[0], y[:, :2])
        assert np.allclose(groups[1], y[:, 2:])

    def test_split_into_groups_3(self):
        y = np.array([[10, 20, 30, 40],
                       [5, 15, 25, 35]], dtype=float)
        groups = ep.split_into_groups(y, group=np.array([1, 2, 3, 1]))
        assert len(groups) == 3


# ── estimateTrendedDisp (bin.spline / bin.loess) ────────────────────

class TestEstimateTrendedDisp:
    """estimateTrendedDisp with different methods."""

    def test_bin_spline(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[:5, 1:3] = 0
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_trended_disp(d, method='bin.spline')
        tr = d['trended.dispersion']
        assert not np.any(np.isnan(tr))
        assert np.all(tr > 0)

    def test_bin_loess(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[:5, 1:3] = 0
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_trended_disp(d, method='bin.loess')
        tr = d['trended.dispersion']
        assert not np.any(np.isnan(tr))
