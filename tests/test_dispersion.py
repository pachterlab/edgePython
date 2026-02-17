# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for dispersion estimation: common, tagwise, trended, GLM, robust, fixed prior.df."""

import os

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def part1_disp():
    """22 genes x 4 samples with common dispersion estimated."""
    y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
    group = np.array([1, 1, 2, 2])
    lib_size = np.array([1001, 1002, 1003, 1004])
    d = ep.make_dgelist(counts=y, group=group, lib_size=lib_size)
    d['genes'] = pd.DataFrame({'gene': [f"Tag.{i+1}" for i in range(y.shape[0])]})
    d = ep.estimate_common_disp(d)
    return d


@pytest.fixture(scope="module")
def part4_data():
    """100 genes x 3 samples, first 5 genes zeroed in cols 1-2."""
    y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
    y[0:5, 1:3] = 0
    group = np.array([1, 2, 2])
    design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
    d = ep.make_dgelist(counts=y, group=group)
    return d, y, group, design


# ── Common Dispersion ────────────────────────────────────────────────

class TestCommonDispersion:
    """estimateCommonDisp."""

    def test_common_dispersion(self, part1_disp):
        # R: 0.210292
        assert abs(part1_disp['common.dispersion'] - 0.210292) < 0.001

    def test_common_disp_with_zeros(self, part4_data):
        d, y, group, design = part4_data
        d2 = ep.estimate_common_disp(d)
        # R: 0.2408
        assert abs(d2['common.dispersion'] - 0.2408) < 0.01


# ── Tagwise Dispersion ──────────────────────────────────────────────

class TestTagwiseDispersion:
    """estimateTagwiseDisp (various trends)."""

    def test_tagwise_trend_none(self, part1_disp):
        d2 = ep.estimate_tagwise_disp(part1_disp, trend='none', prior_df=20)
        tw = d2['tagwise.dispersion']
        # R: Min=0.1757, Median=0.1989, Max=0.2677
        assert abs(np.min(tw) - 0.1757) < 0.01
        assert abs(np.median(tw) - 0.1989) < 0.01
        assert abs(np.max(tw) - 0.2677) < 0.02

    def test_tagwise_trend_movingave(self, part1_disp):
        d2 = ep.estimate_tagwise_disp(part1_disp, trend='movingave',
                                       span=0.4, prior_df=20)
        tw = d2['tagwise.dispersion']
        # R: Min=0.1005, Median=0.2064, Max=0.3164
        assert abs(np.min(tw) - 0.1005) < 0.02
        assert abs(np.median(tw) - 0.2064) < 0.02
        assert abs(np.max(tw) - 0.3164) < 0.02

    def test_tagwise_trend_loess(self, part1_disp):
        d2 = ep.estimate_tagwise_disp(part1_disp, trend='loess',
                                       span=0.8, prior_df=20)
        tw = d2['tagwise.dispersion']
        # R: Min=0.1165, Median=0.1832, Max=0.2825
        assert abs(np.min(tw) - 0.1165) < 0.02
        assert abs(np.median(tw) - 0.1832) < 0.02
        assert abs(np.max(tw) - 0.2825) < 0.02

    def test_exact_test_smallp_rejection(self, part1_disp):
        d2 = ep.estimate_tagwise_disp(part1_disp, trend='movingave',
                                       span=0.4, prior_df=20)
        de = ep.exact_test(d2, rejection_region='smallp')
        pv = de['table']['PValue'].values
        # R: Min=0.02428, Median=0.55662, Mean=0.54319
        assert abs(np.min(pv) - 0.02428) < 0.005
        assert abs(np.median(pv) - 0.55662) < 0.02
        assert abs(np.mean(pv) - 0.54319) < 0.02


# ── GLM Dispersion ──────────────────────────────────────────────────

class TestGLMDispersion:
    """estimateGLMCommonDisp, estimateGLMTagwiseDisp, estimateGLMTrendedDisp."""

    def test_glm_common_disp(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        dglm = ep.estimate_glm_common_disp(part1_disp, design=design)
        # R: 0.2033282
        assert abs(dglm['common.dispersion'] - 0.2033282) < 0.001

    def test_glm_tagwise_disp(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        dglm = ep.estimate_glm_common_disp(part1_disp, design=design)
        dglm = ep.estimate_glm_tagwise_disp(dglm, design=design,
                                              prior_df=20, trend=False)
        tw = dglm['tagwise.dispersion']
        # R: Min=0.1756, Median=0.1998, Max=0.2578
        assert abs(np.min(tw) - 0.1756) < 0.01
        assert abs(np.median(tw) - 0.1998) < 0.01
        assert abs(np.max(tw) - 0.2578) < 0.02

    def test_glm_trended_disp(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        dglm = ep.estimate_glm_common_disp(part1_disp, design=design)
        dglm = ep.estimate_glm_trended_disp(dglm, design=design)
        tr = dglm['trended.dispersion']
        assert len(tr) == part1_disp['counts'].shape[0]
        assert not np.any(np.isnan(tr))
        assert np.all(tr > 0)

    def test_glm_common_disp_with_zeros(self, part4_data):
        d, y, group, design = part4_data
        d2 = ep.estimate_glm_common_disp(d, design=design)
        # R: 0.2181
        assert abs(d2['common.dispersion'] - 0.2181) < 0.001


# ── estimateDisp ────────────────────────────────────────────────────

class TestEstimateDisp:
    """estimateDisp (unified dispersion estimation)."""

    def test_estimate_disp(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        dglm = ep.estimate_disp(part1_disp, design=design)
        tw = dglm['tagwise.dispersion']
        # R: Min=0.1652, Median=0.1821, Max=0.2259
        assert abs(np.min(tw) - 0.1652) < 0.02
        assert abs(np.median(tw) - 0.1821) < 0.02
        assert abs(np.max(tw) - 0.2259) < 0.02

    def test_estimate_disp_prior_df(self, part4_data):
        d, y, group, design = part4_data
        d2 = ep.estimate_disp(d, design=design)
        # R: prior.df=2.1818
        prior_df = d2.get('prior.df')
        if prior_df is not None:
            val = np.median(np.atleast_1d(prior_df))
            assert abs(val - 2.1818) < 0.05


# ── glmQLFit (dispersion aspects) ──────────────────────────────────

class TestGlmQLFitDispersion:
    """glmQLFit with legacy modes (dispersion-focused)."""

    def test_ql_unit_deviance(self, part4_data):
        d, y, group, design = part4_data
        fit = ep.glm_ql_fit(d, design=design, legacy=False,
                             keep_unit_mat=True)
        disp = fit.get('dispersion')
        if isinstance(disp, np.ndarray):
            disp_val = np.median(disp)
        else:
            disp_val = disp
        # R: dispersion=0.1913
        assert abs(disp_val - 0.1913) < 0.01
        # R: s2.post: Min=0.3227, Median=1.0467, Max=3.2255
        s2p = fit['s2.post']
        assert abs(np.min(s2p) - 0.3227) < 0.05
        assert abs(np.median(s2p) - 1.0467) < 0.05
        assert abs(np.max(s2p) - 3.2255) < 0.1

    def test_legacy_false(self, part4_data):
        d, y, group, design = part4_data
        d2 = ep.estimate_disp(d, design=design)
        fit = ep.glm_ql_fit(d2, design=design, legacy=False,
                             keep_unit_mat=True)
        # R: s2.post: Min=0.3180, Median=1.0289, Max=3.1867
        s2p = fit['s2.post']
        assert abs(np.min(s2p) - 0.3180) < 0.05
        assert abs(np.median(s2p) - 1.0289) < 0.05
        assert abs(np.max(s2p) - 3.1867) < 0.15

    def test_legacy_true(self, part4_data):
        d, y, group, design = part4_data
        d2 = ep.estimate_disp(d, design=design)
        fit = ep.glm_ql_fit(d2, design=design, legacy=True)
        # R: dispersion: Min=0.1958, Median=0.2098, Max=0.2685
        disp = np.asarray(fit['dispersion'])
        assert abs(np.min(disp) - 0.1958) < 0.01
        assert abs(np.median(disp) - 0.2098) < 0.01
        assert abs(np.max(disp) - 0.2685) < 0.02
        # R: s2.post: Min=0.0618, Median=0.7988, Max=2.5983
        s2p = fit['s2.post']
        assert abs(np.min(s2p) - 0.0618) < 0.02
        assert abs(np.median(s2p) - 0.7988) < 0.05
        assert abs(np.max(s2p) - 2.5983) < 0.2


# ── Robust Estimation ──────────────────────────────────────────────

class TestRobustEstimation:
    """Robust glmQLFit and estimateDisp."""

    @pytest.fixture(scope="class")
    def d4_robust(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        return d, design

    def test_robust_ql_s2post(self, d4_robust):
        d, design = d4_robust
        fit = ep.glm_ql_fit(d, design=design, robust=True)
        s2p = fit['s2.post']
        # R: Min=0.479, Median=1.530, Max=4.711
        assert abs(np.min(s2p) - 0.479) < 0.05
        assert abs(np.median(s2p) - 1.530) < 0.1
        assert abs(np.max(s2p) - 4.711) < 0.2

    def test_robust_ql_top_genes(self, d4_robust):
        d, design = d4_robust
        fit = ep.glm_ql_fit(d, design=design, robust=True)
        qlf = ep.glm_ql_ftest(fit, coef=1)
        tt = ep.top_tags(qlf)['table']
        # R: gene5 F=8.359, PValue=0.00471
        top = tt.iloc[0]
        assert abs(top['F'] - 8.359) < 0.1
        assert abs(top['PValue'] - 0.00471) < 0.002

    def test_robust_estimate_disp(self, d4_robust):
        d, design = d4_robust
        d_fresh = ep.make_dgelist(counts=d['counts'],
                                  group=np.array([1, 2, 2]))
        d_rob = ep.estimate_disp(d_fresh, design, robust=True)
        prior_df = d_rob.get('prior.df')
        if prior_df is not None:
            pdf = np.asarray(prior_df)
            # R: Min=2.060, Median=2.060, Max=2.183
            assert abs(np.min(pdf) - 2.060) < 0.05
            assert abs(np.median(pdf) - 2.060) < 0.05
        tw = d_rob['tagwise.dispersion']
        # R: Min=0.0972, Median=0.191, Max=0.631
        assert abs(np.min(tw) - 0.0972) < 0.01
        assert abs(np.median(tw) - 0.191) < 0.01

    def test_estimate_glm_robust_disp(self, part4_data):
        d, _, _, design = part4_data
        out = ep.estimate_glm_robust_disp(d, design=design, maxit=2, record=True)

        assert 'tagwise.dispersion' in out
        assert out['tagwise.dispersion'].shape[0] == out['counts'].shape[0]
        assert np.all(np.isfinite(out['tagwise.dispersion']))

        assert 'weights' in out
        assert out['weights'].shape == out['counts'].shape
        assert np.all(np.isfinite(out['weights']))
        assert np.all(out['weights'] > 0)
        assert np.all(out['weights'] <= 1.0)

        assert 'record' in out
        assert 'weights' in out['record']
        assert 'iteration_0' in out['record']['weights']
        assert 'iteration_1' in out['record']['weights']


# ── Dispersion Trend Methods ───────────────────────────────────────

class TestDispersionTrends:
    """Dispersion trend methods (locfit, movingave, none)."""

    @pytest.fixture(scope="class")
    def base_data(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        group = np.array([1, 2, 2])
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        return y, group, design

    def test_trend_none(self, base_data):
        y, group, design = base_data
        d = ep.make_dgelist(counts=y.copy(), group=group)
        d = ep.estimate_disp(d, design, trend_method='none')
        # R: common=0.2185181
        assert abs(d['common.dispersion'] - 0.2185) < 0.01

    def test_trend_locfit(self, base_data):
        y, group, design = base_data
        d = ep.make_dgelist(counts=y.copy(), group=group)
        d = ep.estimate_disp(d, design, trend_method='locfit')
        trended = d.get('trended.dispersion')
        assert trended is not None
        assert not np.any(np.isnan(trended))

    def test_trend_movingave(self, base_data):
        y, group, design = base_data
        d = ep.make_dgelist(counts=y.copy(), group=group)
        d = ep.estimate_disp(d, design, trend_method='movingave')
        trended = d.get('trended.dispersion')
        assert trended is not None
        assert not np.any(np.isnan(trended))


# ── Fixed prior.df ──────────────────────────────────────────────────

class TestFixedPriorDF:
    """estimateDisp with fixed prior.df."""

    @pytest.fixture(scope="class")
    def base_data46(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3),
                                   np.array([0, 1, 1], dtype=float)])
        return y, design

    def test_prior_df_10(self, base_data46):
        y, design = base_data46
        d = ep.make_dgelist(counts=y.copy(), group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design, prior_df=10)
        tw = d['tagwise.dispersion']
        # R: range=[0.166, 0.363], sd=0.04523
        assert np.min(tw) > 0.1
        assert np.max(tw) < 0.5
        assert abs(np.std(tw) - 0.04523) < 0.01

    def test_prior_df_50(self, base_data46):
        y, design = base_data46
        d = ep.make_dgelist(counts=y.copy(), group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design, prior_df=50)
        tw = d['tagwise.dispersion']
        # R: range=[0.187, 0.289], sd=0.03122
        assert np.min(tw) > 0.1
        assert np.max(tw) < 0.4
        assert abs(np.std(tw) - 0.03122) < 0.01

    def test_higher_prior_df_more_shrinkage(self, base_data46):
        y, design = base_data46
        d10 = ep.make_dgelist(counts=y.copy(), group=np.array([1, 2, 2]))
        d10 = ep.estimate_disp(d10, design, prior_df=10)
        d50 = ep.make_dgelist(counts=y.copy(), group=np.array([1, 2, 2]))
        d50 = ep.estimate_disp(d50, design, prior_df=50)
        sd10 = np.std(d10['tagwise.dispersion'])
        sd50 = np.std(d50['tagwise.dispersion'])
        assert sd50 < sd10
