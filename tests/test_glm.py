# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for GLM fitting: glmFit, glmLRT, glmQLFit, glmQLFTest, mglmOneWay,
contrasts, offsets, multi-factor designs, paired designs, interactions.
"""

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


# ── mglmOneWay ──────────────────────────────────────────────────────

class TestMglmOneWay:
    """mglmOneWay fitted values and coefficients."""

    def test_fitted_values(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        result = ep.mglm_one_way(part1_disp['counts'][:10, :], design=design,
                                  dispersion=0.2)
        fv = result['fitted.values']
        assert fv.shape == (10, 4)
        assert not np.any(np.isnan(fv))

    def test_zero_dispersion(self, part1_disp):
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        result = ep.mglm_one_way(part1_disp['counts'][:10, :], design=design,
                                  dispersion=0)
        assert result['coefficients'].shape == (10, 2)
        assert not np.any(np.isnan(result['fitted.values']))


# ── mglmOneGroup ────────────────────────────────────────────────────

class TestMglmOneGroup:
    """mglmOneGroup coefficient estimation."""

    def test_mglm_one_group(self):
        y = np.array([[10, 20, 30],
                       [5, 15, 25],
                       [100, 200, 300],
                       [0, 0, 0],
                       [50, 50, 50]], dtype=float)
        offset = np.log(np.array([1e6, 1.5e6, 2e6]))
        coefs = ep.mglm_one_group(y, dispersion=0.1, offset=offset)
        assert coefs.shape == (5,)
        assert not np.any(np.isnan(coefs))
        assert coefs[3] < -10

    def test_mglm_one_group_weights(self):
        y = np.array([[10, 20, 30],
                       [5, 15, 25]], dtype=float)
        offset = np.log(np.array([1e6, 1.5e6, 2e6]))
        w = np.array([[1, 1, 2], [1, 1, 2]], dtype=float)
        coefs = ep.mglm_one_group(y, dispersion=0.1, offset=offset,
                                   weights=w)
        assert coefs.shape == (2,)
        assert not np.any(np.isnan(coefs))


# ── glmFit + glmLRT ────────────────────────────────────────────────

class TestGlmFitLRT:
    """glmFit + glmLRT on 22-gene data."""

    def test_glm_lrt_top_gene(self, part1_disp):
        d = part1_disp
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        fit = ep.glm_fit(d, design=design,
                         dispersion=d['common.dispersion'],
                         prior_count=0.5/4)
        lrt = ep.glm_lrt(fit, coef=1)
        tt = ep.top_tags(lrt)['table']
        # R: Tag.17 logFC=2.045096, LR=6.0485, PValue=0.01392
        top = tt.iloc[0]
        assert abs(top['logFC'] - 2.045096) < 0.001
        assert abs(top['LR'] - 6.0485) < 0.01
        assert abs(top['PValue'] - 0.01392) < 0.001

    def test_glm_fit_coefficients(self, part1_disp):
        d = part1_disp
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        fit = ep.glm_fit(d, design=design,
                         dispersion=d['common.dispersion'],
                         prior_count=0.5)
        coef = fit['coefficients']
        # R: Intercept: Min=-7.604, Median=-4.713, Max=-4.107
        assert abs(np.min(coef[:, 0]) - (-7.604)) < 0.01
        assert abs(np.median(coef[:, 0]) - (-4.713)) < 0.01
        assert abs(np.max(coef[:, 0]) - (-4.107)) < 0.01
        # R: group2: Min=-1.137, Median=0.1508, Max=1.609
        assert abs(np.min(coef[:, 1]) - (-1.137)) < 0.01
        assert abs(np.median(coef[:, 1]) - 0.1508) < 0.01
        assert abs(np.max(coef[:, 1]) - 1.609) < 0.01


# ── Continuous covariate ────────────────────────────────────────────

class TestContinuousCovariate:
    """Continuous covariate (1000 genes x 3 samples)."""

    @pytest.fixture(scope="class")
    def part2_data(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part2.csv").values
        x = np.array([0, 1, 2])
        design = np.column_stack([np.ones(3), x])
        d = ep.make_dgelist(counts=y, group=np.ones(3, dtype=int))
        d['genes'] = pd.DataFrame({'gene': [f"Gene{i+1}" for i in range(1000)]})
        return d, y, design

    def test_tmm_norm_factors(self, part2_data):
        d, y, design = part2_data
        d = ep.calc_norm_factors(d, method='TMM')
        nf = d['samples']['norm.factors'].values
        # R: [1.0009, 1.0014, 0.9977]
        assert np.allclose(nf, [1.0009, 1.0014, 0.9977], atol=0.001)

    def test_glm_lrt_top_gene(self, part2_data):
        d, y, design = part2_data
        d = ep.calc_norm_factors(d, method='TMM')
        fit = ep.glm_fit(d, design=design, dispersion=0.1,
                         prior_count=0.5/3)
        lrt = ep.glm_lrt(fit, coef=1)
        tt = ep.top_tags(lrt)['table']
        # R: Gene1 logFC=2.9070, LR=38.739, PValue=4.846e-10
        top = tt.iloc[0]
        assert abs(top['logFC'] - 2.9070) < 0.01
        assert abs(top['LR'] - 38.739) < 0.1
        assert top['PValue'] < 1e-8

    def test_glm_common_disp(self, part2_data):
        d, y, design = part2_data
        d = ep.calc_norm_factors(d, method='TMM')
        d = ep.estimate_glm_common_disp(d, design=design)
        # R: 0.10253
        assert abs(d['common.dispersion'] - 0.10253) < 0.001

    def test_tmmwsp_norm_factors(self, part2_data):
        d, y, design = part2_data
        nf = ep.calc_norm_factors(d['counts'], method='TMMwsp')
        # R: [0.9992, 1.0077, 0.9931]
        assert np.allclose(nf, [0.9992, 1.0077, 0.9931], atol=0.005)


# ── glmFit with contrasts ──────────────────────────────────────────

class TestGlmContrasts:
    """glmFit with contrast vectors and matrices."""

    def test_glm_fit_3group_coefficients(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part3b.csv").values
        y[0, 2:4] = 0
        group = np.array([1, 1, 2, 2, 3, 3, 3])
        design = np.column_stack([np.ones(7),
                                   (group == 2).astype(float),
                                   (group == 3).astype(float)])
        d = ep.make_dgelist(counts=y, group=group)
        fit = ep.glm_fit(d, design=design, dispersion=2/3,
                         prior_count=0.5/7)
        coef = fit['coefficients']
        # R: Intercept: Min=-1.817, Median=-1.712, Max=-1.356
        assert abs(np.min(coef[:, 0]) - (-1.817)) < 0.01
        assert abs(np.median(coef[:, 0]) - (-1.712)) < 0.01

    def test_glm_lrt_contrast_matrix(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part3b.csv").values
        y[0, 2:4] = 0
        group = np.array([1, 1, 2, 2, 3, 3, 3])
        design = np.column_stack([np.ones(7),
                                   (group == 2).astype(float),
                                   (group == 3).astype(float)])
        d = ep.make_dgelist(counts=y, group=group)
        fit = ep.glm_fit(d, design=design, dispersion=2/3,
                         prior_count=0.5/7)
        contrast = np.array([[0, 0], [1, 0], [0, 1]])
        lrt = ep.glm_lrt(fit, contrast=contrast)
        tt = ep.top_tags(lrt)['table']
        # R: Row1 LR=10.771, PValue=0.004582
        top = tt.iloc[0]
        assert abs(top['LR'] - 10.771) < 0.1
        assert abs(top['PValue'] - 0.004582) < 0.001


# ── Multi-factor design ────────────────────────────────────────────

class TestMultiFactor:
    """Multi-factor design (50 genes x 6 samples)."""

    @pytest.fixture(scope="class")
    def d29(self):
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
        return d, design

    def test_lrt_treatment(self, d29):
        d, design = d29
        fit = ep.glm_fit(d, design=design)
        lrt = ep.glm_lrt(fit, coef=3)
        tt = ep.top_tags(lrt, n=10)['table']
        # R: gene1 logFC=1.887, PValue=0.00540
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.887) < 0.05
        assert top['PValue'] < 0.01

    def test_ql_treatment(self, d29):
        d, design = d29
        fit_ql = ep.glm_ql_fit(d, design=design)
        qlf = ep.glm_ql_ftest(fit_ql, coef=3)
        tt = ep.top_tags(qlf, n=10)['table']
        # R: gene1 logFC=1.883, F=8.247, PValue=0.01453
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.883) < 0.05
        assert abs(top['PValue'] - 0.01453) < 0.005

    def test_lrt_batch(self, d29):
        d, design = d29
        fit = ep.glm_fit(d, design=design)
        lrt_batch = ep.glm_lrt(fit, coef=[1, 2])
        tt = ep.top_tags(lrt_batch, n=5)['table']
        top = tt.iloc[0]
        assert top['PValue'] < 0.1


# ── Complex contrasts ──────────────────────────────────────────────

class TestComplexContrasts:
    """Complex contrasts (50 genes x 6 samples, 3 groups)."""

    @pytest.fixture(scope="class")
    def fit30(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part30.csv").values
        grp = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        design = np.column_stack([
            (grp == 'A').astype(float),
            (grp == 'B').astype(float),
            (grp == 'C').astype(float)
        ])
        d = ep.make_dgelist(counts=y, group=grp)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        return fit, fit_ql, design

    def test_b_vs_a(self, fit30):
        fit, fit_ql, design = fit30
        lrt = ep.glm_lrt(fit, contrast=np.array([-1, 1, 0]))
        tt = ep.top_tags(lrt, n=5)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.727) < 0.05
        assert abs(top['PValue'] - 0.00742) < 0.005

    def test_c_vs_b(self, fit30):
        fit, fit_ql, design = fit30
        lrt = ep.glm_lrt(fit, contrast=np.array([0, -1, 1]))
        tt = ep.top_tags(lrt, n=5)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - (-2.440)) < 0.1
        assert abs(top['PValue'] - 0.000710) < 0.001

    def test_anova_2df(self, fit30):
        fit, fit_ql, design = fit30
        contrast_mat = np.array([[-1, -1], [1, 0], [0, 1]])
        lrt = ep.glm_lrt(fit, contrast=contrast_mat)
        tt = ep.top_tags(lrt, n=5)['table']
        top = tt.iloc[0]
        assert top['PValue'] < 1e-3

    def test_ql_b_vs_a(self, fit30):
        fit, fit_ql, design = fit30
        qlf = ep.glm_ql_ftest(fit_ql, contrast=np.array([-1, 1, 0]))
        tt = ep.top_tags(qlf, n=5)['table']
        top = tt.iloc[0]
        assert abs(top['F'] - 7.201) < 0.1
        assert abs(top['PValue'] - 0.00810) < 0.003


# ── GLM with offsets ────────────────────────────────────────────────

class TestGLMWithOffsets:
    """GLM with offsets."""

    def test_glm_with_offsets(self):
        from edgepython.utils import scale_offset
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        gene_lengths = np.linspace(500, 11000, y.shape[0])
        lib_sizes = y.sum(axis=0).astype(float)
        offset_mat = (np.log(gene_lengths)[:, None] +
                      np.log(lib_sizes)[None, :] - np.log(1e6))
        scaled_off = scale_offset(lib_sizes, offset_mat)

        d_off = dict(d)
        d_off['offset'] = scaled_off
        fit = ep.glm_fit(d_off, design=design, dispersion=0.1)
        lrt = ep.glm_lrt(fit, coef=1)
        tt = ep.top_tags(lrt, n=10)['table']
        # R: Tag.17 logFC=1.9153, PValue=0.004224
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.9153) < 0.05
        assert abs(top['PValue'] - 0.004224) < 0.002

    def test_glm_with_offset_deviance(self):
        from edgepython.utils import scale_offset
        y = pd.read_csv(f"{CSV_DIR}/test_data_part1.csv").values
        design = np.column_stack([np.ones(4), np.array([0, 0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 2, 2]))
        gene_lengths = np.linspace(500, 11000, y.shape[0])
        lib_sizes = y.sum(axis=0).astype(float)
        offset_mat = (np.log(gene_lengths)[:, None] +
                      np.log(lib_sizes)[None, :] - np.log(1e6))
        scaled_off = scale_offset(lib_sizes, offset_mat)

        d_off = dict(d)
        d_off['offset'] = scaled_off
        fit = ep.glm_fit(d_off, design=design, dispersion=0.1)
        # R deviance: [0, 0.005708, 2.440329, 9.260947, 2.588451]
        expected_dev = [0, 0.005708, 2.440329, 9.260947, 2.588451]
        assert np.allclose(fit['deviance'][:5], expected_dev, atol=0.01)


# ── Paired/blocked design ──────────────────────────────────────────

class TestPairedDesign:
    """Paired/blocked design (50 genes x 12 samples)."""

    @pytest.fixture(scope="class")
    def d34(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part34.csv").values
        patient = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        condition = np.array(['Control'] * 6 + ['Treated'] * 6)
        design = np.column_stack([
            np.ones(12),
            (patient == 2).astype(float),
            (patient == 3).astype(float),
            (patient == 4).astype(float),
            (patient == 5).astype(float),
            (patient == 6).astype(float),
            (condition == 'Treated').astype(float)
        ])
        d = ep.make_dgelist(counts=y, group=condition)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        return d, design

    def test_common_dispersion(self, d34):
        d, design = d34
        # R: 0.1834267
        assert abs(d['common.dispersion'] - 0.1834267) < 0.01

    def test_lrt_treatment_top_gene(self, d34):
        d, design = d34
        fit = ep.glm_fit(d, design=design)
        lrt = ep.glm_lrt(fit, coef=6)
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - (-1.2881)) < 0.05
        assert top['PValue'] < 1e-3

    def test_ql_treatment_top_gene(self, d34):
        d, design = d34
        fit_ql = ep.glm_ql_fit(d, design=design)
        qlf = ep.glm_ql_ftest(fit_ql, coef=6)
        tt = ep.top_tags(qlf, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['F'] - 17.77) < 0.5
        assert abs(top['PValue'] - 0.000903) < 0.001


# ── glmQLFit options ────────────────────────────────────────────────

class TestGlmQLFitOptions:
    """glmQLFit with various option combinations."""

    @pytest.fixture(scope="class")
    def d_p38(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        return d, design

    def test_abundance_trend_true(self, d_p38):
        d, design = d_p38
        fit = ep.glm_ql_fit(d, design=design, abundance_trend=True)
        dp = np.atleast_1d(fit.get('df.prior', 0))
        assert abs(np.median(dp) - 12.183) < 1.0

    def test_legacy_true(self, d_p38):
        d, design = d_p38
        fit = ep.glm_ql_fit(d, design=design, legacy=True)
        dp = np.atleast_1d(fit.get('df.prior', 0))
        assert abs(np.median(dp) - 2.1826) < 0.1


# ── Large-scale robust ─────────────────────────────────────────────

class TestLargeScaleRobust:
    """Large-scale robust estimation (500 genes x 6 samples)."""

    @pytest.fixture(scope="class")
    def d41(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part41.csv").values
        group = np.array([1, 1, 1, 2, 2, 2])
        design = np.column_stack([np.ones(6),
                                   np.array([0, 0, 0, 1, 1, 1], dtype=float)])
        return y, group, design

    def test_non_robust_disp(self, d41):
        y, group, design = d41
        d = ep.make_dgelist(counts=y, group=group)
        d = ep.estimate_disp(d, design, robust=False)
        # R: common=0.2436
        assert abs(d['common.dispersion'] - 0.2436) < 0.01
        pdf = d.get('prior.df')
        if pdf is not None:
            assert abs(np.median(np.atleast_1d(pdf)) - 24.86) < 2

    def test_robust_ql_top_ranking(self, d41):
        y, group, design = d41
        d = ep.make_dgelist(counts=y, group=group)
        d = ep.estimate_disp(d, design, robust=True)
        fit = ep.glm_ql_fit(d, design=design, robust=True)
        qlf = ep.glm_ql_ftest(fit, coef=1)
        tt = ep.top_tags(qlf, n=10)['table']
        top_genes = tt.index.tolist()
        gene32_names = [g for g in top_genes[:3] if '32' in str(g)]
        assert len(gene32_names) > 0 or tt.iloc[0]['PValue'] < 1e-6


# ── Contrast-based testing ─────────────────────────────────────────

class TestContrastTesting:
    """Contrast-based testing (3-group, 50 genes x 9 samples)."""

    @pytest.fixture(scope="class")
    def fits42(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part42.csv").values
        group = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        design = np.column_stack([
            np.ones(9),
            (group == 2).astype(float),
            (group == 3).astype(float)
        ])
        d = ep.make_dgelist(counts=y, group=group)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        return fit, fit_ql, design

    def test_lrt_group2_vs_group3(self, fits42):
        fit, fit_ql, design = fits42
        lrt = ep.glm_lrt(fit, contrast=np.array([0, 1, -1]))
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - (-2.960)) < 0.1
        assert top['PValue'] < 1e-5

    def test_ql_group2_vs_group3(self, fits42):
        fit, fit_ql, design = fits42
        qlf = ep.glm_ql_ftest(fit_ql, contrast=np.array([0, 1, -1]))
        tt = ep.top_tags(qlf, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['F'] - 23.91) < 1.0
        assert top['PValue'] < 1e-4

    def test_contrast_matches_coef(self, fits42):
        fit, fit_ql, design = fits42
        lrt_contrast = ep.glm_lrt(fit, contrast=np.array([0, 0, 1]))
        lrt_coef = ep.glm_lrt(fit, coef=2)
        assert np.allclose(
            lrt_contrast['table']['LR'].values,
            lrt_coef['table']['LR'].values, atol=0.01)


# ── Interaction design ─────────────────────────────────────────────

class TestInteraction:
    """Multi-factor design with interaction (2x2 factorial)."""

    @pytest.fixture(scope="class")
    def fit43(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part43.csv").values
        geno = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
        treat = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float)
        design = np.column_stack([np.ones(12), geno, treat, geno * treat])
        d = ep.make_dgelist(counts=y)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        return fit, d, design

    def test_interaction_top_gene(self, fit43):
        fit, d, design = fit43
        lrt = ep.glm_lrt(fit, coef=3)
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.837) < 0.1
        assert abs(top['PValue'] - 0.00569) < 0.005

    def test_genotype_top_gene(self, fit43):
        fit, d, design = fit43
        lrt = ep.glm_lrt(fit, coef=1)
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - (-1.727)) < 0.1
        assert abs(top['PValue'] - 0.00106) < 0.005

    def test_treatment_top_gene(self, fit43):
        fit, d, design = fit43
        lrt = ep.glm_lrt(fit, coef=2)
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - 1.935) < 0.1
        assert abs(top['PValue'] - 4.72e-4) < 5e-4


# ── Custom contrasts (average of groups) ───────────────────────────

class TestCustomContrasts:
    """Custom contrasts: average of groups vs reference."""

    @pytest.fixture(scope="class")
    def fits44(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part42.csv").values
        group = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        design = np.column_stack([
            np.ones(9),
            (group == 2).astype(float),
            (group == 3).astype(float)
        ])
        d = ep.make_dgelist(counts=y, group=group)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design=design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        return fit, fit_ql

    def test_lrt_avg_vs_group1(self, fits44):
        fit, fit_ql = fits44
        lrt = ep.glm_lrt(fit, contrast=np.array([-1, 0.5, 0.5]))
        tt = ep.top_tags(lrt, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['logFC'] - 7.370) < 0.5
        assert top['PValue'] < 1e-9

    def test_ql_avg_vs_group1(self, fits44):
        fit, fit_ql = fits44
        qlf = ep.glm_ql_ftest(fit_ql, contrast=np.array([-1, 0.5, 0.5]))
        tt = ep.top_tags(qlf, n=10)['table']
        top = tt.iloc[0]
        assert abs(top['F'] - 45.49) < 2.0
        assert top['PValue'] < 1e-8
