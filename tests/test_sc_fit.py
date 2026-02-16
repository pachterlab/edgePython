# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for glm_sc_fit and glm_sc_test (NEBULA-LN single-cell mixed model)."""

import numpy as np
import pandas as pd
import pytest

import edgepython as ep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.RandomState(42)


@pytest.fixture(scope="module")
def sc_data(rng):
    """Synthetic single-cell data: 200 genes x 600 cells, 6 subjects, 2 groups.

    First 20 genes are upregulated in group B (3x fold-change).
    """
    ngenes = 200
    ncells = 600
    nsubjects = 6
    cells_per_subject = ncells // nsubjects  # 100

    # Subject IDs and group assignment
    # Subjects 1-3 = group A, subjects 4-6 = group B
    sample_ids = np.repeat([f"S{i}" for i in range(1, nsubjects + 1)],
                           cells_per_subject)
    group = np.repeat([0, 0, 0, 1, 1, 1], cells_per_subject)

    # Base expression (genes x cells)
    base_mean = rng.uniform(1, 20, size=ngenes)
    counts = np.zeros((ngenes, ncells), dtype=np.float64)
    for g in range(ngenes):
        mu = base_mean[g]
        if g < 20:
            # Upregulated in group B
            mu_vec = np.where(group == 1, mu * 3, mu)
        else:
            mu_vec = np.full(ncells, mu)
        # Add subject-level random effect (small)
        subject_effect = np.repeat(rng.normal(0, 0.2, nsubjects),
                                   cells_per_subject)
        mu_cell = mu_vec * np.exp(subject_effect)
        counts[g, :] = rng.poisson(np.maximum(mu_cell, 0.1))

    return {
        'counts': counts,
        'sample_ids': sample_ids,
        'group': group,
        'ngenes': ngenes,
        'ncells': ncells,
        'nsubjects': nsubjects,
    }


@pytest.fixture(scope="module")
def intercept_fit(sc_data):
    """Intercept-only fit."""
    return ep.glm_sc_fit(
        sc_data['counts'],
        sample=sc_data['sample_ids'],
        norm_method='none',
        verbose=False,
    )


@pytest.fixture(scope="module")
def twogp_design(sc_data):
    """Two-group design matrix (intercept + group)."""
    ncells = sc_data['ncells']
    return pd.DataFrame({
        'Intercept': np.ones(ncells),
        'groupB': sc_data['group'].astype(float),
    }, columns=['Intercept', 'groupB'])


@pytest.fixture(scope="module")
def twogp_fit(sc_data, twogp_design):
    """Two-group fit with TMM normalization."""
    return ep.glm_sc_fit(
        sc_data['counts'],
        design=twogp_design,
        sample=sc_data['sample_ids'],
        norm_method='TMM',
        verbose=False,
    )


# ---------------------------------------------------------------------------
# glm_sc_fit: output structure
# ---------------------------------------------------------------------------

class TestGlmScFitStructure:
    """Test that glm_sc_fit returns all expected keys with correct shapes."""

    def test_required_keys(self, intercept_fit):
        required = [
            'coefficients', 'se', 'dispersion', 'sigma_sample',
            'convergence', 'design', 'offset', 'genes', 'gene_mask',
            'method', 'predictor_names', 'ncells', 'nsamples',
        ]
        for key in required:
            assert key in intercept_fit, f"Missing key: {key}"

    def test_method_tag(self, intercept_fit):
        assert intercept_fit['method'] == 'nebula_ln'

    def test_coef_shape_intercept(self, intercept_fit):
        ngenes = intercept_fit['coefficients'].shape[0]
        assert intercept_fit['coefficients'].shape == (ngenes, 1)
        assert intercept_fit['se'].shape == (ngenes, 1)

    def test_coef_shape_twogp(self, twogp_fit):
        ngenes = twogp_fit['coefficients'].shape[0]
        assert twogp_fit['coefficients'].shape == (ngenes, 2)
        assert twogp_fit['se'].shape == (ngenes, 2)

    def test_vector_shapes(self, twogp_fit):
        ngenes = twogp_fit['coefficients'].shape[0]
        assert twogp_fit['dispersion'].shape == (ngenes,)
        assert twogp_fit['sigma_sample'].shape == (ngenes,)
        assert twogp_fit['convergence'].shape == (ngenes,)

    def test_ncells_nsamples(self, intercept_fit, sc_data):
        assert intercept_fit['ncells'] == sc_data['ncells']
        assert intercept_fit['nsamples'] == sc_data['nsubjects']

    def test_predictor_names_intercept(self, intercept_fit):
        # When no design is provided, auto-generated intercept column is 'V1'
        assert len(intercept_fit['predictor_names']) == 1

    def test_predictor_names_twogp(self, twogp_fit):
        assert twogp_fit['predictor_names'] == ['Intercept', 'groupB']


# ---------------------------------------------------------------------------
# glm_sc_fit: numerical correctness
# ---------------------------------------------------------------------------

class TestGlmScFitValues:
    """Test that fitted values are numerically sensible."""

    def test_convergence(self, twogp_fit):
        conv = twogp_fit['convergence']
        frac = (conv == 1).sum() / len(conv)
        assert frac > 0.8, f"Only {frac:.0%} of genes converged"

    def test_no_nan_coefficients(self, twogp_fit):
        conv_mask = twogp_fit['convergence'] == 1
        assert not np.any(np.isnan(
            twogp_fit['coefficients'][conv_mask]
        )), "NaN coefficients in converged genes"

    def test_se_positive(self, twogp_fit):
        conv_mask = twogp_fit['convergence'] == 1
        se = twogp_fit['se'][conv_mask]
        assert np.all(se > 0), "Non-positive SE in converged genes"

    def test_dispersion_positive(self, twogp_fit):
        conv_mask = twogp_fit['convergence'] == 1
        disp = twogp_fit['dispersion'][conv_mask]
        assert np.all(disp > 0), "Non-positive dispersion in converged genes"

    def test_sigma_nonneg(self, twogp_fit):
        conv_mask = twogp_fit['convergence'] == 1
        sigma = twogp_fit['sigma_sample'][conv_mask]
        assert np.all(sigma >= 0), "Negative sigma_sample"

    def test_de_genes_detected(self, twogp_fit, sc_data):
        """First 20 genes should have large positive groupB coefficient."""
        gene_mask = twogp_fit['gene_mask']
        # Map original gene indices through the filter
        original_indices = np.where(gene_mask)[0]
        groupB_coefs = twogp_fit['coefficients'][:, 1]

        # Find where original DE genes ended up after filtering
        de_coefs = []
        for orig_idx in range(20):
            filtered_pos = np.where(original_indices == orig_idx)[0]
            if len(filtered_pos) > 0:
                de_coefs.append(groupB_coefs[filtered_pos[0]])

        if len(de_coefs) > 0:
            mean_de_logfc = np.nanmean(de_coefs)
            # 3x fold-change = log(3) ~ 1.1 on log scale
            assert mean_de_logfc > 0.5, (
                f"Mean logFC for DE genes = {mean_de_logfc:.3f}, expected > 0.5"
            )


# ---------------------------------------------------------------------------
# glm_sc_fit: gene filtering
# ---------------------------------------------------------------------------

class TestGlmScFitFiltering:
    """Test gene filtering behaviour."""

    def test_gene_mask_shape(self, intercept_fit, sc_data):
        assert intercept_fit['gene_mask'].shape == (sc_data['ngenes'],)

    def test_gene_mask_dtype(self, intercept_fit):
        assert intercept_fit['gene_mask'].dtype == bool

    def test_filtered_count_consistent(self, intercept_fit):
        ngenes_kept = intercept_fit['gene_mask'].sum()
        assert intercept_fit['coefficients'].shape[0] == ngenes_kept

    def test_strict_filtering(self, sc_data):
        """With very strict filtering, fewer genes pass."""
        fit = ep.glm_sc_fit(
            sc_data['counts'],
            sample=sc_data['sample_ids'],
            norm_method='none',
            cpc=5.0,  # very strict
            verbose=False,
        )
        n_strict = fit['gene_mask'].sum()
        assert n_strict < sc_data['ngenes']


# ---------------------------------------------------------------------------
# glm_sc_fit: normalization
# ---------------------------------------------------------------------------

class TestGlmScFitNorm:
    """Test normalization options."""

    def test_norm_none(self, intercept_fit):
        """norm_method='none' should still produce valid offsets."""
        assert intercept_fit['offset'] is not None
        assert not np.any(np.isnan(intercept_fit['offset']))

    def test_norm_tmm(self, twogp_fit):
        """TMM normalization should produce valid offsets."""
        assert twogp_fit['offset'] is not None
        assert not np.any(np.isnan(twogp_fit['offset']))
        # TMM offsets should differ from uniform
        assert np.std(twogp_fit['offset']) > 0


# ---------------------------------------------------------------------------
# glm_sc_fit: input validation
# ---------------------------------------------------------------------------

class TestGlmScFitValidation:

    def test_missing_sample_raises(self, sc_data):
        with pytest.raises(ValueError, match="sample"):
            ep.glm_sc_fit(sc_data['counts'], verbose=False)

    def test_sample_length_mismatch(self, sc_data):
        with pytest.raises(ValueError, match="[Ll]ength"):
            ep.glm_sc_fit(
                sc_data['counts'],
                sample=sc_data['sample_ids'][:10],
                verbose=False,
            )


# ---------------------------------------------------------------------------
# glm_sc_test
# ---------------------------------------------------------------------------

class TestGlmScTest:
    """Test glm_sc_test Wald test."""

    def test_output_structure(self, twogp_fit):
        result = ep.glm_sc_test(twogp_fit, coef=1)
        assert 'table' in result
        tab = result['table']
        for col in ['logFC', 'SE', 'z', 'PValue', 'FDR',
                     'sigma_sample', 'dispersion', 'converged']:
            assert col in tab.columns, f"Missing column: {col}"

    def test_default_coef_is_last(self, twogp_fit):
        """Default coef should be the last column (groupB)."""
        result_default = ep.glm_sc_test(twogp_fit)
        result_explicit = ep.glm_sc_test(twogp_fit, coef=1)
        np.testing.assert_array_equal(
            result_default['table']['logFC'].values,
            result_explicit['table']['logFC'].values,
        )

    def test_coef_intercept(self, twogp_fit):
        """Testing intercept (coef=0) should work."""
        result = ep.glm_sc_test(twogp_fit, coef=0)
        tab = result['table']
        assert len(tab) == twogp_fit['coefficients'].shape[0]
        # Intercept should be mostly significant (nonzero expression)
        assert (tab['PValue'] < 0.05).sum() > len(tab) * 0.5

    def test_pvalue_range(self, twogp_fit):
        result = ep.glm_sc_test(twogp_fit, coef=1)
        pvals = result['table']['PValue'].dropna()
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_fdr_range(self, twogp_fit):
        result = ep.glm_sc_test(twogp_fit, coef=1)
        fdr = result['table']['FDR'].dropna()
        assert np.all(fdr >= 0)
        assert np.all(fdr <= 1)

    def test_fdr_ge_pvalue(self, twogp_fit):
        """FDR should be >= raw p-value for each gene."""
        result = ep.glm_sc_test(twogp_fit, coef=1)
        tab = result['table'].dropna(subset=['PValue', 'FDR'])
        assert np.all(tab['FDR'].values >= tab['PValue'].values - 1e-15)

    def test_z_equals_logfc_over_se(self, twogp_fit):
        result = ep.glm_sc_test(twogp_fit, coef=1)
        tab = result['table']
        expected_z = tab['logFC'] / tab['SE']
        np.testing.assert_allclose(tab['z'].values, expected_z.values,
                                   rtol=1e-10)

    def test_contrast(self, twogp_fit):
        """Test with a custom contrast vector."""
        contrast = np.array([0.0, 1.0])
        result = ep.glm_sc_test(twogp_fit, contrast=contrast)
        tab = result['table']
        # Should be equivalent to coef=1
        result2 = ep.glm_sc_test(twogp_fit, coef=1)
        np.testing.assert_allclose(
            tab['logFC'].values,
            result2['table']['logFC'].values,
            rtol=1e-10,
        )

    def test_contrast_sum(self, twogp_fit):
        """Contrast [1, 1] should give intercept + groupB."""
        contrast = np.array([1.0, 1.0])
        result = ep.glm_sc_test(twogp_fit, contrast=contrast)
        tab = result['table']
        expected_logfc = twogp_fit['coefficients'] @ contrast
        np.testing.assert_allclose(tab['logFC'].values, expected_logfc,
                                   rtol=1e-10)

    def test_nrow_matches_ngenes(self, twogp_fit):
        result = ep.glm_sc_test(twogp_fit, coef=1)
        assert len(result['table']) == twogp_fit['coefficients'].shape[0]


# ---------------------------------------------------------------------------
# top_tags integration with SC fits
# ---------------------------------------------------------------------------

class TestTopTagsSC:
    """Test that top_tags works correctly on glm_sc_fit results."""

    def test_top_tags_with_coef_int(self, twogp_fit):
        tt = ep.top_tags(twogp_fit, n=10, coef=1)
        assert 'table' in tt
        assert len(tt['table']) == 10
        assert tt['test'] == 'wald'

    def test_top_tags_with_coef_name(self, twogp_fit):
        tt = ep.top_tags(twogp_fit, n=10, coef='groupB')
        assert len(tt['table']) == 10

    def test_top_tags_sorted_by_pvalue(self, twogp_fit):
        ngenes = twogp_fit['coefficients'].shape[0]
        tt = ep.top_tags(twogp_fit, n=ngenes, coef=1)
        pvals = tt['table']['PValue'].values
        # Should be sorted ascending (with NaN at end)
        valid = ~np.isnan(pvals)
        valid_pvals = pvals[valid]
        assert np.all(valid_pvals[:-1] <= valid_pvals[1:] + 1e-15)

    def test_top_tags_fdr_column(self, twogp_fit):
        tt = ep.top_tags(twogp_fit, n=10, coef=1)
        assert 'FDR' in tt['table'].columns

    def test_top_tags_default_coef_is_last(self, twogp_fit):
        """Default coef for SC results should be last column."""
        tt_default = ep.top_tags(twogp_fit, n=5)
        tt_explicit = ep.top_tags(twogp_fit, n=5, coef=1)
        np.testing.assert_allclose(
            tt_default['table']['logFC'].values,
            tt_explicit['table']['logFC'].values,
            rtol=1e-10,
        )

    def test_top_tags_p_value_filter(self, twogp_fit):
        ngenes = twogp_fit['coefficients'].shape[0]
        tt = ep.top_tags(twogp_fit, n=ngenes, coef=1, p_value=0.05)
        if len(tt['table']) > 0:
            assert np.all(tt['table']['FDR'].values <= 0.05 + 1e-15)

    def test_top_tags_with_gene_names(self, sc_data, twogp_design):
        """Verify gene names propagate through top_tags."""
        gene_names = [f"Gene_{i}" for i in range(sc_data['ngenes'])]
        fit = ep.glm_sc_fit(
            sc_data['counts'],
            design=twogp_design,
            sample=sc_data['sample_ids'],
            norm_method='none',
            verbose=False,
        )
        gene_mask = fit['gene_mask']
        fit['genes'] = pd.DataFrame(
            {'gene': np.array(gene_names)[gene_mask]}
        )
        tt = ep.top_tags(fit, n=5, coef=1)
        assert 'gene' in tt['table'].columns


# ---------------------------------------------------------------------------
# glm_sc_test and top_tags consistency
# ---------------------------------------------------------------------------

class TestScTestTopTagsConsistency:
    """Verify glm_sc_test and top_tags give the same Wald statistics."""

    def test_same_pvalues(self, twogp_fit):
        """The raw p-values should match between glm_sc_test and top_tags."""
        test_result = ep.glm_sc_test(twogp_fit, coef=1)
        ngenes = twogp_fit['coefficients'].shape[0]
        tt = ep.top_tags(twogp_fit, n=ngenes, coef=1, sort_by='none')

        np.testing.assert_allclose(
            test_result['table']['PValue'].values,
            tt['table']['PValue'].values,
            rtol=1e-10,
        )

    def test_same_logfc(self, twogp_fit):
        test_result = ep.glm_sc_test(twogp_fit, coef=1)
        ngenes = twogp_fit['coefficients'].shape[0]
        tt = ep.top_tags(twogp_fit, n=ngenes, coef=1, sort_by='none')

        np.testing.assert_allclose(
            test_result['table']['logFC'].values,
            tt['table']['logFC'].values,
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# shrink_sc_disp
# ---------------------------------------------------------------------------

class TestShrinkScDisp:
    """Test shrink_sc_disp empirical Bayes shrinkage of cell-level NB dispersion."""

    def _shrink(self, fit, **kwargs):
        """Helper: copy fit and apply shrinkage."""
        import copy
        f = copy.deepcopy(fit)
        return ep.shrink_sc_disp(f, **kwargs)

    def test_output_keys(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        for key in ['phi_raw', 'phi_post', 'phi_prior', 'df_residual',
                     'df_prior_phi', 'dispersion_shrunk']:
            assert key in fit, f"Missing key: {key}"

    def test_phi_raw_is_reciprocal_of_dispersion(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        conv = fit['convergence'] == 1
        finite = np.isfinite(fit['phi_raw'][conv])
        expected = 1.0 / fit['dispersion'][conv][finite]
        np.testing.assert_allclose(
            fit['phi_raw'][conv][finite], expected, rtol=1e-10
        )

    def test_dispersion_shrunk_is_reciprocal_of_phi_post(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        conv = fit['convergence'] == 1
        pos = fit['phi_post'][conv] > 0
        expected = 1.0 / fit['phi_post'][conv][pos]
        np.testing.assert_allclose(
            fit['dispersion_shrunk'][conv][pos], expected, rtol=1e-10
        )

    def test_phi_post_positive(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        conv = fit['convergence'] == 1
        assert np.all(fit['phi_post'][conv] > 0)

    def test_shrinkage_reduces_variance(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        conv = fit['convergence'] == 1
        phi_raw_conv = fit['phi_raw'][conv]
        phi_post_conv = fit['phi_post'][conv]
        finite = np.isfinite(phi_raw_conv) & (phi_raw_conv > 0)
        raw_var = np.var(np.log(phi_raw_conv[finite]))
        post_var = np.var(np.log(phi_post_conv[finite]))
        assert post_var < raw_var, (
            f"Shrinkage did not reduce variance: {post_var:.4f} >= {raw_var:.4f}"
        )

    def test_df_residual_correct(self, twogp_fit, sc_data):
        fit = self._shrink(twogp_fit)
        expected_df = (sc_data['ncells']
                       - twogp_fit['design'].shape[1]
                       - (sc_data['nsubjects'] - 1))
        assert fit['df_residual'] == expected_df

    def test_df_prior_positive(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        df_prior = fit['df_prior_phi']
        if np.isscalar(df_prior):
            assert df_prior > 0 or np.isinf(df_prior)
        else:
            assert np.all(df_prior[np.isfinite(df_prior)] > 0)

    def test_with_covariate(self, twogp_fit):
        fit = self._shrink(twogp_fit)
        assert 'phi_post' in fit
        assert np.all(np.isfinite(fit['phi_post'][fit['convergence'] == 1]))

    def test_no_covariate(self, twogp_fit):
        import copy
        f = copy.deepcopy(twogp_fit)
        f.pop('ave_log_abundance', None)
        ep.shrink_sc_disp(f, covariate=None, counts=None)
        assert 'phi_post' in f
        assert np.all(np.isfinite(f['phi_post'][f['convergence'] == 1]))

    def test_robust_vs_nonrobust(self, twogp_fit):
        fit_r = self._shrink(twogp_fit, robust=True)
        fit_nr = self._shrink(twogp_fit, robust=False)
        conv = fit_r['convergence'] == 1
        assert np.all(np.isfinite(fit_r['phi_post'][conv]))
        assert np.all(np.isfinite(fit_nr['phi_post'][conv]))

    def test_original_fields_unchanged(self, twogp_fit):
        import copy
        f = copy.deepcopy(twogp_fit)
        orig_disp = f['dispersion'].copy()
        orig_coef = f['coefficients'].copy()
        orig_se = f['se'].copy()
        ep.shrink_sc_disp(f)
        np.testing.assert_array_equal(f['dispersion'], orig_disp)
        np.testing.assert_array_equal(f['coefficients'], orig_coef)
        np.testing.assert_array_equal(f['se'], orig_se)
