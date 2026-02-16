# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for edgePython class constructors and CompressedMatrix."""

import numpy as np
import pandas as pd
import pytest

import edgepython as ep


class TestCompressedMatrix:
    """CompressedMatrix construction and arithmetic."""

    def test_scalar(self):
        cm = ep.CompressedMatrix(5.0, dims=(3, 4))
        assert cm.shape == (3, 4)
        mat = cm.as_matrix()
        assert mat.shape == (3, 4)
        assert np.all(mat == 5.0)

    def test_row(self):
        cm = ep.CompressedMatrix(np.array([[1.0, 2.0, 3.0]]),
                                  dims=(4, 3))
        mat = cm.as_matrix()
        assert mat.shape == (4, 3)
        assert np.all(mat == mat[0])

    def test_col(self):
        cm = ep.CompressedMatrix(np.array([[10.0], [20.0], [30.0]]),
                                  dims=(3, 5), byrow=False)
        mat = cm.as_matrix()
        assert mat.shape == (3, 5)
        assert np.all(mat.T == mat[:, 0])

    def test_full(self):
        cm = ep.CompressedMatrix(np.array([[1, 2], [3, 4]], dtype=float))
        mat = cm.as_matrix()
        assert np.allclose(mat, [[1, 2], [3, 4]])

    def test_arithmetic(self):
        cm = ep.CompressedMatrix(5.0, dims=(3, 4))
        cm_add = cm + 1.0
        assert cm_add.as_matrix()[0, 0] == 6.0
        cm_mul = cm * 2.0
        assert cm_mul.as_matrix()[0, 0] == 10.0


class TestDGEListClass:
    """DGEList class construction and subsetting."""

    def test_dgelist(self):
        counts = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        samples = pd.DataFrame({
            'group': pd.Categorical([1, 2]),
            'lib.size': [90.0, 120.0],
            'norm.factors': [1.0, 1.0]
        })
        dge = ep.DGEList({'counts': counts, 'samples': samples})
        assert dge.nrow == 3
        assert dge.ncol == 2
        assert dge.shape == (3, 2)
        sub = dge[np.array([0, 2]), np.array([1])]
        assert sub['counts'].shape == (2, 1)


class TestDGEExactClass:
    """DGEExact class construction."""

    def test_dge_exact(self):
        table = pd.DataFrame({
            'logFC': [1.5, -0.5, 0.1],
            'logCPM': [10.0, 8.0, 12.0],
            'PValue': [0.001, 0.05, 0.9]
        }, index=['gene1', 'gene2', 'gene3'])
        de = ep.DGEExact({'table': table, 'comparison': [1, 2],
                          'genes': None})
        assert de['table'].shape == (3, 3)
        assert de['comparison'] == [1, 2]


class TestDGEGLMClass:
    """DGEGLM class construction."""

    def test_dgeglm(self):
        glm = ep.DGEGLM({
            'coefficients': np.array([[1.0, 0.5], [2.0, -0.3]]),
            'fitted.values': np.ones((2, 3)),
            'deviance': np.array([0.1, 0.2]),
            'design': np.array([[1, 0], [1, 0], [1, 1]]),
            'dispersion': 0.1,
            'df.residual': np.array([1.0, 1.0]),
        })
        assert glm['coefficients'].shape == (2, 2)


class TestDGELRTClass:
    """DGELRT class construction."""

    def test_dgelrt(self):
        lrt = ep.DGELRT({
            'table': pd.DataFrame({
                'logFC': [1.0, -0.5],
                'logCPM': [10.0, 8.0],
                'LR': [5.0, 1.0],
                'PValue': [0.025, 0.317]
            }),
            'coefficients': np.array([[1.0, 0.5], [2.0, -0.3]]),
            'comparison': 'group2',
        })
        assert lrt['table'].shape == (2, 4)


class TestTopTagsClass:
    """TopTags class construction."""

    def test_toptags(self):
        tt = ep.TopTags({
            'table': pd.DataFrame({
                'logFC': [1.5, -0.5],
                'logCPM': [10.0, 8.0],
                'PValue': [0.001, 0.05],
                'FDR': [0.01, 0.1]
            }),
            'adjust.method': 'BH',
            'comparison': [1, 2],
            'test': 'exact',
        })
        assert tt['table'].shape == (2, 4)
        assert tt['adjust.method'] == 'BH'


class TestWLEB:
    """WLEB (weighted likelihoods empirical Bayes)."""

    def test_wleb(self):
        theta_grid = np.linspace(-2, 2, 50)
        ntags = 20
        true_theta = np.linspace(-1, 1, ntags)
        loglik = np.zeros((ntags, len(theta_grid)))
        for i in range(ntags):
            loglik[i] = -0.5 * (theta_grid - true_theta[i]) ** 2
        covariate = np.linspace(0, 10, ntags)
        result = ep.WLEB(theta_grid, loglik, prior_n=5,
                          covariate=covariate, trend_method='locfit')
        assert 'overall' in result
        assert result['trend'].shape == (ntags,)
        assert result['individual'].shape == (ntags,)
