# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for visualization functions: plotBCV, plotMDS, plotQLDisp, plotSmear, plotMD, maPlot."""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestMDSComputation:
    """MDS distance matrix computation."""

    def test_mds_distance_properties(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        lib_size = y.sum(axis=0)
        log_cpm = np.log2(y / lib_size[None, :] * 1e6 + 0.5)
        nsamples = y.shape[1]
        var_ = np.var(log_cpm, axis=1)
        top_idx = np.argsort(var_)[::-1][:100]
        log_cpm_top = log_cpm[top_idx]

        dist_mat = np.zeros((nsamples, nsamples))
        for i in range(nsamples):
            for j in range(i + 1, nsamples):
                d_val = np.sqrt(np.mean(
                    (log_cpm_top[:, i] - log_cpm_top[:, j]) ** 2))
                dist_mat[i, j] = d_val
                dist_mat[j, i] = d_val
        # Distance matrix symmetric
        assert np.allclose(dist_mat, dist_mat.T)
        # All distances non-negative
        assert np.all(dist_mat >= 0)


class TestVisualizationSmoke:
    """Smoke tests for plotting functions."""

    @pytest.fixture(scope="class")
    def viz_data(self):
        import matplotlib.pyplot as plt
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3),
                                   np.array([0, 1, 1], dtype=float)])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit_ql = ep.glm_ql_fit(d, design=design)
        qlf = ep.glm_ql_ftest(fit_ql, coef=1)
        de = ep.exact_test(d)
        return d, fit_ql, qlf, de

    def test_plot_bcv(self, viz_data):
        import matplotlib.pyplot as plt
        d, fit_ql, qlf, de = viz_data
        fig, ax = ep.plot_bcv(d)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_mds(self, viz_data):
        import matplotlib.pyplot as plt
        d, fit_ql, qlf, de = viz_data
        fig, ax = ep.plot_mds(d)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_ql_disp(self, viz_data):
        import matplotlib.pyplot as plt
        d, fit_ql, qlf, de = viz_data
        fig, ax = ep.plot_ql_disp(fit_ql)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_smear(self, viz_data):
        import matplotlib.pyplot as plt
        d, fit_ql, qlf, de = viz_data
        fig, ax = ep.plot_smear(de)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_md(self, viz_data):
        import matplotlib.pyplot as plt
        d, fit_ql, qlf, de = viz_data
        fig, ax = ep.plot_md(qlf)
        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestMAPlot:
    """MA plot function."""

    def test_ma_plot(self):
        import matplotlib.pyplot as plt
        a = np.random.RandomState(42).randn(100) * 2 + 10
        m = np.random.RandomState(43).randn(100) * 0.5
        fig, ax = ep.ma_plot(a, m)
        assert fig is not None
        plt.close(fig)

    def test_ma_plot_with_de_tags(self):
        import matplotlib.pyplot as plt
        a = np.random.RandomState(42).randn(100) * 2 + 10
        m = np.random.RandomState(43).randn(100) * 0.5
        fig, ax = ep.ma_plot(a, m, de_tags=np.array([0, 1, 2, 3, 4]))
        assert fig is not None
        plt.close(fig)
