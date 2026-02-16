# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for gene set testing: camera, fry, roast, mroast, romer, goana, kegga."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── conftest fixtures used: dgelist_disp, design6, gene_sets, small_counts ──


# ======================================================================
# Camera (conftest fixtures)
# ======================================================================

class TestCamera:
    """Tests for camera()."""

    def test_camera_dgelist_returns_dataframe(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {'NGenes', 'Direction', 'PValue', 'FDR'}

    def test_camera_ngenes_correct(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6, contrast=1)
        for name, indices in gene_sets.items():
            assert result.loc[name, 'NGenes'] == len(indices)

    def test_camera_direction_values(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert all(d in ('Up', 'Down') for d in result['Direction'])

    def test_camera_pvalues_valid(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert (result['PValue'] >= 0).all() and (result['PValue'] <= 1).all()
        assert (result['FDR'] >= 0).all() and (result['FDR'] <= 1).all()

    def test_camera_logcpm_matrix(self, dgelist_disp, design6, gene_sets):
        logcpm = ep.cpm(dgelist_disp, log=True)
        result = ep.camera(logcpm, gene_sets, design=design6, contrast=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gene_sets)

    def test_camera_use_ranks(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6,
                           contrast=1, use_ranks=True)
        assert isinstance(result, pd.DataFrame)

    def test_camera_inter_gene_cor_zero(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6,
                           contrast=1, inter_gene_cor=0)
        assert isinstance(result, pd.DataFrame)

    def test_camera_unsorted(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6,
                           contrast=1, sort=False)
        assert list(result.index) == list(gene_sets.keys())

    def test_camera_sorted_by_pvalue(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6,
                           contrast=1, sort=True)
        assert list(result['PValue']) == sorted(result['PValue'])

    def test_camera_contrast_vector(self, dgelist_disp, design6, gene_sets):
        result = ep.camera(dgelist_disp, gene_sets, design=design6,
                           contrast=np.array([0.0, 1.0]))
        assert isinstance(result, pd.DataFrame)


# ======================================================================
# Camera (Part 20 data - 1000 genes x 3 samples)
# ======================================================================

class TestCameraPart20:
    """camera gene set tests with 1000-gene data."""

    @pytest.fixture(scope="class")
    def d2_camera(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part2.csv").values
        design = np.column_stack([np.ones(3), np.array([0, 1, 2])])
        d = ep.make_dgelist(counts=y)
        d = ep.calc_norm_factors(d)
        d = ep.estimate_disp(d, design)
        fit = ep.glm_ql_fit(d, design)
        gene_sets = {
            'set1': list(range(0, 50)),
            'set2': list(range(50, 150)),
            'set3': list(range(0, 20)) + list(range(500, 520)),
            'set4': list(range(900, 1000)),
        }
        return fit, design, gene_sets

    def test_camera_default(self, d2_camera):
        fit, design, gene_sets = d2_camera
        cam = ep.camera(fit, gene_sets, design=design, contrast=1)
        assert cam.loc['set3', 'Direction'] == 'Up'
        assert abs(cam.loc['set3', 'PValue'] - 0.1140) < 0.05

    def test_camera_fixed_cor(self, d2_camera):
        fit, design, gene_sets = d2_camera
        cam = ep.camera(fit, gene_sets, design=design, contrast=1,
                        inter_gene_cor=0.05)
        assert cam.loc['set3', 'Direction'] == 'Up'
        assert abs(cam.loc['set3', 'PValue'] - 0.2741) < 0.05

    def test_camera_ranks(self, d2_camera):
        fit, design, gene_sets = d2_camera
        cam = ep.camera(fit, gene_sets, design=design, contrast=1,
                        use_ranks=True)
        assert cam.loc['set3', 'Direction'] == 'Up'
        assert cam.loc['set3', 'PValue'] < 0.5


# ======================================================================
# Fry (conftest fixtures)
# ======================================================================

class TestFry:
    """Tests for fry()."""

    def test_fry_returns_dataframe(self, dgelist_disp, design6, gene_sets):
        result = ep.fry(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert isinstance(result, pd.DataFrame)
        assert 'PValue' in result.columns
        assert 'Direction' in result.columns

    def test_fry_has_mixed_pvalue(self, dgelist_disp, design6, gene_sets):
        result = ep.fry(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert 'PValue.Mixed' in result.columns
        assert 'FDR.Mixed' in result.columns

    def test_fry_pvalues_valid(self, dgelist_disp, design6, gene_sets):
        result = ep.fry(dgelist_disp, gene_sets, design=design6, contrast=1)
        assert (result['PValue'] >= 0).all() and (result['PValue'] <= 1).all()
        assert (result['PValue.Mixed'] >= 0).all() and (result['PValue.Mixed'] <= 1).all()

    def test_fry_ngenes_correct(self, dgelist_disp, design6, gene_sets):
        result = ep.fry(dgelist_disp, gene_sets, design=design6, contrast=1)
        for name, indices in gene_sets.items():
            assert result.loc[name, 'NGenes'] == len(indices)

    def test_fry_single_set(self, dgelist_disp, design6):
        result = ep.fry(dgelist_disp, {'OnlySet': list(range(20))},
                        design=design6, contrast=1)
        assert len(result) == 1


# ======================================================================
# Fry/Roast stubs (Part 36 data)
# ======================================================================

class TestGeneSetStubs:
    """fry and roast stub implementations."""

    def test_fry_runs(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=np.array([1, 2, 2]))
        d = ep.estimate_disp(d, design)
        gene_sets = {'set1': list(range(0, 20)),
                     'set2': list(range(50, 80))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ep.fry(d, gene_sets, design=design, contrast=1)
        assert isinstance(result, pd.DataFrame)
        assert 'NGenes' in result.columns
        assert 'Direction' in result.columns


# ======================================================================
# Roast (conftest fixtures)
# ======================================================================

class TestRoast:
    """Tests for roast()."""

    def test_roast_returns_dataframe(self, dgelist_disp, design6):
        np.random.seed(42)
        result = ep.roast(dgelist_disp, list(range(10)),
                          design=design6, contrast=1, nrot=99)
        assert isinstance(result, pd.DataFrame)

    def test_roast_has_expected_rows(self, dgelist_disp, design6):
        np.random.seed(42)
        result = ep.roast(dgelist_disp, list(range(10)),
                          design=design6, contrast=1, nrot=99)
        expected_rows = {'Down', 'Up', 'UpOrDown', 'Mixed'}
        assert expected_rows <= set(result.index)

    def test_roast_has_pvalue_and_proportion(self, dgelist_disp, design6):
        np.random.seed(42)
        result = ep.roast(dgelist_disp, list(range(10)),
                          design=design6, contrast=1, nrot=99)
        assert 'P.Value' in result.columns
        assert 'Active.Prop' in result.columns

    def test_roast_pvalues_valid(self, dgelist_disp, design6):
        np.random.seed(42)
        result = ep.roast(dgelist_disp, list(range(10)),
                          design=design6, contrast=1, nrot=99)
        assert (result['P.Value'] >= 0).all() and (result['P.Value'] <= 1).all()

    def test_roast_with_dict_index(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.roast(dgelist_disp, gene_sets,
                          design=design6, contrast=1, nrot=99)
        assert isinstance(result, pd.DataFrame)


# ======================================================================
# Mroast (conftest fixtures)
# ======================================================================

class TestMroast:
    """Tests for mroast()."""

    def test_mroast_returns_dataframe(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.mroast(dgelist_disp, gene_sets,
                           design=design6, contrast=1, nrot=99)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(gene_sets)

    def test_mroast_columns(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.mroast(dgelist_disp, gene_sets,
                           design=design6, contrast=1, nrot=99)
        expected = {'NGenes', 'PropDown', 'PropUp', 'Direction', 'PValue',
                    'FDR', 'PValue.Mixed', 'FDR.Mixed'}
        assert expected <= set(result.columns)

    def test_mroast_pvalues_valid(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.mroast(dgelist_disp, gene_sets,
                           design=design6, contrast=1, nrot=99)
        assert (result['PValue'] >= 0).all() and (result['PValue'] <= 1).all()
        assert (result['FDR'] >= result['PValue']).all()

    def test_mroast_ngenes(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.mroast(dgelist_disp, gene_sets,
                           design=design6, contrast=1, nrot=99)
        for name, indices in gene_sets.items():
            assert result.loc[name, 'NGenes'] == len(indices)


# ======================================================================
# Romer (conftest fixtures)
# ======================================================================

class TestRomer:
    """Tests for romer()."""

    def test_romer_returns_dataframe(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.romer(dgelist_disp, gene_sets,
                          design=design6, contrast=1, nrot=99)
        assert isinstance(result, pd.DataFrame)

    def test_romer_columns(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.romer(dgelist_disp, gene_sets,
                          design=design6, contrast=1, nrot=99)
        assert {'Up', 'Down', 'Mixed'} <= set(result.columns)

    def test_romer_pvalues_valid(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.romer(dgelist_disp, gene_sets,
                          design=design6, contrast=1, nrot=99)
        for col in ['Up', 'Down', 'Mixed']:
            assert (result[col] >= 0).all() and (result[col] <= 1).all()

    def test_romer_ngenes(self, dgelist_disp, design6, gene_sets):
        np.random.seed(42)
        result = ep.romer(dgelist_disp, gene_sets,
                          design=design6, contrast=1, nrot=99)
        for name, indices in gene_sets.items():
            assert result.loc[name, 'NGenes'] == len(indices)


# ======================================================================
# Romer stub (Part 51 data)
# ======================================================================

class TestRomerStub:
    """Romer with small inline data."""

    def test_romer_stub(self):
        y = np.array([[10, 20, 30], [5, 15, 25],
                       [100, 200, 300]], dtype=float)
        d = ep.make_dgelist(counts=y, group=[1, 1, 2])
        d = ep.calc_norm_factors(d)
        design = np.column_stack([np.ones(3), [0, 0, 1]])
        d = ep.estimate_disp(d, design)
        fit = ep.glm_fit(d, design)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ep.romer(fit, {'set1': [0, 1, 2]},
                              design=design, contrast=1)
        assert isinstance(result, pd.DataFrame)


# ======================================================================
# goana / kegga (conftest fixtures)
# ======================================================================

class TestGoanaKegga:
    """Tests for goana() and kegga()."""

    def test_goana_returns_dataframe(self):
        result = ep.goana(None)
        assert isinstance(result, pd.DataFrame)

    def test_kegga_returns_dataframe(self):
        result = ep.kegga(None)
        assert isinstance(result, pd.DataFrame)


# ======================================================================
# goana / kegga stubs (Part 51 data)
# ======================================================================

class TestGoanaKeggaStubs:
    """goana/kegga stub implementations."""

    def test_goana_stub(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ep.goana(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_kegga_stub(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ep.kegga(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
