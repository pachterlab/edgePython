# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for readBismark2DGE, modelMatrixMeth, nearestTSS, nearestRefToX."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import edgepython as ep
from edgepython.utils import nearest_ref_to_x

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def bismark_dir():
    """Create temporary Bismark .cov files for testing."""
    tmpdir = tempfile.mkdtemp(prefix='bismark_test_')

    s1 = pd.DataFrame({
        0: ['chr1'] * 5 + ['chr2'] * 3,
        1: [100, 200, 300, 400, 500, 1000, 2000, 3000],
        2: [101, 201, 301, 401, 501, 1001, 2001, 3001],
        3: [50.0, 60.0, 40.0, 70.0, 55.0, 45.0, 65.0, 35.0],
        4: [10, 12, 8, 14, 11, 9, 13, 7],
        5: [10, 8, 12, 6, 9, 11, 7, 13],
    })

    s2 = pd.DataFrame({
        0: ['chr1'] * 5 + ['chr2'] * 2 + ['chr3'] * 2,
        1: [100, 200, 300, 400, 500, 1000, 2000, 500, 1500],
        2: [101, 201, 301, 401, 501, 1001, 2001, 501, 1501],
        3: [45.0, 55.0, 35.0, 65.0, 50.0, 40.0, 60.0, 30.0, 70.0],
        4: [9, 11, 7, 13, 10, 8, 12, 6, 14],
        5: [11, 9, 13, 7, 10, 12, 8, 14, 6],
    })

    f1 = os.path.join(tmpdir, 'sample1.cov')
    f2 = os.path.join(tmpdir, 'sample2.cov')
    s1.to_csv(f1, sep='\t', header=False, index=False)
    s2.to_csv(f2, sep='\t', header=False, index=False)

    yield tmpdir, [f1, f2]

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def tss_database():
    """Synthetic TSS database for nearest_tss tests."""
    return pd.DataFrame({
        'chr': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr3'],
        'tss': [1000, 5000, 9000, 2000, 7000, 3000],
        'gene_id': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
        'gene_name': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5', 'GENE6'],
        'strand': ['+', '-', '+', '+', '-', '+'],
        'width': [500, 800, 600, 700, 900, 400],
    })


# ======================================================================
# readBismark2DGE (self-contained)
# ======================================================================

class TestReadBismark2DGE:
    """Tests for read_bismark2dge()."""

    def test_returns_dgelist(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        assert isinstance(y, dict)
        assert 'counts' in y
        assert 'samples' in y
        assert 'genes' in y

    def test_column_count(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        assert y['counts'].shape[1] == 4

    def test_interleaved_column_names(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        cols = list(y['samples'].index)
        assert cols == ['S1-Me', 'S1-Un', 'S2-Me', 'S2-Un']

    def test_genes_have_chr_locus(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        assert 'Chr' in y['genes'].columns
        assert 'Locus' in y['genes'].columns

    def test_union_of_loci(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        assert y['counts'].shape[0] == 10

    def test_counts_nonnegative(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        assert np.all(y['counts'] >= 0)

    def test_zeros_for_missing_loci(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        genes = y['genes']
        chr3_mask = genes['Chr'] == 'chr3'
        chr3_counts = y['counts'][chr3_mask.values]
        assert np.all(chr3_counts[:, 0] == 0)
        assert np.all(chr3_counts[:, 1] == 0)
        assert np.all(chr3_counts[:, 2] > 0)

    def test_sample_names_from_filenames(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, verbose=False)
        cols = list(y['samples'].index)
        assert 'sample1-Me' in cols[0]
        assert 'sample2-Me' in cols[2] or 'sample2-Me' in cols[-2]

    def test_row_names_format(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        for idx in y['genes'].index:
            assert '-' in str(idx)


# ======================================================================
# modelMatrixMeth (self-contained)
# ======================================================================

class TestModelMatrixMeth:
    """Tests for model_matrix_meth()."""

    def test_shape_with_intercept(self):
        design = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        assert result.shape == (6, 5)

    def test_shape_no_intercept(self):
        design = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        assert result.shape == (6, 5)

    def test_sample_indicators(self):
        design = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        left = result[:, :3]
        expected_left = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ])
        np.testing.assert_array_equal(left, expected_left)

    def test_treatment_block_me_only(self):
        design = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        right = result[:, 3:]
        np.testing.assert_array_equal(right[0], [1, 0])
        np.testing.assert_array_equal(right[2], [1, 0])
        np.testing.assert_array_equal(right[4], [1, 1])
        np.testing.assert_array_equal(right[1], [0, 0])
        np.testing.assert_array_equal(right[3], [0, 0])
        np.testing.assert_array_equal(right[5], [0, 0])

    def test_two_samples(self):
        design = np.array([[1, 0], [1, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        assert result.shape == (4, 4)

    def test_single_column_design(self):
        design = np.array([[1], [1], [1]], dtype=np.float64)
        result = ep.model_matrix_meth(design)
        assert result.shape == (6, 4)

    def test_dgelist_input(self, bismark_dir):
        _, files = bismark_dir
        y = ep.read_bismark2dge(files, sample_names=['S1', 'S2'], verbose=False)
        design_treat = np.array([[1, 0], [1, 1]], dtype=np.float64)
        result = ep.model_matrix_meth(y, design=design_treat)
        assert result.shape == (4, 4)


# ======================================================================
# nearestRefToX (self-contained)
# ======================================================================

class TestNearestRefToX:
    """Tests for the core nearest_ref_to_x function."""

    def test_exact_match(self):
        ref = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        result = nearest_ref_to_x(np.array([200.0]), ref)
        assert result[0] == 1

    def test_midpoint_left(self):
        ref = np.array([100, 200, 300], dtype=np.float64)
        result = nearest_ref_to_x(np.array([140.0]), ref)
        assert result[0] == 0

    def test_midpoint_right(self):
        ref = np.array([100, 200, 300], dtype=np.float64)
        result = nearest_ref_to_x(np.array([160.0]), ref)
        assert result[0] == 1

    def test_below_range(self):
        ref = np.array([100, 200, 300], dtype=np.float64)
        result = nearest_ref_to_x(np.array([10.0]), ref)
        assert result[0] == 0

    def test_above_range(self):
        ref = np.array([100, 200, 300], dtype=np.float64)
        result = nearest_ref_to_x(np.array([500.0]), ref)
        assert result[0] == 2

    def test_multiple_queries(self):
        ref = np.array([100, 300, 500], dtype=np.float64)
        query = np.array([50, 200, 400, 600], dtype=np.float64)
        result = nearest_ref_to_x(query, ref)
        assert list(result) == [0, 0, 1, 2]


# ======================================================================
# nearestTSS (self-contained)
# ======================================================================

class TestNearestTSS:
    """Tests for nearest_tss()."""

    def test_returns_dataframe(self, tss_database):
        result = ep.nearest_tss(['chr1', 'chr2'], [3000, 5000],
                                tss_data=tss_database)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, tss_database):
        result = ep.nearest_tss(['chr1'], [3000], tss_data=tss_database)
        expected = {'gene_id', 'gene_name', 'strand', 'tss', 'distance'}
        assert expected <= set(result.columns)

    def test_correct_nearest_gene(self, tss_database):
        result = ep.nearest_tss(['chr1'], [3000], tss_data=tss_database)
        assert result.iloc[0]['gene_id'] in ('G1', 'G2')

    def test_correct_chromosome(self, tss_database):
        result = ep.nearest_tss(['chr2'], [5000], tss_data=tss_database)
        assert result.iloc[0]['gene_id'] in ('G4', 'G5')

    def test_distance_sign_positive_strand(self, tss_database):
        result = ep.nearest_tss(['chr1'], [1200], tss_data=tss_database)
        if result.iloc[0]['gene_id'] == 'G1':
            assert result.iloc[0]['distance'] == 200

    def test_distance_sign_negative_strand(self, tss_database):
        result = ep.nearest_tss(['chr1'], [4800], tss_data=tss_database)
        if result.iloc[0]['gene_id'] == 'G2':
            assert result.iloc[0]['distance'] == 200

    def test_multiple_queries(self, tss_database):
        chrs = ['chr1', 'chr2', 'chr3']
        loci = [3000, 5000, 2000]
        result = ep.nearest_tss(chrs, loci, tss_data=tss_database)
        assert len(result) == 3

    def test_unmatched_chromosome(self, tss_database):
        result = ep.nearest_tss(['chrX'], [1000], tss_data=tss_database)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]['gene_id'])

    def test_missing_tss_data_raises(self):
        with pytest.raises((ImportError, ValueError)):
            ep.nearest_tss(['chr1'], [1000])


# ======================================================================
# R vs Python CSV comparisons
# ======================================================================

def _load(name):
    return pd.read_csv(f"{CSV_DIR}/{name}")

def _load_idx(name):
    return pd.read_csv(f"{CSV_DIR}/{name}", index_col=0)


class TestReadBismark2DGERvsPy:
    """readBismark2DGE: R vs Python counts and genes."""

    def test_counts_shape_match(self):
        r = _load_idx("R_bismark_counts.csv")
        p = _load_idx("Py_bismark_counts.csv")
        assert r.shape == p.shape

    def test_counts_column_names(self):
        r = _load_idx("R_bismark_counts.csv")
        p = _load_idx("Py_bismark_counts.csv")
        assert list(r.columns) == list(p.columns)

    def test_counts_common_rows(self):
        r = _load_idx("R_bismark_counts.csv")
        p = _load_idx("Py_bismark_counts.csv")
        common = sorted(set(r.index) & set(p.index))
        assert len(common) == len(r)
        assert len(common) == len(p)

    def test_counts_exact_match(self):
        r = _load_idx("R_bismark_counts.csv")
        p = _load_idx("Py_bismark_counts.csv")
        common = sorted(set(r.index) & set(p.index))
        rc = r.loc[common].sort_index()
        pc = p.loc[common].sort_index()
        diff = np.abs(rc.values - pc.values)
        assert np.max(diff) == 0

    def test_genes_chr_match(self):
        r = _load_idx("R_bismark_genes.csv")
        p = _load_idx("Py_bismark_genes.csv")
        common = sorted(set(r.index) & set(p.index))
        rg = r.loc[common].sort_index()
        pg = p.loc[common].sort_index()
        assert (rg["Chr"].values == pg["Chr"].values).all()

    def test_genes_locus_match(self):
        r = _load_idx("R_bismark_genes.csv")
        p = _load_idx("Py_bismark_genes.csv")
        common = sorted(set(r.index) & set(p.index))
        rg = r.loc[common].sort_index()
        pg = p.loc[common].sort_index()
        assert (rg["Locus"].values == pg["Locus"].values).all()


class TestModelMatrixMethRvsPy:
    """modelMatrixMeth: R vs Python design matrices."""

    def test_design_with_intercept_shape(self):
        r = _load_idx("R_design_meth.csv").values
        p = _load_idx("Py_design_meth.csv").values
        assert r.shape == p.shape

    def test_design_with_intercept_exact(self):
        r = _load_idx("R_design_meth.csv").values
        p = _load_idx("Py_design_meth.csv").values
        assert np.allclose(r, p)

    def test_design_no_intercept_shape(self):
        r = _load_idx("R_design_meth_noint.csv").values
        p = _load_idx("Py_design_meth_noint.csv").values
        assert r.shape == p.shape

    def test_design_no_intercept_exact(self):
        r = _load_idx("R_design_meth_noint.csv").values
        p = _load_idx("Py_design_meth_noint.csv").values
        assert np.allclose(r, p)


class TestNearestRefRvsPy:
    """nearestReftoX: R vs Python basic lookup."""

    def test_nearest_index_match(self):
        r = _load("R_nearest_ref.csv")
        p = _load("Py_nearest_ref.csv")
        assert (r["nearest_idx"].values == p["nearest_idx"].values).all()

    def test_nearest_tss_value_match(self):
        r = _load("R_nearest_ref.csv")
        p = _load("Py_nearest_ref.csv")
        assert (r["nearest_tss"].values == p["nearest_tss"].values).all()

    def test_distance_match(self):
        r = _load("R_nearest_ref.csv")
        p = _load("Py_nearest_ref.csv")
        assert (r["distance"].values == p["distance"].values).all()


class TestNearestTSSRvsPy:
    """nearestTSS: R vs Python per-chromosome lookup."""

    def test_gene_id_match(self):
        r = _load("R_nearest_tss.csv")
        p = _load("Py_nearest_tss.csv")
        assert len(r) == len(p)
        assert (r["gene_id"].values == p["gene_id"].values).all()

    def test_tss_match(self):
        r = _load("R_nearest_tss.csv")
        p = _load("Py_nearest_tss.csv")
        assert (r["tss"].values == p["tss"].values).all()

    def test_distance_match(self):
        r = _load("R_nearest_tss.csv")
        p = _load("Py_nearest_tss.csv")
        assert (r["distance"].values == p["distance"].values).all()

    def test_all_fields_match(self):
        r = _load("R_nearest_tss.csv")
        p = _load("Py_nearest_tss.csv")
        for i in range(len(r)):
            assert r["gene_id"].iloc[i] == p["gene_id"].iloc[i]
            assert r["tss"].iloc[i] == p["tss"].iloc[i]
            assert r["distance"].iloc[i] == p["distance"].iloc[i]
