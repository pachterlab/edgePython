# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for I/O functions: readDGE, read_data, I/O round-trip."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestReadDGE:
    """readDGE function."""

    def test_read_dge(self):
        dge = ep.read_dge([
            f"{CSV_DIR}/test_readDGE_s1.txt",
            f"{CSV_DIR}/test_readDGE_s2.txt"
        ])
        # R: geneA=[10,15], geneB=[20,25], geneC=[30,35]
        assert dge['counts'].shape == (3, 2)
        assert np.allclose(dge['counts'][:, 0], [10, 20, 30])
        assert np.allclose(dge['counts'][:, 1], [15, 25, 35])
        # R: lib.size=[60, 75]
        assert np.allclose(dge['samples']['lib.size'].values, [60, 75])


class TestIORoundTrip:
    """I/O round-trip: write then read DGE files."""

    def test_write_read_dge(self):
        y = np.array([[10, 20, 30],
                       [5, 15, 25],
                       [100, 200, 300],
                       [0, 0, 0],
                       [50, 50, 50]], dtype=float)
        gene_names = ['geneA', 'geneB', 'geneC', 'geneD', 'geneE']

        tmpdir = tempfile.mkdtemp()
        files = []
        try:
            for j in range(3):
                fpath = os.path.join(tmpdir, f'sample{j+1}.txt')
                with open(fpath, 'w') as f:
                    f.write('Gene\tCount\n')
                    for i in range(5):
                        f.write(f'{gene_names[i]}\t{int(y[i, j])}\n')
                files.append(fpath)

            dge = ep.read_dge(files)
            assert dge['counts'].shape == (5, 3)
            assert np.allclose(dge['counts'], y)
            assert np.allclose(dge['samples']['lib.size'].values,
                               y.sum(axis=0))
        finally:
            for fpath in files:
                if os.path.exists(fpath):
                    os.remove(fpath)
            if os.path.exists(tmpdir):
                os.rmdir(tmpdir)


class TestReadData:
    """Universal read_data() function."""

    def test_ndarray_input(self):
        mat = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        dge = ep.read_data(mat, group=[1, 2])
        assert dge['counts'].shape == (3, 2)
        assert np.allclose(dge['counts'], mat)
        assert dge['samples']['group'].values.tolist() == [1, 2]

    def test_dataframe_input(self):
        df = pd.DataFrame({'S1': [10, 30, 50], 'S2': [20, 40, 60]},
                          index=['gene1', 'gene2', 'gene3'])
        dge = ep.read_data(df, group=[1, 2])
        assert dge['counts'].shape == (3, 2)
        assert np.allclose(dge['counts'], df.values)

    def test_csv_input(self):
        csv_path = os.path.join(tempfile.mkdtemp(), 'counts.csv')
        try:
            df = pd.DataFrame({'S1': [10, 30, 50], 'S2': [20, 40, 60]},
                              index=['g1', 'g2', 'g3'])
            df.to_csv(csv_path)
            dge = ep.read_data(csv_path, group=[1, 2], verbose=False)
            assert dge['counts'].shape == (3, 2)
            assert np.allclose(dge['counts'], df.values)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
                os.rmdir(os.path.dirname(csv_path))

    def test_dgelist_passthrough(self):
        existing = ep.make_dgelist(np.array([[1, 2], [3, 4]], dtype=float))
        result = ep.read_data(existing)
        assert result is existing
