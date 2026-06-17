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


class TestTxConstructors:
    """DGEList construction from tximport/tximeta-style objects."""

    def test_dgelist_from_tximport_raw_counts_offsets_and_labels(self):
        counts = pd.DataFrame(
            [[10.0, 20.0], [0.0, 0.0], [5.0, 15.0]],
            index=['tx1', 'tx2', 'tx3'],
            columns=['s1', 's2'],
        )
        length = pd.DataFrame(
            [[100.0, 120.0], [80.0, 80.0], [200.0, 100.0]],
            index=counts.index,
            columns=counts.columns,
        )
        txi = {
            'counts': counts,
            'length': length,
            'countsFromAbundance': 'no',
        }

        dge = ep.dgelist_from_tximport(txi, group=['A', 'B'], remove_zeros=True)

        assert dge['counts'].shape == (2, 2)
        assert dge['genes'].index.tolist() == ['tx1', 'tx3']
        assert dge['samples'].index.tolist() == ['s1', 's2']
        assert dge['tximport.counts'] == 'raw'
        assert dge['divided.counts'] is False
        log_length = np.log(length.values[[0, 2]])
        expected_offset = log_length - log_length.mean(axis=1)[:, None]
        assert np.allclose(dge['offset.prior'], expected_offset)
        assert np.allclose(
            dge['genes']['AveLength'].values,
            np.exp(log_length.mean(axis=1)),
        )

    def test_dgelist_from_tximport_infreps_and_divide(self):
        counts = np.array([[10.0, 20.0], [30.0, 40.0]])
        length = np.array([[100.0, 110.0], [200.0, 210.0]])
        inf_reps = [
            np.array([[8.0, 10.0, 12.0], [28.0, 30.0, 32.0]]),
            np.array([[18.0, 20.0, 22.0], [38.0, 40.0, 42.0]]),
        ]
        txi = {
            'counts': counts,
            'length': length,
            'countsFromAbundance': 'lengthScaledTPM',
            'infReps': inf_reps,
        }

        dge = ep.dgelist_from_tximport(txi, divide=True)

        assert 'Overdispersion' in dge['genes'].columns
        assert np.all(dge['genes']['Overdispersion'].values >= 1.0)
        assert dge['divided.counts'] is True
        assert dge['tximport.counts'] == 'lengthScaledTPM'
        assert 'offset.prior' not in dge
        assert np.allclose(
            dge['counts'],
            counts / dge['genes']['Overdispersion'].values[:, None],
        )

    def test_dgelist_from_tximeta_dict(self):
        counts = pd.DataFrame(
            [[10.0, 15.0], [20.0, 30.0]],
            index=['gene1', 'gene2'],
            columns=['s1', 's2'],
        )
        length = pd.DataFrame(
            [[100.0, 100.0], [150.0, 200.0]],
            index=counts.index,
            columns=counts.columns,
        )
        assays = {
            'counts': counts,
            'length': length,
            'infRep1': counts.values + 1.0,
            'infRep2': counts.values + 2.0,
            'infRep3': counts.values + 3.0,
        }
        txm = {
            'assays': assays,
            'row_data': pd.DataFrame({'gene_id': [1, 2]}),
            'metadata': {'countsFromAbundance': 'no'},
        }

        dge = ep.dgelist_from_tximeta(txm)

        assert dge['counts'].shape == (2, 2)
        assert dge['genes']['gene_id'].tolist() == ['1', '2']
        assert 'Overdispersion' in dge['genes'].columns
        assert dge['tximport.counts'] == 'raw'
        assert dge['offset.prior'].shape == counts.shape

    def test_dgelist_from_tximeta_lightweight_object(self):
        class TxmLike:
            def __init__(self):
                self._assays = {
                    'counts': np.array([[5.0, 10.0], [0.0, 0.0]]),
                    'length': np.array([[90.0, 100.0], [50.0, 50.0]]),
                }

            def assay_names(self):
                return list(self._assays)

            def assay(self, name):
                return self._assays[name]

            def row_data(self):
                return pd.DataFrame({'gene_id': ['g1', 'g2']})

            def metadata(self):
                return {'countsFromAbundance': 'scaledTPM'}

        dge = ep.dgelist_from_tximeta(TxmLike(), remove_zeros=True)

        assert dge['counts'].shape == (1, 2)
        assert dge['genes']['gene_id'].tolist() == ['g1']
        assert dge['tximport.counts'] == 'scaledTPM'
        assert 'offset.prior' not in dge


class TestCatchOarfish:
    """Oarfish quantification reader."""

    def _write_oarfish_sample(self, tmpdir, prefix, counts):
        quant = pd.DataFrame({
            'tname': ['tx1', 'tx2'],
            'len': [100, 200],
            'num_reads': counts,
        })
        quant.to_csv(os.path.join(tmpdir, f'{prefix}.quant'), sep='\t', index=False)

    def test_catch_oarfish_list_output(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self._write_oarfish_sample(tmpdir, 's1', [10.0, 20.0])
            self._write_oarfish_sample(tmpdir, 's2', [15.0, 25.0])

            out = ep.catch_oarfish(path=tmpdir, verbose=False)

            assert np.allclose(out['counts'], [[10.0, 15.0], [20.0, 25.0]])
            assert out['annotation'].index.tolist() == ['tx1', 'tx2']
            assert out['annotation']['Length'].tolist() == [100, 200]
            assert out['resample.type'] == ['bootstrap', 'bootstrap']
            assert out['divided.counts'] is False
            assert np.isnan(out['overdispersion.prior'])
            assert np.all(np.isnan(out['annotation']['Overdispersion'].values))
        finally:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)

    def test_catch_oarfish_dgelist_output(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self._write_oarfish_sample(tmpdir, 'sampleA', [5.0, 7.0])

            dge = ep.catch_oarfish(prefixes=['sampleA'], path=tmpdir,
                                   dge_list=True, verbose=False)

            assert dge['counts'].shape == (2, 1)
            assert dge['genes'].index.tolist() == ['tx1', 'tx2']
            assert dge['samples'].index.tolist() == [os.path.join(tmpdir, 'sampleA')]
            assert dge['resample.type'] == ['bootstrap']
            assert dge['divided.counts'] is False
        finally:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)


class TestCatchRSEM:
    """RSEM quantification reader."""

    def _write_rsem_sample(self, tmpdir, filename, expected, post_mean=None,
                           post_sd=None):
        if post_mean is None:
            post_mean = expected
        if post_sd is None:
            post_sd = [0.0] * len(expected)
        quant = pd.DataFrame({
            'transcript_id': ['tx1', 'tx2'],
            'gene_id': ['g1', 'g2'],
            'length': [100.0, 200.0],
            'effective_length': [90.0, 180.0],
            'expected_count': expected,
            'TPM': [1.0, 2.0],
            'FPKM': [3.0, 4.0],
            'IsoPct': [50.0, 50.0],
            'posterior_mean_count': post_mean,
            'posterior_standard_deviation_of_count': post_sd,
        })
        quant.to_csv(os.path.join(tmpdir, filename), sep='\t', index=False)

    def test_catch_rsem_path_discovery_and_gibbs_metadata(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self._write_rsem_sample(
                tmpdir, 's1.isoforms.results', [10.0, 20.0],
                post_mean=[10.0, 20.0], post_sd=[2.0, 4.0])
            self._write_rsem_sample(
                tmpdir, 's2.isoforms.results', [15.0, 25.0],
                post_mean=[15.0, 25.0], post_sd=[3.0, 5.0])

            out = ep.catch_rsem(path=tmpdir, ngibbs=[5, 5], verbose=False)

            assert np.allclose(out['counts'], [[10.0, 15.0], [20.0, 25.0]])
            assert out['annotation'].index.tolist() == ['tx1', 'tx2']
            assert out['annotation']['Length'].tolist() == [100.0, 200.0]
            assert np.allclose(out['annotation']['AveLength'].values, [90.0, 180.0])
            assert np.allclose(out['annotation']['Max2MinLength'].values, [1.0, 1.0])
            assert np.all(out['annotation']['Overdispersion'].values >= 1.0)
            assert out['overdispersion.prior'] >= 1.0
            assert out['resample.type'] == ['gibbs', 'gibbs']
            assert out['divided.counts'] is False
        finally:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)

    def test_catch_rsem_dgelist_divide_and_edgeR_alias(self):
        tmpdir = tempfile.mkdtemp()
        try:
            self._write_rsem_sample(
                tmpdir, 'sample.isoforms.results', [12.0, 24.0],
                post_mean=[12.0, 24.0], post_sd=[3.0, 6.0])

            dge = ep.catch_rsem(
                files=['sample.isoforms.results'], path=tmpdir, ngibbs=6,
                divide=True, DGEList=True, verbose=False)

            assert dge['genes'].index.tolist() == ['tx1', 'tx2']
            assert list(dge['samples'].index) == [
                os.path.join(tmpdir, 'sample')]
            assert dge['resample.type'] == ['gibbs']
            assert dge['divided.counts'] is True
            assert np.allclose(
                dge['counts'],
                np.array([[12.0], [24.0]]) /
                dge['genes']['Overdispersion'].values[:, None],
            )
            assert dge['overdispersion.prior'] >= 1.0
        finally:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)

    def test_catch_rsem_without_gibbs_columns_uses_nan_overdispersion(self):
        tmpdir = tempfile.mkdtemp()
        try:
            quant = pd.DataFrame({
                'transcript_id': ['tx1', 'tx2'],
                'length': [100.0, 200.0],
                'effective_length': [90.0, 180.0],
                'expected_count': [5.0, 8.0],
            })
            quant.to_csv(
                os.path.join(tmpdir, 'nogibbs.isoforms.results'),
                sep='\t', index=False)

            out = ep.catch_rsem(path=tmpdir, verbose=False)

            assert np.allclose(out['counts'], [[5.0], [8.0]])
            assert np.isnan(out['overdispersion.prior'])
            assert np.all(np.isnan(out['annotation']['Overdispersion'].values))
        finally:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
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
