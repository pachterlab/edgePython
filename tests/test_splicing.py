# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""Tests for differential splicing: diffSplice, spliceVariants."""

import os

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestDiffSplice:
    """diffSplice on QL-fitted data."""

    @pytest.fixture(scope="class")
    def part4_data(self):
        y = pd.read_csv(f"{CSV_DIR}/test_data_part4.csv").values
        y[0:5, 1:3] = 0
        group = np.array([1, 2, 2])
        design = np.column_stack([np.ones(3), np.array([0, 1, 1])])
        d = ep.make_dgelist(counts=y, group=group)
        return d, y, group, design

    def test_diff_splice_runs(self, part4_data):
        d, y, group, design = part4_data
        fit = ep.glm_ql_fit(d, design=design, legacy=False,
                             keep_unit_mat=True)
        GeneID = np.repeat(np.arange(1, 11), 10)
        ds = ep.diff_splice(fit, geneid=GeneID)
        assert 'gene.p.value' in ds
        assert 'gene.Simes.p.value' in ds
        assert 'exon.p.value' in ds
        assert len(ds['gene.p.value']) == 10
        assert len(ds['exon.p.value']) == 100


class TestSpliceVariants:
    """spliceVariants function."""

    def test_splice_variants(self):
        exon_counts = pd.read_csv(
            f"{CSV_DIR}/test_data_part22_exons.csv").values.astype(float)
        geneids = np.array([f'Gene{i+1}' for i in range(5)
                            for _ in range(10)])
        group = np.array([1, 1, 2, 2])
        d = ep.make_dgelist(counts=exon_counts, group=group)
        d = ep.calc_norm_factors(d)
        sv = ep.splice_variants(d, geneids)
        # R: Gene3 is most significant
        sorted_sv = sv.sort_values('PValue')
        assert sorted_sv['GeneID'].iloc[0] == 'Gene3'


class TestDiffSpliceDGE:
    """diff_splice_dge on small exon-level DGEList."""

    def test_diff_splice_dge(self):
        y = np.array([
            [50, 60, 55, 100, 110, 105],
            [40, 45, 42, 40, 38, 43],
            [30, 35, 33, 30, 32, 28],
            [80, 75, 85, 80, 78, 82],
            [20, 22, 18, 20, 21, 19],
            [60, 65, 58, 120, 130, 125],
            [50, 55, 52, 50, 48, 53],
            [45, 40, 48, 45, 43, 47],
        ], dtype=float)
        geneid = np.array(['gene1'] * 3 + ['gene2'] * 2 + ['gene3'] * 3)
        exonid = np.array(['e1', 'e2', 'e3', 'e1', 'e2',
                           'e1', 'e2', 'e3'])
        d = ep.make_dgelist(counts=y, group=np.array([1, 1, 1, 2, 2, 2]))
        d = ep.estimate_disp(d,
                              np.column_stack([np.ones(6),
                                               [0, 0, 0, 1, 1, 1]]))
        result = ep.diff_splice_dge(d, geneid=geneid, exonid=exonid,
                                     dispersion=0.1)
        assert result['gene.table'].shape[0] == 3
        assert result['exon.table'].shape[0] == 8
