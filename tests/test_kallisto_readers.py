"""kallisto and kb count readers: kb's t2g.txt as tx2gene, kallisto run
information, and read_kb for kb count single-cell output.

The fixtures in data/kb are real kallisto 0.52.0 / kb-python 0.30.2 output
on a tiny simulated genome, written by data/kb/make_kb_files.sh.
"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import edgepython as ep

KB = Path(__file__).parent / "data" / "kb"
QUANT = Path(__file__).parent / "data" / "quant"


def _layer_names(a):
    # anndata >= 0.13 also lists X in layers, under the key None
    return {k for k in a.layers.keys() if k is not None}


def _mtx(path):
    from scipy.io import mmread
    return mmread(str(path)).tocsr()


# ===== kb's t2g.txt as tx2gene =====

def test_catch_kallisto_gene_reads_kb_t2g():
    pytest.importorskip("h5py")
    y = ep.catch_kallisto_gene(str(KB / "bulk"), tx2gene=str(KB / "t2g.txt"), verbose=False)
    t2g = pd.read_csv(KB / "t2g.txt", sep="\t", header=None)
    # Same counts as the two-column table, plus kb's gene names
    ref = ep.catch_kallisto_gene(str(KB / "bulk"), tx2gene=t2g.iloc[:, :2], verbose=False)
    np.testing.assert_array_equal(y["counts"], ref["counts"])
    assert "GeneName" not in ref["genes"]
    assert list(y["genes"].index) == [f"ENSG{g:011d}" for g in range(1, 7)]
    assert list(y["genes"]["GeneName"]) == [f"GENE{g}" for g in range(1, 7)]
    assert list(y["genes"]["NTx"]) == [2, 1, 2, 1, 2, 1]


def test_t2g_gene_names_follow_the_gene_order(tmp_path):
    pytest.importorskip("h5py")
    # Shuffled rows and a two-column file: names are matched, and only a third column adds them
    t2g = pd.read_csv(KB / "t2g.txt", sep="\t", header=None).sample(frac=1, random_state=1)
    t2g.to_csv(tmp_path / "t2g.txt", sep="\t", header=False, index=False)
    t2g.iloc[:, :2].to_csv(tmp_path / "t2g2.txt", sep="\t", header=False, index=False)
    y = ep.catch_kallisto_gene(str(KB / "bulk"), tx2gene=str(tmp_path / "t2g.txt"), verbose=False)
    assert list(y["genes"]["GeneName"]) == [f"GENE{g}" for g in range(1, 7)]
    y2 = ep.catch_kallisto_gene(str(KB / "bulk"), tx2gene=tmp_path / "t2g2.txt", verbose=False)
    assert "GeneName" not in y2["genes"]
    np.testing.assert_array_equal(y["counts"], y2["counts"])


# ===== kallisto run information =====

def test_run_info_from_run_info_json():
    pytest.importorskip("h5py")
    for y in (ep.catch_kallisto(str(KB / "bulk"), verbose=False),
              ep.catch_kallisto_gene(str(KB / "bulk"), tx2gene=str(KB / "t2g.txt"), verbose=False),
              ep.catch_kallisto(str(KB / "bulk"), dgelist=False, verbose=False)):
        info = y["run.info"]
        assert list(info.index) == ["s1", "s2", "s3"]
        assert list(info["n_processed"]) == [20000] * 3
        assert list(info["n_bootstraps"]) == [8] * 3
        assert list(info["kallisto_version"]) == ["0.52.0"] * 3
        assert info.loc["s2", "call"].startswith("kallisto quant -i std.idx -o bulk/s2 -b 8")
    # samples stays as edgeR's
    assert list(ep.catch_kallisto(str(KB / "bulk"), verbose=False)["samples"].columns) == \
        ["group", "lib.size", "norm.factors"]


def test_run_info_falls_back_to_h5_aux_and_warns_on_versions(tmp_path):
    pytest.importorskip("h5py")
    shutil.copytree(KB / "bulk", tmp_path / "bulk")
    (tmp_path / "bulk" / "s2" / "run_info.json").unlink()
    p = tmp_path / "bulk" / "s3" / "run_info.json"
    p.write_text(p.read_text().replace('"index_version": 13', '"index_version": 12'))
    with pytest.warns(UserWarning, match="different index_versions: s1=13, s2=13, s3=12"):
        y = ep.catch_kallisto(str(tmp_path / "bulk"), verbose=False)
    info = y["run.info"]
    # abundance.h5 has no pseudoalignment counts
    assert np.isnan(info.loc["s2", "n_pseudoaligned"])
    assert info.loc["s2", "n_processed"] == 20000 and info.loc["s2", "n_bootstraps"] == 8
    assert info.loc["s2", "kallisto_version"] == "0.52.0"
    assert info.loc["s2", "call"].startswith("kallisto quant -i std.idx -o bulk/s2")


def test_samples_must_share_transcripts(tmp_path):
    shutil.copytree(QUANT / "kallisto_tsv", tmp_path / "k")
    p = tmp_path / "k" / "k2" / "abundance.tsv"
    q = pd.read_csv(p, sep="\t")
    q.loc[0, "target_id"] = "ENST_OTHER"
    q.to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="k2 does not have the same transcripts"):
        ep.catch_kallisto(str(tmp_path / "k"), verbose=False)


# ===== read_kb =====

def test_read_kb_standard_workflow(tmp_path):
    pytest.importorskip("anndata")
    a = ep.read_kb(str(KB / "sc_std"), verbose=False)
    d = KB / "sc_std" / "counts_unfiltered"
    assert a.shape == (50, 6)
    assert a.X.format == "csr" and a.X.dtype == np.float64
    assert (a.X != _mtx(d / "cells_x_genes.mtx")).nnz == 0
    assert list(a.obs_names) == (d / "cells_x_genes.barcodes.txt").read_text().split()
    assert list(a.var_names) == [f"ENSG{g:011d}.1" for g in range(1, 7)]
    assert list(a.var["gene_name"]) == [f"GENE{g}" for g in range(1, 7)]
    assert set(a.uns["kb"]) == {"run_info", "inspect", "kb_info"}
    assert a.uns["kb"]["run_info"] == json.loads((KB / "sc_std" / "run_info.json").read_text())
    assert _layer_names(a) == set()
    # The counts directory itself can be given; the run's json files are above it
    b = ep.read_kb(str(d), verbose=False)
    assert (a.X != b.X).nnz == 0 and set(b.uns["kb"]) == set(a.uns["kb"])
    # and it saves as h5ad
    a.write_h5ad(tmp_path / "a.h5ad")


def test_read_kb_nac_workflow():
    pytest.importorskip("anndata")
    d = KB / "sc_nac" / "counts_unfiltered"
    m, n, amb = (_mtx(d / f"cells_x_genes.{k}.mtx") for k in ("mature", "nascent", "ambiguous"))
    a = ep.read_kb(str(KB / "sc_nac"), filtered=False, verbose=False)
    assert a.shape == (50, 6)
    assert _layer_names(a) == {"ambiguous", "mature", "nascent"}
    # X is mature + nascent + ambiguous, as in kb's own h5ad
    assert (a.X != m + n + amb).nnz == 0
    assert (a.layers["nascent"] != n).nnz == 0
    b = ep.read_kb(str(KB / "sc_nac"), layer="mature", filtered=False, verbose=False)
    assert (b.X != m).nnz == 0


def test_read_kb_prefers_counts_filtered():
    pytest.importorskip("anndata")
    a = ep.read_kb(str(KB / "sc_nac"), verbose=False)
    assert a.shape == (45, 6)
    filt = KB / "sc_nac" / "counts_filtered"
    assert list(a.obs_names) == (filt / "cells_x_genes.barcodes.txt").read_text().split()
    # counts_filtered/ has no gene names; they come from counts_unfiltered/
    assert not (filt / "cells_x_genes.genes.names.txt").exists()
    assert list(a.var["gene_name"]) == [f"GENE{g}" for g in range(1, 7)]


def test_read_kb_summed_matrices():
    pytest.importorskip("anndata")
    d = KB / "sc_nac_sum" / "counts_unfiltered"
    a = ep.read_kb(str(KB / "sc_nac_sum"), layer="total", verbose=False)
    assert (a.X != _mtx(d / "cells_x_genes.total.mtx")).nnz == 0
    assert _layer_names(a) == {"ambiguous", "mature", "nascent", "cell", "nucleus", "total"}


def test_read_kb_errors(tmp_path):
    pytest.importorskip("anndata")
    with pytest.raises(ValueError, match="holds a single matrix"):
        ep.read_kb(str(KB / "sc_std"), layer="mature", verbose=False)
    with pytest.raises(ValueError, match="Layer 'spliced' not found"):
        ep.read_kb(str(KB / "sc_nac"), layer="spliced", verbose=False)
    with pytest.raises(FileNotFoundError, match="counts_filtered/ not found"):
        ep.read_kb(str(KB / "sc_std"), filtered=True, verbose=False)
    with pytest.raises(FileNotFoundError, match="No kb count output"):
        ep.read_kb(str(tmp_path), verbose=False)


def test_read_data_detects_kb_output():
    pytest.importorskip("anndata")
    with pytest.warns(UserWarning, match="Densifying sparse AnnData matrix"):
        y = ep.read_data(str(KB / "sc_std"), verbose=False)
    assert y["counts"].shape == (6, 50)
    np.testing.assert_array_equal(
        y["counts"], _mtx(KB / "sc_std" / "counts_unfiltered" / "cells_x_genes.mtx").toarray().T)
    assert list(y["genes"]["gene_name"]) == [f"GENE{g}" for g in range(1, 7)]
    with pytest.warns(UserWarning):
        n = ep.read_data(str(KB / "sc_nac"), layer="nascent", verbose=False)
    np.testing.assert_array_equal(
        n["counts"],
        _mtx(KB / "sc_nac" / "counts_filtered" / "cells_x_genes.nascent.mtx").toarray().T)


def test_read_kb_feeds_seurat_to_pb():
    pytest.importorskip("anndata")
    a = ep.read_kb(str(KB / "sc_std"), verbose=False)
    a.obs["sample"] = np.repeat(["s1", "s2"], 25)
    a.obs["cluster"] = np.tile(["0", "1"], 25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pb = ep.seurat_to_pb(a, sample="sample", cluster="cluster")
    assert pb["counts"].shape == (6, 4)
    np.testing.assert_allclose(pb["counts"].sum(), a.X.sum())
