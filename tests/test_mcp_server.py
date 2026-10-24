"""Tests for the MCP server, calling its tools through a FastMCP client."""

import asyncio
import gzip
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

fastmcp = pytest.importorskip("fastmcp")

import edgepython as ep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT = os.path.join(ROOT, "tests", "data", "quant")
KB = os.path.join(ROOT, "tests", "data", "kb")

_spec = importlib.util.spec_from_file_location(
    "edgepython_server", os.path.join(ROOT, "MCP", "edgepython_server.py"))
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


def call(*calls):
    """Run (tool, args) calls in one client session; return their texts.
    Fails on the first tool error unless the call is marked with a third
    element 'error', in which case the error text is returned."""
    async def run():
        out = []
        async with fastmcp.Client(srv.mcp) as client:
            for c in calls:
                name, args = c[0], c[1]
                expect_error = len(c) > 2
                r = await client.call_tool(name, args, raise_on_error=False)
                text = "\n".join(getattr(b, "text", "") for b in r.content)
                assert r.is_error == expect_error, f"{name}: {text}"
                out.append(text)
        return out
    srv._state["results"] = {}
    return asyncio.run(run())


@pytest.fixture
def bulk_files(tmp_path):
    rng = np.random.default_rng(1)
    ngenes, n = 1000, 6
    mu = rng.gamma(2, 50, ngenes)[:, None] * np.ones(n)
    mu[:60, 3:] *= 4
    counts = rng.negative_binomial(5, 5 / (5 + mu))
    ids = [f"ENSG{i:011d}" for i in range(ngenes)]
    samples = [f"S{i}" for i in range(n)]
    pd.DataFrame(counts, index=ids, columns=samples).to_csv(tmp_path / "counts.csv")
    pd.DataFrame({"sample": samples, "group": ["A"] * 3 + ["B"] * 3,
                  "batch": ["x", "y", "z"] * 2}).to_csv(tmp_path / "samples.csv", index=False)
    return tmp_path, ids, counts


def test_bulk_pipeline(bulk_files):
    d, ids, counts = bulk_files
    sets = {"up": ids[:60], "rest": ids[100:200], "absent": ["nope"]}
    out = call(
        ("load_data", {"counts_path": str(d / "counts.csv"),
                       "sample_info_path": str(d / "samples.csv"), "group_column": "group"}),
        ("filter_genes", {"preset": "moderate"}),
        ("normalize", {}),
        ("set_design", {"formula": "~group"}),
        ("estimate_dispersion", {}),
        ("gene_set_test", {"gene_sets": sets, "coef": 1}),
        ("get_result_table", {}),
        ("voom_transform", {"sample_weights": True, "block_column": "batch"}),
        ("gene_set_test", {"gene_sets": sets, "coef": 1, "method": "fry"}),
        ("fit_model", {"sample_weights": True}),
        ("test_contrast", {"contrast": "groupB"}),
        ("test_coef", {"coef": 1, "method": "qlf_threshold", "lfc": 1.0, "name": "thr"}),
        ("test_coef", {"coef": 1, "method": "treat", "lfc": 1.0, "name": "treat"}),
        ("test_coef", {"coef": 1, "method": "bogus"}, "error"),
        ("get_top_genes", {"name": "thr", "n": 5}),
    )
    assert "Filtered: 1,000 →" in out[1]
    assert "camera on DGEList" in out[5] and "Sets tested: 2" in out[5]
    assert "Span used" in out[7] and "Sample-weight range" in out[7]
    assert "fry on voom" in out[8]
    assert "Sample weights: S0" in out[9]
    assert "logFC" in out[14]
    assert out[6].startswith("Set\t")

    # The threshold test matches calling edgePython directly
    dge = srv._state["dgelist"]
    fit = ep.glm_ql_fit(dge, design=srv._state["design"], robust=True)
    fit = ep.glm_ql_fit(dge, design=srv._state["design"], robust=True,
                        weights=ep.sample_weights(fit))
    direct = ep.glm_ql_ftest(fit, coef=1, lfc=1.0)["table"]["PValue"].to_numpy()
    via_mcp = srv._state["results"]["thr"]["table"]["PValue"].to_numpy()
    np.testing.assert_allclose(via_mcp, direct, rtol=1e-10)


def test_load_data_auto(bulk_files):
    d, _, _ = bulk_files
    salmon = [os.path.join(QUANT, "salmon", f"s{i}") for i in (1, 2, 3)]
    out = call(
        ("load_data_auto", {"data": str(d / "counts.csv"), "sep": ",",
                            "group": ["A"] * 3 + ["B"] * 3}),
        ("load_data_auto", {"data_list": salmon, "source": "salmon", "divide": True}),
    )
    assert "Loaded 1,000 genes × 6 samples" in out[0]
    assert "Samples: s1, s2, s3" in out[1]
    assert "Feature level: transcript" in out[1]


def test_load_quant_and_dtu():
    salmon = [os.path.join(QUANT, "salmon", f"s{i}") for i in (1, 2, 3)]
    out = call(
        ("load_quant", {"source": "salmon", "paths": salmon, "group": ["A", "A", "B"]}),
        ("dtu_diff_splice_dge", {}, "error"),
        ("estimate_dispersion", {}),
        ("dtu_diff_splice_dge", {}),
        ("get_top_genes", {"n": 3}),
        ("dtu_splice_variants", {}),
        ("load_quant", {"source": "salmon", "paths": salmon, "level": "gene"}),
        ("load_quant", {"source": "rsem", "paths": [os.path.join(QUANT, "rsem", f"r{i}.isoforms.results")
                                                    for i in (1, 2, 3)]}),
        ("load_quant", {"source": "nope", "paths": salmon}, "error"),
    )
    assert "Overdispersion" in out[0] and "(counts divided)" in out[0]
    assert "estimate_dispersion" in out[1]
    assert "DTU: diff_splice_dge completed" in out[3]
    assert "NExons" in out[4]
    assert "DTU: splice_variants completed" in out[5]
    assert "Loaded 14 genes × 3 samples" in out[6]
    assert "Loaded 40 genes × 3 samples" in out[7]


def _write_cov(path, chrom, loci, meth, unmeth):
    with gzip.open(path, "wt") as f:
        for pos, m, u in zip(loci, meth, unmeth):
            if m + u > 0:
                f.write(f"{chrom}\t{pos}\t{pos}\t{100 * m / (m + u):.4f}\t{m}\t{u}\n")


def test_methylation(tmp_path):
    rng = np.random.default_rng(3)
    nloci = 300
    loci = np.arange(1, nloci + 1) * 100
    p = rng.beta(2, 2, nloci)
    files = []
    for j in range(6):
        pj = p.copy()
        if j >= 3:
            pj[:30] = np.clip(pj[:30] + 0.4, 0, 0.98)
        cov = rng.poisson(30, nloci)
        meth = rng.binomial(cov, pj)
        f = tmp_path / f"s{j}.cov.gz"
        _write_cov(f, "chr1", loci, meth, cov - meth)
        files.append(str(f))
    out = call(
        ("load_bismark", {"files": files, "group": ["A"] * 3 + ["B"] * 3}),
        ("fit_bin_model", {}),
        ("test_bin", {}),
        ("test_bin", {"coef": 1, "lor": 1.0, "name": "lor1"}),
        ("get_top_genes", {"n": 5}),
        ("save_results", {"output_path": str(tmp_path / "meth.tsv")}),
    )
    assert "6 samples" in out[0]
    assert "Coefficients: Intercept, group[T.B]" in out[1]
    assert "Differentially methylated loci" in out[2]
    assert "logOR" in out[4]
    saved = pd.read_csv(tmp_path / "meth.tsv", sep="\t")
    assert {"logOR", "PValue", "FDR"} <= set(saved.columns)


def _sc_data():
    rng = np.random.default_rng(2)
    ncell, ngene = 120, 60
    X = rng.negative_binomial(2, 2 / (2 + rng.gamma(2, 2, ngene)), size=(ncell, ngene)).astype(float)
    obs = pd.DataFrame({"sid": np.repeat(["a", "b", "c", "d"], 30),
                        "ct": np.tile(["T", "B"], 60)}, index=[f"c{i}" for i in range(ncell)])
    return X, obs, [f"g{i}" for i in range(ngene)]


def _sc_calls(path):
    fed = [1.0 if s in ("a", "b") else 0.0 for s in _sc_data()[1]["sid"]]
    return call(
        ("load_sc_data", {"h5ad_path": path, "sample_col": "sid", "layer": "X",
                          "log1p_transform": False, "cell_type_col": "ct", "cell_type_filter": ["T"]}),
        ("load_sc_data", {"h5ad_path": path, "sample_col": "sid", "layer": "X", "log1p_transform": False}),
        ("set_sc_design", {"covariates": {"Intercept": [1.0] * len(fed), "fed": fed}}),
        ("fit_sc_model", {"verbose": False}),
        ("test_sc_coef", {"coef": "fed"}),
    )


def test_load_sc_data_current_anndata(tmp_path):
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp
    X, obs, genes = _sc_data()
    obs = obs.astype("category")
    path = str(tmp_path / "sc.h5ad")
    ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(path)
    out = _sc_calls(path)
    assert "Loaded 60 genes × 60 cells, 4 samples" in out[0]
    assert "Loaded 60 genes × 120 cells" in out[1]
    assert "Wald test: fed" in out[4]


def test_load_sc_data_legacy_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    X, obs, genes = _sc_data()
    path = str(tmp_path / "sc.h5ad")
    with h5py.File(path, "w") as h:
        h["X"] = X
        h.create_group("var")["_index"] = np.array([g.encode() for g in genes])
        o = h.create_group("obs")
        o["_index"] = np.array([i.encode() for i in obs.index])
        cats = h.create_group("obs/__categories")
        for col in ("sid", "ct"):
            levels = sorted(obs[col].unique())
            cats[col] = np.array([v.encode() for v in levels])
            o[col] = np.array([levels.index(v) for v in obs[col]], dtype=np.int8)
    out = _sc_calls(path)
    assert "Loaded 60 genes × 60 cells, 4 samples" in out[0]
    assert "Wald test: fed" in out[4]


def test_load_quant_reads_kb_t2g_and_reports_runs(tmp_path):
    pytest.importorskip("h5py")
    import shutil
    out = call(("load_quant", {"source": "kallisto", "parent_dir": os.path.join(KB, "bulk"),
                               "level": "gene", "tx2gene_path": os.path.join(KB, "t2g.txt")}))
    assert "Loaded 6 genes × 3 samples" in out[0]
    assert "kallisto runs:" in out[0]
    assert "s2: 20,000 reads, 100.0% pseudoaligned, kallisto 0.52.0" in out[0]
    assert list(srv._state["dgelist"]["genes"]["GeneName"]) == [f"GENE{g}" for g in range(1, 7)]
    shutil.copytree(os.path.join(KB, "bulk"), tmp_path / "bulk")
    p = tmp_path / "bulk" / "s3" / "run_info.json"
    p.write_text(p.read_text().replace('"index_version": 13', '"index_version": 12'))
    out = call(("load_quant", {"source": "kallisto", "parent_dir": str(tmp_path / "bulk")}))
    assert "Warning: Samples were quantified with different index_versions" in out[0]


def test_load_kb_single_cell():
    pytest.importorskip("anndata")
    dirs = [os.path.join(KB, d) for d in ("sc_std", "sc_nac", "sc_nac_sum", "sc_std")]
    out = call(
        ("load_kb", {"kb_dirs": dirs, "sample_names": ["a", "b", "c", "d"]}),
        ("load_kb", {"kb_dirs": dirs, "sample_names": ["a", "b", "c", "d"], "min_counts": 50}),
        ("set_sc_design", {"sample_covariates": {"fed": {"a": 1, "b": 0, "c": 1}}}, "error"),
        ("set_sc_design", {"sample_covariates": {"fed": {"a": 1, "b": 0, "c": 1, "d": 0}}}),
        ("fit_sc_model", {"verbose": False}),
        ("test_sc_coef", {"coef": "fed"}),
        ("load_kb", {"kb_dirs": dirs[:2], "sample_names": ["a", "a"]}, "error"),
        ("load_kb", {"kb_dirs": [dirs[0], os.path.join(KB, "bulk")]}, "error"),
    )
    assert "Loaded 6 genes × 195 cells, 4 samples" in out[0]
    assert "a: 50 cells" in out[0] and "b: 45 cells" in out[0]
    assert "empty droplets, for a, c, d" in out[0]
    assert "Loaded 6 genes × 180 cells, 4 samples" in out[1]
    assert "a: 45 cells (of 50 barcodes)" in out[1] and "empty droplets" not in out[1]
    assert "no value for: d" in out[2]
    assert "Columns: Intercept, fed" in out[3]
    assert srv._sc_state["design"]["fed"].tolist() == [1.0] * 45 + [0.0] * 45 + [1.0] * 45 + [0.0] * 45
    assert "Wald test: fed" in out[5]
    assert "unique" in out[6]
    assert "No kb count output" in out[7]


def test_load_kb_pseudobulk_and_load_data_auto():
    pytest.importorskip("anndata")
    from scipy.io import mmread
    dirs = [os.path.join(KB, d) for d in ("sc_std", "sc_nac", "sc_nac_sum")]
    out = call(
        ("load_kb", {"kb_dirs": dirs, "mode": "pseudobulk", "layer": "nascent"}, "error"),
        ("load_kb", {"kb_dirs": dirs[1:], "mode": "pseudobulk", "layer": "nascent",
                     "group": ["x", "y"]}),
    )
    assert "single matrix" in out[0]
    assert "Loaded 6 genes × 2 samples" in out[1]
    assert "Context: pseudobulk" in out[1] and "isoform" not in out[1]
    y = srv._state["dgelist"]
    nascent = mmread(os.path.join(KB, "sc_nac", "counts_filtered", "cells_x_genes.nascent.mtx"))
    np.testing.assert_allclose(y["counts"][:, 0], np.asarray(nascent.sum(axis=0)).ravel())
    assert list(y["samples"].index) == ["sc_nac", "sc_nac_sum"]
    assert list(y["samples"]["group"].astype(str)) == ["x", "y"]
    assert list(y["genes"]["gene_name"]) == [f"GENE{g}" for g in range(1, 7)]
    out = call(("load_data_auto", {"data": os.path.join(KB, "sc_std"), "verbose": False}))
    assert "Loaded 6 genes × 50 samples" in out[0] and "Context: single_cell (kb count output)" in out[0]
