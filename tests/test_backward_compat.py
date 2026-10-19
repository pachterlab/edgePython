"""Calls written for earlier edgePython versions (0.2.x) keep working.

Each test uses a function the way 0.2.6 documented it: positional
arguments in their old order, old keyword names, and old result keys.
"""

from pathlib import Path

import numpy as np
import pytest

import edgepython as ep

QUANT = Path(__file__).parent / "data" / "quant"
rng = np.random.default_rng(7)
Y = rng.negative_binomial(5, 0.05, (400, 6)).astype(float)
GROUP = np.repeat(["A", "B"], 3)
X = np.column_stack([np.ones(6), GROUP == "B"]).astype(float)


@pytest.fixture(scope="module")
def dge():
    return ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(Y, group=GROUP)), X)


@pytest.mark.parametrize("reader, folder", [("catch_salmon", "salmon"), ("catch_kallisto", "kallisto_tsv")])
def test_old_catch_calls(reader, folder):
    dirs = sorted(str(p) for p in (QUANT / folder).iterdir())
    f = getattr(ep, reader)
    for y in (f(dirs), f(dirs, False), f(paths=dirs, verbose=False)):
        assert y["counts"].shape[1] == 3
        assert list(y["annotation"].columns) == ["GeneID", "Length"]
        assert list(y["samples"]["files"]) == dirs
        assert len(y["annotation"]) == y["counts"].shape[0]


def test_old_catch_rsem_calls():
    iso = sorted(str(p) for p in QUANT.glob("rsem/*.isoforms.results"))
    genes = sorted(str(p) for p in QUANT.glob("rsem/*.genes.results"))
    for y in (ep.catch_rsem(iso), ep.catch_rsem(iso, False), ep.catch_rsem(files=iso, verbose=False)):
        assert list(y["annotation"].columns) == ["GeneID"]
        assert y["annotation"]["GeneID"].iloc[0].count("|") >= 8
    y = ep.catch_rsem(genes, False)
    assert y["counts"].shape == (14, 3)
    assert y["annotation"]["GeneID"].iloc[0].startswith("ENSG")


def test_old_read_data_quantifier_output():
    dirs = sorted(str(p) for p in (QUANT / "salmon").iterdir())
    y = ep.read_data(dirs, source="salmon", verbose=False)
    assert {"Length", "EffectiveLength", "Overdispersion"} <= set(y["genes"].columns)
    # Transcript names are kept as read (not unpacked from Gencode)
    assert y["genes"].index[0].count("|") >= 8
    k = ep.read_data(sorted(str(p) for p in (QUANT / "kallisto").iterdir()), source="kallisto",
                     format="h5", labels=["a", "b", "c"], group=[0, 0, 1], verbose=False)
    assert list(k["samples"].index) == ["a", "b", "c"]
    assert "overdispersion.prior" in k
    g = ep.read_data(sorted(str(p) for p in QUANT.glob("rsem/*.genes.results")), source="rsem",
                     ngibbs=50, verbose=False)
    assert g["counts"].shape == (14, 3)


def test_old_positional_arguments(dge):
    lib = dge["samples"]["lib.size"].to_numpy()
    v = ep.voom(Y, X, lib, "none", 0.5, False)  # counts, design, lib_size, normalize_method, span, adaptive_span
    assert v["span"] == 0.5
    assert "correlation" in v and "sample_weights" in v
    vl = ep.voom_lmfit(Y, X, None, None, False, None, None, 10.0, lib, "none", 0.5, False)
    assert vl["coefficients"].shape == (400, 2)
    sets = {"s1": list(range(20)), "s2": list(range(50, 90))}
    cam = ep.camera(dge, sets, X, 1, None, False, False, 0.01, False)  # ..., inter_gene_cor, sort
    assert list(cam.index) == ["s1", "s2"]
    fr = ep.fry(dge, sets, X, 1, False)  # y, index, design, contrast, sort
    assert list(fr.index) == ["s1", "s2"]
    fit = ep.glm_ql_fit(dge, X)
    assert "top.proportion" in fit
    res = ep.glm_ql_ftest(fit, 1, None, True)  # glmfit, coef, contrast, poisson_bound
    assert {"logFC", "logCPM", "F", "PValue"} <= set(res["table"].columns)
