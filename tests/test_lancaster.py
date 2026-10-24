"""Lancaster aggregation of transcript p-values to genes (Yi et al. 2018).

lancaster() is checked against lancaster() and fisher() of the aggregation
R package (1.0.2); the reference values come from data/lancaster.R on the
seeded inputs built here, and are hard-coded so the tests need no R.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import edgepython as ep
from edgepython.results import _lancaster_by

QUANT = Path(__file__).parent / "data" / "quant"
KB = Path(__file__).parent / "data" / "kb"


def _lancaster_inputs():
    # 40 groups of 1-6 p-values with NaNs, ones, tiny values, zero, negative
    # and (group 7) NaN weights; data/lancaster.R embeds the same arrays
    rng = np.random.default_rng(2018)
    sizes = rng.integers(1, 7, 40)
    codes = np.repeat(np.arange(40), sizes)
    n = codes.size
    p = rng.uniform(size=n) ** 3
    p[rng.choice(n, 6, replace=False)] = np.nan
    p[rng.choice(n, 3, replace=False)] = 1.0
    p[rng.choice(n, 4, replace=False)] = [1e-30, 1e-100, 1e-300, 1e-12]
    w = rng.gamma(2.0, 2.0, n)
    w[rng.choice(n, 5, replace=False)] = 0.0
    w[rng.choice(n, 3, replace=False)] = -1.5
    w[codes == 7] = np.where(np.arange((codes == 7).sum()) == 0, np.nan, w[codes == 7])
    return codes, p, w


REF_LANCASTER = np.array([0.31328085717075649, 0.053331371308293476, 0.016998798108103074, 0.59824747611538309, 0.002087078366341599, 0.00062308738245586204, 0.9850707074352445, np.nan, 0.061807059065321149, 0.44290101339386306, 0.004037192326012711, 0.0024239798284883605, 0.825889585633102, 3.8608762058875967e-07, 0.0087824371277248972, 0.075739506608146434, 0.0058064409078288984, 2.1030523667357328e-05, 0.0049962107218430203, 1.8781910136183736e-06, 8.2644321583095961e-27, 0.10846320807306438, 0.47891501369708289, 2.605415631083842e-05, 8.5107548566799422e-10, 0.019969386152483431, 0.55470977283912659, 7.7621765646077318e-06, 0.22482356526706504, np.nan, 0.13316667165562524, 0.064570730074416646, 5.7643195671485535e-05, 0.48516335371075214, 0.0078728093798031609, 6.2785379268778969e-99, 0.081264457101297885, 1.3586972643883374e-07, 7.8629184002578158e-298, 0.30979000948098467])
REF_FISHER = np.array([0.32032283462863598, 0.055895440605873759, 0.016998798108103074, 0.60797182760557145, 0.002087078366341599, 0.00030223475068754405, 0.98470124589558272, 9.3303212263899482e-13, 0.061807059065321149, 0.44290101339386306, 0.0011443797055225636, 0.0013697966105190821, 0.7441379673992552, 5.1312267807549842e-08, 0.0016666571744357213, 0.0065527505239959035, 0.0026411141303893841, 2.5228271895537907e-05, 0.0026956736928291703, 7.9332974492369715e-07, 6.4752536130481643e-28, 0.13360574176961473, 0.47891501369708289, 5.2489993313129474e-05, 1.7461066626454428e-09, 0.019217994072305747, 0.0003543698254489768, 1.4869114144670773e-06, 0.14608303267474862, 0.61815134964773533, 0.092406049994305975, 0.064570730074416646, 6.0347478552261007e-05, 0.48516335371075214, 0.0080054449395000286, 9.6287374004007616e-100, 0.039304475166038702, 7.3022051347755868e-09, 1.5410356988680245e-295, 0.30979000948098467])


def _close(a, b, rtol=1e-12):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                               rtol=rtol, atol=0, equal_nan=True)


def test_lancaster_matches_aggregation_package():
    codes, p, w = _lancaster_inputs()
    _close(_lancaster_by(p, w, codes, 40), REF_LANCASTER)
    _close([ep.lancaster(p[codes == g], w[codes == g]) for g in range(40)], REF_LANCASTER)
    # Weights of 2 are Fisher's method
    _close(_lancaster_by(p, np.full_like(p, 2.0), codes, 40), REF_FISHER)


def test_lancaster_special_cases():
    # One p-value is returned as it is, whatever its weight, as in R
    assert ep.lancaster([0.3], [5.0]) == 0.3
    assert ep.lancaster([np.nan, 0.2], [1.0, np.nan]) == 0.2
    assert np.isnan(ep.lancaster([0.1, 0.2], [1.0, np.nan]))
    assert np.isnan(ep.lancaster([0.1, 0.2], [0.0, -1.0]))
    assert ep.lancaster([0.0, 0.5], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="Length of weights"):
        ep.lancaster([0.1, 0.2], [1.0])


def _salmon_exact():
    y = ep.catch_salmon(str(QUANT / "salmon"), verbose=False)
    y["samples"]["group"] = pd.Categorical(["A", "A", "B"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = ep.estimate_disp(y)
    return y, ep.exact_test(y)


def test_aggregate_tx_pvalues_with_gencode_names():
    y, res = _salmon_exact()
    out = ep.aggregate_tx_pvalues(res, y=y)
    genes = res["genes"]["EnsG"].to_numpy()
    assert list(out.index) == list(pd.unique(genes))
    assert list(out.columns) == ["GeneID", "GeneName", "NTx", "PValue", "FDR", "TopTx", "TopTx.logFC"]
    # Default weights: mean log normalized count (+ 0.5), as sleuth
    lib = (y["samples"]["lib.size"] * y["samples"]["norm.factors"]).to_numpy()
    w = np.log(y["counts"] * (np.exp(np.log(lib).mean()) / lib) + 0.5).mean(axis=1)
    p = res["table"]["PValue"].to_numpy()
    for g in out.index[:6]:
        m = genes == g
        _close([out.loc[g, "PValue"]], [ep.lancaster(p[m], w[m])])
        assert out.loc[g, "NTx"] == np.sum(w[m] > 0)
        top = np.flatnonzero(m)[np.argmin(p[m])]
        assert out.loc[g, "TopTx"] == res["genes"].index[top]
        assert out.loc[g, "TopTx.logFC"] == res["table"]["logFC"].iloc[top]
    assert out.loc["ENSG00000000001", "GeneName"] == "GENE1"
    from statsmodels.stats.multitest import multipletests
    ok = out["PValue"].notna()
    _close(out.loc[ok, "FDR"], multipletests(out.loc[ok, "PValue"], method="fdr_bh")[1])


def test_aggregate_tx_pvalues_weights_and_gene_ids():
    y, res = _salmon_exact()
    genes = res["genes"]["EnsG"].to_numpy()
    p = res["table"]["PValue"].to_numpy()
    fisher = ep.aggregate_tx_pvalues(res, weights="equal")
    for g in fisher.index[:5]:
        m = genes == g
        expected = p[m][0] if m.sum() == 1 else ep.lancaster(p[m], np.full(m.sum(), 2.0))
        assert fisher.loc[g, "PValue"] == pytest.approx(expected, rel=1e-12)
    w = np.linspace(1, 3, len(p))
    a = ep.aggregate_tx_pvalues(res, weights=w)
    b = ep.aggregate_tx_pvalues(res, gene=genes, weights=w)
    c = ep.aggregate_tx_pvalues(res, gene="EnsG", weights=w)
    _close(a["PValue"], b["PValue"])
    _close(a["PValue"], c["PValue"])


def test_aggregate_tx_pvalues_with_kb_t2g():
    pytest.importorskip("h5py")
    y = ep.catch_kallisto(str(KB / "bulk"), verbose=False)
    y["samples"]["group"] = pd.Categorical(["A", "A", "B"])
    res = ep.exact_test(y, dispersion=0.05)
    with pytest.raises(ValueError, match="give gene= or tx2gene="):
        ep.aggregate_tx_pvalues(res, y=y)
    out = ep.aggregate_tx_pvalues(res, y=y, tx2gene=str(KB / "t2g.txt"))
    assert list(out.index) == [f"ENSG{g:011d}" for g in range(1, 7)]
    assert list(out["GeneName"]) == [f"GENE{g}" for g in range(1, 7)]
    assert out["TopTx"].str.startswith("ENST").all()


def test_aggregate_tx_pvalues_errors():
    y, res = _salmon_exact()
    with pytest.raises(ValueError, match="Give y"):
        ep.aggregate_tx_pvalues(res)
    with pytest.raises(ValueError, match="weights has 3 values"):
        ep.aggregate_tx_pvalues(res, weights=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="'nope' not found"):
        ep.aggregate_tx_pvalues(res, y=y, gene="nope")
    with pytest.raises(ValueError, match="weights must be"):
        ep.aggregate_tx_pvalues(res, weights="sleuth")


def test_lancaster_finds_isoform_switches():
    # Two isoforms switch from 80/20 to 20/80 of the gene: the gene total is
    # unchanged, so a gene-level count test misses it, while the aggregated
    # p-values find it
    rng = np.random.default_rng(5)
    G, n = 600, 8
    ntx = rng.integers(2, 5, G)
    gene = np.repeat(np.arange(G), ntx)
    mu = np.outer(rng.gamma(2.0, 100, gene.size), np.ones(n))
    grp = np.repeat([0, 1], 4)
    switch = np.zeros(G, dtype=bool)
    switch[:60] = True
    for g in np.flatnonzero(switch):
        a, b = np.flatnonzero(gene == g)[:2]
        total = mu[a, 0] + mu[b, 0]
        mu[a], mu[b] = np.where(grp == 0, 0.8, 0.2) * total, np.where(grp == 0, 0.2, 0.8) * total
    counts = rng.negative_binomial(20, 20 / (20 + mu)).astype(float)
    design = np.c_[np.ones(n), grp]

    def ql(c, ann):
        yy = ep.calc_norm_factors(ep.make_dgelist(c, group=grp, genes=ann))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yy = ep.estimate_disp(yy, design)
            return yy, ep.glm_ql_ftest(ep.glm_ql_fit(yy, design), coef=1)

    y, res = ql(counts, pd.DataFrame({"gene": gene.astype(str)}))
    agg = ep.aggregate_tx_pvalues(res, y=y)
    gc = pd.DataFrame(counts).groupby(gene).sum().to_numpy()
    _, gres = ql(gc, pd.DataFrame({"gene": np.arange(G).astype(str)}))
    gene_fdr = ep.top_tags(gres, n=G, sort_by="none")["table"]["FDR"].to_numpy()
    found = agg.loc[np.arange(G).astype(str), "FDR"].to_numpy() < 0.05
    assert found[switch].mean() > 0.8
    assert found[~switch].mean() < 0.02
    assert (gene_fdr[switch] < 0.05).mean() < 0.1
    # Null genes are calibrated
    assert abs((agg["PValue"].to_numpy()[~switch] < 0.05).mean() - 0.05) < 0.03
