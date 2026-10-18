"""Parity with edgeR 4.8.2 and limma 3.66.0 on the glmQLFit / glmQLFTest path.

The reference values in data/edger_ql_parity.json come from
data/edger_ql_parity.R. The last test runs the whole pipeline against R
when Rscript, edgeR and jsonlite are installed.
"""

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import edgepython as ep
from edgepython.dispersion import WLEB
from edgepython.glm_levenberg import nbinom_deviance
from edgepython.limma_port import squeeze_var
from edgepython.smoothing import locfit_by_col

DATA = Path(__file__).parent / "data"
FIXTURE = json.loads((DATA / "edger_ql_parity.json").read_text())
INPUTS = FIXTURE["inputs"]
EDGER = {k: np.asarray(v, dtype=np.float64) for k, v in FIXTURE["edgeR"].items()}


def _array(name):
    return np.asarray(INPUTS[name], dtype=np.float64)


def test_tmm_matches_edger_on_tied_counts():
    # Small integer counts tie exactly; the log-ratio must round as edgeR's
    # log2((obs/nO)/(ref/nR)) does, or ties are ranked and trimmed differently
    y = _array("tmm")
    dge = ep.calc_norm_factors(ep.make_dgelist(y, lib_size=y.sum(axis=0)), method="TMM")
    np.testing.assert_allclose(dge["samples"]["norm.factors"], EDGER["tmm"], rtol=1e-12)


@pytest.mark.parametrize("legacy_span, expected", [(False, 0), (True, 1)])
def test_wleb_span_matches_choose_lowess_span(legacy_span, expected):
    rng = np.random.default_rng(1)
    theta = np.arange(-10, 11, dtype=np.float64)
    loglik = -(theta[None, :] / 5) ** 2 + rng.normal(0, 0.1, (105, 1))
    out = WLEB(theta, loglik, covariate=rng.normal(size=105),
               legacy_span=legacy_span, individual=False)
    assert out["span"] == pytest.approx(EDGER["span105"][expected], rel=1e-12)


def test_locfit_degree0_matches_locfit_tree_evaluation():
    fitted = locfit_by_col(_array("loc_y"), _array("loc_x"),
                           weights=_array("loc_w"), span=0.6, degree=0)
    np.testing.assert_allclose(fitted, EDGER["locfit"], rtol=1e-12, atol=1e-12)


def test_locfit_degree0_handles_more_than_1000_points():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1500)
    fitted = locfit_by_col(np.column_stack([np.sin(x), x]), x, span=0.3, degree=0)
    assert fitted.shape == (1500, 2)
    assert np.all(np.isfinite(fitted))


def test_robust_squeeze_var_takes_limma_infinite_df_branch():
    assert np.isinf(EDGER["nonrobust_df"]).all()
    out = squeeze_var(_array("var"), df=6, covariate=_array("cov"), robust=True)
    np.testing.assert_allclose(out["var_prior"], EDGER["robust_var_prior"], rtol=1e-8)
    assert np.isinf(out["df_prior"]).all()
    assert np.isinf(EDGER["robust_df_prior"]).all()


def test_robust_squeeze_var_unequal_df_at_infinite_df():
    out = squeeze_var(_array("var"), df=_array("df_unequal"), covariate=_array("cov"),
                      robust=True, legacy=True)
    np.testing.assert_allclose(out["var_prior"], EDGER["unequal_var_prior"], rtol=1e-8)
    np.testing.assert_array_equal(out["df_prior"], EDGER["unequal_df_prior"])


def test_robust_squeeze_var_holds_prior_constant_beyond_covariate_range():
    # Rows with zero df lie at both ends of the covariate; limma fills in
    # their prior with approx(rule = 2), which does not extrapolate
    out = squeeze_var(_array("var"), df=_array("df_zero"), covariate=_array("cov"),
                      robust=True)
    np.testing.assert_allclose(out["var_prior"], EDGER["zero_var_prior"], rtol=1e-8)


def test_glm_ql_fit_passes_legacy_to_squeeze_var(monkeypatch):
    glm_fit_module = importlib.import_module("edgepython.glm_fit")
    seen = {}
    original = glm_fit_module.squeeze_var

    def spy(*args, **kwargs):
        seen["legacy"] = kwargs.get("legacy")
        return original(*args, **kwargs)

    monkeypatch.setattr(glm_fit_module, "squeeze_var", spy)
    y = _array("zg")
    design = np.repeat(np.eye(3), 2, axis=0)
    dge = ep.estimate_disp(ep.make_dgelist(y), design)
    for legacy in (True, False):
        ep.glm_ql_fit(dge, design=design, legacy=legacy)
        assert seen["legacy"] is legacy


def test_estimate_disp_with_an_all_zero_group_matches_edger():
    # The first three rows are zero in the third group. edgeR drops those
    # libraries and the redundant column before computing the grid.
    y = _array("zg")
    design = np.column_stack([np.ones(6), np.repeat(np.eye(3)[:, 1:], 2, axis=0)])
    dge = ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(y)), design)
    assert dge["common.dispersion"] == pytest.approx(float(EDGER["zg_common"][0]), rel=1e-5)
    np.testing.assert_allclose(dge["trended.dispersion"], EDGER["zg_trended"], rtol=1e-4)
    np.testing.assert_allclose(dge["tagwise.dispersion"], EDGER["zg_tagwise"], rtol=1e-3)


def test_zero_dispersion_array_is_poisson():
    rng = np.random.default_rng(3)
    y = rng.poisson(3.0, size=(20, 6)).astype(np.float64)
    mu = np.full_like(y, 3.0)
    np.testing.assert_allclose(nbinom_deviance(y, mu, np.zeros(20)),
                               nbinom_deviance(y, mu, 0.0))
    fit = ep.glm_fit(y, design=np.repeat(np.eye(3), 2, axis=0),
                     dispersion=np.zeros(20), prior_count=0)
    assert np.all(np.isfinite(fit["deviance"]))


def test_poisson_bound_only_raises_flagged_p_values():
    y = _array("zg")
    design = np.repeat(np.eye(3), 2, axis=0)
    dge = ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(y)), design)
    # As in edgeR, only legacy fits apply the Poisson bound
    fit = ep.glm_ql_fit(dge, design=design, legacy=True)
    # Put alternate rows just below and just above the bound,
    # s2.post * (1 + dispersion * fitted) < 1 in some library
    mu_min = np.asarray(fit["fitted.values"]).min(axis=1)
    below = np.arange(len(mu_min)) % 2 == 0
    fit["s2.post"] = np.where(below, 0.9, 1.1) / (1 + np.asarray(fit["dispersion"]) * mu_min)
    contrast = np.array([0.0, 1.0, -1.0])
    bounded = ep.glm_ql_ftest(fit, contrast=contrast)["table"]["PValue"].to_numpy()
    plain = ep.glm_ql_ftest(fit, contrast=contrast, poisson_bound=False)["table"]["PValue"].to_numpy()
    assert np.all(bounded >= plain)
    np.testing.assert_array_equal(bounded[~below], plain[~below])
    assert np.any(bounded[below] > plain[below])


def _r_available():
    if shutil.which("Rscript") is None:
        return False
    out = subprocess.run(
        ["Rscript", "-e", "cat(all(sapply(c('edgeR', 'jsonlite'), requireNamespace, quietly=TRUE)))"],
        capture_output=True, text=True)
    return out.stdout.strip() == "TRUE"


def _simulate(rng, ngenes, design, beta, dispersion=0.01):
    base = np.exp(rng.normal(2.0, 1.2, ngenes))
    base[:60] = np.exp(rng.uniform(0.0, 1.2, 60))
    mu = base[:, None] * np.exp(design @ beta)[None, :]
    size = 1 / dispersion
    return rng.poisson(mu * rng.gamma(size, 1 / size, mu.shape))


@pytest.mark.skipif(not _r_available(), reason="Rscript with edgeR and jsonlite required")
def test_ql_pipeline_matches_edger(tmp_path):
    """TMM, estimateDisp, glmQLFit(robust, legacy) and glmQLFTest against edgeR."""
    rng = np.random.default_rng(20260617)
    batch = np.repeat([0, 1], 6)
    condition = np.tile(np.repeat([0, 1], 3), 2)
    level = np.repeat([0, 1, 2], 4)
    cases = {
        "batch_condition": (np.column_stack([np.ones(12), batch, condition]),
                            [0, 0.3, 0.4], {"coef": 2}),
        "three_levels": (np.column_stack([np.ones(12), level == 1, level == 2]).astype(float),
                         [0, 0.3, -0.2], {"contrast": [0, 1, -1]}),
        "covariate": (np.column_stack([np.ones(12), np.linspace(-1, 1, 12), condition]),
                      [0, 0.2, 0.4], {"coef": 2}),
        "many_rows": (np.column_stack([np.ones(12), condition]), [0, 0.4], {"coef": 1}),
    }
    payload = {}
    for name, (design, beta, test) in cases.items():
        ngenes = 1500 if name == "many_rows" else 600
        payload[name] = {"counts": _simulate(rng, ngenes, design, np.asarray(beta)).tolist(),
                         "design": design.tolist(), **{k: v for k, v in test.items()}}
    (tmp_path / "in.json").write_text(json.dumps(payload))
    script = tmp_path / "run.R"
    script.write_text(f"""
suppressMessages({{library(edgeR); library(jsonlite)}})
cases <- fromJSON("{tmp_path / 'in.json'}", simplifyMatrix = TRUE)
out <- lapply(cases, function(c) {{
  y <- estimateDisp(calcNormFactors(DGEList(c$counts)), c$design)
  fit <- glmQLFit(y, c$design, robust = TRUE, legacy = TRUE)
  res <- if (!is.null(c$contrast)) glmQLFTest(fit, contrast = c$contrast)
         else glmQLFTest(fit, coef = c$coef + 1)
  list(trended = y$trended.dispersion, F = res$table$F, PValue = res$table$PValue)
}})
write_json(out, "{tmp_path / 'out.json'}", digits = NA)
""")
    subprocess.run(["Rscript", str(script)], check=True)
    expected = json.loads((tmp_path / "out.json").read_text())

    for name, (design, _, test) in cases.items():
        dge = ep.estimate_disp(
            ep.calc_norm_factors(ep.make_dgelist(np.asarray(payload[name]["counts"]))), design)
        fit = ep.glm_ql_fit(dge, design=design, robust=True, legacy=True)
        kwargs = {"contrast": np.asarray(test["contrast"], dtype=float)} if "contrast" in test \
            else {"coef": test["coef"]}
        table = ep.glm_ql_ftest(fit, **kwargs)["table"]
        r = expected[name]
        np.testing.assert_allclose(dge["trended.dispersion"], r["trended"], rtol=1e-2,
                                   err_msg=name)
        np.testing.assert_allclose(table["PValue"], r["PValue"], rtol=1e-3, atol=1e-12,
                                   err_msg=name)
