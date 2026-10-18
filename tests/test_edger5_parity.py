"""Parity with edgeR 5.0 (Bioconductor devel edgeR 4.99.8) and limma 3.99.2.

The reference values in data/edger5_parity.json.gz come from
data/edger5_parity.R, which also writes the inputs.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import edgepython as ep
from edgepython.glm_fit import mglm_one_group
from edgepython.glm_levenberg import mglm_levenberg, nbinom_deviance

DATA = Path(__file__).parent / "data"
FIXTURE = json.loads(gzip.decompress((DATA / "edger5_parity.json.gz").read_bytes()))
INPUTS = FIXTURE["inputs"]
EDGER = FIXTURE["edgeR"]


def _as_array(x):
    """JSON values as float arrays, with R's NA as NaN."""
    if isinstance(x, list):
        return np.array([_as_array(v) for v in x], dtype=np.float64)
    return np.nan if x == "NA" else float(x)


COUNTS = _as_array(INPUTS["counts"])
OFFSET = _as_array(INPUTS["offset"])
DISPERSION = _as_array(INPUTS["dispersion"])
WEIGHTS = _as_array(INPUTS["weights"])


def _edger(*keys):
    x = EDGER
    for k in keys:
        x = x[k]
    return _as_array(x)


def _assert_close(actual, expected, rtol, atol=1e-12):
    """Agreement to rtol, or to atol near zero; NA exactly where edgeR has NA."""
    actual = np.atleast_1d(np.asarray(actual, dtype=np.float64))
    expected = np.atleast_1d(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    ok = ~np.isnan(expected)
    np.testing.assert_allclose(actual[ok], expected[ok], rtol=rtol, atol=atol)


@pytest.mark.parametrize("design", ["oneway", "additive"])
@pytest.mark.parametrize("case, kwargs", [
    ("", {}),
    ("_weights", {"weights": WEIGHTS}),
    ("_poisson", {"dispersion": 0.0}),
])
def test_glm_fit_matches_edger(design, case, kwargs):
    # One-way designs use the groupwise Newton-Raphson fit, others the
    # Levenberg fit; the counts include all-zero genes (NA coefficients and
    # zero fitted values in edgeR) and genes with an all-zero group
    args = {"dispersion": DISPERSION, "offset": OFFSET, **kwargs}
    fit = ep.glm_fit(COUNTS, _as_array(INPUTS[design]), **args)
    key = f"glmfit_{design}{case}"
    _assert_close(fit["fitted.values"], _edger(key, "fitted"), rtol=1e-9)
    _assert_close(fit["deviance"], _edger(key, "deviance"), rtol=1e-9)
    _assert_close(fit["coefficients"], _edger(key, "coefficients"), rtol=1e-8)
    # Coefficients of all-zero groups are near -28 or -1e8, determined only
    # to the fitter's tolerance
    _assert_close(fit["unshrunk.coefficients"], _edger(key, "unshrunk"), rtol=1e-5)


def test_all_zero_genes_have_zero_fitted_values():
    fit = ep.glm_fit(COUNTS, _as_array(INPUTS["additive"]), dispersion=DISPERSION, offset=OFFSET)
    zero = COUNTS.sum(axis=1) == 0
    assert np.all(fit["fitted.values"][zero] == 0)
    assert np.all(np.isnan(fit["unshrunk.coefficients"][zero]))


def test_glm_lrt_contrast_with_an_all_zero_group():
    # The null model must not be started from coefficients of -1e8
    fit = ep.glm_fit(COUNTS, _as_array(INPUTS["oneway"]), dispersion=DISPERSION, offset=OFFSET)
    fit["AveLogCPM"] = np.zeros(COUNTS.shape[0])
    res = ep.glm_lrt(fit, contrast=np.array([0, 1, -1.0]))
    _assert_close(res["table"]["PValue"], _edger("glmlrt_contrast"), rtol=1e-7)


def test_pred_fc_and_add_prior_count_match_edger():
    pfc = ep.pred_fc(COUNTS, _as_array(INPUTS["additive"]), prior_count=1,
                     offset=OFFSET, dispersion=DISPERSION)
    _assert_close(pfc, _edger("pred_fc"), rtol=1e-8)
    apc = ep.add_prior_count(COUNTS, offset=OFFSET, prior_count=2)
    _assert_close(apc["y"], _edger("add_prior_count", "y"), rtol=1e-12)
    _assert_close(np.broadcast_to(apc["offset"], COUNTS.shape),
                  _edger("add_prior_count", "offset"), rtol=1e-12)


def test_mglm_one_group_and_levenberg_match_edger():
    _assert_close(mglm_one_group(COUNTS, dispersion=DISPERSION, offset=OFFSET),
                  _edger("mglm_one_group"), rtol=1e-12)
    fit = mglm_levenberg(COUNTS, _as_array(INPUTS["additive"]), dispersion=DISPERSION,
                         offset=OFFSET)
    _assert_close(fit["coefficients"], _edger("mglm_levenberg"), rtol=1e-5)


def test_nbinom_deviance_matches_edger_c_unit_deviance():
    fitted = _edger("glmfit_additive", "fitted")
    _assert_close(nbinom_deviance(COUNTS, fitted, DISPERSION), _edger("nbinom_deviance"),
                  rtol=1e-11)
    # Dispersions below 1e-4 use edgeR's Taylor expansion
    y = COUNTS[:40]
    mean = np.tile(y.mean(axis=1)[:, None] + 0.5, (1, y.shape[1]))
    _assert_close(nbinom_deviance(y, mean, 1e-6), _edger("nbinom_deviance_small_disp"),
                  rtol=1e-12)


QL_KEYS = [("dispersion", "dispersion"), ("average.ql.dispersion", "average_ql"),
           ("df.residual.adj", "df_adj"), ("deviance.adj", "deviance_adj"),
           ("s2.prior", "s2_prior"), ("df.prior", "df_prior"), ("s2.post", "s2_post")]


def _qlfit(case):
    design = _as_array(INPUTS["additive"])
    if case in ("qlfit_dgelist", "qlfit_nonrobust", "qlfit_no_trend"):
        dge = ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(COUNTS)), design)
        kwargs = {"qlfit_dgelist": {"robust": True}, "qlfit_nonrobust": {},
                  "qlfit_no_trend": {"abundance_trend": False}}[case]
        return ep.glm_ql_fit(dge, design, **kwargs)
    if case == "qlfit_weights":
        return ep.glm_ql_fit(COUNTS, design, offset=OFFSET, weights=WEIGHTS, robust=True)
    return ep.glm_ql_fit(COUNTS, design, offset=OFFSET, dispersion=10.0)


@pytest.mark.parametrize("case", ["qlfit_dgelist", "qlfit_nonrobust", "qlfit_no_trend",
                                  "qlfit_weights", "qlfit_disp10"])
def test_glm_ql_fit_matches_edger5(case):
    # edgeR 5.0 ignores dispersions stored in a DGEList, no longer caps a
    # preset dispersion at 4, and weights the hat matrix by sqrt(mu * w / ...)
    fit = _qlfit(case)
    for key, rkey in QL_KEYS:
        expected = _edger(case, rkey)
        actual = np.broadcast_to(fit[key], np.shape(expected)) if np.ndim(expected) else fit[key]
        _assert_close(np.ravel(actual)[:np.size(expected)], np.ravel(expected), rtol=1e-6)
    if case != "qlfit_disp10":
        res = ep.glm_ql_ftest(fit, coef=4)
        _assert_close(res["table"]["PValue"], _edger(case, "pvalue"), rtol=1e-6)


@pytest.mark.parametrize("robust", [False, True])
def test_unequal_df_squeeze_var_matches_limma(robust):
    from edgepython.limma_port import squeeze_var
    s = squeeze_var(_as_array(INPUTS["sv_var"]), df=_as_array(INPUTS["sv_df"]),
                    covariate=_as_array(INPUTS["sv_cov"]), robust=robust)
    key = "squeezevar_unequal" + ("_robust" if robust else "")
    for name in ["var_prior", "df_prior", "var_post"]:
        _assert_close(np.broadcast_to(s[name], np.shape(_edger(key, name))), _edger(key, name),
                      rtol=1e-10)


def test_unequal_df_squeeze_var_with_loess_span_below_one():
    # limma's chooseLowessSpan(n, small.n = 500) uses min.span = 0.3
    from edgepython.limma_port import choose_lowess_span, squeeze_var
    np.testing.assert_allclose([choose_lowess_span(20000, small_n=500), choose_lowess_span(1000)],
                               _edger("choose_lowess_span"), rtol=1e-14)
    s = squeeze_var(_as_array(INPUTS["sv2_var"]), df=_as_array(INPUTS["sv2_df"]),
                    covariate=_as_array(INPUTS["sv2_cov"]))
    _assert_close(s["var_prior"], _edger("squeezevar_unequal_2000", "var_prior"), rtol=1e-10)
    _assert_close(s["df_prior"], _edger("squeezevar_unequal_2000", "df_prior"), rtol=1e-10)


def test_optimize_brent_reproduces_r_optimize():
    # At a bound, R's answer depends on where Brent's search stops
    from edgepython.utils import optimize_brent
    found = [optimize_brent(lambda x: (x - 0.3) ** 2 + np.sin(5 * x), 0, 2),
             optimize_brent(lambda x: -np.log(x), 0.5, 0.9998)]
    # (the JSON file holds 15 significant digits)
    np.testing.assert_allclose(found, _edger("optimize"), rtol=1e-14)


def test_cox_reid_dispersion_and_adjusted_profile_lik_match_edger():
    from edgepython.dispersion_lowlevel import adjusted_profile_lik, disp_cox_reid
    offset = np.tile(OFFSET, (COUNTS.shape[0], 1))
    for i, design in enumerate(["oneway", "additive"]):
        assert disp_cox_reid(COUNTS, _as_array(INPUTS[design]), offset=OFFSET) == \
            pytest.approx(_edger("disp_cox_reid")[i], rel=1e-6)
    # All-zero genes have zero fitted values, which contribute no likelihood,
    # and a Cox-Reid term of -log(1e-10) per coefficient
    for design, key in [("additive", "apl"), ("oneway", "apl_oneway")]:
        _assert_close(adjusted_profile_lik(0.1, COUNTS, _as_array(INPUTS[design]), offset),
                      _edger(key), rtol=1e-9, atol=1e-7)
