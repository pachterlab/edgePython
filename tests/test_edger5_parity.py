"""Parity with edgeR 5.0 (Bioconductor devel edgeR 4.99.8) and limma 3.99.2.

The reference values in data/edger5_parity.json.gz come from
data/edger5_parity.R, which also writes the inputs.
"""

import gzip
import json
import re
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


def _ftest_fits():
    design = _as_array(INPUTS["additive"])
    dge = ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(COUNTS)), design)
    return (ep.glm_ql_fit(dge, design, robust=True),
            ep.glm_ql_fit(dge, design, robust=True, legacy=True))


FTEST_CASES = {
    "ftest_new": ("new", {"coef": 4}),
    "ftest_new_upshot": ("new", {"coef": 4, "lfc": np.log2(1.5)}),
    "ftest_new_treat": ("new", {"coef": 4, "lfc": np.log2(1.5), "upshot": False}),
    "ftest_new_fc": ("new", {"coef": 4, "fc": 1.5}),
    "ftest_new_contrast": ("new", {"contrast": np.array([0, 0, 0, 1, -1.0]), "lfc": np.log2(1.5)}),
    "ftest_new_twocoef": ("new", {"coef": [3, 4], "lfc": np.log2(1.5)}),
    "ftest_new_twolfc": ("new", {"coef": [3, 4], "lfc": [0.3, 0.8], "upshot": False}),
    "ftest_new_contrast2": ("new", {"contrast": "contrast2", "lfc": np.log2(1.5)}),
    "ftest_legacy": ("legacy", {"coef": 4}),
    "ftest_legacy_upshot": ("legacy", {"coef": 4, "lfc": np.log2(1.5)}),
}


@pytest.fixture(scope="module")
def ftest_fits():
    return dict(zip(["new", "legacy"], _ftest_fits()))


@pytest.mark.parametrize("case", list(FTEST_CASES))
def test_glm_ql_ftest_thresholds_match_edger5(ftest_fits, case):
    # TREAT and UPSHOT refit the null model with the tested coefficients
    # pinned at the corners of the threshold box
    which, kwargs = FTEST_CASES[case]
    kwargs = dict(kwargs)
    if isinstance(kwargs.get("contrast"), str):
        kwargs["contrast"] = _as_array(INPUTS[kwargs["contrast"]])
    res = ep.glm_ql_ftest(ftest_fits[which], **kwargs)
    table = res["table"]
    rtol = 1e-6
    logfc = table[[c for c in table.columns if c.startswith("logFC")]].to_numpy()
    _assert_close(logfc.reshape(np.shape(_edger(case, "logFC"))), _edger(case, "logFC"),
                  rtol=rtol, atol=1e-9)
    _assert_close(table["F"], _edger(case, "F"), rtol=rtol, atol=1e-7)
    _assert_close(table["PValue"], _edger(case, "pvalue"), rtol=rtol, atol=1e-12)
    _assert_close(res["df.total"], _edger(case, "df_total"), rtol=1e-10)


def test_contrast_as_coef_matches_limma_with_pivoting():
    # A rank-2 contrast matrix whose second column is a multiple of the
    # first: R's qr() moves it to the end
    from edgepython.limma_port import contrast_as_coef
    design = _as_array(INPUTS["additive"])
    contrast = np.zeros((5, 3))
    contrast[1] = [1, 2, 0]
    contrast[2] = [0, 0, 1]
    reform = contrast_as_coef(design, contrast, first=True)
    assert reform["rank"] == 2 and reform["coef"] == [0, 1]
    # The reparametrized design reproduces the fitted values of the original
    beta = np.linalg.lstsq(reform["design"], design @ np.arange(1.0, 6.0), rcond=None)[0]
    np.testing.assert_allclose(reform["design"] @ beta, design @ np.arange(1.0, 6.0), atol=1e-12)
    np.testing.assert_allclose(beta[:2], contrast[:, [0, 2]].T @ np.arange(1.0, 6.0), atol=1e-12)


@pytest.fixture(scope="module")
def treat_fits():
    design = _as_array(INPUTS["additive"])
    dge = ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(COUNTS)), design)
    return {"lrt": ep.glm_fit(dge, design), "lrt0": ep.glm_fit(dge, design, prior_count=0),
            "ql": ep.glm_ql_fit(dge, design, robust=True),
            "qllegacy": ep.glm_ql_fit(dge, design, robust=True, legacy=True)}


TREAT_CASES = {
    "treat_lrt": ("lrt", {}),
    "treat_lrt_worst": ("lrt", {"null": "worst.case"}),
    "treat_lrt_unshrunk": ("lrt0", {}),
    "treat_ql": ("ql", {}),
    "treat_ql_worst": ("ql", {"null": "worst.case"}),
    "treat_ql_contrast": ("ql", {"coef": None, "contrast": np.array([0, 0, 0, 1, -1.0])}),
    "treat_qllegacy": ("qllegacy", {}),
}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("case", list(TREAT_CASES))
def test_glm_treat_matches_edger5(treat_fits, case, legacy):
    # legacy=False is edgeR 5.0's new default. The references for legacy
    # QL fits and for legacy=True come from edgeR with two bugs worked around
    # (see edger5_parity.R): poisson.bound unset for legacy QL fits, and a
    # misaligned index when some genes are NA.
    which, kwargs = TREAT_CASES[case]
    key = case + ("_legacy" if legacy else "")
    args = {"coef": 4, "lfc": np.log2(1.5), "legacy": legacy, **kwargs}
    table = ep.glm_treat(treat_fits[which], **args)["table"]
    rtol = 1e-7 if which == "ql" else 1e-6
    _assert_close(table["PValue"], _edger(key, "pvalue"), rtol=rtol, atol=1e-12)
    # Without prior counts logFC is the unshrunk coefficient (see below)
    _assert_close(table["logFC"], _edger(key, "logFC"),
                  rtol=max(rtol, 1e-5) if which == "lrt0" else rtol, atol=1e-9)
    if which == "lrt0":
        assert "unshrunk.logFC" not in table.columns
    else:
        # Unshrunk coefficients of genes with an all-zero group (about -25 on
        # the log2 scale) are determined only to the fitter's tolerance
        _assert_close(table["unshrunk.logFC"], _edger(key, "unshrunk_logFC"),
                      rtol=max(rtol, 1e-5), atol=1e-9)


def test_glm_treat_legacy_p_values_are_genewise(treat_fits):
    # A gene's p-value must not depend on whether other genes are NA
    fit = treat_fits["lrt"]
    nonzero = COUNTS.sum(axis=1) > 0
    full = ep.glm_treat(fit, coef=4, lfc=np.log2(1.5), legacy=True)["table"]["PValue"].to_numpy()
    sub = {k: (v[nonzero] if isinstance(v, np.ndarray) and v.shape[:1] == (COUNTS.shape[0],) else v)
           for k, v in fit.items()}
    part = ep.glm_treat(sub, coef=4, lfc=np.log2(1.5), legacy=True)["table"]["PValue"].to_numpy()
    assert np.all(np.isnan(full[~nonzero]))
    np.testing.assert_allclose(full[nonzero], part, rtol=1e-12)


def _fresh_dgelist(weights=None):
    dge = ep.calc_norm_factors(ep.make_dgelist(COUNTS, group=np.repeat(["A", "B", "C"], 3)))
    if weights is not None:
        dge["weights"] = weights
    return dge


DISP_CASES = {
    "disp_glm": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"])),
    "disp_glm_robust": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"]),
                                                robust=True),
    "disp_glm_weights": lambda: ep.estimate_disp(_fresh_dgelist(WEIGHTS), _as_array(INPUTS["additive"])),
    "disp_classic": lambda: ep.estimate_disp(_fresh_dgelist()),
    "disp_classic_robust": lambda: ep.estimate_disp(_fresh_dgelist(), robust=True),
    "disp_classic_onegroup": lambda: ep.estimate_disp(ep.calc_norm_factors(ep.make_dgelist(COUNTS))),
    "disp_trend_none": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"]),
                                                trend_method="none"),
    "disp_trend_movingave": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"]),
                                                     trend_method="movingave"),
    "disp_trend_loess": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"]),
                                                 trend_method="loess"),
    "disp_trend_locfit.mixed": lambda: ep.estimate_disp(_fresh_dgelist(), _as_array(INPUTS["additive"]),
                                                        trend_method="locfit.mixed"),
    # edgeR 5.0 does not use the classic model with genewise offsets (and then
    # fails); the reference is the GLM with the classic group design
    "disp_genewise_offset": lambda: ep.estimate_disp(COUNTS, group=np.repeat(["A", "B", "C"], 3),
                                                     offset=_as_array(INPUTS["genewise_offset"])),
}


@pytest.mark.parametrize("case", list(DISP_CASES))
def test_estimate_disp_matches_edger5(case):
    # The APL grid warm-starts each dispersion from the previous one's
    # coefficients, as edgeR does
    e = DISP_CASES[case]()
    # The classic estimators maximize with R's optimize() (tol eps^0.25), so a last-bit
    # change in scipy's norm.logsf (scipy >= 1.16) moves them by ~1e-8; 1e-5 is still
    # far inside the optimizer's own precision.
    rtol = 1e-5 if case.startswith("disp_classic") else 1e-8
    for key, rkey in [("common.dispersion", "common"), ("trended.dispersion", "trended"),
                      ("tagwise.dispersion", "tagwise"), ("prior.df", "prior_df")]:
        if EDGER[case].get(rkey) in (None, [], {}):
            continue
        expected = _edger(case, rkey)
        actual = np.broadcast_to(np.asarray(e[key], dtype=np.float64), np.shape(expected))
        _assert_close(actual, expected, rtol=rtol)


def test_estimate_common_disp_classic_matches_edger():
    assert ep.estimate_common_disp(_fresh_dgelist())["common.dispersion"] == \
        pytest.approx(_edger("common_disp_classic")[0], rel=1e-8)


def test_locfit_degree1_matches_locfit_tree_evaluation():
    from edgepython.smoothing import locfit_by_col
    fitted = locfit_by_col(_as_array(INPUTS["loc1_y"]), _as_array(INPUTS["loc1_x"]),
                           weights=_as_array(INPUTS["loc1_w"]), span=0.3, degree=1)
    np.testing.assert_allclose(fitted, _edger("locfit_degree1"), rtol=1e-11, atol=1e-11)


# Tolerances for the older estimators. Cox-Reid dispersions on designs that
# need the Levenberg fit agree to R's Brent tolerance (1e-5 on
# dispersion^0.25), as their APL sums differ at 1e-9 through the rounding of
# the fits' QR start values. In the one-way bin with the lowest counts the
# Cox-Reid dispersion is about 1e-7, where the APL is flat and R's own value
# is set by rounding in lgamma(y + 1/dispersion).
OLD_ESTIMATOR_TOL = {
    "oneway": {"exact": (1e-9, 0), "coxreid": (1e-3, 5e-4)},
    "additive": {"exact": (1e-9, 0), "coxreid": (1e-3, 1e-5)},
}


@pytest.mark.parametrize("design", ["oneway", "additive"])
def test_glm_common_disp_methods_match_edger(design):
    X = _as_array(INPUTS[design])
    for method in ["Pearson", "deviance"]:
        d = ep.estimate_glm_common_disp(_fresh_dgelist(), X, method=method)["common.dispersion"]
        assert d == pytest.approx(_edger(f"glmcommon_{method}_{design}")[0], rel=1e-12)
    d = ep.estimate_glm_common_disp(_fresh_dgelist(), X)["common.dispersion"]
    rtol = 1e-9 if design == "oneway" else 1e-4
    assert d == pytest.approx(_edger(f"glmcommon_CoxReid_{design}")[0], rel=rtol)


@pytest.mark.parametrize("design", ["oneway", "additive"])
@pytest.mark.parametrize("method", ["bin.spline", "bin.loess", "power", "spline"])
def test_glm_trended_disp_matches_edger(design, method):
    # power and spline use R's Nelder-Mead and R's ns() basis
    y = ep.estimate_glm_trended_disp(_fresh_dgelist(), _as_array(INPUTS[design]), method=method)
    kind = "coxreid" if method.startswith("bin") else "exact"
    rtol, atol = OLD_ESTIMATOR_TOL[design][kind]
    _assert_close(y["trended.dispersion"], _edger(f"glmtrended_{method}_{design}"), rtol=rtol, atol=atol)


@pytest.mark.parametrize("method", ["bin.spline", "bin.loess"])
def test_binned_trends_with_many_bins_match_edger(method):
    big = _as_array(INPUTS["big_counts"])
    fresh = lambda: ep.calc_norm_factors(ep.make_dgelist(big, group=np.repeat(["A", "B", "C"], 3)))
    # dispBinTrend with 20 bins fits the spline or loess trend
    y = ep.estimate_glm_trended_disp(fresh(), _as_array(INPUTS["additive"]), method=method)
    _assert_close(y["trended.dispersion"], _edger(f"bigtrended_{method}"), rtol=1e-3)
    # The classic estimateTrendedDisp, with 50 bins of estimateCommonDisp. Each bin is
    # an optimize() fit, so scipy >= 1.16's last-bit change in norm.logsf moves the
    # trend by ~1e-6 (see test_estimate_disp_matches_edger5)
    y = ep.estimate_trended_disp(ep.estimate_common_disp(fresh()), method=method)
    _assert_close(y["trended.dispersion"], _edger(f"classic_trended_{method}"), rtol=1e-5)


@pytest.mark.parametrize("design", ["oneway", "additive"])
def test_glm_tagwise_and_robust_disp_match_edger(design):
    X = _as_array(INPUTS[design])
    rtol, atol = OLD_ESTIMATOR_TOL[design]["coxreid"]
    y = ep.estimate_glm_trended_disp(ep.estimate_glm_common_disp(_fresh_dgelist(), X), X)
    trended = y["trended.dispersion"].copy()
    y = ep.estimate_glm_tagwise_disp(y, X)
    # The trended dispersions must not be overwritten
    np.testing.assert_array_equal(y["trended.dispersion"], trended)
    _assert_close(y["tagwise.dispersion"], _edger(f"glmtagwise_{design}"), rtol=rtol, atol=atol)
    y = ep.estimate_glm_tagwise_disp(ep.estimate_glm_common_disp(_fresh_dgelist(), X), X, trend=False)
    notrend_rtol = 1e-9 if design == "oneway" else 1e-4
    _assert_close(y["tagwise.dispersion"], _edger(f"glmtagwise_notrend_{design}"), rtol=notrend_rtol)
    y = ep.estimate_glm_robust_disp(_fresh_dgelist(), X)
    _assert_close(y["tagwise.dispersion"], _edger(f"glmrobust_{design}"), rtol=rtol, atol=atol)


@pytest.mark.parametrize("case, kwargs", [
    ("classic_tagwise", {}),
    ("classic_tagwise_loess", {"trend": "loess", "span": 0.8, "prior_df": 20}),
    ("classic_tagwise_none", {"trend": "none"}),
])
def test_classic_tagwise_disp_matches_edger(case, kwargs):
    y = ep.estimate_tagwise_disp(ep.estimate_common_disp(_fresh_dgelist()), **kwargs)
    _assert_close(y["tagwise.dispersion"], _edger(case), rtol=1e-6)


def test_r_optimizers_reproduce_r():
    # Nelder-Mead and uniroot, as optim() and uniroot() give them in R
    from edgepython.utils import optim_nelder_mead, uniroot_brent
    par, value = optim_nelder_mead(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
                                   [-1.2, 1])
    np.testing.assert_allclose(par, [1.0002601387256695, 1.000505999303765], rtol=1e-15)
    assert value == pytest.approx(8.8252410967227472e-08, rel=1e-13)
    assert uniroot_brent(lambda x: x ** 3 - 2 * x - 5, 2, 3, tol=1e-5) == 2.0945512665841388


def test_cpm_matches_edger():
    # Raw CPM is y * 1e6 / lib.size; filter_by_expr compares it with a cutoff
    # that it can equal exactly, which the filter tests check. (The JSON file
    # holds 15 significant digits.)
    lib_size = _as_array(INPUTS["lib_size"])
    np.testing.assert_allclose(ep.cpm(COUNTS, lib_size=lib_size), _edger("cpm_raw"), rtol=1e-13)
    np.testing.assert_allclose(ep.cpm(COUNTS, lib_size=lib_size, log=True, prior_count=2),
                               _edger("cpm_log"), rtol=1e-13)


FILTER_CASES = {
    "filter_group": {},
    "filter_design": {"design": "additive"},
    "filter_min_sample_size": {"min_sample_size": 2},
    "filter_stringent": {"preset": "stringent"},
    "filter_moderate": {"preset": "moderate"},
    "filter_lenient": {"preset": "lenient"},
}


@pytest.mark.parametrize("case", list(FILTER_CASES))
def test_filter_by_expr_matches_edger5(case):
    kwargs = dict(FILTER_CASES[case])
    if "design" in kwargs:
        kwargs["design"] = _as_array(INPUTS[kwargs["design"]])
    dge = ep.make_dgelist(COUNTS, group=np.repeat(["A", "B", "C"], 3))
    np.testing.assert_array_equal(ep.filter_by_expr(dge, **kwargs), _edger(case).astype(bool))


def test_filter_by_expr_design_without_intercept():
    # R's hat() adds an intercept column to the design
    design = _as_array(INPUTS["filter_covariate"]).reshape(-1, 1)
    np.testing.assert_array_equal(ep.filter_by_expr(COUNTS, design=design),
                                  _edger("filter_covariate").astype(bool))


def test_filter_by_expr_caps_min_sample_size():
    # Two groups of 70: 10 + 0.7 * 60 = 52 samples, capped at max.n.samples = 40
    y = _as_array(INPUTS["filter_large"])
    group = _as_array(INPUTS["filter_large_group"])
    np.testing.assert_array_equal(ep.filter_by_expr(y, group=group),
                                  _edger("filter_large").astype(bool))
    np.testing.assert_array_equal(ep.filter_by_expr(y, group=group, max_n_samples=np.inf),
                                  _edger("filter_large_nocap").astype(bool))
    with pytest.warns(UserWarning, match="Ignoring arguments set individually"):
        ep.filter_by_expr(y, group=group, preset="lenient", min_count=5)


def _offset_prior_dgelist():
    import pandas as pd
    dge = ep.calc_norm_factors(ep.make_dgelist(
        COUNTS, group=np.repeat(["A", "B", "C"], 3),
        genes=pd.DataFrame({"AveLength": _as_array(INPUTS["ave_length"])})))
    dge["offset.prior"] = _as_array(INPUTS["offset_prior"])
    return dge


def test_offset_prior_in_get_offset_cpm_and_rpkm():
    from edgepython.dgelist import get_offset
    dge = _offset_prior_dgelist()
    np.testing.assert_allclose(get_offset(dge), _edger("op_get_offset"), rtol=1e-13)
    np.testing.assert_allclose(ep.cpm(dge), _edger("op_cpm"), rtol=1e-13)
    np.testing.assert_allclose(ep.cpm(dge, log=True), _edger("op_cpm_log"), rtol=1e-13)
    # rpkm finds gene lengths in an AveLength column
    np.testing.assert_allclose(ep.rpkm(dge), _edger("op_rpkm"), rtol=1e-13)
    np.testing.assert_allclose(ep.cpm(dge[10:60, 1:8]), _edger("op_subset_cpm"), rtol=1e-13)
    with pytest.warns(UserWarning, match="offset_prior is ignored"):
        ep.cpm(COUNTS, offset=np.log(COUNTS.sum(axis=0)), offset_prior=_as_array(INPUTS["offset_prior"]))


def test_models_from_dgelist_use_offset_prior():
    design = _as_array(INPUTS["additive"])
    e = ep.estimate_disp(_offset_prior_dgelist(), design)
    assert e["common.dispersion"] == pytest.approx(_edger("op_disp", "common")[0], rel=1e-10)
    _assert_close(e["trended.dispersion"], _edger("op_disp", "trended"), rtol=1e-10)
    res = ep.glm_ql_ftest(ep.glm_ql_fit(e, design), coef=4)
    _assert_close(res["table"]["PValue"], _edger("op_qlftest"), rtol=1e-7, atol=1e-12)


def test_cbind_and_rbind_keep_offset_prior():
    # edgeR's cbind/rbind keep only the first object's offset.prior
    from edgepython.classes import cbind_dgelist, rbind_dgelist
    dge = _offset_prior_dgelist()
    both = cbind_dgelist(dge[:, :4], dge[:, 4:])
    np.testing.assert_array_equal(both["offset.prior"], dge["offset.prior"])
    both = rbind_dgelist(dge[:100], dge[100:])
    np.testing.assert_array_equal(both["offset.prior"], dge["offset.prior"])


# limma 3.99: voom, voomLmFit, arrayWeights, duplicateCorrelation, camera, fry

BIG = _as_array(INPUTS["big_counts"])
GROUP = np.repeat(["A", "B", "C"], 3)
ADDITIVE = _as_array(INPUTS["additive"])
ONEWAY = _as_array(INPUTS["oneway"])
SETS = {k: np.atleast_1d(np.asarray(v, dtype=int)) - 1 for k, v in INPUTS["sets"].items()}


def _big_dgelist():
    return ep.calc_norm_factors(ep.make_dgelist(BIG, group=GROUP))


@pytest.fixture(scope="module")
def big_voom():
    return ep.voom(_big_dgelist(), ADDITIVE)


def test_voom_matches_limma(big_voom):
    _assert_close(big_voom["E"][:200], _edger("voom", "E"), rtol=1e-12)
    _assert_close(big_voom["weights"][:200], _edger("voom", "weights"), rtol=1e-8)
    assert big_voom["span"] == pytest.approx(_edger("voom", "span")[0], rel=1e-14)
    _assert_close(ep.voom(_big_dgelist(), ADDITIVE, adaptive_span=False)["weights"][:200],
                  _edger("voom_fixed_span"), rtol=1e-8)
    _assert_close(ep.voom(_offset_prior_dgelist(), ADDITIVE)["weights"],
                  _edger("voom_offset_prior"), rtol=1e-8)


@pytest.mark.parametrize("case, kwargs", [
    ("voomlmfit", {}),
    ("voomlmfit_sw", {"sample_weights": True}),
    ("voomlmfit_block", {"design": ONEWAY, "block": _as_array(INPUTS["block"])}),
    ("voomlmfit_zeros", {}),
])
def test_voom_lmfit_matches_limma(case, kwargs):
    dge = _big_dgelist()
    if case == "voomlmfit_zeros":
        # Structural zeros: an all-zero group in the first 20 genes
        dge["counts"] = dge["counts"].copy()
        dge["counts"][:20, 6:] = 0
    kwargs = {"design": ADDITIVE, **kwargs}
    fit = ep.voom_lmfit(dge, **kwargs)
    _assert_close(fit["coefficients"][:200], _edger(case, "coefficients"), rtol=1e-8, atol=1e-10)
    _assert_close(fit["sigma"][:200], _edger(case, "sigma"), rtol=1e-8)
    _assert_close(fit["df_residual"], _edger(case, "df_residual"), rtol=1e-12)
    _assert_close(fit["stdev_unscaled"][:200], _edger(case, "stdev_unscaled"), rtol=1e-8)


def test_array_weights_and_duplicate_correlation_match_limma(big_voom):
    _assert_close(ep.array_weights(big_voom["E"], ADDITIVE, weights=big_voom["weights"]),
                  _edger("array_weights"), rtol=1e-9)
    v = ep.voom(_big_dgelist(), ONEWAY)
    dc = ep.duplicate_correlation(v["E"], ONEWAY, np.tile([1, 2, 3], 3), weights=v["weights"])
    assert dc["consensus_correlation"] == pytest.approx(_edger("duplicate_correlation")[0], rel=1e-8)


def _assert_gene_set_table(tab, case, names=None):
    names = list(SETS) if names is None else names
    expected = EDGER[case]
    for col, values in expected.items():
        actual = tab.loc[names, col]
        if col == "Direction":
            np.testing.assert_array_equal((actual == "Up").astype(int), values)
        else:
            _assert_close(actual.to_numpy(float), _as_array(values), rtol=1e-8, atol=1e-15)


def _weighted_sets():
    import pandas as pd
    return {"w1": pd.DataFrame({"id": np.arange(40), "w": _as_array(INPUTS["set_weights"])})}


CAMERA_CASES = {
    "camera": lambda v: ep.camera(v, SETS, ADDITIVE),
    "camera_array_weights": lambda v: ep.camera(v["E"], SETS, ADDITIVE, weights=_as_array(INPUTS["array_w"])),
    "camera_correlation": lambda v: ep.camera(v, SETS, ADDITIVE, inter_gene_cor=None),
    "camera_ranks": lambda v: ep.camera(v, SETS, ADDITIVE, use_ranks=True),
    "camera_nondirectional": lambda v: ep.camera(v, SETS, ADDITIVE, directional=False),
    "camera_contrast": lambda v: ep.camera(v, SETS, ADDITIVE, contrast=np.array([0, 1, -1, 0, 0])),
    "camera_weighted_set": lambda v: ep.camera(v, _weighted_sets(), ADDITIVE, inter_gene_cor=None),
    "camera_dgelist": lambda v: ep.camera(ep.estimate_disp(_big_dgelist(), ADDITIVE), SETS, ADDITIVE),
    "fry": lambda v: ep.fry(v, SETS, ADDITIVE),
    "fry_none": lambda v: ep.fry(v, SETS, ADDITIVE, standardize="none"),
    "fry_residual.sd": lambda v: ep.fry(v, SETS, ADDITIVE, standardize="residual.sd"),
    "fry_p2": lambda v: ep.fry(v, SETS, ADDITIVE, standardize="p2"),
    "fry_robust": lambda v: ep.fry(v, SETS, ADDITIVE, robust=True),
    "fry_trend": lambda v: ep.fry(v["E"], SETS, ADDITIVE, trend_var=True),
    "fry_array_weights": lambda v: ep.fry(v["E"], SETS, ADDITIVE, array_weights=_as_array(INPUTS["array_w"])),
    "fry_gene_weights": lambda v: ep.fry(v["E"], SETS, ADDITIVE, gene_weights=np.linspace(0.5, 2, 2000)),
    "fry_contrast": lambda v: ep.fry(v, SETS, ADDITIVE, contrast=np.array([0, 1, -1, 0, 0])),
    "fry_block": lambda v: ep.fry(v, SETS, ONEWAY, block=_as_array(INPUTS["block"]), correlation=0.3),
    "fry_weighted_set": lambda v: ep.fry(v, _weighted_sets(), ADDITIVE),
    "fry_dgelist": lambda v: ep.fry(ep.estimate_disp(_big_dgelist()), SETS),
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("case", list(CAMERA_CASES))
def test_camera_and_fry_match_limma(big_voom, case):
    names = ["w1"] if "weighted_set" in case else None
    _assert_gene_set_table(CAMERA_CASES[case](big_voom), case, names)


# sampleWeights, glmLRT on a matrix fit, normLibSizes

BIG_WEIGHTS = (0.5 + (np.arange(1, 2000 * 9 + 1) * 7919 % 1000) / 1000).reshape(9, 2000).T


@pytest.fixture(scope="module")
def big_disp():
    return ep.estimate_disp(_big_dgelist(), ADDITIVE)


def test_sample_weights_match_edger5(big_disp):
    fit = ep.glm_ql_fit(big_disp, ADDITIVE)
    w = ep.sample_weights(fit)
    assert w.shape == BIG.shape
    _assert_close(w[0], _edger("sample_weights"), rtol=1e-10)
    np.testing.assert_array_equal(w, np.tile(w[0], (BIG.shape[0], 1)))

    weighted = dict(big_disp, weights=BIG_WEIGHTS)
    _assert_close(ep.sample_weights(ep.glm_ql_fit(weighted, ADDITIVE))[0],
                  _edger("sample_weights_obs"), rtol=1e-10)

    lib_size = big_disp["samples"]["lib.size"].to_numpy()
    fit_m = ep.glm_ql_fit(BIG, ONEWAY, dispersion=0.1, offset=np.log(lib_size))
    _assert_close(ep.sample_weights(fit_m)[0], _edger("sample_weights_matrix"), rtol=1e-10)

    refit = ep.glm_ql_fit(dict(big_disp, weights=w), ADDITIVE)
    _assert_close(ep.glm_ql_ftest(refit, coef=4)["table"]["PValue"][:200],
                  _edger("sample_weights_refit"), rtol=1e-7, atol=1e-12)


def test_sample_weights_need_a_new_ql_fit(big_disp):
    with pytest.raises(ValueError, match="legacy"):
        ep.sample_weights(ep.glm_fit(big_disp, ADDITIVE))


def test_glm_lrt_on_a_matrix_fit_matches_edger():
    fit = ep.glm_fit(COUNTS, ADDITIVE, dispersion=DISPERSION, offset=OFFSET)
    tab = ep.glm_lrt(fit, coef=4)["table"]
    _assert_close(tab["logCPM"], _edger("lrt_matrix", "logCPM"), rtol=1e-10)
    _assert_close(tab["LR"], _edger("lrt_matrix", "LR"), rtol=1e-8, atol=1e-10)
    _assert_close(tab["PValue"], _edger("lrt_matrix", "PValue"), rtol=1e-8, atol=1e-14)


def test_norm_lib_sizes_is_calc_norm_factors():
    # edgeR 4.99.10 renamed calcNormFactors -> normLibSizes, keeping the old name as a
    # deprecated alias, and put the negative-count guard only on normLibSizes.default
    # (normLibSizes.R:36).  The two names give the same factors; the guard is not shared.
    assert ep.norm_lib_sizes is not ep.calc_norm_factors
    np.testing.assert_array_equal(ep.norm_lib_sizes(COUNTS), ep.calc_norm_factors(COUNTS))
    negative = np.array([[5.0, 3.0], [2.0, -1.0], [0.0, 0.0], [8.0, 4.0]])
    factors = ep.calc_norm_factors(negative)          # the alias has no such guard
    assert factors.shape == (2,) and np.isfinite(factors).all()
    with pytest.raises(ValueError, match="Negative counts not allowed"):
        ep.norm_lib_sizes(negative)


# DGEListFromTximport and DGEListFromTximeta

TX_COUNTS = _as_array(INPUTS["tx_counts"])
TX_LENGTH = _as_array(INPUTS["tx_length"])
TX_INFREPS = list(_as_array(INPUTS["tx_infreps"]))


def _txi(**kwargs):
    return {"counts": TX_COUNTS, "length": TX_LENGTH, "countsFromAbundance": "no",
            "infReps": TX_INFREPS, **kwargs}


def _assert_tx_dgelist(y, case):
    # R's list(x = NULL) keeps x, written as {}: those are absent
    expected = {k: v for k, v in EDGER[case].items() if v != {}}
    _assert_close(np.asarray(y["counts"], dtype=float), _as_array(expected["counts"]), rtol=1e-13)
    for col in ("AveLength", "Max2MinLength", "Overdispersion"):
        if col in expected:
            _assert_close(y["genes"][col], _as_array(expected[col]), rtol=1e-12)
        else:
            assert col not in y["genes"]
    _assert_close(y["samples"]["lib.size"], _as_array(expected["lib_size"]), rtol=1e-13)
    if "offset_prior" in expected:
        _assert_close(y["offset.prior"], _as_array(expected["offset_prior"]), rtol=1e-12, atol=1e-14)
    else:
        assert y.get("offset.prior") is None


def test_dgelist_from_tximport_matches_edger():
    y = ep.dgelist_from_tximport(_txi())
    _assert_tx_dgelist(y, "tximport")
    assert y["divided.counts"] is True and y["tximport.counts"] == "raw"
    y = ep.dgelist_from_tximport(_txi(), divide=False, remove_zeros=True)
    _assert_tx_dgelist(y, "tximport_undivided_zeros")
    assert y["divided.counts"] is False
    y = ep.dgelist_from_tximport(_txi(countsFromAbundance="lengthScaledTPM", infReps=None))
    _assert_tx_dgelist(y, "tximport_scaled")
    # No replicates: nothing to divide by
    assert y["divided.counts"] is False and y["tximport.counts"] == "lengthScaledTPM"


def _txm():
    import pandas as pd
    assays = {"counts": TX_COUNTS, "length": TX_LENGTH}
    for k in range(5):
        assays[f"infRep{k + 1}"] = np.column_stack([b[:, k] for b in TX_INFREPS])
    return {"assays": assays, "metadata": {"countsFromAbundance": "no"},
            "rowData": pd.DataFrame({"gene_id": [f"g{i + 1}" for i in range(100)]})}


def test_dgelist_from_tximeta_matches_edger():
    y = ep.dgelist_from_tximeta(_txm())
    _assert_tx_dgelist(y, "tximeta")
    assert list(y["genes"]["gene_id"][:2]) == ["g1", "g2"]
    _assert_tx_dgelist(ep.dgelist_from_tximeta(_txm(), remove_zeros=True), "tximeta_zeros")


def test_dgelist_from_tximeta_reads_anndata():
    anndata = pytest.importorskip("anndata")
    txm = _txm()
    adata = anndata.AnnData(X=txm["assays"]["counts"].T, var=txm["rowData"].set_index("gene_id", drop=False),
                            layers={k: v.T for k, v in txm["assays"].items() if k != "counts"},
                            uns={"countsFromAbundance": "no"})
    y = ep.dgelist_from_tximeta(adata)
    _assert_tx_dgelist(y, "tximeta")


def test_dgelist_from_tximport_checks_components():
    with pytest.raises(ValueError, match="countsFromAbundance"):
        ep.dgelist_from_tximport({"counts": TX_COUNTS, "length": TX_LENGTH})


@pytest.mark.parametrize("key", [np.array([1, 3, 50]), np.arange(300) > 280, [0, 2],
                                 (np.array([1, 3]), slice(None)), ([1, 2], [0, 3])])
def test_dgelist_subsetting_with_index_arrays(key):
    dge = _offset_prior_dgelist()
    dge["weights"] = WEIGHTS
    sub = dge[key]
    i, j = key if isinstance(key, tuple) else (key, slice(None))
    i = np.flatnonzero(i) if np.asarray(i).dtype == bool else np.asarray(i)
    for k in ("counts", "offset.prior", "weights"):
        np.testing.assert_array_equal(sub[k], np.asarray(dge[k])[i][:, j])
    assert sub["genes"].shape[0] == len(i)


# catch* readers on simulated quantifier output (data/quant, from make_quant_files.py)

QUANT = DATA / "quant"


def _assert_catch(y, case, rtol=1e-12, strip_versions=False):
    expected = EDGER[case]
    ids = [re.sub(r"\.[0-9]+", "", i, count=1) for i in expected["ids"]] if strip_versions else expected["ids"]
    assert list(y["genes"].index) == ids
    assert list(y["samples"].index) == expected["samples"]
    _assert_close(y["counts"], _as_array(expected["counts"]), rtol=rtol)
    assert list(y["genes"].columns) == list(expected["genes"])
    for col, values in expected["genes"].items():
        if values and isinstance(values[0], str):
            assert list(y["genes"][col].astype(str)) == values, col
        else:
            _assert_close(y["genes"][col].to_numpy(float), _as_array(values), rtol=rtol)
    for key, ours in (("offset_prior", y.get("offset.prior")),
                      ("effective_length", y.get("other", {}).get("effective.length"))):
        if expected[key] == {}:
            assert ours is None, key
        else:
            _assert_close(np.asarray(ours, dtype=float), _as_array(expected[key]), rtol=rtol, atol=1e-13)
    assert y["overdispersion.prior"] == pytest.approx(_as_array(expected["overdispersion_prior"]), rel=rtol)
    assert y["resample.type"] == expected["resample_type"]
    assert y["divided.counts"] == expected["divided"][0]


@pytest.mark.parametrize("case, reader, kwargs", [
    ("catch_salmon", "catch_salmon", {}),
    ("catch_salmon_raw", "catch_salmon", {"divide": False, "impute_eff_len": False, "offset_prior": False}),
    ("catch_salmon_gene_moderate", "catch_salmon_gene", {}),
    ("catch_salmon_gene_tximport", "catch_salmon_gene", {"gene_length": "tximport"}),
    ("catch_salmon_gene_simple", "catch_salmon_gene", {"gene_length": "simple"}),
    ("catch_kallisto", "catch_kallisto", {}),
    ("catch_oarfish", "catch_oarfish", {}),
])
def test_catch_readers_match_edger5(case, reader, kwargs):
    folder = {"salmon": "salmon", "kallisto": "kallisto", "oarfish": "oarfish"}[reader.split("_")[1]]
    y = getattr(ep, reader)(str(QUANT / folder), verbose=False, **kwargs)
    if reader == "catch_salmon_gene":
        # edgePython adds Max2MinEffLen, which catchKallistoGene has and catchSalmonGene lacks
        y["genes"] = y["genes"].drop(columns="Max2MinEffLen")
    _assert_catch(y, case)


@pytest.mark.parametrize("case, kwargs", [("catch_kallisto_gene", {}),
                                          ("catch_kallisto_gene_simple", {"gene_length": "simple"})])
def test_catch_kallisto_gene_matches_edger5(case, kwargs):
    import pandas as pd
    tx2gene = pd.read_csv(QUANT / "tx2gene.tsv", sep="\t")
    y = ep.catch_kallisto_gene(str(QUANT / "kallisto"), tx2gene=tx2gene, verbose=False, **kwargs)
    # catchKallistoGene keeps tx2gene's gene versions; remove_version removes
    # them, as documented for catchSalmonGene and as catchSalmonGene does
    _assert_catch(y, case, strip_versions=True)


def test_catch_rsem_matches_edger5():
    _assert_catch(ep.catch_rsem(str(QUANT / "rsem"), ngibbs=50, verbose=False), "catch_rsem")


def test_split_and_collapse_gencode_match_edger5():
    import pandas as pd
    names = pd.read_csv(QUANT / "salmon" / "s1" / "quant.sf", sep="\t")["Name"]
    sp = ep.split_gencode_tx_names(names)
    for col, values in EDGER["split_gencode"].items():
        assert list(sp[col]) == values
    raw = ep.split_gencode_tx_names(names[:3], remove_version=False)
    np.testing.assert_array_equal(raw[["EnsT", "EnsG"]].to_numpy(), np.array(EDGER["split_gencode_version"]))
    cg = ep.collapse_gencode_tx2gene(sp)
    expected = EDGER["collapse_gencode"]
    assert list(cg.index) == expected["ids"]
    assert list(cg["GeneName"]) == expected["GeneName"]
    assert list(cg["TxTypes"]) == expected["TxTypes"]
    assert list(cg["NPCTx"]) == expected["NPCTx"]


@pytest.mark.parametrize("layout", ["dirs", "flat"])
def test_catch_lr_kallisto_matches_edger_kallisto_reader(layout):
    # Per-sample quant-tcc output is kallisto's: abundance_<n>/abundance.h5
    # (--matrix-to-directories), or flat plaintext abundance_<n>.tsv with
    # bs_abundance_<n>_<b>.tsv (--matrix-to-files --plaintext)
    y = ep.catch_lr_kallisto(str(QUANT / "lr_kallisto" / layout), verbose=False)
    _assert_catch(y, "catch_lr_kallisto")
    g = ep.catch_lr_kallisto_gene(str(QUANT / "lr_kallisto" / layout), verbose=False)
    _assert_catch(g, "catch_lr_kallisto_gene")


def test_catch_lr_kallisto_reads_the_matrix_without_bootstraps():
    path = str(QUANT / "lr_kallisto" / "matrix")
    with pytest.warns(UserWarning, match="no bootstraps"):
        y = ep.catch_lr_kallisto(path, sample_names=["a", "b", "c"], verbose=False)
    # The matrix has no transcript lengths, so effective lengths can't be imputed
    raw = ep.catch_lr_kallisto(str(QUANT / "lr_kallisto" / "dirs"), divide=False, impute_eff_len=False,
                               verbose=False)
    np.testing.assert_array_equal(y["counts"], raw["counts"])
    assert list(y["samples"].index) == ["a", "b", "c"]
    assert y["divided.counts"] is False and np.all(np.isnan(y["genes"]["Overdispersion"]))
    # Effective lengths are only written for transcripts with reads; the
    # others get the transcript's geometric mean over samples
    eff = y["other"]["effective.length"].to_numpy()
    assert np.all(eff > 0)
    has_reads = raw["counts"] > 0
    np.testing.assert_allclose(eff[has_reads], raw["other"]["effective.length"].to_numpy()[has_reads],
                               rtol=1e-15)
    with pytest.warns(UserWarning):
        g = ep.catch_lr_kallisto_gene(path, verbose=False)
    np.testing.assert_allclose(g["counts"], ep.catch_lr_kallisto_gene(
        str(QUANT / "lr_kallisto" / "dirs"), divide=False, verbose=False)["counts"], rtol=1e-13)


def test_catch_kallisto_reads_plaintext_bootstraps():
    import pandas.testing as pdt
    h5 = ep.catch_kallisto(str(QUANT / "kallisto"), verbose=False)
    tsv = ep.catch_kallisto(str(QUANT / "kallisto_tsv"), verbose=False)
    np.testing.assert_allclose(tsv["counts"], h5["counts"], rtol=1e-14)
    pdt.assert_frame_equal(tsv["genes"], h5["genes"], rtol=1e-14)


def _moderated(od, df):
    from scipy.stats import f as f_dist
    i = df > 0
    od = od.copy()
    od[i] /= df[i]
    prior = max(np.median(od[i]) / f_dist.ppf(0.5, np.median(df[i]), 3), 1)
    od[i] = (3 * prior + df[i] * od[i]) / (3 + df[i])
    od = np.maximum(od, 1)
    od[~i] = prior
    return od


def test_catch_oarfish_gene_sums_counts_and_bootstraps():
    import pandas as pd
    tx = ep.catch_oarfish(str(QUANT / "oarfish"), divide=False, verbose=False)
    g = ep.catch_oarfish_gene(str(QUANT / "oarfish"), verbose=False)
    codes, genes = pd.factorize(tx["genes"]["EnsG"])
    assert list(g["genes"].index) == list(genes)
    assert list(g["genes"].columns) == ["GeneName", "TxTypes", "NPCTx", "NTx", "MaxTxLen", "Overdispersion"]
    od, df = np.zeros(len(genes)), np.zeros(len(genes))
    for j in (1, 2, 3):
        b = pd.read_parquet(QUANT / "oarfish" / f"o{j}.infreps.pq").to_numpy()
        s = np.zeros((len(genes), b.shape[1]))
        np.add.at(s, codes, b)
        m = s.mean(axis=1)
        pos = m > 0
        od[pos] += ((s[pos] - m[pos, None]) ** 2).sum(axis=1) / m[pos]
        df[pos] += b.shape[1] - 1
    od = _moderated(od, df)
    np.testing.assert_allclose(g["genes"]["Overdispersion"], od, rtol=1e-13)
    summed = np.zeros((len(genes), 3))
    np.add.at(summed, codes, tx["counts"])
    np.testing.assert_allclose(g["counts"], summed / od[:, None], rtol=1e-13)
    assert g.get("offset.prior") is None


def test_catch_rsem_gene_reads_gene_results():
    import pandas as pd
    tabs = [pd.read_csv(QUANT / "rsem" / f"r{j}.genes.results", sep="\t") for j in (1, 2, 3)]
    y = ep.catch_rsem_gene(str(QUANT / "rsem"), ngibbs=50, verbose=False)
    assert list(y["genes"].index) == [re.sub(r"\.[0-9]+", "", i, count=1) for i in tabs[0]["gene_id"]]
    assert list(y["samples"].index) == ["r1", "r2", "r3"]
    np.testing.assert_array_equal(y["genes"]["NTx"], tabs[0]["transcript_id(s)"].str.count(",") + 1)
    # Overdispersions from RSEM's genewise Gibbs summaries, as catchRSEM does for transcripts
    od, df = np.zeros(len(tabs[0])), np.zeros(len(tabs[0]))
    for t in tabs:
        m, s = t["posterior_mean_count"].to_numpy(), t["posterior_standard_deviation_of_count"].to_numpy()
        pos = m > 0
        od[pos] += 49 * s[pos] ** 2 / m[pos]
        df[pos] += 49
    od = _moderated(od, df)
    np.testing.assert_allclose(y["genes"]["Overdispersion"], od, rtol=1e-13)
    counts = np.column_stack([t["expected_count"] for t in tabs])
    np.testing.assert_allclose(y["counts"], counts / od[:, None], rtol=1e-13)
    # The gene with no reads has zero effective lengths, replaced by 1
    eff = y["other"]["effective.length"].to_numpy()
    assert np.all(eff > 0) and np.all(np.isfinite(y["offset.prior"]))


def test_catch_readers_where_edger5_fails(tmp_path):
    import shutil
    import pandas as pd
    # Gene readers with verbose=False (edgeR stops: the else binds to if(verbose))
    assert ep.catch_salmon_gene(str(QUANT / "salmon"), verbose=False)["counts"].shape == (14, 3)
    # No resamples: counts are left undivided (edgeR divides by NA and fails)
    for j in (1, 2, 3):
        shutil.copytree(QUANT / "salmon" / f"s{j}", tmp_path / f"s{j}")
        meta = tmp_path / f"s{j}" / "aux_info" / "meta_info.json"
        m = json.loads(meta.read_text())
        m["num_bootstraps"] = 0
        meta.write_text(json.dumps(m))
    y = ep.catch_salmon(str(tmp_path), verbose=False)
    assert y["divided.counts"] is False and np.isnan(y["overdispersion.prior"])
    np.testing.assert_array_equal(
        y["counts"], ep.catch_salmon(str(QUANT / "salmon"), divide=False, verbose=False)["counts"])
    # gene_length='tximport' when no gene is all zero (edgeR returns the
    # counts); the gene given reads here has no TPM, so gets the unweighted average
    for j in (1, 2, 3):
        q = tmp_path / f"s{j}" / "quant.sf"
        t = pd.read_csv(q, sep="\t")
        t.loc[t["NumReads"] == 0, "NumReads"] = 5.0
        t.to_csv(q, sep="\t", index=False, float_format="%.17g")
    g = ep.catch_salmon_gene(str(tmp_path), gene_length="tximport", divide=False, verbose=False)
    eff = g["other"]["effective.length"].to_numpy()
    assert not np.allclose(eff, g["counts"])
    assert np.all((eff > 0) & np.isfinite(eff))


def test_catch_readers_old_calls_and_plain_lists():
    dirs = [str(QUANT / "salmon" / f"s{j}") for j in (1, 2, 3)]
    y = ep.catch_salmon(dirs, verbose=False)
    _assert_catch(y, "catch_salmon")
    out = ep.catch_salmon(str(QUANT / "salmon"), dgelist=False, verbose=False)
    assert set(out) == {"counts", "effective.length", "annotation", "overdispersion.prior",
                        "resample.type", "divided.counts"}
    np.testing.assert_array_equal(out["counts"].to_numpy(), y["counts"])


# Binomial GLMs for paired counts: PCList, binFit, binQLFit, binQLFTest

METH = _as_array(INPUTS["meth"])
UNMETH = _as_array(INPUTS["unmeth"])
BIN_ONEWAY = _as_array(INPUTS["bin_oneway"])
BIN_ADDITIVE = _as_array(INPUTS["bin_additive"])
_k = np.arange(1, 400 * 6 + 1)
BIN_WEIGHTS = (0.5 + (_k * 7919 % 1000) / 1000).reshape(6, 400).T
BIN_OFFSET = ((_k * 104729 % 2000) / 5000 - 0.2).reshape(6, 400).T
# Loci whose weighted design is rank deficient (no coverage in a batch or
# group): edgeR returns their coefficients in LINPACK's pivoted column
# order, while edgePython keeps the design's order (0 for the column set
# aside), so their coefficients, and TREAT tests that use them, differ
BIN_RANKDEF = np.array([(c > 0).sum() == 0 or np.linalg.matrix_rank(BIN_ADDITIVE[c > 0]) < 4
                        for c in METH + UNMETH])


def _pclist():
    return ep.make_pclist(METH, UNMETH, group=np.repeat(["A", "B"], 3))


@pytest.fixture(scope="module")
def bin_fits():
    return {"oneway": ep.bin_ql_fit(_pclist()),
            "additive": ep.bin_ql_fit(_pclist(), design=BIN_ADDITIVE),
            "weights": ep.bin_ql_fit(METH, UNMETH, design=BIN_ADDITIVE, weights=BIN_WEIGHTS)}


def test_pclist_ave_log_cpm_and_rowsum_match_edger():
    _assert_close(ep.ave_log_cpm(_pclist()), _edger("pclist_avelogcpm"), rtol=1e-13)
    rs = ep.rowsum_pclist(_pclist(), np.repeat(np.arange(1, 101), 4))
    np.testing.assert_array_equal(rs["counts"], _edger("pclist_rowsum", "counts"))
    np.testing.assert_array_equal(rs["counts2"], _edger("pclist_rowsum", "counts2"))
    assert list(rs["genes"].index) == [str(i) for i in range(1, 101)]


@pytest.mark.parametrize("case, args, kwargs", [
    ("binfit_oneway", "pclist", {}),
    ("binfit_additive", "pclist", {"design": BIN_ADDITIVE}),
    ("binfit_weights", "matrix", {"design": BIN_ADDITIVE, "weights": BIN_WEIGHTS}),
    ("binfit_offset", "matrix", {"design": BIN_ONEWAY, "offset": BIN_OFFSET, "prior_count": 0}),
])
def test_bin_fit_matches_edger5(case, args, kwargs):
    fit = ep.bin_fit(_pclist(), **kwargs) if args == "pclist" else ep.bin_fit(METH, UNMETH, **kwargs)
    expected = EDGER[case]
    _assert_close(fit["coefficients"], _as_array(expected["coefficients"]), rtol=1e-11, atol=1e-11)
    _assert_close(fit["deviance"], _as_array(expected["deviance"]), rtol=1e-10, atol=1e-12)
    if expected["fitted"] != {}:
        _assert_close(fit["fitted.values"], _as_array(expected["fitted"]), rtol=1e-12, atol=1e-14)
    if expected["unshrunk"] != {}:
        full_rank = ~BIN_RANKDEF if fit["design"].shape[1] == 4 else slice(None)
        _assert_close(fit["unshrunk.coefficients"][full_rank], _as_array(expected["unshrunk"])[full_rank],
                      rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("case", ["oneway", "additive", "weights", "plain"])
def test_bin_ql_fit_matches_edger5(bin_fits, case):
    fit = bin_fits.get(case) or ep.bin_ql_fit(_pclist(), design=BIN_ADDITIVE, robust=False,
                                              abundance_trend=False)
    expected = EDGER[f"binqlfit_{case}"]
    for key, ours in [("df_adj", "df.residual.adj"), ("dev_adj", "deviance.adj"), ("s2_post", "s2.post"),
                      ("s2_prior", "s2.prior"), ("df_prior", "df.prior"),
                      ("average_ql", "average.ql.dispersion")]:
        _assert_close(fit[ours], _as_array(expected[key]), rtol=1e-8, atol=1e-12)
    if case != "weights":
        _assert_close(fit["AveLogCPM"], _edger("pclist_avelogcpm"), rtol=1e-13)


@pytest.mark.parametrize("case, fit, kwargs", [
    ("binqlftest_oneway", "oneway", {}),
    ("binqlftest_coef", "additive", {"coef": 1}),
    ("binqlftest_multi", "additive", {"coef": [1, 2, 3]}),
    ("binqlftest_contrast", "additive", {"contrast": np.array([0, 1, -1, 0])}),
    ("binqlftest_upshot", "additive", {"lor": 0.5}),
    ("binqlftest_treat", "additive", {"lor": 0.5, "upshot": False}),
    ("binqlftest_multi_lor", "additive", {"coef": [1, 2], "lor": [0.3, 0.6]}),
    ("binqlftest_weights", "weights", {"lor": 0.4}),
])
def test_bin_ql_ftest_matches_edger5(bin_fits, case, fit, kwargs):
    tab = ep.bin_ql_ftest(bin_fits[fit], **kwargs)["table"]
    rows = ~BIN_RANKDEF if "lor" in kwargs else slice(None)
    for col in ("F", "PValue"):
        _assert_close(tab[col].to_numpy()[rows], _edger(case, col)[rows], rtol=1e-6, atol=1e-12)
    logor = [c for c in tab.columns if c.startswith("logOR")]
    coefs = bin_fits[fit]["coefficients"] / np.log(2)
    if "contrast" in kwargs:
        np.testing.assert_allclose(tab["logOR"], coefs @ kwargs["contrast"], rtol=1e-13, atol=1e-14)
    else:
        coef = np.atleast_1d(kwargs.get("coef", coefs.shape[1] - 1))
        np.testing.assert_allclose(tab[logor].to_numpy(), coefs[:, coef], rtol=1e-13)


def test_bin_ql_ftest_treat_with_edgers_pivoted_coefficients(bin_fits):
    # Given edgeR's (pivoted) unshrunk coefficients, the rank-deficient loci agree too
    fit = ep.DGEBIN(bin_fits["additive"])
    fit["unshrunk.coefficients"] = _edger("binfit_additive", "unshrunk")
    tab = ep.bin_ql_ftest(fit, lor=0.5)["table"]
    _assert_close(tab["PValue"], _edger("binqlftest_upshot", "PValue"), rtol=1e-6, atol=1e-12)


def test_bin_sample_weights_match_edger5(bin_fits):
    _assert_close(ep.sample_weights(bin_fits["additive"])[0], _edger("bin_sample_weights"), rtol=1e-12)
    _assert_close(ep.sample_weights(bin_fits["weights"])[0], _edger("bin_sample_weights_obs"), rtol=1e-12)


def test_mbin_one_way_and_iwls_match_edger():
    r = ep.mbin_one_way(METH, UNMETH, design=BIN_ONEWAY)
    _assert_close(r["coefficients"], _edger("mbin_oneway", "coefficients"), rtol=1e-12, atol=1e-12)
    _assert_close(r["deviance"], _edger("mbin_oneway", "deviance"), rtol=1e-12, atol=1e-12)
    _assert_close(ep.mbin_one_way(METH, UNMETH, group=np.repeat(["A", "B"], 3))["coefficients"],
                  _edger("mbin_oneway_group"), rtol=1e-12, atol=1e-12)
    r = ep.mbin_iwls(METH, UNMETH, BIN_ADDITIVE)
    _assert_close(r["coefficients"][~BIN_RANKDEF], _edger("mbin_iwls", "coefficients")[~BIN_RANKDEF],
                  rtol=1e-8, atol=1e-8)
    _assert_close(r["deviance"], _edger("mbin_iwls", "deviance"), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(r["iter"], _edger("mbin_iwls", "iter"))


def test_read_bismark2pc_matches_edger():
    import os
    old = os.getcwd()
    try:
        # edgeR's sample names keep the paths given, so read from the same place
        os.chdir(DATA)
        y = ep.read_bismark2pc(["bismark/sample1.cov", "bismark/sample2.cov", "bismark/sample3.cov.gz"],
                               verbose=False)
    finally:
        os.chdir(old)
    expected = EDGER["bismark2pc"]
    assert list(y["genes"].index) == expected["ids"]
    assert list(y["samples"].index) == expected["samples"]
    np.testing.assert_array_equal(y["counts"], _as_array(expected["counts"]))
    np.testing.assert_array_equal(y["counts2"], _as_array(expected["counts2"]))
    assert list(y["genes"]["Chr"].astype(str)) == expected["Chr"]
    assert list(y["genes"]["Chr"].cat.categories) == expected["chr_levels"]
    np.testing.assert_array_equal(y["genes"]["Locus"], _as_array(expected["Locus"]))


def test_pclist_subsetting_and_top_tags(bin_fits):
    pc = _pclist()
    sub = pc[np.arange(10, 20), [0, 1, 2]]
    np.testing.assert_array_equal(sub["counts2"], UNMETH[10:20, :3])
    assert list(sub["samples"]["group"].cat.categories) == ["A"]
    res = ep.bin_ql_ftest(bin_fits["additive"])
    top = ep.top_tags(res, n=5)["table"]
    assert list(top.columns[:2]) == ["logOR", "logCPM"]
    assert np.all(np.diff(top["PValue"].to_numpy()) >= 0)
    de = ep.decide_tests(res, lfc=1)
    assert set(np.unique(de)) <= {-1, 0, 1}
    assert np.all(de[np.abs(res["table"]["logOR"].to_numpy()) < 1] == 0)
    # A DGEBIN row subset keeps counts2 alongside counts
    s = bin_fits["additive"][np.arange(5), None]
    assert s["counts2"].shape == (5, 6) and s["coefficients"].shape == (5, 4)


def test_pclist_mds_and_dgebin_md_plots(bin_fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep.plot_mds(_pclist())
    assert plt.gcf().axes[0].get_xlabel().startswith("Leading logOR dim 1")
    ep.plot_md(bin_fits["weights"])
    assert plt.gcf().axes[0].get_ylabel() == "log-odds-ratio"
    plt.close("all")


_WEIGHTED_CALLS = {
    "glm_fit": (lambda y, w: ep.glm_fit(y, _as_array(INPUTS["oneway"]), weights=w), "coefficients"),
    "glm_ql_fit": (lambda y, w: ep.glm_ql_fit(y, _as_array(INPUTS["oneway"]), weights=w), "coefficients"),
    "estimate_disp": (lambda y, w: ep.estimate_disp(y, _as_array(INPUTS["oneway"]), weights=w), "tagwise.dispersion"),
    "estimate_glm_common_disp": (lambda y, w: ep.estimate_glm_common_disp(y, _as_array(INPUTS["oneway"]), weights=w),
                                 "common.dispersion"),
    "estimate_glm_trended_disp": (lambda y, w: ep.estimate_glm_trended_disp(y, _as_array(INPUTS["oneway"]), weights=w),
                                  "trended.dispersion"),
    "ave_log_cpm": (lambda y, w: ep.ave_log_cpm(y, weights=w), None),
    "cpm_by_group": (lambda y, w: ep.cpm_by_group(y, weights=w), None),
    "bin_fit": (lambda y, w: ep.bin_fit(y, weights=w), "coefficients"),
    "bin_ql_fit": (lambda y, w: ep.bin_ql_fit(y, weights=w), "s2.post"),
}


@pytest.mark.parametrize("name", list(_WEIGHTED_CALLS))
def test_weights_argument_with_a_dgelist(name):
    # Weights passed alongside a DGEList (or PCList) act as its own weights.
    # edgeR's DGEList methods fail with "matched by multiple actual
    # arguments"; edgePython used to ignore them silently.
    call, key = _WEIGHTED_CALLS[name]
    if name.startswith("bin"):
        y = _pclist()
        w = BIN_WEIGHTS
    else:
        y = ep.calc_norm_factors(ep.make_dgelist(COUNTS, group=np.repeat(["A", "B", "C"], 3)))
        if name == "glm_fit" or name == "glm_ql_fit":
            y = ep.estimate_disp(y)
        w = WEIGHTS
    value = (lambda r: np.asarray(r if key is None else r[key], dtype=float))
    stored = type(y)(y)
    stored["weights"] = w
    with_arg = value(call(y, w))
    np.testing.assert_allclose(with_arg, value(call(stored, None)), rtol=1e-13)
    assert not np.allclose(with_arg, value(call(y, None)))


@pytest.mark.parametrize("source, path, reader", [
    ("salmon", "salmon", lambda p: ep.catch_salmon(p, divide=False, verbose=False)),
    ("kallisto", "kallisto", lambda p: ep.catch_kallisto(p, divide=False, verbose=False)),
    ("oarfish", "oarfish", lambda p: ep.catch_oarfish(p, divide=False, verbose=False)),
    ("rsem", "rsem", lambda p: ep.catch_rsem(p, ngibbs=50, divide=False, verbose=False)),
])
def test_read_data_uses_the_catch_readers(source, path, reader):
    import os
    folder = str(QUANT / path)
    expected = reader(folder)
    if source == "oarfish":
        args = folder
    elif source == "rsem":
        args = sorted(str(QUANT / path / f) for f in os.listdir(folder) if f.endswith(".isoforms.results"))
    else:
        args = sorted(str(QUANT / path / d) for d in os.listdir(folder))
    y = ep.read_data(args, source=source, ngibbs=50, group=["a", "b", "b"], verbose=False)
    # Counts as quantified (divide=False) and the readers' overdispersions;
    # read_data keeps transcript names as read and its EffectiveLength column
    np.testing.assert_array_equal(y["counts"], expected["counts"])
    np.testing.assert_array_equal(y["genes"]["Overdispersion"], expected["genes"]["Overdispersion"])
    assert y["genes"].index[0].count("|") >= 8 or source == "kallisto"
    assert ("EffectiveLength" in y["genes"]) == (source != "oarfish")
    assert list(y["samples"]["group"]) == ["a", "b", "b"]
    np.testing.assert_allclose(ep.read_data(args, source=source, ngibbs=50, divide=True, verbose=False)["counts"],
                               expected["counts"] / expected["genes"]["Overdispersion"].to_numpy()[:, None])


def test_read_data_detects_lr_kallisto_and_kallisto_tsv():
    y = ep.read_data(str(QUANT / "lr_kallisto" / "dirs"), labels=["x", "y", "z"], verbose=False)
    np.testing.assert_array_equal(
        y["counts"], ep.catch_lr_kallisto(str(QUANT / "lr_kallisto" / "dirs"), divide=False, verbose=False)["counts"])
    assert list(y["samples"].index) == ["x", "y", "z"]
    dirs = [str(QUANT / "kallisto" / f"k{j}") for j in (1, 2, 3)]
    h5 = ep.read_data(dirs, source="kallisto", format="h5", verbose=False)
    tsv = ep.read_data([str(QUANT / "kallisto_tsv" / f"k{j}") for j in (1, 2, 3)], source="kallisto",
                       format="tsv", verbose=False)
    np.testing.assert_allclose(tsv["genes"]["Overdispersion"], h5["genes"]["Overdispersion"], rtol=1e-14)
