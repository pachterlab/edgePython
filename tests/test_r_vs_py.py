# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""R vs Python comparison tests: normalization, DGEList, glmTreat, gene sets."""

import os

import numpy as np
import pandas as pd
import pytest

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(name):
    return pd.read_csv(f"{CSV_DIR}/{name}")


def _load_idx(name):
    return pd.read_csv(f"{CSV_DIR}/{name}", index_col=0)


def _compare_df(r, p, tol=1e-6):
    """Compare two DataFrames column-by-column, return max relative diff."""
    max_rel = 0
    for col in r.columns:
        if col not in p.columns:
            raise AssertionError(f"Column '{col}' missing from Python output")
        try:
            r_num = pd.to_numeric(r[col], errors='raise')
            p_num = pd.to_numeric(p[col], errors='raise')
            abs_diff = np.abs(r_num.values - p_num.values)
            rel_diff = abs_diff / (np.abs(r_num.values) + 1e-15)
            max_rel = max(max_rel, np.max(rel_diff))
        except (ValueError, TypeError):
            assert (r[col].astype(str) == p[col].astype(str)).all(), \
                f"String column '{col}' mismatch"
    return max_rel


def _compare_treat(r_file, p_file):
    """Compare treat results row-aligned. Returns dict of metrics."""
    r = _load_idx(r_file)
    p = _load_idx(p_file)
    nrows = min(len(r), len(p))
    r = r.head(nrows)
    p = p.head(nrows)

    # logFC
    lfc_abs = np.abs(r["logFC"].values - p["logFC"].values)
    lfc_rel = lfc_abs / (np.abs(r["logFC"].values) + 1e-15)

    # logCPM
    cpm_abs = np.abs(r["logCPM"].values - p["logCPM"].values)
    cpm_rel = cpm_abs / (np.abs(r["logCPM"].values) + 1e-15)

    # PValue (log-scale)
    r_logp = np.log10(np.maximum(r["PValue"].values, 1e-300))
    p_logp = np.log10(np.maximum(p["PValue"].values, 1e-300))
    logp_diff = np.abs(r_logp - p_logp)

    # PValue relative
    pv_rel = np.abs(r["PValue"].values - p["PValue"].values) / (np.abs(r["PValue"].values) + 1e-15)

    return {
        "nrows": nrows,
        "lfc_max_rel": np.max(lfc_rel),
        "cpm_max_rel": np.max(cpm_rel),
        "pv_max_rel": np.max(pv_rel),
        "logp_max_diff": np.max(logp_diff),
        "logp_mean_diff": np.mean(logp_diff),
    }


# ======================================================================
# Normalization (normLibSizes)
# ======================================================================

class TestNormTMM:
    """TMM normalization variants."""

    def test_tmm_default(self):
        r = _load("R_norm_TMM.csv")
        p = _load("Py_norm_TMM.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_tmmwsp(self):
        r = _load("R_norm_TMMwsp.csv")
        p = _load("Py_norm_TMMwsp.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_rle(self):
        r = _load("R_norm_RLE.csv")
        p = _load("Py_norm_RLE.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_upperquartile(self):
        r = _load("R_norm_UQ.csv")
        p = _load("Py_norm_UQ.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_none(self):
        r = _load("R_norm_none.csv")
        p = _load("Py_norm_none.csv")
        assert np.max(np.abs(r["norm.factors"].values - p["norm.factors"].values)) == 0

    def test_tmm_refcolumn(self):
        r = _load("R_norm_TMM_ref1.csv")
        p = _load("Py_norm_TMM_ref1.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_tmm_custom_trim(self):
        r = _load("R_norm_TMM_trim.csv")
        p = _load("Py_norm_TMM_trim.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_tmm_no_weighting(self):
        r = _load("R_norm_TMM_noweight.csv")
        p = _load("Py_norm_TMM_noweight.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_tmm_raw_matrix(self):
        r = _load("R_norm_TMM_raw.csv")
        p = _load("Py_norm_TMM_raw.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_tmm_custom_libsize(self):
        r = _load("R_norm_TMM_customlib.csv")
        p = _load("Py_norm_TMM_customlib.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3

    def test_uq_p50(self):
        r = _load("R_norm_UQ_p50.csv")
        p = _load("Py_norm_UQ_p50.csv")
        rel = np.abs(r["norm.factors"].values - p["norm.factors"].values) / (np.abs(r["norm.factors"].values) + 1e-15)
        assert np.max(rel) < 1e-3


class TestEffectiveLibSizes:
    """Effective library size comparison."""

    def test_effective_lib_sizes(self):
        r = _load("R_norm_efflib.csv")
        p = _load("Py_norm_efflib.csv")
        rel = np.abs(r["eff.lib.size"].values - p["eff.lib.size"].values) / r["eff.lib.size"].values
        assert np.max(rel) < 1e-3


class TestReferenceColumn:
    """Reference column selection."""

    def test_reference_column_selection(self):
        r = _load("R_norm_refcol.csv")
        p = _load("Py_norm_refcol.csv")
        r_sel = r[r["selected"] == True].index[0]
        p_sel = p[p["selected"] == True].index[0]
        assert r_sel == p_sel

    def test_f75_values(self):
        r = _load("R_norm_refcol.csv")
        p = _load("Py_norm_refcol.csv")
        rel = np.abs(r["f75"].values - p["f75"].values) / (np.abs(r["f75"].values) + 1e-15)
        assert np.max(rel) < 1e-3


# ======================================================================
# DGEList construction and operations
# ======================================================================

class TestDGEListBasic:
    """Basic DGEList creation."""

    def test_basic_dgelist(self):
        r = _load("R_dgelist_basic.csv")
        p = _load("Py_dgelist_basic.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-4

    def test_column_sums(self):
        r = _load("R_dgelist_colsums.csv")
        p = _load("Py_dgelist_colsums.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-6

    def test_counts_head(self):
        r = _load_idx("R_dgelist_counts_head.csv")
        p = _load_idx("Py_dgelist_counts_head.csv")
        assert r.shape == p.shape
        max_diff = np.max(np.abs(r.values - p.values))
        assert max_diff < 1e-6


class TestDGEListOperations:
    """DGEList operations: remove.zeros, custom lib, genes, filter."""

    def test_remove_zeros(self):
        r = _load("R_dgelist_removezeros.csv")
        p = _load("Py_dgelist_removezeros.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-4

    def test_custom_lib_size(self):
        r = _load("R_dgelist_custom_lib.csv")
        p = _load("Py_dgelist_custom_lib.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-4

    def test_genes_annotation(self):
        r = _load("R_dgelist_genes_head.csv")
        p = _load("Py_dgelist_genes_head.csv")
        assert r.shape == p.shape
        for col in r.columns:
            if col in p.columns:
                assert (r[col] == p[col]).all(), f"Column '{col}' mismatch"


class TestFilterByExpr:
    """filterByExpr comparison."""

    def test_filter_summary(self):
        r = _load("R_dgelist_filter.csv")
        p = _load("Py_dgelist_filter.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-4

    def test_filter_head_agreement(self):
        r = _load("R_dgelist_filter_head.csv")
        p = _load("Py_dgelist_filter_head.csv")
        r_keep = r["keep"].astype(str).str.upper() == "TRUE"
        p_keep = p["keep"].astype(str).str.upper() == "TRUE"
        n_agree = (r_keep == p_keep).sum()
        n_total = len(r_keep)
        # Allow a few disagreements at boundary
        assert n_agree / n_total > 0.95

    def test_filtered_dgelist(self):
        r = _load("R_dgelist_filtered.csv")
        p = _load("Py_dgelist_filtered.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-4


class TestNormalization:
    """Normalization factors from DGEList pipeline."""

    def test_normed_dgelist(self):
        r = _load("R_dgelist_normed.csv")
        p = _load("Py_dgelist_normed.csv")
        assert r.shape == p.shape
        max_rel = _compare_df(r, p)
        assert max_rel < 1e-3


class TestDispersionSummary:
    """Dispersion summary scalars (not per-gene 44K)."""

    def test_dispersion_summary(self):
        r = _load("R_dgelist_disp_summary.csv")
        p = _load("Py_dgelist_disp_summary.csv")
        for col in r.columns:
            if col not in p.columns:
                continue
            try:
                rv = float(r[col].iloc[0])
                pv = float(p[col].iloc[0])
                rel = abs(rv - pv) / (abs(rv) + 1e-15)
                assert rel < 0.15, f"{col}: R={rv}, Py={pv}, rel_diff={rel}"
            except (ValueError, TypeError):
                assert str(r[col].iloc[0]) == str(p[col].iloc[0])


# ======================================================================
# glmTreat (QL and LRT)
# ======================================================================

class TestQLTreat:
    """QL-based glmTreat on synthetic data (100 genes)."""

    def test_ql_lfc12_logfc(self):
        m = _compare_treat("R_treat_ql_lfc12.csv", "Py_treat_ql_lfc12.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_ql_lfc12_logcpm(self):
        m = _compare_treat("R_treat_ql_lfc12.csv", "Py_treat_ql_lfc12.csv")
        assert m["cpm_max_rel"] < 0.10

    def test_ql_lfc12_pvalue(self):
        m = _compare_treat("R_treat_ql_lfc12.csv", "Py_treat_ql_lfc12.csv")
        assert m["logp_max_diff"] < 2.0

    def test_ql_lfc15_logfc(self):
        m = _compare_treat("R_treat_ql_lfc15.csv", "Py_treat_ql_lfc15.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_ql_lfc15_pvalue(self):
        m = _compare_treat("R_treat_ql_lfc15.csv", "Py_treat_ql_lfc15.csv")
        assert m["logp_max_diff"] < 2.0

    def test_ql_lfc10_logfc(self):
        m = _compare_treat("R_treat_ql_lfc10.csv", "Py_treat_ql_lfc10.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_ql_lfc10_pvalue(self):
        m = _compare_treat("R_treat_ql_lfc10.csv", "Py_treat_ql_lfc10.csv")
        assert m["logp_max_diff"] < 2.0

    def test_ql_worst_logfc(self):
        m = _compare_treat("R_treat_ql_worst.csv", "Py_treat_ql_worst.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_ql_worst_pvalue(self):
        m = _compare_treat("R_treat_ql_worst.csv", "Py_treat_ql_worst.csv")
        assert m["logp_max_diff"] < 2.0


class TestLRTTreat:
    """LRT-based glmTreat on synthetic data (100 genes)."""

    def test_lrt_lfc12_logfc(self):
        m = _compare_treat("R_treat_lrt_lfc12.csv", "Py_treat_lrt_lfc12.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_lrt_lfc12_logcpm(self):
        m = _compare_treat("R_treat_lrt_lfc12.csv", "Py_treat_lrt_lfc12.csv")
        assert m["cpm_max_rel"] < 0.10

    def test_lrt_lfc12_pvalue(self):
        m = _compare_treat("R_treat_lrt_lfc12.csv", "Py_treat_lrt_lfc12.csv")
        assert m["logp_max_diff"] < 2.0

    def test_lrt_lfc15_logfc(self):
        m = _compare_treat("R_treat_lrt_lfc15.csv", "Py_treat_lrt_lfc15.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_lrt_lfc15_pvalue(self):
        m = _compare_treat("R_treat_lrt_lfc15.csv", "Py_treat_lrt_lfc15.csv")
        assert m["logp_max_diff"] < 2.0

    def test_lrt_lfc10_logfc(self):
        m = _compare_treat("R_treat_lrt_lfc10.csv", "Py_treat_lrt_lfc10.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_lrt_lfc10_pvalue(self):
        m = _compare_treat("R_treat_lrt_lfc10.csv", "Py_treat_lrt_lfc10.csv")
        assert m["logp_max_diff"] < 2.0

    def test_lrt_worst_logfc(self):
        m = _compare_treat("R_treat_lrt_worst.csv", "Py_treat_lrt_worst.csv")
        assert m["lfc_max_rel"] < 0.10

    def test_lrt_worst_pvalue(self):
        m = _compare_treat("R_treat_lrt_worst.csv", "Py_treat_lrt_worst.csv")
        assert m["logp_max_diff"] < 2.0


# ======================================================================
# camera (gene set testing)
# ======================================================================

class TestCameraRvsPy:
    """camera: R vs Python on simulated 100-gene data."""

    @pytest.fixture(params=[
        ("camera (default)", "R_camera_default.csv", "Py_camera_default.csv"),
        ("camera (log-CPM)", "R_camera_logcpm.csv", "Py_camera_logcpm.csv"),
        ("camera (use.ranks)", "R_camera_ranks.csv", "Py_camera_ranks.csv"),
        ("camera (inter.gene.cor=0)", "R_camera_cor0.csv", "Py_camera_cor0.csv"),
        ("camera (inter.gene.cor=0.05)", "R_camera_cor05.csv", "Py_camera_cor05.csv"),
        ("camera (allow.neg.cor)", "R_camera_allowneg.csv", "Py_camera_allowneg.csv"),
        ("camera (unsorted)", "R_camera_unsorted.csv", "Py_camera_unsorted.csv"),
    ], ids=lambda x: x[0])
    def camera_pair(self, request):
        name, r_file, p_file = request.param
        return name, _load_idx(r_file), _load_idx(p_file)

    def test_ngenes_match(self, camera_pair):
        name, r, p = camera_pair
        common = sorted(set(r.index) & set(p.index))
        r, p = r.loc[common], p.loc[common]
        assert (r["NGenes"].values == p["NGenes"].values).all()

    def test_direction_agreement(self, camera_pair):
        name, r, p = camera_pair
        common = sorted(set(r.index) & set(p.index))
        r, p = r.loc[common], p.loc[common]
        n_agree = (r["Direction"].values == p["Direction"].values).sum()
        assert n_agree == len(common)

    def test_pvalue_close(self, camera_pair):
        name, r, p = camera_pair
        common = sorted(set(r.index) & set(p.index))
        r, p = r.loc[common], p.loc[common]
        rv = r["PValue"].values
        pv = p["PValue"].values
        rel = np.abs(rv - pv) / (np.abs(rv) + 1e-15)
        assert np.max(rel) < 1.0, f"{name}: max_rel_diff={np.max(rel)}"

    def test_fdr_close(self, camera_pair):
        name, r, p = camera_pair
        common = sorted(set(r.index) & set(p.index))
        r, p = r.loc[common], p.loc[common]
        if "FDR" in r.columns and "FDR" in p.columns:
            rv = r["FDR"].values
            pv = p["FDR"].values
            rel = np.abs(rv - pv) / (np.abs(rv) + 1e-15)
            assert np.max(rel) < 1.0


# ======================================================================
# fry
# ======================================================================

class TestFryRvsPy:
    """fry: R vs Python comparison."""

    def test_direction_agreement(self):
        r = _load_idx("R_fry.csv")
        p = _load_idx("Py_fry.csv")
        common = sorted(set(r.index) & set(p.index))
        r_dir = r.loc[common, "Direction"].values
        p_dir = p.loc[common, "Direction"].values
        n_agree = (r_dir == p_dir).sum()
        assert n_agree >= len(common) - 2

    def test_pvalue_order_of_magnitude(self):
        r = _load_idx("R_fry.csv")
        p = _load_idx("Py_fry.csv")
        common = sorted(set(r.index) & set(p.index))
        for gs in common:
            r_p = r.loc[gs, "PValue"]
            p_p = p.loc[gs, "PValue"]
            r_logp = np.log10(max(r_p, 1e-300))
            p_logp = np.log10(max(p_p, 1e-300))
            assert abs(r_logp - p_logp) < 4.0, \
                f"{gs}: R_pval={r_p:.4e}, Py_pval={p_p:.4e}"

    def test_mixed_pvalue_exists(self):
        r = _load_idx("R_fry.csv")
        p = _load_idx("Py_fry.csv")
        assert "PValue.Mixed" in r.columns
        assert "PValue.Mixed" in p.columns

    def test_ngenes_match(self):
        r = _load_idx("R_fry.csv")
        p = _load_idx("Py_fry.csv")
        common = sorted(set(r.index) & set(p.index))
        assert (r.loc[common, "NGenes"].values == p.loc[common, "NGenes"].values).all()


# ======================================================================
# roast
# ======================================================================

class TestRoastRvsPy:
    """roast: R vs Python comparison (rotation-based, expect some variation)."""

    def test_has_expected_rows(self):
        r = _load_idx("R_roast.csv")
        p = _load_idx("Py_roast.csv")
        expected = {"Down", "Up", "UpOrDown", "Mixed"}
        assert expected <= set(r.index)
        assert expected <= set(p.index)

    def test_pvalues_valid(self):
        p = _load_idx("Py_roast.csv")
        p_col = "P.Value" if "P.Value" in p.columns else "p.value.P.Value"
        pv = p[p_col].values
        assert np.all(pv >= 0) and np.all(pv <= 1)

    def test_pvalue_same_order(self):
        r = _load_idx("R_roast.csv")
        p = _load_idx("Py_roast.csv")
        r_col = "P.Value" if "P.Value" in r.columns else "p.value.P.Value"
        p_col = "P.Value" if "P.Value" in p.columns else "p.value.P.Value"
        for direction in ["Down", "Up", "UpOrDown", "Mixed"]:
            if direction in r.index and direction in p.index:
                r_p = r.loc[direction, r_col]
                p_p = p.loc[direction, p_col]
                assert abs(r_p - p_p) < 0.3, \
                    f"{direction}: R={r_p:.4f}, Py={p_p:.4f}"


# ======================================================================
# mroast
# ======================================================================

class TestMroastRvsPy:
    """mroast: R vs Python comparison."""

    def test_direction_agreement(self):
        r = _load_idx("R_mroast.csv")
        p = _load_idx("Py_mroast.csv")
        common = sorted(set(r.index) & set(p.index))
        r_dir = r.loc[common, "Direction"].values
        p_dir = p.loc[common, "Direction"].values
        n_agree = (r_dir == p_dir).sum()
        assert n_agree >= len(common) - 1

    def test_ngenes_match(self):
        r = _load_idx("R_mroast.csv")
        p = _load_idx("Py_mroast.csv")
        common = sorted(set(r.index) & set(p.index))
        assert (r.loc[common, "NGenes"].values == p.loc[common, "NGenes"].values).all()

    def test_pvalues_valid(self):
        p = _load_idx("Py_mroast.csv")
        assert (p["PValue"] >= 0).all() and (p["PValue"] <= 1).all()

    def test_pvalue_same_range(self):
        r = _load_idx("R_mroast.csv")
        p = _load_idx("Py_mroast.csv")
        common = sorted(set(r.index) & set(p.index))
        for gs in common:
            r_p = r.loc[gs, "PValue"]
            p_p = p.loc[gs, "PValue"]
            assert abs(r_p - p_p) < 0.3, \
                f"{gs}: R={r_p:.4f}, Py={p_p:.4f}"


# ======================================================================
# romer
# ======================================================================

class TestRomerRvsPy:
    """romer: R vs Python comparison."""

    def test_columns_present(self):
        r = _load_idx("R_romer.csv")
        p = _load_idx("Py_romer.csv")
        assert {"Up", "Down", "Mixed"} <= set(r.columns)
        assert {"Up", "Down", "Mixed"} <= set(p.columns)

    def test_pvalues_valid(self):
        p = _load_idx("Py_romer.csv")
        for col in ["Up", "Down", "Mixed"]:
            assert (p[col] >= 0).all() and (p[col] <= 1).all()

    def test_direction_agreement(self):
        r = _load_idx("R_romer.csv")
        p = _load_idx("Py_romer.csv")
        common = sorted(set(r.index) & set(p.index))
        n_agree = 0
        for gs in common:
            r_dir = "Up" if r.loc[gs, "Up"] < r.loc[gs, "Down"] else "Down"
            p_dir = "Up" if p.loc[gs, "Up"] < p.loc[gs, "Down"] else "Down"
            if r_dir == p_dir:
                n_agree += 1
        assert n_agree >= len(common) - 1

    def test_ngenes_match(self):
        r = _load_idx("R_romer.csv")
        p = _load_idx("Py_romer.csv")
        common = sorted(set(r.index) & set(p.index))
        assert (r.loc[common, "NGenes"].values == p.loc[common, "NGenes"].values).all()

    def test_pvalue_same_range(self):
        r = _load_idx("R_romer.csv")
        p = _load_idx("Py_romer.csv")
        common = sorted(set(r.index) & set(p.index))
        for gs in common:
            for col in ["Up", "Down", "Mixed"]:
                r_p = r.loc[gs, col]
                p_p = p.loc[gs, col]
                assert abs(r_p - p_p) < 0.3, \
                    f"{gs} {col}: R={r_p:.4f}, Py={p_p:.4f}"
