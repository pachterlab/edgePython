# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
edgePython MCP Server — Differential expression analysis via tool calls.

Wraps edgePython's core pipeline so AI agents can run RNA-seq DE analysis
step-by-step: load data → filter → normalize → estimate dispersions →
fit model → test contrasts → get results → generate plots.

Usage:
    python3 edgepython_server.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from typing import Optional

from fastmcp import FastMCP

# Ensure edgePython is importable
_edgepython_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _edgepython_root not in sys.path:
    sys.path.insert(0, _edgepython_root)

import edgepython as ep

# ---------------------------------------------------------------------------
# Server + state
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "edgepython",
    instructions=(
        "RNA-seq differential expression analysis server powered by edgePython. "
        "Bulk workflow: load_data → filter_genes → normalize → set_design → "
        "estimate_dispersion → fit_model → test_contrast → get_top_genes / generate_plot. "
        "Single-cell workflow: load_sc_data → fit_sc_model → test_sc_coef → get_sc_top_genes. "
        "Use describe() or describe_sc() to see current pipeline state."
    ),
)

# Module-level state — mutated by tools across a session.
_state: dict = {
    "dgelist": None,       # Current DGEList
    "design": None,        # Design matrix (ndarray)
    "design_info": None,   # Column names for the design matrix
    "fit": None,           # Fitted QL-GLM (DGEGLM-like dict)
    "glm_fit": None,       # Fitted GLM (non-QL)
    "results": {},         # name → DGELRT-like dict
    "last_result": None,   # Name of most recent test result
    "filtered": False,
    "normalized": False,
    "dispersions_estimated": False,
}


def _require(key: str, label: str):
    """Raise if a pipeline stage hasn't been completed yet."""
    if _state[key] is None:
        raise ValueError(
            f"No {label} available. Run the appropriate step first."
        )


def _gene_name(row):
    """Extract gene name from a top_tags result row."""
    # top_tags prepends gene annotation columns; check for GeneID first
    for col in ("GeneID", "gene_id", "gene_name", "Symbol"):
        if col in row.index:
            return str(row[col])
    # Fall back to row index
    return str(row.name)


# ---------------------------------------------------------------------------
# Tool 1: load_data
# ---------------------------------------------------------------------------

@mcp.tool()
def load_data(
    counts_path: str,
    sample_info_path: Optional[str] = None,
    group_column: Optional[str] = None,
    separator: Optional[str] = None,
) -> str:
    """Load a count matrix (and optional sample metadata) to create a DGEList.

    Args:
        counts_path: Path to count matrix file (TSV/CSV). Genes as rows,
            samples as columns. First column is gene IDs.
        sample_info_path: Optional path to sample metadata file (TSV/CSV).
            Must have a column matching sample names in the count matrix.
        group_column: Column in sample info to use as group factor.
            Auto-detected if not provided.
        separator: Column separator ('\\t' or ','). Auto-detected from
            file extension if not provided.
    """
    # Auto-detect separator
    if separator is None:
        separator = "," if counts_path.endswith(".csv") else "\t"

    # Read counts
    counts_df = pd.read_csv(counts_path, sep=separator, index_col=0)
    gene_ids = list(counts_df.index)
    sample_names = list(counts_df.columns)
    counts = counts_df.values.astype(np.float64)

    genes_df = pd.DataFrame({"GeneID": gene_ids})

    # Read sample info if provided
    group = None
    samples_df = None
    if sample_info_path is not None:
        sep2 = "," if sample_info_path.endswith(".csv") else "\t"
        samples_df = pd.read_csv(sample_info_path, sep=sep2)

        # Try to align rows to sample order
        for col in samples_df.columns:
            vals = [str(v) for v in samples_df[col]]
            if set(sample_names).issubset(set(vals)):
                samples_df = samples_df.set_index(col).loc[sample_names].reset_index()
                break

        # Extract group
        if group_column is not None:
            group = list(samples_df[group_column].astype(str))
        else:
            # Auto-detect: first column with few unique values
            for col in samples_df.columns:
                n_unique = samples_df[col].nunique()
                if 2 <= n_unique <= len(sample_names) // 2:
                    group = list(samples_df[col].astype(str))
                    group_column = col
                    break

    dgelist = ep.make_dgelist(
        counts=counts,
        group=group,
        genes=genes_df,
        samples=samples_df,
    )
    # Attach sample/gene names for display
    dgelist["_sample_names"] = sample_names
    dgelist["_gene_ids"] = gene_ids

    _state["dgelist"] = dgelist
    _state["filtered"] = False
    _state["normalized"] = False
    _state["dispersions_estimated"] = False
    _state["design"] = None
    _state["design_info"] = None
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None

    # Build summary
    lib_sizes = dgelist["samples"]["lib.size"].values
    lines = [
        f"Loaded {counts.shape[0]:,} genes × {counts.shape[1]} samples",
        f"Samples: {', '.join(sample_names)}",
    ]
    if group is not None:
        from collections import Counter
        gc = Counter(group)
        lines.append(f"Groups ({group_column}): " + ", ".join(
            f"{k} (n={v})" for k, v in gc.items()
        ))
    lines.append(
        f"Library sizes: {int(lib_sizes.min()):,} – {int(lib_sizes.max()):,} "
        f"(median {int(np.median(lib_sizes)):,})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1b: load_data_auto
# ---------------------------------------------------------------------------

@mcp.tool()
def load_data_auto(
    data: Optional[str] = None,
    data_list: Optional[list] = None,
    source: Optional[str] = None,
    format: Optional[str] = None,
    path: Optional[str] = None,
    group: Optional[list] = None,
    labels: Optional[list] = None,
    columns: Optional[list] = None,
    sep: str = "\t",
    obs_col: Optional[str] = None,
    layer: Optional[str] = None,
    ngibbs: int = 100,
    verbose: bool = True,
) -> str:
    """Load data using edgePython's flexible read_data() loader.

    Args:
        data: Path to file/dir or other supported value. If data_list is provided,
            this is ignored.
        data_list: Optional list of paths (e.g. quantification dirs).
        source: Data source (kallisto, salmon, oarfish, rsem, 10x, table,
            anndata, rds, sparse, matrix, dataframe). Auto-detected if None.
        format: Source-specific format (e.g. 'h5' or 'tsv' for kallisto).
        path: Base path prefix for relative paths.
        group: Optional group assignments.
        labels: Optional sample labels.
        columns: For table format: (gene_id_col, count_col) 0-based indices.
        sep: Field separator for table/CSV files.
        obs_col: For AnnData: column in .obs to use as group factor.
        layer: For AnnData: layer name to use instead of .X.
        ngibbs: For RSEM: number of Gibbs samples per sample.
        verbose: Print progress messages.
    """
    if data_list is not None:
        data_input = data_list
    elif data is not None:
        data_input = data
    else:
        raise ValueError("Provide data or data_list.")

    dgelist = ep.read_data(
        data_input,
        source=source,
        format=format,
        path=path,
        group=group,
        labels=labels,
        columns=columns,
        sep=sep,
        obs_col=obs_col,
        layer=layer,
        ngibbs=ngibbs,
        verbose=verbose,
    )

    _state["dgelist"] = dgelist
    _state["filtered"] = False
    _state["normalized"] = False
    _state["dispersions_estimated"] = False
    _state["design"] = None
    _state["design_info"] = None
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None

    counts = dgelist["counts"]
    gene_ids = None
    if dgelist.get("genes") is not None:
        gdf = dgelist["genes"]
        for col in ("GeneID", "gene_id", "gene_name", "Symbol"):
            if col in gdf.columns:
                gene_ids = list(gdf[col].astype(str))
                break
    if gene_ids is not None:
        dgelist["_gene_ids"] = gene_ids

    sample_names = None
    if dgelist.get("samples") is not None:
        sdf = dgelist["samples"]
        if sdf.index is not None and len(sdf.index) == counts.shape[1]:
            sample_names = [str(x) for x in sdf.index]
    if sample_names is not None:
        dgelist["_sample_names"] = sample_names

    lines = [f"Loaded {counts.shape[0]:,} genes × {counts.shape[1]} samples"]
    if sample_names is not None:
        lines.append(f"Samples: {', '.join(sample_names)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: describe
# ---------------------------------------------------------------------------

@mcp.tool()
def describe() -> str:
    """Report the current state of the analysis pipeline."""
    lines = []

    d = _state["dgelist"]
    if d is None:
        return "No data loaded. Use load_data() first."

    counts = d["counts"]
    lines.append(f"Data: {counts.shape[0]:,} genes × {counts.shape[1]} samples")
    lines.append(f"Samples: {', '.join(d.get('_sample_names', [str(i) for i in range(counts.shape[1])]))}")
    lines.append(f"Filtered: {'yes' if _state['filtered'] else 'no'}")
    lines.append(f"Normalized: {'yes' if _state['normalized'] else 'no'}")

    if _state["normalized"]:
        nf = d["samples"]["norm.factors"].values
        lines.append(f"  Norm factors: {', '.join(f'{v:.4f}' for v in nf)}")

    lines.append(f"Dispersions estimated: {'yes' if _state['dispersions_estimated'] else 'no'}")
    if _state["dispersions_estimated"]:
        cd = d.get("common.dispersion")
        if cd is not None:
            lines.append(f"  Common dispersion: {cd:.4f} (BCV: {np.sqrt(cd):.4f})")

    if _state["design"] is not None:
        dm = _state["design"]
        lines.append(f"Design: {dm.shape[1]} coefficients — {', '.join(_state['design_info'])}")

    if _state["fit"] is not None:
        fit = _state["fit"]
        pdf = fit.get("df.prior")
        if pdf is not None:
            if np.isscalar(pdf):
                lines.append(f"Model fitted (QL): prior df = {pdf:.2f}")
            else:
                lines.append(f"Model fitted (QL): median prior df = {np.median(pdf):.2f}")

    if _state["results"]:
        lines.append(f"Test results: {', '.join(_state['results'].keys())}")
    else:
        lines.append("No test results yet.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2b: reset_state
# ---------------------------------------------------------------------------

@mcp.tool()
def reset_state() -> str:
    """Reset all pipeline state."""
    _state["dgelist"] = None
    _state["design"] = None
    _state["design_info"] = None
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None
    _state["filtered"] = False
    _state["normalized"] = False
    _state["dispersions_estimated"] = False
    return "State cleared."


# ---------------------------------------------------------------------------
# Tool 3: filter_genes
# ---------------------------------------------------------------------------

@mcp.tool()
def filter_genes(
    min_count: int = 10,
    min_total_count: int = 15,
) -> str:
    """Filter out lowly-expressed genes using edgeR's filterByExpr logic.

    Uses adaptive CPM thresholds based on library sizes and group structure.

    Args:
        min_count: Minimum count threshold (used to derive CPM cutoff).
            Default: 10.
        min_total_count: Minimum total count across all samples. Default: 15.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    group = d["samples"].get("group")
    keep = ep.filter_by_expr(
        d, group=group, min_count=min_count, min_total_count=min_total_count
    )

    n_before = d["counts"].shape[0]
    n_after = int(keep.sum())

    # Apply filter
    d["counts"] = d["counts"][keep]
    if "genes" in d and d["genes"] is not None:
        d["genes"] = d["genes"].iloc[keep.nonzero()[0]].reset_index(drop=True)
    gene_ids = d.get("_gene_ids")
    if gene_ids is not None:
        d["_gene_ids"] = [gene_ids[i] for i in range(n_before) if keep[i]]

    # Clear downstream state
    _state["dispersions_estimated"] = False
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None
    _state["filtered"] = True

    # Recompute lib sizes after filtering
    d["samples"]["lib.size"] = d["counts"].sum(axis=0).astype(np.float64)

    return (
        f"Filtered: {n_before:,} → {n_after:,} genes "
        f"({n_before - n_after:,} removed)"
    )


# ---------------------------------------------------------------------------
# Tool 4: normalize
# ---------------------------------------------------------------------------

@mcp.tool()
def normalize(method: str = "TMM") -> str:
    """Apply library-size normalization.

    Args:
        method: Normalization method. One of 'TMM' (default), 'TMMwsp',
            'RLE', 'upperquartile', or 'none'.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    d = ep.calc_norm_factors(d, method=method)
    _state["dgelist"] = d
    _state["normalized"] = True

    nf = d["samples"]["norm.factors"].values
    eff = d["samples"]["lib.size"].values * nf

    lines = [
        f"Normalization: {method}",
        "Sample | Norm Factor | Effective Lib Size",
        "-------|------------|-------------------",
    ]
    names = d.get("_sample_names", [f"S{i+1}" for i in range(len(nf))])
    for i, name in enumerate(names):
        lines.append(f"{name} | {nf[i]:.4f} | {int(eff[i]):,}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4b: normalize_chip — ChIP-Seq normalization to input
# ---------------------------------------------------------------------------

@mcp.tool()
def normalize_chip(
    input_csv: str,
    dispersion: float = 0.01,
    niter: int = 6,
    loss: str = "p",
) -> str:
    """Normalize ChIP-Seq counts to input control and test for enrichment.

    Reads input control counts from a CSV file and normalizes the currently
    loaded ChIP response counts against them, setting offsets for downstream
    GLM fitting.

    Args:
        input_csv: Path to a CSV/TSV file of input control counts with the
            same genes (rows) and samples (columns) as the loaded data.
        dispersion: Negative binomial dispersion (default 0.01).
        niter: Number of iterations for scaling-factor estimation.
        loss: Loss function — 'p' (cumulative probabilities) or 'z' (z-value).
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    sep = "\t" if input_csv.endswith(".tsv") else ","
    inp_df = pd.read_csv(input_csv, sep=sep, index_col=0)
    inp_mat = inp_df.values.astype(np.float64)

    if inp_mat.shape[0] != d["counts"].shape[0]:
        raise ValueError(
            f"Row count mismatch: input has {inp_mat.shape[0]} rows but "
            f"loaded data has {d['counts'].shape[0]} genes."
        )

    d = ep.calc_norm_offsets_for_chip(
        inp_mat, d, dispersion=dispersion, niter=niter, loss=loss,
    )
    _state["dgelist"] = d

    lines = [
        f"ChIP normalization applied (dispersion={dispersion}, loss='{loss}').",
        f"Offset matrix set ({d['offset'].shape[0]:,} genes × "
        f"{d['offset'].shape[1]} samples).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4c: chip_enrichment_test — per-sample ChIP enrichment p-values
# ---------------------------------------------------------------------------

@mcp.tool()
def chip_enrichment_test(
    input_csv: str,
    sample: int = 0,
    dispersion: float = 0.01,
    niter: int = 6,
    n_top: int = 20,
) -> str:
    """Test for ChIP-Seq enrichment relative to input for one sample.

    Returns the top enriched features ranked by p-value.

    Args:
        input_csv: Path to input control counts (CSV/TSV).
        sample: Sample index (0-based) to test.
        dispersion: Negative binomial dispersion.
        niter: Number of iterations.
        n_top: Number of top enriched features to report.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    sep = "\t" if input_csv.endswith(".tsv") else ","
    inp_df = pd.read_csv(input_csv, sep=sep, index_col=0)
    inp_mat = inp_df.values.astype(np.float64)

    if sample < 0 or sample >= d["counts"].shape[1]:
        raise ValueError(f"sample must be 0..{d['counts'].shape[1]-1}")

    out = ep.normalize_chip_to_input(
        inp_mat[:, sample], d["counts"][:, sample],
        dispersion=dispersion, niter=niter,
    )

    gene_ids = d.get("_gene_ids")
    if gene_ids is None:
        gene_ids = [str(i) for i in range(d["counts"].shape[0])]

    order = np.argsort(out['p_value'])
    n_show = min(n_top, len(order))

    names = d.get("_sample_names", [f"S{i}" for i in range(d["counts"].shape[1])])
    lines = [
        f"ChIP enrichment test — sample '{names[sample]}'",
        f"Scaling factor: {out['scaling_factor']:.4f}",
        f"Proportion enriched: {out['prop_enriched']:.4f}",
        "",
        f"Top {n_show} enriched features:",
        "Gene | p-value | mid-p",
        "-----|---------|------",
    ]
    for i in range(n_show):
        idx = order[i]
        lines.append(
            f"{gene_ids[idx]} | {out['p_value'][idx]:.4e} | "
            f"{out['pmid_value'][idx]:.4e}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: set_design
# ---------------------------------------------------------------------------

@mcp.tool()
def set_design(formula: str) -> str:
    """Set the experimental design using an R-style formula.

    Args:
        formula: R-style formula, e.g. '~ 0 + group', '~ group + batch'.
            Variables must match column names in the sample metadata.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    # Build data for patsy
    samples = d.get("samples")
    if samples is None:
        raise ValueError("No sample metadata — load sample info with load_data().")

    design = ep.model_matrix(formula, data=samples)

    # Extract column names from patsy
    import patsy
    dinfo = patsy.dmatrix(formula, data=samples, return_type="dataframe")
    col_names = list(dinfo.columns)

    _state["design"] = design
    _state["design_info"] = col_names
    # Clear downstream
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None

    lines = [
        f"Design matrix: {design.shape[0]} samples × {design.shape[1]} coefficients",
        f"Columns: {', '.join(col_names)}",
        f"Rank: {np.linalg.matrix_rank(design)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5b: set_design_matrix
# ---------------------------------------------------------------------------

@mcp.tool()
def set_design_matrix(matrix: list, columns: list) -> str:
    """Set the design matrix directly.

    Args:
        matrix: 2D list of floats with shape (samples, coefficients).
        columns: Column names for the design matrix.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]
    counts = d["counts"]

    design = np.asarray(matrix, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError("Design matrix must be 2D.")
    if design.shape[0] != counts.shape[1]:
        raise ValueError(
            f"Design has {design.shape[0]} rows but data has {counts.shape[1]} samples."
        )
    if len(columns) != design.shape[1]:
        raise ValueError(
            f"Design has {design.shape[1]} columns but {len(columns)} column names provided."
        )

    _state["design"] = design
    _state["design_info"] = list(columns)
    _state["fit"] = None
    _state["glm_fit"] = None
    _state["results"] = {}
    _state["last_result"] = None

    lines = [
        f"Design matrix set: {design.shape[0]} samples × {design.shape[1]} coefficients",
        f"Columns: {', '.join(columns)}",
        f"Rank: {np.linalg.matrix_rank(design)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: estimate_dispersion
# ---------------------------------------------------------------------------

@mcp.tool()
def estimate_dispersion(robust: bool = True) -> str:
    """Estimate NB dispersions (common, trended, tagwise).

    Args:
        robust: Use robust empirical Bayes. Default True.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    kwargs = {"robust": robust}
    if _state["design"] is not None:
        kwargs["design"] = _state["design"]

    d = ep.estimate_disp(d, **kwargs)
    _state["dgelist"] = d
    _state["dispersions_estimated"] = True

    cd = d.get("common.dispersion", None)
    prior_df = d.get("prior.df", None)
    lines = []
    if cd is not None:
        lines.append(f"Common dispersion: {cd:.4f}")
        lines.append(f"BCV: {np.sqrt(cd):.4f}")
    if prior_df is not None:
        if np.isscalar(prior_df):
            lines.append(f"Prior df: {prior_df:.2f}")
        else:
            lines.append(f"Prior df (median): {np.median(prior_df):.2f}")
    td = d.get("trended.dispersion")
    if td is not None:
        lines.append(f"Trended dispersion range: {td.min():.4f} – {td.max():.4f}")

    return "\n".join(lines) if lines else "Dispersions estimated."


# ---------------------------------------------------------------------------
# Tool 7: fit_model
# ---------------------------------------------------------------------------

@mcp.tool()
def fit_model(robust: bool = True) -> str:
    """Fit a quasi-likelihood negative binomial GLM.

    Args:
        robust: Use robust empirical Bayes for QL dispersion. Default True.
    """
    _require("dgelist", "DGEList")
    _require("design", "design matrix")
    d = _state["dgelist"]

    fit = ep.glm_ql_fit(d, design=_state["design"], robust=robust)
    _state["fit"] = fit
    # Clear old results
    _state["results"] = {}
    _state["last_result"] = None

    n_genes = fit["coefficients"].shape[0]
    n_coef = fit["coefficients"].shape[1]
    pdf = fit.get("df.prior")
    s2 = fit.get("s2.prior")

    lines = [f"Fitted QL-GLM: {n_genes:,} genes, {n_coef} coefficients"]
    if pdf is not None:
        if np.isscalar(pdf):
            lines.append(f"Prior df: {pdf:.2f}")
        else:
            lines.append(f"Prior df (median): {np.median(pdf):.2f}")
    if s2 is not None:
        if np.isscalar(s2):
            lines.append(f"Prior QL dispersion: {s2:.4f}")
        else:
            lines.append(f"Prior QL dispersion (median): {np.median(s2):.4f}")
    lines.append(f"Coefficients: {', '.join(_state['design_info'])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7b: fit_glm
# ---------------------------------------------------------------------------

@mcp.tool()
def fit_glm() -> str:
    """Fit a negative binomial GLM (non-QL)."""
    _require("dgelist", "DGEList")
    _require("design", "design matrix")
    d = _state["dgelist"]

    fit = ep.glm_fit(d, design=_state["design"])
    _state["glm_fit"] = fit
    _state["results"] = {}
    _state["last_result"] = None

    n_genes = fit["coefficients"].shape[0]
    n_coef = fit["coefficients"].shape[1]
    lines = [f"Fitted GLM: {n_genes:,} genes, {n_coef} coefficients"]
    lines.append(f"Coefficients: {', '.join(_state['design_info'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8: test_contrast
# ---------------------------------------------------------------------------

def _parse_contrast(contrast_str: str, col_names: list) -> np.ndarray:
    """Parse a contrast string like 'A - B' into a numeric vector.

    Supports:
        'A - B'           → simple difference
        '(A + B)/2 - C'   → average vs reference
        'A'               → single coefficient test
    """
    contrast = np.zeros(len(col_names))

    # Build lookup map: multiple aliases for each column index.
    # Patsy generates names like 'group[A]', 'group[T.A]', 'Intercept'.
    # Users may type 'groupA', 'group[A]', 'A', etc.
    clean_map = {}
    for i, cn in enumerate(col_names):
        clean_map[cn] = i  # exact name, e.g. 'group[A]'

        # Strip patsy treatment-coded wrapper: 'group[T.xxx]' → 'xxx'
        m = re.match(r'^(.+)\[T\.\s*(.+)\]$', cn)
        if m:
            prefix, val = m.group(1), m.group(2)
            clean_map[val] = i              # 'xxx'
            clean_map[prefix + val] = i     # 'groupxxx'

        # Strip patsy no-intercept wrapper: 'group[xxx]' → 'xxx'
        m2 = re.match(r'^(.+)\[(.+)\]$', cn)
        if m2:
            prefix, val = m2.group(1), m2.group(2)
            clean_map[val] = i              # 'xxx'  (e.g. 'A')
            clean_map[prefix + val] = i     # 'groupA'

    def _find_col(term: str) -> int:
        term = term.strip()
        if term in clean_map:
            return clean_map[term]
        # Case-insensitive fallback
        for k, v in clean_map.items():
            if k.lower() == term.lower():
                return v
        raise ValueError(
            f"Term '{term}' not found in design columns: {col_names}\n"
            f"Recognized aliases: {sorted(clean_map.keys())}"
        )

    # Try simple 'A - B' pattern first
    parts = re.split(r'\s*-\s*', contrast_str)
    if len(parts) == 2 and '(' not in contrast_str and '+' not in contrast_str:
        i_pos = _find_col(parts[0])
        i_neg = _find_col(parts[1])
        contrast[i_pos] = 1
        contrast[i_neg] = -1
        return contrast

    # General parser: split into +/- terms
    # Add '+' at start if not starting with '-'
    s = contrast_str.strip()
    if not s.startswith('-'):
        s = '+' + s

    # Tokenize: find all (+/-)(coefficient_or_group)
    tokens = re.findall(r'([+-])\s*(?:(\d*\.?\d*)\s*\*?\s*)?([A-Za-z_][\w.]*)', s)
    for sign, coeff, name in tokens:
        idx = _find_col(name)
        c = float(coeff) if coeff else 1.0
        if sign == '-':
            c = -c
        contrast[idx] += c

    if np.all(contrast == 0):
        raise ValueError(
            f"Could not parse contrast '{contrast_str}'. "
            f"Available columns: {col_names}"
        )

    return contrast


@mcp.tool()
def test_contrast(
    contrast: str,
    name: Optional[str] = None,
    method: str = "qlf",
    lfc: float = 0.585,
) -> str:
    """Test a contrast for differential expression.

    Args:
        contrast: Contrast string, e.g. 'groupB - groupA'. Terms must
            match design matrix column names (shown by set_design/describe).
        name: Label for this result set. Default: auto-generated.
        method: Test method — 'qlf' (quasi-likelihood F-test, default)
            or 'treat' (TREAT with log-FC threshold).
        lfc: Log2-fold-change threshold for TREAT method. Default: log2(1.5) ≈ 0.585.
    """
    _require("fit", "fitted model")
    fit = _state["fit"]
    col_names = _state["design_info"]

    contrast_vec = _parse_contrast(contrast, col_names)

    if method == "treat":
        res = ep.glm_treat(fit, contrast=contrast_vec, lfc=lfc)
    else:
        res = ep.glm_ql_ftest(fit, contrast=contrast_vec)

    if name is None:
        name = contrast.replace(" ", "")

    _state["results"][name] = res
    _state["last_result"] = name

    # Summary
    tt = ep.top_tags(res, n=res["table"].shape[0])
    table = tt["table"]
    fdr = table["FDR"].values
    logfc = table["logFC"].values

    n_up = int(((fdr < 0.05) & (logfc > 0)).sum())
    n_down = int(((fdr < 0.05) & (logfc < 0)).sum())
    n_ns = int((fdr >= 0.05).sum())

    lines = [
        f"Test: {method.upper()} — {contrast}",
        f"DE genes (FDR < 0.05): {n_up} up, {n_down} down, {n_ns} NS",
        "",
        "Top 5 genes:",
    ]

    top5 = ep.top_tags(res, n=5)["table"]
    # Format top genes
    lines.append(f"{'Gene':<20} {'logFC':>8} {'logCPM':>8} {'PValue':>10} {'FDR':>10}")
    lines.append("-" * 60)
    for _, row in top5.iterrows():
        gene = _gene_name(row)
        if len(gene) > 19:
            gene = gene[:17] + ".."
        lines.append(
            f"{gene:<20} {row['logFC']:>8.3f} {row['logCPM']:>8.2f} "
            f"{row['PValue']:>10.2e} {row['FDR']:>10.2e}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8b: test_coef
# ---------------------------------------------------------------------------

@mcp.tool()
def test_coef(
    coef: int,
    name: Optional[str] = None,
    method: str = "qlf",
    lfc: float = 0.585,
) -> str:
    """Test a single coefficient by index.

    Args:
        coef: 0-based coefficient index.
        name: Label for this result set. Default: auto-generated.
        method: 'qlf' (default) or 'treat'.
        lfc: Log2-fold-change threshold for TREAT method.
    """
    _require("fit", "fitted model")
    fit = _state["fit"]

    if method == "treat":
        res = ep.glm_treat(fit, coef=coef, lfc=lfc)
    else:
        res = ep.glm_ql_ftest(fit, coef=coef)

    if name is None:
        name = f"coef{coef}"

    _state["results"][name] = res
    _state["last_result"] = name

    tt = ep.top_tags(res, n=res["table"].shape[0])
    table = tt["table"]
    fdr = table["FDR"].values
    logfc = table["logFC"].values

    n_up = int(((fdr < 0.05) & (logfc > 0)).sum())
    n_down = int(((fdr < 0.05) & (logfc < 0)).sum())
    n_ns = int((fdr >= 0.05).sum())

    lines = [
        f"Test: {method.upper()} — coef {coef}",
        f"DE genes (FDR < 0.05): {n_up} up, {n_down} down, {n_ns} NS",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8c: glm_lrt_test
# ---------------------------------------------------------------------------

@mcp.tool()
def glm_lrt_test(
    coef: Optional[int] = None,
    contrast: Optional[list] = None,
    name: Optional[str] = None,
    use_ql_fit: bool = False,
) -> str:
    """Run a likelihood-ratio test (LRT) from a GLM fit.

    Args:
        coef: 0-based coefficient index (default: last).
        contrast: Optional contrast vector (same length as coefficients).
        name: Label for this result set. Default: auto-generated.
        use_ql_fit: If True, uses the QL fit from fit_model().
    """
    fit = _state["fit"] if use_ql_fit else _state["glm_fit"]
    if fit is None:
        raise ValueError("No GLM fit available. Run fit_glm() first.")

    res = ep.glm_lrt(fit, coef=coef, contrast=contrast)

    if name is None:
        if contrast is not None:
            name = "lrt_contrast"
        elif coef is not None:
            name = f"lrt_coef{coef}"
        else:
            name = "lrt"

    _state["results"][name] = res
    _state["last_result"] = name

    tt = ep.top_tags(res, n=res["table"].shape[0])
    table = tt["table"]
    fdr = table["FDR"].values
    logfc = table["logFC"].values if "logFC" in table.columns else np.zeros(len(table))

    n_up = int(((fdr < 0.05) & (logfc > 0)).sum())
    n_down = int(((fdr < 0.05) & (logfc < 0)).sum())
    n_ns = int((fdr >= 0.05).sum())

    lines = [
        "Test: LRT",
        f"DE genes (FDR < 0.05): {n_up} up, {n_down} down, {n_ns} NS",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8d: exact_test
# ---------------------------------------------------------------------------

@mcp.tool()
def exact_test(
    pair: Optional[list] = None,
    dispersion: str = "auto",
    rejection_region: str = "doubletail",
    big_count: int = 900,
    prior_count: float = 0.125,
    name: Optional[str] = None,
) -> str:
    """Run edgeR-style exact test for two-group DE.

    Args:
        pair: Two group labels to compare. Default: first two groups.
        dispersion: 'auto', 'common', 'trended', 'tagwise', or numeric.
        rejection_region: 'doubletail', 'deviance', or 'smallp'.
        big_count: Threshold for beta approximation.
        prior_count: Prior count for logFC calculation.
        name: Label for this result set. Default: auto-generated.
    """
    _require("dgelist", "DGEList")
    d = _state["dgelist"]

    res = ep.exact_test(
        d,
        pair=pair,
        dispersion=dispersion,
        rejection_region=rejection_region,
        big_count=big_count,
        prior_count=prior_count,
    )

    if name is None:
        if pair is None:
            name = "exact"
        else:
            name = f"exact_{pair[0]}_vs_{pair[1]}"

    _state["results"][name] = res
    _state["last_result"] = name

    tt = ep.top_tags(res, n=res["table"].shape[0])
    table = tt["table"]
    fdr = table["FDR"].values
    logfc = table["logFC"].values

    n_up = int(((fdr < 0.05) & (logfc > 0)).sum())
    n_down = int(((fdr < 0.05) & (logfc < 0)).sum())
    n_ns = int((fdr >= 0.05).sum())

    lines = [
        "Test: Exact",
        f"DE genes (FDR < 0.05): {n_up} up, {n_down} down, {n_ns} NS",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 9: get_top_genes
# ---------------------------------------------------------------------------

@mcp.tool()
def get_top_genes(
    name: Optional[str] = None,
    n: int = 20,
    fdr_threshold: float = 0.05,
) -> str:
    """Get the top differentially expressed genes from a test result.

    Args:
        name: Which result set to query. Default: most recent.
        n: Number of top genes to return. Default: 20.
        fdr_threshold: FDR cutoff for significance. Default: 0.05.
    """
    if name is None:
        name = _state["last_result"]
    if name is None or name not in _state["results"]:
        available = list(_state["results"].keys())
        if not available:
            return "No test results available. Run test_contrast() first."
        return f"Result '{name}' not found. Available: {', '.join(available)}"

    res = _state["results"][name]
    tt = ep.top_tags(res, n=n)
    table = tt["table"]

    # Filter by FDR
    sig = table[table["FDR"] < fdr_threshold]

    lines = [
        f"Result: {name} (showing top {min(n, len(table))} genes, "
        f"{len(sig)} with FDR < {fdr_threshold})",
        "",
    ]

    # Determine which stat column to use (F or LR)
    stat_col = "F" if "F" in table.columns else "LR" if "LR" in table.columns else None
    header = f"{'Gene':<20} {'logFC':>8} {'logCPM':>8}"
    if stat_col:
        header += f" {stat_col:>8}"
    header += f" {'PValue':>10} {'FDR':>10} {'Sig':>3}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in table.iterrows():
        gene = _gene_name(row)
        if len(gene) > 19:
            gene = gene[:17] + ".."
        sig_mark = "***" if row["FDR"] < 0.001 else "**" if row["FDR"] < 0.01 else "*" if row["FDR"] < fdr_threshold else ""
        line = f"{gene:<20} {row['logFC']:>8.3f} {row['logCPM']:>8.2f}"
        if stat_col:
            line += f" {row[stat_col]:>8.2f}"
        line += f" {row['PValue']:>10.2e} {row['FDR']:>10.2e} {sig_mark:>3}"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 9b: get_result_table
# ---------------------------------------------------------------------------

@mcp.tool()
def get_result_table(
    name: Optional[str] = None,
    format: str = "tsv",
    max_rows: Optional[int] = None,
) -> str:
    """Return a full result table as TSV/CSV/JSON text.

    Args:
        name: Which result set to query. Default: most recent.
        format: 'tsv', 'csv', or 'json'.
        max_rows: Optional row cap.
    """
    if name is None:
        name = _state["last_result"]
    if name is None or name not in _state["results"]:
        available = list(_state["results"].keys())
        if not available:
            return "No test results available."
        return f"Result '{name}' not found. Available: {', '.join(available)}"

    res = _state["results"][name]
    tt = ep.top_tags(res, n=res["table"].shape[0])
    table = tt["table"]
    if max_rows is not None:
        table = table.iloc[:max_rows]

    if format == "json":
        return table.to_json(orient="records")
    if format == "csv":
        return table.to_csv(index=False)
    if format == "tsv":
        return table.to_csv(sep="\t", index=False)
    return "Invalid format. Use 'tsv', 'csv', or 'json'."


# ---------------------------------------------------------------------------
# Tool 9c: save_results
# ---------------------------------------------------------------------------

@mcp.tool()
def save_results(
    output_path: str,
    name: Optional[str] = None,
    format: str = "tsv",
) -> str:
    """Save a full result table to disk.

    Args:
        output_path: Destination file path.
        name: Which result set to save. Default: most recent.
        format: 'tsv', 'csv', or 'json'.
    """
    content = get_result_table(name=name, format=format)
    if content.startswith("No test results") or content.startswith("Result '"):
        return content

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved results to: {output_path}"


# ---------------------------------------------------------------------------
# Tool 10: generate_plot
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_plot(
    plot_type: str,
    result_name: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate and save a visualization.

    Args:
        plot_type: Type of plot. One of:
            'mds' — multi-dimensional scaling of samples,
            'bcv' — biological coefficient of variation vs abundance,
            'ql_dispersion' — quasi-likelihood dispersion plot,
            'md' — mean-difference (MA) plot for a test result,
            'volcano' — volcano plot (logFC vs -log10 p-value),
            'heatmap' — heatmap of top DE genes.
        result_name: Which test result to plot (for md/volcano/heatmap).
            Default: most recent result.
        output_path: Path to save the PNG. Default: auto-generated
            in the current working directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_types = {"mds", "bcv", "ql_dispersion", "md", "volcano", "heatmap"}
    if plot_type not in valid_types:
        return f"Invalid plot_type '{plot_type}'. Choose from: {', '.join(sorted(valid_types))}"

    if output_path is None:
        suffix = f"_{result_name}" if result_name else ""
        output_path = os.path.join(os.getcwd(), f"{plot_type}{suffix}.png")

    d = _state["dgelist"]

    if plot_type == "mds":
        _require("dgelist", "DGEList")
        fig, ax = ep.plot_mds(d)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"MDS plot saved to: {output_path}"

    elif plot_type == "bcv":
        _require("dgelist", "DGEList")
        if not _state["dispersions_estimated"]:
            return "Dispersions not estimated yet. Run estimate_dispersion() first."
        fig, ax = ep.plot_bcv(d)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"BCV plot saved to: {output_path}"

    elif plot_type == "ql_dispersion":
        _require("fit", "fitted model")
        fig, ax = ep.plot_ql_disp(_state["fit"])
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"QL dispersion plot saved to: {output_path}"

    elif plot_type == "md":
        if result_name is None:
            result_name = _state["last_result"]
        if result_name is None or result_name not in _state["results"]:
            return "No test result available. Run test_contrast() first."
        res = _state["results"][result_name]
        status = ep.decide_tests(res)
        fig, ax = ep.plot_md(res, status=status)
        ax.set_title(f"MD plot: {result_name}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"MD plot saved to: {output_path}"

    elif plot_type == "volcano":
        if result_name is None:
            result_name = _state["last_result"]
        if result_name is None or result_name not in _state["results"]:
            return "No test result available. Run test_contrast() first."
        res = _state["results"][result_name]
        tt = ep.top_tags(res, n=res["table"].shape[0])
        table = tt["table"]

        fig, ax = plt.subplots(figsize=(8, 6))
        logfc = table["logFC"].values
        pval = table["PValue"].values
        neg_log_p = -np.log10(np.clip(pval, 1e-300, 1))
        fdr = table["FDR"].values

        # Color by significance
        colors = np.where(
            (fdr < 0.05) & (logfc > 0), "red",
            np.where((fdr < 0.05) & (logfc < 0), "blue", "grey")
        )
        ax.scatter(logfc, neg_log_p, c=colors, s=4, alpha=0.5)
        ax.set_xlabel("log2 fold change")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(f"Volcano: {result_name}")
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.5)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"Volcano plot saved to: {output_path}"

    elif plot_type == "heatmap":
        if result_name is None:
            result_name = _state["last_result"]
        if result_name is None or result_name not in _state["results"]:
            return "No test result available. Run test_contrast() first."
        res = _state["results"][result_name]
        tt = ep.top_tags(res, n=30)
        table = tt["table"]

        # Get log-CPM for top genes
        logcpm = ep.cpm(d, log=True)

        # The table from top_tags has gene annotation columns (e.g. GeneID)
        # plus stat columns. The original row positions in the result table
        # correspond to gene indices in the count matrix.
        # Use the original table's row ordering from the test result.
        all_tt = ep.top_tags(res, n=res["table"].shape[0])
        all_genes = [_gene_name(row) for _, row in all_tt["table"].iterrows()]
        top_genes = [_gene_name(row) for _, row in table.iterrows()]

        gene_ids = d.get("_gene_ids", [])
        idx = []
        gene_labels = []
        for gn in top_genes:
            if gn in gene_ids:
                idx.append(gene_ids.index(gn))
                gene_labels.append(gn)
        if not idx:
            # Fallback: use row positions from the sorted result
            idx = list(range(min(30, logcpm.shape[0])))
            gene_labels = [str(i) for i in idx]

        mat = logcpm[idx, :]
        # Z-score per gene
        mat_z = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-10)

        fig, ax = plt.subplots(figsize=(max(6, len(d.get("_sample_names", [])) * 0.6), max(6, len(idx) * 0.3)))
        im = ax.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        sample_names = d.get("_sample_names", [f"S{i+1}" for i in range(mat.shape[1])])
        ax.set_xticks(range(len(sample_names)))
        ax.set_xticklabels(sample_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(gene_labels)))
        ax.set_yticklabels(gene_labels, fontsize=7)
        ax.set_title(f"Top DE genes: {result_name}")
        fig.colorbar(im, ax=ax, label="z-score (log-CPM)", shrink=0.6)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"Heatmap saved to: {output_path}"


# ===========================================================================
# Single-cell tools
# ===========================================================================

_sc_state: dict = {
    "counts": None,       # genes × cells ndarray
    "design": None,       # cells × predictors DataFrame
    "sample": None,       # per-cell sample IDs (array)
    "gene_names": None,   # gene name array
    "fit": None,          # glm_sc_fit result dict
    "n_cells": 0,
    "n_genes": 0,
    "n_samples": 0,
}


# ---------------------------------------------------------------------------
# SC Tool 1: load_sc_data
# ---------------------------------------------------------------------------

@mcp.tool()
def load_sc_data(
    h5ad_path: str,
    sample_col: str,
    cell_type_col: Optional[str] = None,
    cell_type_filter: Optional[list] = None,
    layer: str = "raw",
    log1p_transform: bool = True,
) -> str:
    """Load single-cell count data from an h5ad file.

    Reads counts, cell metadata, and gene names. Optionally subsets
    to specific cell types.

    Args:
        h5ad_path: Path to .h5ad file.
        sample_col: Column in obs for sample/subject IDs (e.g. 'orgID').
        cell_type_col: Optional column in obs for cell type annotations.
        cell_type_filter: Optional list of cell type values to keep.
            Only used if cell_type_col is provided.
        layer: Which count layer to use — 'raw' (from adata.raw.X),
            'X' (from adata.X), or a named layer. Default: 'raw'.
        log1p_transform: If True, reverse log1p transformation on
            the counts (i.e. round(expm1(x))). Default: True.
    """
    import h5py
    import scipy.sparse as sp

    h5 = h5py.File(h5ad_path, "r")

    # --- Read counts ---
    if layer == "raw" and "raw" in h5:
        x_grp = h5["raw"]["X"]
        if isinstance(x_grp, h5py.Dataset):
            # Dense
            data = x_grp[:]
            if log1p_transform:
                data = np.round(np.expm1(data)).astype(np.float64)
            counts_csr = None
            counts_dense = data
        else:
            # Sparse
            raw_data = h5["raw"]["X"]["data"][:]
            if log1p_transform:
                raw_data = np.round(np.expm1(raw_data)).astype(np.float64)
            else:
                raw_data = raw_data.astype(np.float64)
            indices = h5["raw"]["X"]["indices"][:]
            indptr = h5["raw"]["X"]["indptr"][:]
            n_obs = len(indptr) - 1
            # Gene names from raw
            var_key = "index" if "index" in h5["raw"]["var"] else "_index"
            gene_names = np.array([g.decode() if isinstance(g, bytes) else g
                                   for g in h5["raw"]["var"][var_key][:]])
            n_var = len(gene_names)
            counts_csr = sp.csr_matrix((raw_data, indices, indptr),
                                       shape=(n_obs, n_var))
            counts_dense = None
    else:
        x_grp = h5["X"]
        if isinstance(x_grp, h5py.Dataset):
            data = x_grp[:]
            if log1p_transform:
                data = np.round(np.expm1(data)).astype(np.float64)
            counts_csr = None
            counts_dense = data
        else:
            raw_data = h5["X"]["data"][:]
            if log1p_transform:
                raw_data = np.round(np.expm1(raw_data)).astype(np.float64)
            else:
                raw_data = raw_data.astype(np.float64)
            indices = h5["X"]["indices"][:]
            indptr = h5["X"]["indptr"][:]
            n_obs = len(indptr) - 1
            var_key = "index" if "index" in h5["var"] else "_index"
            gene_names = np.array([g.decode() if isinstance(g, bytes) else g
                                   for g in h5["var"][var_key][:]])
            n_var = len(gene_names)
            counts_csr = sp.csr_matrix((raw_data, indices, indptr),
                                       shape=(n_obs, n_var))
            counts_dense = None

    # Gene names (if not set from raw)
    if layer == "raw" and "raw" in h5:
        var_key = "index" if "index" in h5["raw"]["var"] else "_index"
        gene_names = np.array([g.decode() if isinstance(g, bytes) else g
                               for g in h5["raw"]["var"][var_key][:]])
    elif "gene_names" not in dir() or gene_names is None:
        var_key = "index" if "index" in h5["var"] else "_index"
        gene_names = np.array([g.decode() if isinstance(g, bytes) else g
                               for g in h5["var"][var_key][:]])

    # --- Read obs metadata ---
    obs_keys = [k for k in h5["obs"].keys() if k != "__categories"]
    categories = {}
    if "__categories" in h5["obs"]:
        for cat_name in h5["obs"]["__categories"]:
            raw_cats = h5["obs"]["__categories"][cat_name][:]
            categories[cat_name] = [c.decode() if isinstance(c, bytes) else c
                                    for c in raw_cats]

    def _decode_col(name):
        codes = h5["obs"][name][:]
        if name in categories:
            return np.array([categories[name][c] for c in codes])
        if codes.dtype.kind in ("S", "O"):
            return np.array([c.decode() if isinstance(c, bytes) else c
                             for c in codes])
        return codes

    sample_ids = _decode_col(sample_col)

    # --- Cell type filtering ---
    if cell_type_col is not None and cell_type_filter is not None:
        ct_vals = _decode_col(cell_type_col)
        mask = np.isin(ct_vals, cell_type_filter)
        if counts_csr is not None:
            counts_csr = counts_csr[mask, :]
        else:
            counts_dense = counts_dense[mask, :]
        sample_ids = sample_ids[mask]
        n_obs = int(mask.sum())
        ct_counts = {ct: int((ct_vals[mask] == ct).sum()) for ct in cell_type_filter}
    else:
        if counts_csr is not None:
            n_obs = counts_csr.shape[0]
        else:
            n_obs = counts_dense.shape[0]
        ct_counts = None

    h5.close()

    # --- Convert to genes × cells ---
    if counts_csr is not None:
        counts_T = counts_csr.T.toarray().astype(np.float64)
    else:
        counts_T = counts_dense.T.astype(np.float64)

    n_genes = counts_T.shape[0]
    n_samples = len(set(sample_ids))

    _sc_state["counts"] = counts_T
    _sc_state["sample"] = sample_ids
    _sc_state["gene_names"] = gene_names
    _sc_state["design"] = None
    _sc_state["fit"] = None
    _sc_state["n_cells"] = n_obs
    _sc_state["n_genes"] = n_genes
    _sc_state["n_samples"] = n_samples

    lines = [
        f"Loaded {n_genes:,} genes × {n_obs:,} cells, {n_samples} samples",
    ]
    if ct_counts is not None:
        for ct, n in ct_counts.items():
            lines.append(f"  {ct}: {n:,} cells")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SC Tool 2: set_sc_design
# ---------------------------------------------------------------------------

@mcp.tool()
def set_sc_design(
    covariates: dict,
) -> str:
    """Set the design matrix for single-cell analysis.

    Args:
        covariates: Dictionary mapping column names to per-cell values.
            Must include 'Intercept' (all ones). Values can be lists or
            arrays of length n_cells. Example:
            {"Intercept": [1,1,...], "fed": [1,0,1,...]}
    """
    n = _sc_state["n_cells"]
    if n == 0:
        raise ValueError("No SC data loaded. Use load_sc_data() first.")

    cols = list(covariates.keys())
    arrays = {}
    for col in cols:
        arr = np.asarray(covariates[col], dtype=np.float64)
        if len(arr) != n:
            raise ValueError(
                f"Column '{col}' has {len(arr)} values but expected {n} cells."
            )
        arrays[col] = arr

    design = pd.DataFrame(arrays, columns=cols)
    _sc_state["design"] = design
    _sc_state["fit"] = None

    return (
        f"Design matrix set: {n:,} cells × {len(cols)} predictors\n"
        f"Columns: {', '.join(cols)}"
    )


# ---------------------------------------------------------------------------
# SC Tool 3: fit_sc_model
# ---------------------------------------------------------------------------

@mcp.tool()
def fit_sc_model(
    norm_method: str = "TMM",
    verbose: bool = True,
) -> str:
    """Fit NEBULA-LN single-cell mixed model.

    Fits a per-gene negative binomial gamma mixed model with a
    sample-level random intercept. Requires load_sc_data() and
    set_sc_design() to have been called first.

    Args:
        norm_method: Normalization method — 'TMM' (default) or 'none'.
        verbose: Print progress messages.
    """
    counts = _sc_state["counts"]
    if counts is None:
        raise ValueError("No SC data loaded. Use load_sc_data() first.")
    design = _sc_state["design"]
    if design is None:
        raise ValueError("No design matrix set. Use set_sc_design() first.")
    sample = _sc_state["sample"]
    gene_names = _sc_state["gene_names"]

    import time
    t0 = time.perf_counter()

    fit = ep.glm_sc_fit(
        counts,
        design=design,
        sample=sample,
        norm_method=norm_method,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    # Attach gene names
    gene_mask = fit["gene_mask"]
    if fit["genes"] is None and gene_names is not None:
        fit["genes"] = pd.DataFrame({"gene": gene_names[gene_mask]})

    _sc_state["fit"] = fit

    n_tested = fit["coefficients"].shape[0]
    n_conv = int((fit["convergence"] == 1).sum())

    return (
        f"Fitted NEBULA-LN: {n_tested:,} genes in {elapsed:.1f}s\n"
        f"Converged: {n_conv:,}/{n_tested:,}\n"
        f"Predictors: {', '.join(fit['predictor_names'])}"
    )


# ---------------------------------------------------------------------------
# SC Tool 4: test_sc_coef
# ---------------------------------------------------------------------------

@mcp.tool()
def test_sc_coef(
    coef: Optional[str] = None,
    coef_index: Optional[int] = None,
    name: Optional[str] = None,
) -> str:
    """Run a Wald test on a coefficient from the single-cell model.

    Args:
        coef: Coefficient name to test (e.g. 'fed'). Default: last coefficient.
        coef_index: 0-based coefficient index. Alternative to coef.
        name: Label for this result. Default: auto-generated.
    """
    fit = _sc_state["fit"]
    if fit is None:
        raise ValueError("No SC model fitted. Use fit_sc_model() first.")

    if coef is not None:
        c = coef
    elif coef_index is not None:
        c = coef_index
    else:
        c = None  # top_tags defaults to last coefficient

    tt = ep.top_tags(fit, n=fit["coefficients"].shape[0], coef=c)
    table = tt["table"]

    fdr = table["FDR"].values
    logfc = table["logFC"].values
    valid = ~np.isnan(fdr)

    n_up = int(((fdr < 0.05) & (logfc > 0) & valid).sum())
    n_down = int(((fdr < 0.05) & (logfc < 0) & valid).sum())
    n_sig_01 = int(((fdr < 0.1) & valid).sum())

    if name is None:
        name = coef if coef is not None else f"coef{coef_index}"
    _state["results"][name] = {"table": table, "method": "nebula_ln"}
    _state["last_result"] = name

    label = coef if coef is not None else f"coefficient {coef_index}"
    lines = [
        f"Wald test: {label}",
        f"DE genes (FDR < 0.05): {n_up} up, {n_down} down ({n_up + n_down} total)",
        f"DE genes (FDR < 0.10): {n_sig_01}",
        "",
        "Top 10 genes:",
        f"{'Gene':<30} {'logFC':>8} {'SE':>8} {'PValue':>10} {'FDR':>10}",
        "-" * 70,
    ]
    for _, row in table.head(10).iterrows():
        gene = str(row.get("gene", row.name))
        if len(gene) > 29:
            gene = gene[:27] + ".."
        lines.append(
            f"{gene:<30} {row['logFC']:>8.3f} {row['SE']:>8.3f} "
            f"{row['PValue']:>10.2e} {row['FDR']:>10.2e}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SC Tool 5: get_sc_top_genes
# ---------------------------------------------------------------------------

@mcp.tool()
def get_sc_top_genes(
    coef: Optional[str] = None,
    coef_index: Optional[int] = None,
    n: int = 20,
    fdr_threshold: float = 0.05,
) -> str:
    """Get top DE genes from the single-cell model.

    Args:
        coef: Coefficient name to test (e.g. 'fed'). Default: last coefficient.
        coef_index: 0-based coefficient index. Alternative to coef.
        n: Number of top genes to return. Default: 20.
        fdr_threshold: FDR cutoff for marking significance. Default: 0.05.
    """
    fit = _sc_state["fit"]
    if fit is None:
        raise ValueError("No SC model fitted. Use fit_sc_model() first.")

    c = coef if coef is not None else coef_index

    tt = ep.top_tags(fit, n=n, coef=c)
    table = tt["table"]

    label = coef if coef is not None else (
        f"coefficient {coef_index}" if coef_index is not None else "last coefficient"
    )
    lines = [
        f"Top {min(n, len(table))} genes for {label}:",
        "",
        f"{'Gene':<30} {'logFC':>8} {'SE':>8} {'z':>8} {'PValue':>10} {'FDR':>10} {'Sig':>3}",
        "-" * 82,
    ]
    for _, row in table.iterrows():
        gene = str(row.get("gene", row.name))
        if len(gene) > 29:
            gene = gene[:27] + ".."
        fdr_val = row["FDR"]
        sig = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else \
              "*" if fdr_val < fdr_threshold else ""
        lines.append(
            f"{gene:<30} {row['logFC']:>8.3f} {row['SE']:>8.3f} "
            f"{row['z']:>8.2f} {row['PValue']:>10.2e} {fdr_val:>10.2e} {sig:>3}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SC Tool 6: describe_sc
# ---------------------------------------------------------------------------

@mcp.tool()
def describe_sc() -> str:
    """Report the current state of the single-cell analysis pipeline."""
    lines = []
    if _sc_state["counts"] is None:
        return "No single-cell data loaded. Use load_sc_data() first."

    lines.append(
        f"SC data: {_sc_state['n_genes']:,} genes × "
        f"{_sc_state['n_cells']:,} cells, "
        f"{_sc_state['n_samples']} samples"
    )

    if _sc_state["design"] is not None:
        cols = list(_sc_state["design"].columns)
        lines.append(f"Design: {', '.join(cols)}")
    else:
        lines.append("Design: not set")

    fit = _sc_state["fit"]
    if fit is not None:
        n_tested = fit["coefficients"].shape[0]
        n_conv = int((fit["convergence"] == 1).sum())
        lines.append(
            f"Model: NEBULA-LN, {n_tested:,} genes tested, "
            f"{n_conv:,} converged"
        )
        lines.append(f"Predictors: {', '.join(fit['predictor_names'])}")
    else:
        lines.append("Model: not fitted")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SC Tool 7: save_sc_results
# ---------------------------------------------------------------------------

@mcp.tool()
def save_sc_results(
    output_path: str,
    coef: Optional[str] = None,
    coef_index: Optional[int] = None,
    format: str = "csv",
) -> str:
    """Save full single-cell DE results to disk.

    Args:
        output_path: Destination file path.
        coef: Coefficient name to test and save.
        coef_index: 0-based coefficient index. Alternative to coef.
        format: 'csv' (default), 'tsv', or 'json'.
    """
    fit = _sc_state["fit"]
    if fit is None:
        raise ValueError("No SC model fitted. Use fit_sc_model() first.")

    c = coef if coef is not None else coef_index
    tt = ep.top_tags(fit, n=fit["coefficients"].shape[0], coef=c)
    table = tt["table"]

    if format == "json":
        table.to_json(output_path, orient="records")
    elif format == "tsv":
        table.to_csv(output_path, sep="\t")
    else:
        table.to_csv(output_path)

    valid_fdr = table["FDR"].dropna()
    n_sig = int((valid_fdr < 0.05).sum())

    return (
        f"Saved {len(table):,} genes to {output_path}\n"
        f"DE at FDR < 0.05: {n_sig}"
    )


# ---------------------------------------------------------------------------
# SC Tool 8: reset_sc_state
# ---------------------------------------------------------------------------

@mcp.tool()
def reset_sc_state() -> str:
    """Reset all single-cell pipeline state."""
    _sc_state["counts"] = None
    _sc_state["design"] = None
    _sc_state["sample"] = None
    _sc_state["gene_names"] = None
    _sc_state["fit"] = None
    _sc_state["n_cells"] = 0
    _sc_state["n_genes"] = 0
    _sc_state["n_samples"] = 0
    return "Single-cell state cleared."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
