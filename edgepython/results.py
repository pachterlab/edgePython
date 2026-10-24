# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Results processing for edgePython.

Port of edgeR's topTags and decideTests, and Lancaster aggregation of
transcript p-values to genes (Yi et al. 2018).
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2 as _chi2


def _build_sc_table(obj, coef):
    """Build a Wald-test table from a glm_sc_fit result."""
    coefficients = obj['coefficients']
    se_arr = obj['se']
    nb = coefficients.shape[1]

    if coef is None:
        coef = nb - 1
    if isinstance(coef, str):
        predn = obj.get('predictor_names', [])
        if coef in predn:
            coef = predn.index(coef)
        else:
            raise ValueError(f"Coefficient '{coef}' not found in predictor names: {predn}")

    logFC = coefficients[:, coef]
    se = se_arr[:, coef]
    z = logFC / se
    pvalue = _chi2.sf(z ** 2, 1)

    table = pd.DataFrame({
        'logFC': logFC,
        'SE': se,
        'z': z,
        'PValue': pvalue,
    })
    return table


def top_tags(obj, n=10, adjust_method='BH', sort_by='PValue', p_value=1.0,
             coef=None):
    """Summary table of the top differentially expressed genes.

    Port of edgeR's topTags.

    Parameters
    ----------
    obj : dict (DGEExact, DGELRT, or glm_sc_fit result)
        Result from exact_test(), glm_lrt(), glm_ql_ftest(), glm_treat(),
        or glm_sc_fit().
    n : int
        Number of top genes to return.
    adjust_method : str
        Multiple testing adjustment method.
    sort_by : str
        'PValue', 'logFC', or 'none'.
    p_value : float
        P-value cutoff for filtering.
    coef : int or str, optional
        Coefficient to test. Required for glm_sc_fit results (default:
        last coefficient). Ignored for objects that already have a table.

    Returns
    -------
    dict (TopTags-like) with 'table', 'adjust.method', 'comparison', 'test'.
    """
    # Handle single-cell fits (NEBULA-LN): build Wald test table
    if obj.get('method') == 'nebula_ln' and obj.get('table') is None:
        tab = _build_sc_table(obj, coef)
        # Work on a copy with the table injected so we don't mutate the fit
        obj = dict(obj, table=tab)

    if obj.get('table') is None:
        raise ValueError("Need to run exact_test, glm_lrt, or glm_ql_ftest first")

    tab = obj['table'].copy()

    # Determine test type
    if obj.get('method') == 'nebula_ln':
        test = 'wald'
    elif obj.get('comparison') is not None and isinstance(obj['comparison'], list):
        test = 'exact'
    else:
        test = 'glm'

    n = min(n, len(tab))
    if n < 1:
        raise ValueError("No rows to output")

    # Validate adjust_method
    valid_fwer = ['holm', 'hochberg', 'hommel', 'bonferroni']
    valid_fdr = ['BH', 'BY', 'fdr']
    valid_methods = valid_fwer + valid_fdr + ['none']

    if adjust_method == 'fdr':
        adjust_method = 'BH'

    if adjust_method not in valid_methods:
        raise ValueError(f"adjust_method must be one of {valid_methods}")

    # Validate sort_by
    if sort_by == 'p.value':
        sort_by = 'PValue'

    # Compute adjusted p-values (handle NaN p-values)
    raw_pvals = tab['PValue'].values
    if adjust_method != 'none':
        # Map to statsmodels method names
        method_map = {
            'BH': 'fdr_bh', 'BY': 'fdr_by', 'fdr': 'fdr_bh',
            'holm': 'holm', 'hochberg': 'hochberg',
            'hommel': 'hommel', 'bonferroni': 'bonferroni'
        }
        sm_method = method_map.get(adjust_method, adjust_method)
        valid_mask = ~np.isnan(raw_pvals)
        if valid_mask.all():
            _, adj_pvals, _, _ = multipletests(raw_pvals, method=sm_method)
        elif valid_mask.any():
            adj_pvals = np.full_like(raw_pvals, np.nan)
            _, adj_pvals[valid_mask], _, _ = multipletests(
                raw_pvals[valid_mask], method=sm_method)
        else:
            adj_pvals = np.full_like(raw_pvals, np.nan)

        if adjust_method in valid_fwer:
            adj_name = 'FWER'
        else:
            adj_name = 'FDR'
    else:
        adj_pvals = raw_pvals
        adj_name = None

    # Sort
    # Binomial tests (bin_ql_ftest) report log2 odds ratios as logOR
    lfc_col = 'logFC' if 'logFC' in tab.columns else 'logOR'
    if sort_by == 'PValue':
        alfc = np.abs(tab[lfc_col].values) if lfc_col in tab.columns else np.zeros(len(tab))
        o = np.lexsort((-alfc, raw_pvals))
    elif sort_by == 'logFC':
        alfc = np.abs(tab[lfc_col].values) if lfc_col in tab.columns else np.zeros(len(tab))
        o = np.argsort(-alfc)
    else:
        o = np.arange(len(tab))

    tab = tab.iloc[o].copy()

    if adj_name is not None:
        tab[adj_name] = adj_pvals[o]

    # Add gene annotation — preserve original row indices
    genes = obj.get('genes')
    if genes is not None:
        if isinstance(genes, pd.DataFrame):
            orig_idx = tab.index.copy()
            gene_info = genes.iloc[o].reset_index(drop=True)
            tab = pd.concat([gene_info, tab.reset_index(drop=True)], axis=1)
            tab.index = orig_idx

    # Filter by p-value
    if p_value < 1:
        sig = adj_pvals[o] <= p_value
        tab = tab[sig]

    n = min(n, len(tab))
    if n < 1:
        return pd.DataFrame()

    tab = tab.iloc[:n]

    return {
        'table': tab,
        'adjust.method': adjust_method,
        'comparison': obj.get('comparison', ''),
        'test': test
    }


def decide_tests(obj, adjust_method='BH', p_value=0.05, lfc=0):
    """Classify genes as up, down, or not significant.

    Port of edgeR's decideTests.DGEExact / decideTests.DGELRT.

    Parameters
    ----------
    obj : dict (DGEExact or DGELRT-like)
        Result from exact_test() or glm_lrt/glm_ql_ftest/glm_treat().
    adjust_method : str
        Multiple testing adjustment method.
    p_value : float
        Significance threshold.
    lfc : float
        Log-fold-change threshold.

    Returns
    -------
    ndarray of int (-1, 0, 1) indicating direction for each gene.
    """
    # Get raw p-values
    raw_p = obj['table']['PValue'].values

    # Adjust p-values
    if adjust_method != 'none':
        method_map = {
            'BH': 'fdr_bh', 'BY': 'fdr_by', 'fdr': 'fdr_bh',
            'holm': 'holm', 'hochberg': 'hochberg',
            'hommel': 'hommel', 'bonferroni': 'bonferroni'
        }
        sm_method = method_map.get(adjust_method, adjust_method)
        _, adj_p, _, _ = multipletests(raw_p, method=sm_method)
    else:
        adj_p = raw_p

    is_de = (adj_p < p_value).astype(int)

    # Get logFC (logOR for binomial tests, which have no dispersion)
    prefix = 'logOR' if obj.get('dispersion') is None and any(
        c.startswith('logOR') for c in obj['table'].columns) else 'logFC'
    logFC = obj['table'].get(prefix)
    f_test = logFC is None

    if f_test:
        # F-test with multiple logFC columns
        if lfc > 0:
            coef_cols = [c for c in obj['table'].columns if c.startswith(prefix)]
            if coef_cols:
                logFC_mat = obj['table'][coef_cols].values
                small_fc = np.all(np.abs(logFC_mat) < lfc, axis=1)
                is_de[small_fc] = 0
    else:
        logFC = logFC.values if hasattr(logFC, 'values') else np.asarray(logFC)
        # Apply directionality
        is_de[is_de.astype(bool) & (logFC < 0)] = -1
        # Apply lfc threshold
        small_fc = np.abs(logFC) < lfc
        is_de[small_fc] = 0

    return is_de


# =====================================================================
# Transcript-to-gene p-value aggregation
# =====================================================================

def lancaster(pvalues, weights):
    """Lancaster's weighted combination of p-values.

    Port of lancaster() from the aggregation R package (Yi et al. 2018).
    Each p-value p_i with weight w_i contributes the upper chi-squared
    quantile with w_i degrees of freedom; their sum is referred to the
    chi-squared distribution with sum(w_i) degrees of freedom. Weights of 2
    give Fisher's method.

    Parameters
    ----------
    pvalues, weights : array-like
        P-values and their weights, of the same length. NaN p-values and
        weights <= 0 are dropped; a NaN weight gives NaN.

    Returns
    -------
    float
        The combined p-value: NaN if no p-values remain, the p-value
        itself if one remains.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if p.shape != w.shape:
        raise ValueError("Length of weights not equal to length of pvalues")
    out = _lancaster_by(p, w, np.zeros(p.size, dtype=np.intp), 1)
    return float(out[0])


def _lancaster_by(p, w, codes, ngroups):
    """lancaster() within each group of codes (0..ngroups-1)."""
    keep = ~np.isnan(p) & ((w > 0) | np.isnan(w))
    c, pk, wk = codes[keep], p[keep], w[keep]
    n = np.bincount(c, minlength=ngroups)
    t = np.bincount(c, weights=_chi2.isf(pk, wk), minlength=ngroups)
    df = np.bincount(c, weights=wk, minlength=ngroups)
    with np.errstate(invalid="ignore"):
        out = _chi2.sf(t, df)
    single = n == 1
    out[single] = np.bincount(c, weights=pk, minlength=ngroups)[single]
    out[n == 0] = np.nan
    return out


_GENE_COLUMNS = ("EnsG", "GeneID", "gene_id", "gene")


def aggregate_tx_pvalues(res, y=None, gene=None, tx2gene=None, weights=None,
                         remove_version=True, adjust_method="BH"):
    """Gene-level p-values from transcript-level tests, by Lancaster's method.

    An edgePython addition, following Yi et al. (2018): test transcripts
    (e.g. from catch_kallisto, with exact_test or glm_ql_ftest), then
    combine each gene's transcript p-values with lancaster().

    Parameters
    ----------
    res : dict
        Transcript-level test result with a 'table' holding PValue (and,
        optionally, logFC), and 'genes' annotation, as from exact_test,
        glm_ql_ftest or glm_lrt.
    y : DGEList, optional
        The DGEList that was tested, for the default weights.
    gene : str or array-like, optional
        Column of res['genes'] holding gene IDs, or the gene ID of each
        transcript. By default the first of EnsG (unpacked Gencode names),
        GeneID, gene_id and gene that res['genes'] has.
    tx2gene : DataFrame or str, optional
        Transcript-to-gene table, or path to one such as kb ref's t2g.txt,
        matched to the transcript IDs (the row names of res['genes']), as
        in catch_kallisto_gene. Used in place of gene.
    weights : array-like or 'equal', optional
        Weight of each transcript. By default sleuth's: each transcript's
        mean over samples of log(normalized count + 0.5), the counts of y
        scaled to the geometric mean of the effective library sizes;
        transcripts with weight <= 0 are left out. 'equal' gives every
        transcript weight 2, which is Fisher's method.
    remove_version : bool
        For tx2gene: remove version numbers from transcript and gene IDs.
    adjust_method : str
        Multiple testing adjustment over genes, as for top_tags.

    Returns
    -------
    DataFrame, one row per gene in order of first appearance and indexed
    by gene ID, with GeneID, GeneName (when known), NTx (transcripts
    combined), PValue, FDR, and the transcript with the smallest p-value
    (TopTx, with its TopTx.logFC when res has logFC).
    """
    table = res["table"]
    p = table["PValue"].to_numpy(dtype=np.float64)
    ntx = len(p)
    ann = res.get("genes")
    if ann is not None and len(ann) != ntx:
        ann = None
    if ann is not None and not isinstance(ann.index, pd.RangeIndex):
        tx_ids = np.asarray(ann.index.astype(str), dtype=object)
    else:
        tx_ids = np.asarray(table.index.astype(str), dtype=object)

    gene_names = None
    if tx2gene is not None:
        from .quant_readers import _gene_map
        gene_ids, gene_ann = _gene_map(tx_ids, tx2gene, remove_version, verbose=False)
        if gene_ann is not None and "GeneName" in gene_ann:
            gene_names = gene_ann["GeneName"]
    elif gene is not None and not isinstance(gene, str):
        gene_ids = np.asarray(gene, dtype=object)
        if len(gene_ids) != ntx:
            raise ValueError(f"gene has {len(gene_ids)} values for {ntx} transcripts")
    else:
        if gene is None:
            gene = next((c for c in _GENE_COLUMNS if ann is not None and c in ann), None)
            if gene is None:
                raise ValueError("No gene IDs in res['genes']; give gene= or tx2gene=")
        elif ann is None or gene not in ann:
            raise ValueError(f"Column '{gene}' not found in res['genes']")
        gene_ids = ann[gene].astype(str).to_numpy(dtype=object)
    if gene_names is None and ann is not None and "GeneName" in ann:
        gene_names = pd.Series(ann["GeneName"].to_numpy(), index=gene_ids)
        gene_names = gene_names[~gene_names.index.duplicated()]

    if isinstance(weights, str):
        if weights != "equal":
            raise ValueError("weights must be an array or 'equal'")
        w = np.full(ntx, 2.0)
    elif weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != ntx:
            raise ValueError(f"weights has {len(w)} values for {ntx} transcripts")
    else:
        if y is None:
            raise ValueError("Give y, the DGEList that was tested, for the default "
                             "weights, or weights=")
        counts = np.asarray(y["counts"], dtype=np.float64)
        if counts.shape[0] != ntx:
            raise ValueError(f"y has {counts.shape[0]} rows for {ntx} transcripts in res")
        lib = (y["samples"]["lib.size"] * y["samples"]["norm.factors"]).to_numpy(dtype=np.float64)
        scaled = counts * (np.exp(np.mean(np.log(lib))) / lib)
        w = np.mean(np.log(scaled + 0.5), axis=1)

    codes, genes = pd.factorize(pd.Series(gene_ids), sort=False)
    ngene = len(genes)
    pg = _lancaster_by(p, w, codes, ngene)
    n_used = np.bincount(codes[~np.isnan(p) & ((w > 0) | np.isnan(w))], minlength=ngene)

    fdr = np.full(ngene, np.nan)
    ok = ~np.isnan(pg)
    if ok.any():
        method_map = {'BH': 'fdr_bh', 'BY': 'fdr_by', 'fdr': 'fdr_bh',
                      'holm': 'holm', 'hochberg': 'hochberg',
                      'hommel': 'hommel', 'bonferroni': 'bonferroni'}
        if adjust_method == 'none':
            fdr[ok] = pg[ok]
        else:
            _, fdr[ok], _, _ = multipletests(pg[ok], method=method_map.get(adjust_method,
                                                                           adjust_method))

    out = pd.DataFrame({"GeneID": np.asarray(genes, dtype=object)},
                       index=pd.Index(np.asarray(genes, dtype=object)))
    if gene_names is not None:
        out["GeneName"] = gene_names.reindex(out.index).to_numpy()
    out["NTx"] = n_used
    out["PValue"] = pg
    out["FDR"] = fdr
    # Most significant transcript of each gene (first on ties)
    order = np.lexsort((np.arange(ntx), np.where(np.isnan(p), np.inf, p), codes))
    first = order[np.r_[True, codes[order][1:] != codes[order][:-1]]]
    out["TopTx"] = tx_ids[first]
    if "logFC" in table:
        out["TopTx.logFC"] = table["logFC"].to_numpy()[first]
    return out
