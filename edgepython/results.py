# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Results processing for edgePython.

Port of edgeR's topTags and decideTests.
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
    if sort_by == 'PValue':
        alfc = np.abs(tab['logFC'].values) if 'logFC' in tab.columns else np.zeros(len(tab))
        o = np.lexsort((-alfc, raw_pvals))
    elif sort_by == 'logFC':
        alfc = np.abs(tab['logFC'].values) if 'logFC' in tab.columns else np.zeros(len(tab))
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

    # Get logFC
    logFC = obj['table'].get('logFC')
    f_test = logFC is None

    if f_test:
        # F-test with multiple logFC columns
        if lfc > 0:
            coef_cols = [c for c in obj['table'].columns if c.startswith('logFC')]
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
