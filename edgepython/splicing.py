# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Splicing analysis for edgePython.

Port of edgeR's diffSplice, diffSpliceDGE, spliceVariants.
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, chi2
from statsmodels.stats.multitest import multipletests


def diff_splice(glmfit, coef=None, contrast=None, geneid=None, exonid=None,
                prior_count=0.125, robust=None, verbose=True):
    """Test for differential exon/transcript usage.

    Faithful port of edgeR's diffSpliceDGE. Tests whether the log-fold-change
    for each exon differs from the overall gene-level log-fold-change, i.e.
    tests for differential *usage* rather than differential expression.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted GLM from glm_fit() or glm_ql_fit().
    coef : int, optional
        Coefficient to test (0-indexed). Default is last column.
    contrast : ndarray, optional
        Contrast vector.
    geneid : ndarray or str, optional
        Gene IDs for each exon/transcript.
    exonid : ndarray or str, optional
        Exon/transcript IDs.
    prior_count : float
        Prior count for gene-level GLM fit.
    robust : bool or None
        Use robust empirical Bayes for squeezeVar. None = auto-detect.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with gene-level and exon-level test results including:
        - gene.table: DataFrame with GeneID, NExons, gene.F (or gene.LR),
          gene.p.value, gene.Simes.p.value
        - exon.table: DataFrame with GeneID, ExonID, logFC, exon.F (or exon.LR),
          exon.p.value
        - coefficients: exon-level coefficients relative to gene
        - design, comparison
    """
    from .glm_fit import glm_fit
    from .limma_port import squeeze_var, contrast_as_coef
    from .utils import expand_as_matrix

    # --- Detect LRT vs QL ---
    isLRT = glmfit.get('df.prior') is None
    if robust is None and not isLRT:
        df_prior = glmfit['df.prior']
        robust = hasattr(df_prior, '__len__') and len(np.atleast_1d(df_prior)) > 1

    # --- Get gene and exon IDs ---
    exon_genes = glmfit.get('genes')
    nexons = glmfit['counts'].shape[0]
    design = np.asarray(glmfit['design'], dtype=np.float64)

    if exon_genes is None:
        exon_genes = pd.DataFrame({'ExonID': np.arange(nexons)})
    else:
        exon_genes = exon_genes.copy()

    genecolname = 'GeneID'
    if geneid is None:
        if isinstance(exon_genes, pd.DataFrame):
            for col in ['GeneID', 'geneid', 'gene_id', 'Gene']:
                if col in exon_genes.columns:
                    geneid = exon_genes[col].values
                    genecolname = col
                    break
        if geneid is None:
            raise ValueError("geneid must be provided")
    elif isinstance(geneid, str):
        genecolname = geneid
        geneid = exon_genes[geneid].values
    else:
        exon_genes['GeneID'] = geneid
        genecolname = 'GeneID'

    exoncolname = None
    if exonid is not None:
        if isinstance(exonid, str):
            exoncolname = exonid
            exonid = exon_genes[exonid].values
        else:
            exon_genes['ExonID'] = exonid
            exoncolname = 'ExonID'
    else:
        exoncolname = None

    # --- Sort by geneid (+exonid) ---
    geneid = np.asarray(geneid)
    if exonid is not None:
        exonid = np.asarray(exonid)
        o = np.lexsort((exonid, geneid))
    else:
        o = np.argsort(geneid, kind='stable')

    geneid = geneid[o]
    exon_genes = exon_genes.iloc[o].reset_index(drop=True)

    # Subset glmfit arrays by o
    counts = glmfit['counts'][o]
    coefficients = glmfit['coefficients'][o]
    deviance = glmfit['deviance'][o]
    df_residual_orig = glmfit['df.residual'][o]

    # Handle offset: could be matrix or vector
    offset_full = glmfit['offset']
    if offset_full.ndim == 2:
        offset_full = offset_full[o]
    else:
        offset_full = expand_as_matrix(offset_full, (nexons, counts.shape[1]))
        offset_full = offset_full[o]

    weights = glmfit.get('weights')
    if weights is not None:
        if np.ndim(weights) == 2:
            weights = weights[o]
        else:
            weights = expand_as_matrix(weights, (nexons, counts.shape[1]))
            weights = weights[o]

    dispersion_orig = glmfit.get('dispersion')
    if dispersion_orig is not None:
        dispersion_orig = np.atleast_1d(dispersion_orig)
        if len(dispersion_orig) == nexons:
            dispersion_orig = dispersion_orig[o]

    nbeta = design.shape[1]
    if nbeta < 2:
        raise ValueError("Need at least two columns for design")
    coef_names = [f"x{i}" for i in range(nbeta)]

    # --- Handle contrast or coef ---
    if contrast is not None:
        contrast = np.asarray(contrast, dtype=np.float64)
        if contrast.ndim == 2:
            contrast = contrast[:, 0]
        reform = contrast_as_coef(design, contrast, first=True)
        coef_idx = 0
        beta = coefficients @ contrast
        i = contrast != 0
        coef_name = ' '.join(f"{contrast[j]}*{coef_names[j]}" for j in range(len(contrast)) if contrast[j] != 0)
        design = reform['design']
    else:
        if coef is None:
            coef_idx = nbeta - 1
        else:
            coef_idx = coef
        coef_name = coef_names[coef_idx]
        beta = coefficients[:, coef_idx]

    design0 = np.delete(design, coef_idx, axis=1)

    # --- Count exons per gene ---
    unique_genes, gene_idx = np.unique(geneid, return_inverse=True)
    # But we need reorder=False behavior (preserve order of first appearance)
    # R's rowsum(reorder=FALSE) preserves the order of geneid as encountered
    _, first_idx = np.unique(geneid, return_index=True)
    order_by_first = np.argsort(first_idx)
    unique_genes_ordered = unique_genes[order_by_first]
    # Remap gene_idx to match this ordering
    remap = np.empty(len(unique_genes), dtype=int)
    remap[order_by_first] = np.arange(len(unique_genes))
    g = remap[gene_idx]  # gene index for each exon (0-based, ordered by first appearance)

    gene_nexons = np.bincount(g)
    ngenes_total = len(unique_genes_ordered)

    if verbose:
        print(f"Total number of exons:  {nexons}")
        print(f"Total number of genes:  {ngenes_total}")
        print(f"Number of genes with 1 exon:  {np.sum(gene_nexons == 1)}")
        print(f"Mean number of exons in a gene:  {np.round(np.mean(gene_nexons)):.0f}")
        print(f"Max number of exons in a gene:  {np.max(gene_nexons)}")

    # --- Filter to genes with >1 exon ---
    gene_keep = gene_nexons > 1
    ngenes = int(np.sum(gene_keep))
    if ngenes == 0:
        raise ValueError("No genes with more than one exon")

    exon_keep = gene_keep[g]
    geneid = geneid[exon_keep]
    exon_genes = exon_genes.iloc[exon_keep].reset_index(drop=True)
    beta = beta[exon_keep]
    counts = counts[exon_keep]
    offset_full = offset_full[exon_keep]
    deviance = deviance[exon_keep]
    df_residual_orig = df_residual_orig[exon_keep]
    if weights is not None:
        weights = weights[exon_keep]
    if dispersion_orig is not None and len(dispersion_orig) > 1:
        dispersion_orig = dispersion_orig[exon_keep]
    coefficients_full = coefficients[exon_keep]

    gene_nexons = gene_nexons[gene_keep]
    unique_genes_ordered = unique_genes_ordered[gene_keep]
    # Rebuild g for kept exons
    g = np.repeat(np.arange(ngenes), gene_nexons)

    nlib = counts.shape[1]
    nexons_kept = counts.shape[0]

    # --- Gene-level counts and GLM fit ---
    gene_counts = np.zeros((ngenes, nlib), dtype=np.float64)
    np.add.at(gene_counts, g, counts)

    # Gene-level offset: use first row's offset (R uses offset[1,])
    gene_offset = offset_full[0, :]

    fit_gene = glm_fit(gene_counts, design, dispersion=0.05,
                       offset=gene_offset, prior_count=prior_count)

    # --- Gene-level betabar, expand to exon level ---
    gene_betabar = fit_gene['coefficients'][:, coef_idx:coef_idx+1]  # (ngenes, 1)
    gene_betabar_exon = gene_betabar[g]  # (nexons_kept, 1)

    # New offset = original offset + gene_betabar @ design[:,coef].T
    design_coef_col = design[:, coef_idx:coef_idx+1]  # (nlib, 1)
    offset_new = offset_full + gene_betabar_exon @ design_coef_col.T  # (nexons_kept, nlib)

    # --- Relative coefficients ---
    coefficients_rel = beta - gene_betabar_exon.ravel()

    # --- Dispersion for reduced model fit ---
    if glmfit.get('average.ql.dispersion') is not None:
        ave_ql_disp = glmfit['average.ql.dispersion']
        if dispersion_orig is not None:
            dispersion = dispersion_orig / ave_ql_disp
        else:
            dispersion = 0.05
    else:
        dispersion = dispersion_orig if dispersion_orig is not None else 0.05

    # --- Fit reduced model ---
    fit0 = glm_fit(counts, design=design0, offset=offset_new,
                   dispersion=dispersion, weights=weights, prior_count=0)

    # --- Deviance differences ---
    exon_LR = fit0['deviance'] - deviance
    gene_LR = np.zeros(ngenes)
    np.add.at(gene_LR, g, exon_LR)

    exon_df_test = fit0['df.residual'] - df_residual_orig
    gene_df_test = np.zeros(ngenes)
    np.add.at(gene_df_test, g, exon_df_test)

    # --- Get adjusted df/deviance for QL path ---
    if not isLRT:
        if glmfit.get('df.residual.zeros') is not None:
            exon_df_residual = glmfit['df.residual.zeros'][o][exon_keep]
            exon_deviance = glmfit['deviance'][o][exon_keep]
        elif glmfit.get('df.residual.adj') is not None:
            exon_df_residual = glmfit['df.residual.adj'][o][exon_keep]
            exon_deviance = glmfit['deviance.adj'][o][exon_keep]
        else:
            exon_df_residual = df_residual_orig
            exon_deviance = deviance

    # --- Statistical tests ---
    if isLRT:
        # Chi-squared tests
        exon_p_value = chi2.sf(exon_LR, df=exon_df_test)
        gene_p_value = chi2.sf(gene_LR, df=gene_df_test)
    else:
        # QL F-tests
        gene_df_residual = np.zeros(ngenes)
        np.add.at(gene_df_residual, g, exon_df_residual)

        gene_s2_num = np.zeros(ngenes)
        np.add.at(gene_s2_num, g, exon_deviance)
        gene_s2 = gene_s2_num / gene_df_residual

        squeeze = squeeze_var(gene_s2, gene_df_residual, robust=robust)
        gene_df_total = gene_df_residual + squeeze['df_prior']
        gene_df_total = np.minimum(gene_df_total, np.sum(gene_df_residual))
        gene_s2_post = squeeze['var_post']

        # Exon-level F and p-values
        exon_F = exon_LR / exon_df_test / gene_s2_post[g]
        gene_F = gene_LR / gene_df_test / gene_s2_post

        exon_p_value = f_dist.sf(exon_F, dfn=exon_df_test, dfd=gene_df_total[g])
        gene_p_value = f_dist.sf(gene_F, dfn=gene_df_test, dfd=gene_df_total)

        # Clamp exon p-values when s2.post < 1 and df.residual.zeros available
        if glmfit.get('df.residual.zeros') is not None:
            i = gene_s2_post[g] < 1
            if np.any(i):
                chisq_pvalue = chi2.sf(exon_LR[i], df=exon_df_test[i])
                exon_p_value[i] = np.maximum(exon_p_value[i], chisq_pvalue)

    # --- Simes aggregation for gene-level p-values ---
    # R code: sort exon p-values within each gene, compute Simes statistic
    # o <- order(g, exon.p.value)
    simes_order = np.lexsort((exon_p_value, g))
    p_sorted = exon_p_value[simes_order]

    # Build ranks within each gene
    # r = cumsum(1s) - (cumsum at gene boundaries - gene_nexons) repeated
    q = np.ones(nexons_kept, dtype=np.float64)
    cumq = np.cumsum(q)
    gene_boundaries = np.cumsum(gene_nexons)
    # Value at end of each gene
    boundary_vals = cumq[gene_boundaries - 1]
    # Starting value for each gene
    gene_starts = boundary_vals - gene_nexons
    r = cumq - np.repeat(gene_starts, gene_nexons)

    # pp = p * nexons_per_gene / rank
    pp = p_sorted * np.repeat(gene_nexons, gene_nexons) / r

    # Reverse sort to get minimum per gene
    # oo <- order(-g, pp, decreasing=TRUE)
    # This reverses the order so that the first exon of each gene (by the reverse sort)
    # corresponds to the minimum Simes statistic
    oo = np.lexsort((pp, -g))[::-1]
    gene_Simes_p_value = pp[oo][gene_boundaries - 1]

    # --- Build output ---
    result = {}
    result['comparison'] = coef_name
    result['design'] = design
    result['coefficients'] = coefficients_rel
    result['genes'] = exon_genes
    result['genecolname'] = genecolname
    result['exoncolname'] = exoncolname
    result['exon.df.test'] = exon_df_test

    if isLRT:
        result['exon.LR'] = exon_LR
    else:
        result['exon.F'] = exon_F

    result['exon.p.value'] = exon_p_value
    result['gene.df.test'] = gene_df_test

    if isLRT:
        result['gene.LR'] = gene_LR
    else:
        result['gene.df.prior'] = squeeze['df_prior']
        result['gene.df.residual'] = gene_df_residual
        result['gene.F'] = gene_F

    result['gene.p.value'] = gene_p_value
    result['gene.Simes.p.value'] = gene_Simes_p_value

    # --- Gene-level genes table ---
    exon_lastexon = gene_boundaries - 1
    exon_firstexon = exon_lastexon - gene_nexons + 1
    gene_genes = exon_genes.iloc[exon_lastexon].copy().reset_index(drop=True)
    gene_genes['NExons'] = gene_nexons

    # Identify gene-level columns (duplicated across all exons in a gene)
    no = np.zeros(len(exon_genes), dtype=bool)
    for col in exon_genes.columns:
        vals = exon_genes[col].values
        # Check if column is duplicated within each gene (skip first exon of each gene)
        not_first = np.ones(len(exon_genes), dtype=bool)
        not_first[exon_firstexon] = False
        if not_first.sum() > 0:
            # Check if all non-first exons have duplicate values
            shifted = np.zeros(len(exon_genes), dtype=bool)
            for gi in range(ngenes):
                start = exon_firstexon[gi]
                end = exon_lastexon[gi] + 1
                if end - start > 1:
                    first_val = vals[start]
                    for ei in range(start + 1, end):
                        if vals[ei] != first_val:
                            shifted[ei] = True
            no = no | shifted

    isgenelevel = []
    for col in exon_genes.columns:
        vals = exon_genes[col].values
        is_dup = True
        for gi in range(ngenes):
            start = exon_firstexon[gi]
            end = exon_lastexon[gi] + 1
            if end - start > 1:
                first_val = vals[start]
                for ei in range(start + 1, end):
                    if vals[ei] != first_val:
                        is_dup = False
                        break
            if not is_dup:
                break
        isgenelevel.append(is_dup)

    gene_level_cols = [col for col, isg in zip(exon_genes.columns, isgenelevel) if isg]
    gene_genes = exon_genes[gene_level_cols].iloc[exon_lastexon].copy().reset_index(drop=True)
    gene_genes['NExons'] = gene_nexons
    result['gene.genes'] = gene_genes

    return result


def diff_splice_dge(y, geneid=None, exonid=None, group=None,
                     dispersion='auto', prior_count=0.125):
    """Test for differential exon usage between groups using exact test.

    Port of edgeR's diffSpliceDGE.

    Parameters
    ----------
    y : DGEList-like dict
        DGEList with exon-level counts.
    geneid : ndarray or str
        Gene IDs.
    exonid : ndarray or str, optional
        Exon IDs.
    group : ndarray, optional
        Group factor.
    dispersion : str or ndarray
        Dispersion.

    Returns
    -------
    dict with gene-level and exon-level test results.
    """
    from .exact_test import exact_test

    if group is None and isinstance(y, dict):
        group = y['samples']['group'].values

    unique_groups = np.unique(group)
    if len(unique_groups) != 2:
        raise ValueError("Exactly 2 groups required for diffSpliceDGE")

    # Run exact test
    result = exact_test(y, pair=unique_groups[:2].tolist(), dispersion=dispersion,
                        prior_count=prior_count)

    # Get gene IDs
    if isinstance(geneid, str) and y.get('genes') is not None:
        geneid = y['genes'][geneid].values
    geneid = np.asarray(geneid)

    logFC = result['table']['logFC'].values
    p_exon = result['table']['PValue'].values

    # Simes aggregation
    unique_genes = np.unique(geneid)
    ngenes = len(unique_genes)
    gene_pvalue = np.ones(ngenes)
    gene_nexons = np.zeros(ngenes, dtype=int)

    for g_idx, gene in enumerate(unique_genes):
        mask = geneid == gene
        n_exons = np.sum(mask)
        gene_nexons[g_idx] = n_exons
        if n_exons <= 1:
            continue
        p_sorted = np.sort(p_exon[mask])
        gene_pvalue[g_idx] = min(np.min(p_sorted * n_exons / np.arange(1, n_exons + 1)), 1.0)

    _, gene_fdr, _, _ = multipletests(gene_pvalue, method='fdr_bh')

    return {
        'gene.table': pd.DataFrame({
            'GeneID': unique_genes,
            'NExons': gene_nexons,
            'PValue': gene_pvalue,
            'FDR': gene_fdr
        }),
        'exon.table': result['table'],
        'comparison': result.get('comparison')
    }


def splice_variants(y, geneids, dispersion=None):
    """Identify genes with splice variants.

    Port of edgeR's spliceVariants.

    Parameters
    ----------
    y : DGEList-like dict
        Exon-level count data.
    geneids : ndarray
        Gene IDs for each exon.
    dispersion : float or ndarray, optional
        Dispersion values.

    Returns
    -------
    DataFrame with splice variant statistics.
    """
    if isinstance(y, dict) and 'counts' in y:
        counts = y['counts']
    else:
        counts = np.asarray(y, dtype=np.float64)

    geneids = np.asarray(geneids)
    unique_genes = np.unique(geneids)

    results = []
    for gene in unique_genes:
        mask = geneids == gene
        n_exons = np.sum(mask)
        if n_exons <= 1:
            results.append({'GeneID': gene, 'NExons': n_exons,
                           'Chisq': 0, 'PValue': 1.0})
            continue

        gene_counts = counts[mask]
        # Chi-squared test for homogeneity of proportions
        col_totals = gene_counts.sum(axis=0)
        row_totals = gene_counts.sum(axis=1)
        grand_total = gene_counts.sum()

        if grand_total == 0:
            results.append({'GeneID': gene, 'NExons': n_exons,
                           'Chisq': 0, 'PValue': 1.0})
            continue

        expected = np.outer(row_totals, col_totals) / grand_total
        expected = np.maximum(expected, 1e-10)
        chi_sq = np.sum((gene_counts - expected) ** 2 / expected)
        df = (n_exons - 1) * (gene_counts.shape[1] - 1)
        p_value = chi2.sf(chi_sq, df) if df > 0 else 1.0

        results.append({'GeneID': gene, 'NExons': n_exons,
                       'Chisq': chi_sq, 'PValue': p_value})

    return pd.DataFrame(results)
