# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Visualization functions for edgePython.

Port of edgeR's plotMD, plotBCV, plotMDS, plotSmear, plotQLDisp, maPlot, gof.
"""

import numpy as np
import warnings


def plot_md(obj, column=None, coef=None, xlab='Average log CPM',
            ylab='log-fold-change', main=None, status=None,
            values=None, col=None, hl_col=None, **kwargs):
    """Mean-difference plot (MD plot / MA plot).

    Port of edgeR's plotMD.

    Parameters
    ----------
    obj : dict (DGEGLM, DGELRT, DGEExact, or DGEList-like)
        Object containing logCPM and logFC.
    column : int, optional
        Column of coefficients to plot.
    coef : int, optional
        Alias for column.
    xlab, ylab, main : str
        Plot labels.
    status : ndarray, optional
        Status indicators for coloring.
    """
    import matplotlib.pyplot as plt

    if coef is not None:
        column = coef

    # Extract x (logCPM) and y (logFC) values
    if isinstance(obj, dict) and 'table' in obj:
        tab = obj['table']
        x = tab['logCPM'].values if 'logCPM' in tab.columns else tab.get('logCPM', np.zeros(len(tab)))
        y = tab['logFC'].values if 'logFC' in tab.columns else tab.iloc[:, 0].values
    elif isinstance(obj, dict) and 'coefficients' in obj:
        from .expression import ave_log_cpm
        x = obj.get('AveLogCPM')
        if x is None:
            x = ave_log_cpm(obj)
        if column is None:
            column = 0
        y = obj['coefficients'][:, column] / np.log(2)
    elif isinstance(obj, dict) and 'counts' in obj:
        from .expression import ave_log_cpm, cpm
        x = ave_log_cpm(obj)
        cpm_vals = cpm(obj, log=True)
        if column is None:
            column = 0
        y = cpm_vals[:, column] - np.mean(cpm_vals, axis=1)
    else:
        raise ValueError("Unsupported object type for plotMD")

    fig, ax = plt.subplots(figsize=(8, 6))

    if status is not None:
        status = np.asarray(status)
        unique_status = np.unique(status)
        colors = ['grey', 'red', 'blue']
        for i, s in enumerate(unique_status):
            mask = status == s
            c = colors[i % len(colors)] if col is None else (col[i] if isinstance(col, list) else col)
            ax.scatter(x[mask], y[mask], s=2, alpha=0.5, c=c, label=str(s))
        ax.legend()
    else:
        ax.scatter(x, y, s=2, alpha=0.5, c='black')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if main:
        ax.set_title(main)

    plt.tight_layout()
    return fig, ax


def plot_bcv(y, xlab='Average log CPM', ylab='Biological coefficient of variation',
             pch=16, cex=0.2, col_common='red', col_trend='blue',
             col_tagwise='black', **kwargs):
    """Plot biological coefficient of variation.

    Port of edgeR's plotBCV.

    Parameters
    ----------
    y : DGEList-like dict
        Must have dispersion estimates.
    """
    import matplotlib.pyplot as plt
    from .expression import ave_log_cpm

    alc = y.get('AveLogCPM')
    if alc is None:
        alc = ave_log_cpm(y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Tagwise dispersions
    if y.get('tagwise.dispersion') is not None:
        bcv_tagwise = np.sqrt(y['tagwise.dispersion'])
        ax.scatter(alc, bcv_tagwise, s=cex * 10, alpha=0.3, c=col_tagwise, label='Tagwise')

    # Trended dispersion
    if y.get('trended.dispersion') is not None:
        bcv_trend = np.sqrt(y['trended.dispersion'])
        o = np.argsort(alc)
        ax.plot(alc[o], bcv_trend[o], c=col_trend, linewidth=2, label='Trend')

    # Common dispersion
    if y.get('common.dispersion') is not None:
        bcv_common = np.sqrt(y['common.dispersion'])
        ax.axhline(y=bcv_common, color=col_common, linewidth=2, label='Common')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_mds(y, top=500, labels=None, pch=None, cex=1, dim_plot=(1, 2),
             gene_selection='pairwise', xlab=None, ylab=None, main=None,
             **kwargs):
    """Multi-dimensional scaling plot.

    Port of edgeR's plotMDS.

    Parameters
    ----------
    y : DGEList-like dict or ndarray
        Count data.
    top : int
        Number of top genes to use.
    labels : list, optional
        Sample labels.
    dim_plot : tuple
        Dimensions to plot.
    gene_selection : str
        'pairwise' or 'common'.
    """
    import matplotlib.pyplot as plt
    from .expression import cpm

    if isinstance(y, dict) and 'counts' in y:
        counts = y['counts']
        if labels is None:
            labels = y['samples'].index.tolist() if hasattr(y['samples'], 'index') else None
    else:
        counts = np.asarray(y, dtype=np.float64)

    # Log-CPM values
    lib_size = counts.sum(axis=0)
    log_cpm = np.log2(counts / lib_size[None, :] * 1e6 + 0.5)

    nsamples = counts.shape[1]
    if labels is None:
        labels = [f'S{i+1}' for i in range(nsamples)]

    # Select top variable genes
    var = np.var(log_cpm, axis=1)
    top_idx = np.argsort(var)[::-1][:min(top, len(var))]
    log_cpm_top = log_cpm[top_idx]

    # Pairwise distances
    dist_mat = np.zeros((nsamples, nsamples))
    for i in range(nsamples):
        for j in range(i + 1, nsamples):
            d = np.sqrt(np.mean((log_cpm_top[:, i] - log_cpm_top[:, j]) ** 2))
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # Classical MDS
    H = np.eye(nsamples) - np.ones((nsamples, nsamples)) / nsamples
    B = -0.5 * H @ (dist_mat ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Get coordinates for desired dimensions
    d1 = dim_plot[0] - 1
    d2 = dim_plot[1] - 1
    x_coord = eigvecs[:, d1] * np.sqrt(max(eigvals[d1], 0))
    y_coord = eigvecs[:, d2] * np.sqrt(max(eigvals[d2], 0))

    # Variance explained
    total_var = np.sum(np.maximum(eigvals, 0))
    var_exp1 = max(eigvals[d1], 0) / total_var * 100 if total_var > 0 else 0
    var_exp2 = max(eigvals[d2], 0) / total_var * 100 if total_var > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_coord, y_coord, s=50)

    for i, label in enumerate(labels):
        ax.annotate(label, (x_coord[i], y_coord[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    if xlab is None:
        xlab = f'Dimension {dim_plot[0]} ({var_exp1:.1f}%)'
    if ylab is None:
        ylab = f'Dimension {dim_plot[1]} ({var_exp2:.1f}%)'

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if main:
        ax.set_title(main)

    plt.tight_layout()
    return fig, ax


def plot_smear(obj, pair=None, de_tags=None, xlab='Average logCPM',
               ylab='logFC', main='MA Plot', smooth_scatter=False,
               lowess=False, **kwargs):
    """Smear plot (MA plot for DGE).

    Port of edgeR's plotSmear.

    Parameters
    ----------
    obj : DGEList or DGEExact-like dict
        Object to plot.
    pair : list, optional
        Groups to compare.
    de_tags : list or ndarray, optional
        Indices of DE genes to highlight.
    """
    import matplotlib.pyplot as plt

    if isinstance(obj, dict) and 'table' in obj:
        tab = obj['table']
        x = tab['logCPM'].values
        y_vals = tab['logFC'].values
    else:
        raise ValueError("Object must have a 'table' attribute")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y_vals, s=2, alpha=0.3, c='black')

    if de_tags is not None:
        de_tags = np.asarray(de_tags)
        ax.scatter(x[de_tags], y_vals[de_tags], s=4, c='red', alpha=0.5)

    ax.axhline(y=0, color='blue', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main)

    plt.tight_layout()
    return fig, ax


def plot_ql_disp(glmfit, xlab='Average Log2 CPM',
                  ylab='Quarter-Root Mean Deviance',
                  pch=16, cex=0.2, col_shrunk='red', col_trend='blue',
                  col_raw='black', **kwargs):
    """Plot quasi-likelihood dispersions.

    Port of edgeR's plotQLDisp.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted QL GLM from glm_ql_fit().
    """
    import matplotlib.pyplot as plt
    from .expression import ave_log_cpm

    if glmfit.get('s2.post') is None:
        raise ValueError("need to run glm_ql_fit before plot_ql_disp")

    A = glmfit.get('AveLogCPM')
    if A is None:
        A = ave_log_cpm(glmfit)

    if glmfit.get('df.residual.zeros') is None:
        df_residual = glmfit.get('df.residual.adj', glmfit['df.residual'])
        deviance = glmfit.get('deviance.adj', glmfit['deviance'])
    else:
        df_residual = glmfit['df.residual.zeros']
        deviance = glmfit['deviance']

    df_residual = np.asarray(df_residual, dtype=np.float64)
    s2 = deviance / np.maximum(df_residual, 1e-8)
    s2[df_residual < 1e-8] = 0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw
    ax.scatter(A, s2 ** 0.25, s=cex * 10, alpha=0.3, c=col_raw, label='Raw')

    # Squeezed
    ax.scatter(A, np.asarray(glmfit['s2.post']) ** 0.25, s=cex * 10,
               alpha=0.3, c=col_shrunk, label='Squeezed')

    # Trend
    s2_prior = np.atleast_1d(glmfit['s2.prior'])
    if len(s2_prior) == 1:
        ax.axhline(y=s2_prior[0] ** 0.25, color=col_trend, linewidth=2, label='Trend')
    else:
        o = np.argsort(A)
        ax.plot(A[o], s2_prior[o] ** 0.25, c=col_trend, linewidth=2, label='Trend')

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend()

    plt.tight_layout()
    return fig, ax


def ma_plot(x, y, logFC=None, de_tags=None, smooth_scatter=False,
            xlab='A', ylab='M', main='MA Plot', **kwargs):
    """Simple MA plot.

    Parameters
    ----------
    x : ndarray
        Average expression (A values).
    y : ndarray
        Log fold change (M values).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=2, alpha=0.3, c='black')

    if de_tags is not None:
        de_tags = np.asarray(de_tags)
        ax.scatter(x[de_tags], y[de_tags], s=4, c='red', alpha=0.5)

    ax.axhline(y=0, color='blue', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main)

    plt.tight_layout()
    return fig, ax


def gof(glmfit, pcutoff=0.1, adjust='holm', plot=True, main='Goodness of Fit',
        **kwargs):
    """Goodness of fit test for each gene.

    Port of edgeR's gof.

    Parameters
    ----------
    glmfit : dict (DGEGLM-like)
        Fitted GLM.
    pcutoff : float
        P-value cutoff.
    adjust : str
        P-value adjustment method.
    plot : bool
        Whether to plot.

    Returns
    -------
    dict with 'gof.statistics', 'gof.pvalues', 'outlier', 'df'.
    """
    from scipy.stats import chi2
    from statsmodels.stats.multitest import multipletests

    deviance = glmfit['deviance']
    df = glmfit['df.residual']

    df = np.asarray(df, dtype=np.float64)
    gof_pvalues = chi2.sf(deviance, df)

    # Adjust p-values
    method_map = {'holm': 'holm', 'BH': 'fdr_bh', 'bonferroni': 'bonferroni'}
    sm_method = method_map.get(adjust, adjust)
    _, adj_p, _, _ = multipletests(gof_pvalues, method=sm_method)

    outlier = adj_p < pcutoff

    if plot:
        import matplotlib.pyplot as plt
        from scipy.stats import chi2 as chi2_dist

        fig, ax = plt.subplots(figsize=(8, 6))
        # QQ plot
        n = len(deviance)
        theoretical = chi2_dist.ppf(np.arange(1, n + 1) / (n + 1), df[0])
        observed = np.sort(deviance)

        ax.scatter(theoretical, observed, s=2, alpha=0.5)
        max_val = max(np.max(theoretical), np.max(observed))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=0.5)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Observed deviance')
        ax.set_title(main)
        plt.tight_layout()

    return {
        'gof.statistics': deviance,
        'gof.pvalues': gof_pvalues,
        'outlier': outlier,
        'df': df
    }
