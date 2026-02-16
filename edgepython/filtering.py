# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Gene filtering for edgePython.

Port of edgeR's filterByExpr.
"""

import numpy as np
import pandas as pd
from .expression import cpm


def filter_by_expr(y, design=None, group=None, lib_size=None,
                   min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """Filter low-expressed genes.

    Port of edgeR's filterByExpr().

    Parameters
    ----------
    y : array-like or DGEList
        Count matrix or DGEList.
    design : array-like, optional
        Design matrix.
    group : array-like, optional
        Group factor.
    lib_size : array-like, optional
        Library sizes.
    min_count : float
        Minimum count threshold.
    min_total_count : float
        Minimum total count across all samples.
    large_n : int
        Large sample size threshold.
    min_prop : float
        Minimum proportion for large groups.

    Returns
    -------
    ndarray of bool, True for genes to keep.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if design is None and group is None:
            design = dge.get('design')
            if design is None:
                group = dge['samples']['group'].values
        if lib_size is None:
            lib_size = dge['samples']['lib.size'].values * dge['samples']['norm.factors'].values
        counts = dge['counts']
    else:
        counts = np.asarray(y, dtype=np.float64)

    if counts.ndim == 1:
        counts = counts.reshape(-1, 1)

    if lib_size is None:
        lib_size = counts.sum(axis=0)
    lib_size = np.asarray(lib_size, dtype=np.float64)

    # Minimum effective sample size
    if group is None:
        if design is None:
            min_sample_size = counts.shape[1]
        else:
            design = np.asarray(design, dtype=np.float64)
            h = _hat_values(design)
            min_sample_size = 1.0 / np.max(h)
    else:
        group = np.asarray(group)
        _, counts_per_group = np.unique(group, return_counts=True)
        nonzero_counts = counts_per_group[counts_per_group > 0]
        min_sample_size = np.min(nonzero_counts)

    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * min_prop

    # CPM cutoff
    median_lib_size = np.median(lib_size)
    cpm_cutoff = min_count / median_lib_size * 1e6
    cpm_vals = cpm(counts, lib_size=lib_size)

    tol = 1e-14
    keep_cpm = np.sum(cpm_vals >= cpm_cutoff, axis=1) >= (min_sample_size - tol)

    # Total count cutoff
    keep_total = np.sum(counts, axis=1) >= (min_total_count - tol)

    return keep_cpm & keep_total


def _hat_values(design):
    """Compute hat/leverage values for a design matrix."""
    Q, R = np.linalg.qr(design)
    return np.sum(Q ** 2, axis=1)
