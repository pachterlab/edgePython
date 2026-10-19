# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Gene filtering for edgePython.

Port of edgeR's filterByExpr.
"""

import warnings

import numpy as np
import pandas as pd
from .expression import cpm


_PRESETS = {
    'stringent': dict(min_count=10, min_total_count=15, large_n=10, min_prop=0.7, max_n_samples=40),
    'moderate': dict(min_count=2, min_total_count=8, large_n=4, min_prop=0.5, max_n_samples=20),
    'lenient': dict(min_count=1, min_total_count=6, large_n=2, min_prop=0.5, max_n_samples=10),
}


def filter_by_expr(y, design=None, group=None, lib_size=None, min_count=None,
                   min_total_count=None, large_n=None, min_prop=None,
                   min_sample_size=None, max_n_samples=None, preset=None):
    """Filter low-expressed genes.

    Port of edgeR 5.0's filterByExpr(). A gene is kept if it has at least
    min_total_count reads and a CPM of at least min_count / (median library
    size) * 1e6 in at least n samples, where n is the smallest group size
    (or 1 / the largest leverage of the design), reduced above large_n by
    min_prop and capped at max_n_samples.

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
        Minimum count threshold (default 10).
    min_total_count : float
        Minimum total count across all samples (default 15).
    large_n : int
        Large sample size threshold (default 10).
    min_prop : float
        Minimum proportion for large groups (default 0.7).
    min_sample_size : float, optional
        Minimum sample size, overriding the one from group or design.
    max_n_samples : float
        Cap on the minimum sample size (default 40).
    preset : str, optional
        'stringent' (the defaults), 'moderate' or 'lenient'; overrides the
        individual thresholds.

    Returns
    -------
    ndarray of bool, True for genes to keep.
    """
    thresholds = dict(min_count=min_count, min_total_count=min_total_count, large_n=large_n,
                      min_prop=min_prop, max_n_samples=max_n_samples)
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError("preset must be 'stringent', 'moderate' or 'lenient'")
        if any(v is not None for v in thresholds.values()):
            warnings.warn(f"Using {preset} settings. Ignoring arguments set individually.")
        thresholds = dict(_PRESETS[preset])
    else:
        thresholds = {k: _PRESETS['stringent'][k] if v is None else v for k, v in thresholds.items()}

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
    if min_sample_size is None:
        if group is None:
            if design is None:
                min_sample_size = counts.shape[1]
            else:
                min_sample_size = 1.0 / np.max(_hat_values(np.asarray(design, dtype=np.float64)))
        else:
            _, counts_per_group = np.unique(np.asarray(group), return_counts=True)
            min_sample_size = np.min(counts_per_group[counts_per_group > 0])

    large_n = thresholds['large_n']
    if min_sample_size > large_n:
        min_sample_size = large_n + (min_sample_size - large_n) * thresholds['min_prop']
    min_sample_size = min(min_sample_size, thresholds['max_n_samples'])

    # CPM cutoff
    median_lib_size = np.median(lib_size)
    cpm_cutoff = thresholds['min_count'] / median_lib_size * 1e6
    cpm_vals = cpm(counts, lib_size=lib_size)

    tol = 1e-14
    keep_cpm = np.sum(cpm_vals >= cpm_cutoff, axis=1) >= (min_sample_size - tol)

    # Total count cutoff
    keep_total = np.sum(counts, axis=1) >= (thresholds['min_total_count'] - tol)

    return keep_cpm & keep_total


def _hat_values(design):
    """Leverages as R's hat(design) gives them.

    hat() adds an intercept column (intercept = TRUE) and drops columns
    that are linearly dependent by qr()'s rule (tol 1e-7).
    """
    from .ql_weights import _hat_values as _qr_hat
    x = np.column_stack([np.ones(design.shape[0]), design])
    return _qr_hat(np.ascontiguousarray(x), np.ones(design.shape[0]))
