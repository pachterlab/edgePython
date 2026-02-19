# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
DGEList construction, validation, and accessors.

Port of edgeR's DGEList.R, validDGEList.R, getCounts.R, getDispersion.R,
getOffset.R, effectiveLibSizes.R.
"""

import numpy as np
import pandas as pd
import warnings
from .classes import DGEList


def _drop_empty_levels(x):
    """Drop unused levels from a categorical/factor variable."""
    if hasattr(x, 'cat'):
        return x.cat.remove_unused_categories()
    return pd.Categorical(x)


def make_dgelist(counts, lib_size=None, norm_factors=None, samples=None,
                 group=None, genes=None, remove_zeros=False,
                 annotation_columns=None):
    """Construct a DGEList object from components.

    Port of edgeR's DGEList().

    Parameters
    ----------
    counts : array-like or DataFrame
        Matrix of counts (genes x samples).
    lib_size : array-like, optional
        Library sizes. Defaults to column sums.
    norm_factors : array-like, optional
        Normalization factors. Defaults to all ones.
    samples : DataFrame, optional
        Sample-level information.
    group : array-like, optional
        Group memberships.
    genes : DataFrame, optional
        Gene-level annotation.
    remove_zeros : bool
        Whether to remove rows with all zero counts.
    annotation_columns : list, optional
        For DataFrame counts, which columns are annotation (not counts).

    Returns
    -------
    DGEList
    """
    # Handle DataFrame input
    if isinstance(counts, pd.DataFrame):
        if annotation_columns is not None:
            if isinstance(annotation_columns, (list, np.ndarray)):
                ann_cols = annotation_columns
            elif isinstance(annotation_columns, str):
                ann_cols = [annotation_columns]
            else:
                ann_cols = list(annotation_columns)

            if genes is None:
                genes = counts[ann_cols].copy()
            else:
                genes = pd.concat([counts[ann_cols], genes], axis=1)
            counts = counts.drop(columns=ann_cols)
        else:
            # Auto-detect non-numeric columns
            numeric_mask = counts.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
            if not numeric_mask.all():
                non_numeric = counts.columns[~numeric_mask]
                last_non_numeric = non_numeric[-1]
                last_idx = counts.columns.get_loc(last_non_numeric)
                ann_cols = counts.columns[:last_idx + 1].tolist()
                if genes is None:
                    genes = counts[ann_cols].copy()
                else:
                    genes = pd.concat([counts[ann_cols], genes], axis=1)
                counts = counts.iloc[:, last_idx + 1:]

    # Handle scipy sparse matrices
    if hasattr(counts, 'toarray') and hasattr(counts, 'nnz'):
        shape = counts.shape
        nnz = counts.nnz
        density = nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
        warnings.warn(
            f"Densifying sparse matrix ({shape[0]} x {shape[1]}, "
            f"{100*density:.1f}% non-zero, "
            f"{shape[0] * shape[1] * 8 / 1e6:.0f} MB dense). "
            f"edgePython stores counts as dense arrays.",
            stacklevel=2,
        )
        counts = np.asarray(counts.toarray(), dtype=np.float64)
    else:
        counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts.reshape(-1, 1)

    # Validate counts
    if counts.size == 0:
        raise ValueError("'counts' must contain at least one value")
    m = np.nanmin(counts)
    if np.isnan(m):
        raise ValueError("NA counts not allowed")
    if m < 0:
        raise ValueError("Negative counts not allowed")
    if not np.isfinite(np.max(counts)):
        raise ValueError("Infinite counts not allowed")

    nlib = counts.shape[1]
    ntags = counts.shape[0]

    # Column names
    col_names = [f"Sample{i+1}" for i in range(nlib)]
    row_names = [str(i+1) for i in range(ntags)]

    # Library sizes
    if lib_size is None:
        lib_size = counts.sum(axis=0)
        if np.min(lib_size) <= 0:
            warnings.warn("At least one library size is zero")
    else:
        lib_size = np.asarray(lib_size, dtype=np.float64)
        if not np.issubdtype(lib_size.dtype, np.number):
            raise ValueError("'lib_size' must be numeric")
        if len(lib_size) != nlib:
            raise ValueError("length of 'lib_size' must equal number of samples")
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA library sizes not allowed")
        if np.any(lib_size < 0):
            raise ValueError("negative library sizes not allowed")
        if np.any((lib_size == 0) & (counts.sum(axis=0) > 0)):
            raise ValueError("library size set to zero but counts for that sample are nonzero")

    # Normalization factors
    if norm_factors is None:
        norm_factors = np.ones(nlib)
    else:
        norm_factors = np.asarray(norm_factors, dtype=np.float64)
        if len(norm_factors) != nlib:
            raise ValueError("Length of 'norm_factors' must equal number of columns in 'counts'")
        if np.any(np.isnan(norm_factors)):
            raise ValueError("NA norm factors not allowed")
        if np.any(norm_factors <= 0):
            raise ValueError("norm factors must be positive")
        if abs(np.sum(np.log(norm_factors))) > 1e-6:
            warnings.warn("norm factors don't multiply to 1")

    # Samples DataFrame
    if samples is not None:
        samples = pd.DataFrame(samples)
        if nlib != len(samples):
            raise ValueError("Number of rows in 'samples' must equal number of columns in 'counts'")

    # Group
    if group is None and samples is not None and 'group' in samples.columns:
        group = samples['group'].values
        samples = samples.drop(columns=['group'])

    if group is None:
        group = pd.Categorical([1] * nlib)
    else:
        if len(group) != nlib:
            raise ValueError("Length of 'group' must equal number of columns in 'counts'")
        group = _drop_empty_levels(pd.Categorical(group))

    # Build samples DataFrame
    sam = pd.DataFrame({
        'group': group,
        'lib.size': lib_size,
        'norm.factors': norm_factors
    })
    if samples is not None:
        for col in samples.columns:
            sam[col] = samples[col].values
    sam.index = col_names

    # Build DGEList
    x = DGEList()
    x['counts'] = counts
    x['samples'] = sam

    # Gene annotation
    if genes is not None:
        genes = pd.DataFrame(genes)
        if len(genes) != ntags:
            raise ValueError("Counts and genes have different numbers of rows")
        genes.index = row_names
        x['genes'] = genes

    # Remove all-zero rows
    if remove_zeros:
        all_zeros = np.sum(counts > 0, axis=1) == 0
        if np.any(all_zeros):
            keep = ~all_zeros
            x['counts'] = counts[keep]
            if 'genes' in x and x['genes'] is not None:
                x['genes'] = x['genes'].iloc[keep]
            print(f"Removing {np.sum(all_zeros)} rows with all zero counts")

    return x


def valid_dgelist(y):
    """Check and fill standard components of a DGEList.

    Port of edgeR's validDGEList.
    """
    if 'counts' not in y or y['counts'] is None:
        raise ValueError("No count matrix")
    y['counts'] = np.asarray(y['counts'], dtype=np.float64)
    nlib = y['counts'].shape[1]
    if 'samples' not in y:
        y['samples'] = pd.DataFrame()
    if 'group' not in y['samples'].columns:
        y['samples']['group'] = pd.Categorical([1] * nlib)
    if 'lib.size' not in y['samples'].columns:
        y['samples']['lib.size'] = y['counts'].sum(axis=0)
    if 'norm.factors' not in y['samples'].columns:
        y['samples']['norm.factors'] = np.ones(nlib)
    return y


def get_counts(y):
    """Extract count matrix from DGEList.

    Port of edgeR's getCounts.
    """
    return np.asarray(y['counts'])


def get_dispersion(y):
    """Get the most complex dispersion values from a DGEList.

    Port of edgeR's getDispersion.
    Returns tagwise > trended > common > None, with a 'type' attribute.
    """
    if y.get('tagwise.dispersion') is not None:
        disp = np.asarray(y['tagwise.dispersion'])
        disp_type = 'tagwise'
    elif y.get('trended.dispersion') is not None:
        disp = np.asarray(y['trended.dispersion'])
        disp_type = 'trended'
    elif y.get('common.dispersion') is not None:
        disp = np.float64(y['common.dispersion'])
        disp_type = 'common'
    else:
        return None

    # Store type as attribute (Python doesn't have R's attr, use a wrapper)
    result = disp
    # We'll just return the value; callers can check type if needed
    return result


def get_dispersion_type(y):
    """Get the type of the most complex dispersion in a DGEList."""
    if y.get('tagwise.dispersion') is not None:
        return 'tagwise'
    elif y.get('trended.dispersion') is not None:
        return 'trended'
    elif y.get('common.dispersion') is not None:
        return 'common'
    return None


def get_offset(y):
    """Extract offset vector or matrix from a DGEList.

    Port of edgeR's getOffset. Returns log(lib.size * norm.factors) by default.
    """
    if y.get('offset') is not None:
        return y['offset']

    lib_size = y['samples']['lib.size'].values
    if lib_size is None:
        raise ValueError("y is not a valid DGEList object")

    norm_factors = y['samples'].get('norm.factors')
    if norm_factors is not None:
        lib_size = lib_size * norm_factors.values

    if np.any(~np.isfinite(lib_size)) or np.any(lib_size <= 0):
        raise ValueError("library sizes must be positive finite values")

    return np.log(lib_size)


def get_norm_lib_sizes(y, log=False):
    """Get effective (normalized) library sizes.

    Port of edgeR's getNormLibSizes / effectiveLibSizes.
    """
    if isinstance(y, dict):
        if y.get('offset') is not None:
            # For DGEGLM/DGELRT objects, offset is a matrix
            offset = y['offset']
            if hasattr(offset, 'as_matrix'):
                offset = offset.as_matrix()
            if isinstance(offset, np.ndarray) and offset.ndim == 2:
                els = offset[0, :]
            else:
                els = offset
            if not log:
                els = np.exp(els)
            return els
        elif 'samples' in y:
            els = y['samples']['lib.size'].values * y['samples']['norm.factors'].values
            if log:
                els = np.log(els)
            return els
    # Default for matrices
    y = np.asarray(y)
    els = y.sum(axis=0)
    if log:
        els = np.log(els)
    return els
