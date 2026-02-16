# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Expression value computation for edgePython.

Port of edgeR's cpm, rpkm, tpm, aveLogCPM, cpmByGroup, rpkmByGroup.
"""

import numpy as np
import warnings
from .utils import expand_as_matrix, add_prior_count


def cpm(y, lib_size=None, offset=None, log=False, prior_count=2,
        normalized_lib_sizes=True):
    """Counts per million.

    Port of edgeR's cpm().

    Parameters
    ----------
    y : array-like or DGEList
        Count matrix or DGEList.
    lib_size : array-like, optional
        Library sizes.
    offset : array-like, optional
        Log-scale offsets.
    log : bool
        Return log2-CPM?
    prior_count : float
        Prior count for log transformation.
    normalized_lib_sizes : bool
        Use normalized library sizes (for DGEList input).

    Returns
    -------
    ndarray of CPM values.
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        ls = dge['samples']['lib.size'].values

        if dge.get('offset') is not None:
            ls = None
            offset = dge['offset']
        elif normalized_lib_sizes:
            ls = ls * dge['samples']['norm.factors'].values

        return _cpm_default(dge['counts'], lib_size=ls, offset=offset,
                            log=log, prior_count=prior_count)

    return _cpm_default(y, lib_size=lib_size, offset=offset,
                        log=log, prior_count=prior_count)


def _cpm_default(y, lib_size=None, offset=None, log=False, prior_count=2):
    """Core CPM calculation."""
    y = np.asarray(y, dtype=np.float64)
    ymin = np.nanmin(y)
    if np.isnan(ymin):
        raise ValueError("NA counts not allowed")
    if ymin < 0:
        raise ValueError("Negative counts not allowed")

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.size == 0:
        return y.copy()

    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.ndim == 2:
            if offset.shape != y.shape:
                raise ValueError("dimensions not consistent between counts and offset")
        else:
            if len(offset) != y.shape[1]:
                raise ValueError("Length of offset differs from number of libraries")
        lib_size = np.exp(offset) if offset.ndim == 1 else None
        if offset.ndim == 2:
            lib_size = np.exp(offset)
    else:
        if lib_size is None:
            lib_size = y.sum(axis=0)

    lib_size = np.asarray(lib_size, dtype=np.float64)
    if lib_size.ndim == 1:
        if np.any(lib_size <= 0):
            raise ValueError("library sizes should be greater than zero")

    if log:
        out = add_prior_count(y, lib_size=lib_size if lib_size.ndim == 1 else None,
                              offset=np.log(lib_size) if lib_size.ndim == 1 else offset,
                              prior_count=prior_count)
        y_aug = out['y']
        offset_aug = out['offset']

        if isinstance(offset_aug, np.ndarray) and offset_aug.ndim == 1:
            lib_size_aug = np.exp(offset_aug)
        else:
            lib_size_aug = np.exp(offset_aug)

        if lib_size_aug.ndim == 1:
            result = np.log2(y_aug / lib_size_aug[np.newaxis, :] * 1e6)
        else:
            result = np.log2(y_aug / lib_size_aug * 1e6)
        return result
    else:
        if lib_size.ndim == 1:
            return y / lib_size[np.newaxis, :] * 1e6
        else:
            return y / lib_size * 1e6


def rpkm(y, gene_length, lib_size=None, offset=None, log=False, prior_count=2,
         normalized_lib_sizes=True):
    """Reads per kilobase per million.

    Port of edgeR's rpkm().

    Parameters
    ----------
    y : array-like or DGEList
        Count matrix or DGEList.
    gene_length : array-like or str
        Gene lengths in bp.
    lib_size, offset, log, prior_count, normalized_lib_sizes :
        As for cpm().
    """
    # Extract gene_length from DGEList if string
    if isinstance(y, dict) and 'counts' in y:
        if isinstance(gene_length, str):
            gene_length = y['genes'][gene_length].values
        elif gene_length is None:
            for col in ['Length', 'length']:
                if col in y.get('genes', {}).columns if y.get('genes') is not None else False:
                    gene_length = y['genes'][col].values
                    break
            if gene_length is None:
                raise ValueError("Gene lengths not found")

    gene_length = np.asarray(gene_length, dtype=np.float64)
    gene_length_kb = gene_length / 1000

    result = cpm(y, lib_size=lib_size, offset=offset, log=log,
                 prior_count=prior_count, normalized_lib_sizes=normalized_lib_sizes)

    if log:
        return result - np.log2(gene_length_kb[:, np.newaxis])
    else:
        return result / gene_length_kb[:, np.newaxis]


def tpm(y, effective_tx_length, rta_overdispersion=None, shrunk=False):
    """Transcripts per million from a fitted model.

    Port of edgeR's tpm().
    """
    t = cpm(y, log=False)
    A = np.asarray(effective_tx_length, dtype=np.float64)
    if rta_overdispersion is not None:
        A = A / np.asarray(rta_overdispersion)
    if A.ndim == 1:
        t = t / A[:, np.newaxis]
    else:
        t = t / A
    col_sums = t.sum(axis=0)
    avg_col_sum = np.exp(np.mean(np.log(col_sums)))
    return t / avg_col_sum * 1e6


def ave_log_cpm(y, lib_size=None, offset=None, prior_count=2, dispersion=None,
                weights=None, normalized_lib_sizes=True):
    """Average log2-CPM for each gene.

    Port of edgeR's aveLogCPM().
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        ls = dge['samples']['lib.size'].values
        if ls is None:
            ls = dge['counts'].sum(axis=0)
        if normalized_lib_sizes:
            nf = dge['samples'].get('norm.factors')
            if nf is not None:
                ls = ls * nf.values
        if dispersion is None:
            dispersion = dge.get('common.dispersion')
        w = dge.get('weights')
        return _ave_log_cpm_default(dge['counts'], lib_size=ls, offset=offset,
                                    prior_count=prior_count, dispersion=dispersion,
                                    weights=w)

    return _ave_log_cpm_default(y, lib_size=lib_size, offset=offset,
                                prior_count=prior_count, dispersion=dispersion,
                                weights=weights)


def _ave_log_cpm_default(y, lib_size=None, offset=None, prior_count=2,
                         dispersion=None, weights=None):
    """Core aveLogCPM calculation.

    Uses mglmOneGroup to fit intercept-only NB GLM model, then converts
    the fitted coefficient to log2-CPM. This matches R edgeR's aveLogCPM.default
    which calls C++ code internally doing the same thing.
    """
    from .glm_fit import mglm_one_group

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[0] == 0:
        return np.array([], dtype=np.float64)

    if dispersion is None:
        dispersion = 0.05
    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=np.float64))
    if np.all(np.isnan(dispersion)):
        dispersion = np.array([0.05])
    dispersion = np.where(np.isnan(dispersion), np.nanmean(dispersion), dispersion)

    if offset is None:
        if lib_size is None:
            lib_size = y.sum(axis=0)
        offset = np.log(lib_size)
    offset = np.atleast_1d(np.asarray(offset, dtype=np.float64))

    # Add prior counts and adjust offset
    out = add_prior_count(y, offset=offset, prior_count=prior_count)
    y_aug = out['y']
    offset_aug = out['offset']

    # Fit intercept-only NB GLM using mglmOneGroup
    ab = mglm_one_group(y_aug, dispersion=dispersion, offset=offset_aug,
                        weights=weights)

    # Convert fitted coefficient to log2-CPM: (ab + log(1e6)) / log(2)
    result = (ab + np.log(1e6)) / np.log(2)

    return result


def cpm_by_group(y, group=None, dispersion=0.05, offset=None, weights=None,
                 log=False, prior_count=2):
    """Counts per million averaged by group.

    Port of edgeR's cpmByGroup().
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if group is None:
            group = dge['samples']['group'].values
        if offset is None:
            from .dgelist import get_offset
            offset = get_offset(dge)
        w = dge.get('weights')
        return _cpm_by_group_default(dge['counts'], group=group, dispersion=dispersion,
                                     offset=offset, weights=w, log=log,
                                     prior_count=prior_count)

    return _cpm_by_group_default(y, group=group, dispersion=dispersion,
                                 offset=offset, weights=weights, log=log,
                                 prior_count=prior_count)


def _cpm_by_group_default(y, group=None, dispersion=0.05, offset=None,
                          weights=None, log=False, prior_count=2):
    """Core cpmByGroup calculation.

    Uses mglmOneWay to fit NB GLM per group, matching R's cpmByGroup.default.
    """
    from .glm_fit import mglm_one_way

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if group is None:
        group = np.ones(y.shape[1], dtype=int)
    group = np.asarray(group)

    if offset is None:
        offset = np.log(y.sum(axis=0))
    offset = np.atleast_1d(np.asarray(offset, dtype=np.float64))

    if log:
        out = add_prior_count(y, offset=offset, prior_count=prior_count)
        fit = mglm_one_way(out['y'], group=group, dispersion=dispersion,
                           offset=out['offset'], weights=weights)
        return fit['coefficients'] / np.log(2) + np.log2(1e6)
    else:
        fit = mglm_one_way(y, group=group, dispersion=dispersion,
                           offset=offset, weights=weights)
        return np.exp(fit['coefficients']) * 1e6


def rpkm_by_group(y, group=None, gene_length=None, dispersion=0.05,
                  offset=None, weights=None, log=False, prior_count=2):
    """RPKM averaged by group.

    Port of edgeR's rpkmByGroup().
    """
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if gene_length is None:
            for col in ['Length', 'length']:
                if dge.get('genes') is not None and col in dge['genes'].columns:
                    gene_length = dge['genes'][col].values
                    break
        elif isinstance(gene_length, str):
            gene_length = dge['genes'][gene_length].values
        if gene_length is None:
            raise ValueError("Gene lengths not found")

    gene_length = np.asarray(gene_length, dtype=np.float64)
    z = cpm_by_group(y, group=group, dispersion=dispersion, offset=offset,
                     weights=weights, log=log, prior_count=prior_count)

    if log:
        return z - np.log2(gene_length[:, np.newaxis] / 1e3)
    else:
        return z / (gene_length[:, np.newaxis] / 1e3)
