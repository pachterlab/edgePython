# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Expression value computation for edgePython.

Port of edgeR's cpm, rpkm, tpm, aveLogCPM, cpmByGroup, rpkmByGroup.
"""

import numpy as np
from .compressed_matrix import CompressedMatrix
import warnings
from .utils import expand_as_matrix, add_prior_count


def cpm(y, lib_size=None, offset=None, log=False, prior_count=2,
        normalized_lib_sizes=True, offset_prior=None):
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
    offset_prior : array-like, optional
        Genes x samples log-scale adjustments to the library sizes, which
        become exp(log(lib_size) + offset_prior). A DGEList's
        offset.prior is used automatically. Ignored if offset is given.

    Returns
    -------
    ndarray of CPM values.
    """
    # DGEList input. As in edgeR 5.0, a stored offset is used as is;
    # otherwise the library sizes are combined with any offset.prior.
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if dge.get('offset') is not None:
            return _cpm_default(dge['counts'], offset=dge['offset'], log=log,
                                prior_count=prior_count)
        ls = dge['samples']['lib.size'].values
        if normalized_lib_sizes:
            ls = ls * dge['samples']['norm.factors'].values
        return _cpm_default(dge['counts'], lib_size=ls, offset_prior=dge.get('offset.prior'),
                            log=log, prior_count=prior_count)

    return _cpm_default(y, lib_size=lib_size, offset=offset, offset_prior=offset_prior,
                        log=log, prior_count=prior_count)


def _cpm_default(y, lib_size=None, offset=None, log=False, prior_count=2, offset_prior=None):
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

    # Library sizes as in edgeR 5.0's cpm.default: from the offset if given,
    # otherwise lib_size (default column sums), times exp(offset_prior) if
    # given, which makes them genes x samples
    if offset is not None:
        if isinstance(offset, CompressedMatrix):
            offset = offset.as_matrix()
        offset = np.asarray(offset, dtype=np.float64)
        if offset.ndim == 2:
            if offset.shape != y.shape:
                raise ValueError("counts and offset must have equal dimensions")
        elif len(offset) != y.shape[1]:
            raise ValueError("if offset is a vector, its length must be the number of samples")
        if lib_size is not None:
            warnings.warn("lib_size is ignored in the presence of offset")
        if offset_prior is not None:
            warnings.warn("offset_prior is ignored in the presence of offset")
        lib_size = np.exp(offset)
    else:
        if lib_size is None:
            lib_size = y.sum(axis=0)
        lib_size = np.asarray(lib_size, dtype=np.float64)
        if offset_prior is not None:
            offset_prior = np.asarray(offset_prior, dtype=np.float64)
            if offset_prior.shape != y.shape:
                raise ValueError("counts and offset_prior must have equal dimensions.")
            lib_size = np.exp(np.log(lib_size)[None, :] + offset_prior)

    lib_size = np.asarray(lib_size, dtype=np.float64)
    if np.any(np.isnan(lib_size)):
        raise ValueError("NA library sizes not allowed")
    if np.any(lib_size <= 0):
        raise ValueError("Library sizes should be greater than zero")

    if log:
        out = add_prior_count(y, offset=np.log(lib_size), prior_count=prior_count)
        # As edgeR's calc_cpm_log: (log(y + prior) - log(lib + 2 prior)
        # + log(1e6)) / log(2), NaN where y + prior is not positive
        y_aug = out['y']
        offset_aug = np.asarray(out['offset'], dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (np.log(y_aug) - offset_aug + np.log(1e6)) / np.log(2)
        result[~(y_aug > 0)] = np.nan
        return result
    else:
        # As edgeR's calc_cpm_raw: y * 1e6 / lib.size
        return y * 1e6 / (lib_size[np.newaxis, :] if lib_size.ndim == 1 else lib_size)


def rpkm(y, gene_length=None, lib_size=None, offset=None, log=False, prior_count=2,
         normalized_lib_sizes=True, offset_prior=None):
    """Reads per kilobase per million.

    Port of edgeR's rpkm().

    Parameters
    ----------
    y : array-like or DGEList
        Count matrix or DGEList.
    gene_length : array-like or str, optional
        Gene lengths in bp, or the name of a column of y['genes']. For a
        DGEList the default is the Length, length or AveLength column, or
        a single column with 'length' in its name.
    lib_size, offset, log, prior_count, normalized_lib_sizes, offset_prior :
        As for cpm().
    """
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        genes = dge.get('genes')
        if isinstance(gene_length, str):
            if genes is None or gene_length not in genes.columns:
                raise ValueError("gene_length column not found")
            gene_length = genes[gene_length].values
        elif gene_length is None:
            cols = [] if genes is None else list(genes.columns)
            for col in ['Length', 'length', 'AveLength']:
                if col in cols:
                    gene_length = genes[col].values
                    break
            else:
                matches = [c for c in cols if 'length' in str(c).lower()]
                if len(matches) != 1:
                    raise ValueError("Gene lengths not found")
                gene_length = genes[matches[0]].values
        lib = dge['samples']['lib.size'].values
        offset = dge.get('offset')
        if offset is not None:
            off = offset.as_matrix() if isinstance(offset, CompressedMatrix) else np.asarray(offset)
            if np.min(off) > np.max(np.log(lib)) or np.min(np.log(lib)) > np.max(off):
                warnings.warn("Offset may not reflect library sizes. Scaling offset may be required.")
            lib = None
        elif normalized_lib_sizes:
            lib = lib * dge['samples']['norm.factors'].values
        result = _cpm_default(dge['counts'], lib_size=lib, offset=offset,
                              offset_prior=dge.get('offset.prior'), log=log,
                              prior_count=prior_count)
    else:
        result = _cpm_default(y, lib_size=lib_size, offset=offset, offset_prior=offset_prior,
                              log=log, prior_count=prior_count)

    gene_length_kb = np.asarray(gene_length, dtype=np.float64) / 1000
    if log:
        return result - np.log2(gene_length_kb[:, np.newaxis])
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
    if weights is not None and isinstance(y, dict) and 'counts' in y:
        # Weights given explicitly act as the object's own weights
        y = type(y)(y)
        y['weights'] = weights

    # PCList input: log2 CPM of the total coverage (aveLogCPM.PCList)
    if isinstance(y, dict) and 'counts2' in y and 'coefficients' not in y:
        total = np.asarray(y['counts'], dtype=np.float64) + np.asarray(y['counts2'], dtype=np.float64)
        return _ave_log_cpm_default(total, lib_size=total.sum(axis=0), prior_count=prior_count,
                                    dispersion=0, weights=y.get('weights'))

    # DGEGLM input: the fit's offset, dispersion and weights (aveLogCPM.DGEGLM)
    if isinstance(y, dict) and 'counts' in y and 'coefficients' in y:
        if dispersion is None:
            dispersion = y.get('dispersion')
        if dispersion is None:
            counts = np.asarray(y['counts'], dtype=np.float64)
            if y.get('counts2') is not None:
                counts = counts + np.asarray(y['counts2'], dtype=np.float64)
            return _ave_log_cpm_default(counts, prior_count=prior_count,
                                        dispersion=0, weights=y.get('weights'))
        return _ave_log_cpm_default(y['counts'], offset=y.get('offset'),
                                    prior_count=prior_count, dispersion=dispersion,
                                    weights=y.get('weights'))

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

    # As edgeR: the all-zero short-circuit (aveLogCPM.R:71-76), conjunctive in lib.size
    # and offset; without it the matrix reaches mglm_one_group's -1e8 sentinel.
    if y.size and np.all(y == 0):
        ls_ok = lib_size is None or np.max(np.asarray(lib_size, dtype=np.float64)) == 0
        off_ok = offset is None or np.max(np.asarray(offset, dtype=np.float64)) == -np.inf
        if ls_ok and off_ok:
            abundance = (-np.log(y.shape[0]) + np.log(1e6)) / np.log(2)
            return np.full(y.shape[0], abundance, dtype=np.float64)

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


def cpm_by_group(y, group=None, dispersion=None, offset=None, weights=None,
                 log=False, prior_count=2):
    """Counts per million averaged by group.

    Port of edgeR's cpmByGroup().

    ``dispersion`` defaults to ``None`` and is then resolved as edgeR's
    ``getDispersion(y)`` does -- tagwise, then trended, then common, then 0.05
    (``getDispersion.R:6-17``, ``cpmByGroup.R:13-14``, ``:31``).
    """
    # DGEList input
    if isinstance(y, dict) and 'counts' in y:
        dge = y
        if weights is not None:
            # Weights given explicitly act as the DGEList's own weights
            dge = type(dge)(dge)
            dge['weights'] = weights
        if group is None:
            group = dge['samples']['group'].values
        if offset is None:
            from .dgelist import get_offset
            offset = get_offset(dge)
        if dispersion is None:
            # As edgeR: getDispersion(y) -- tagwise, then trended, then common
            # (getDispersion.R:6-17); 0.05 only where the object carries none.
            from .dgelist import get_dispersion
            dispersion = get_dispersion(dge)
            if dispersion is None:
                dispersion = 0.05
        w = dge.get('weights')
        return _cpm_by_group_default(dge['counts'], group=group, dispersion=dispersion,
                                     offset=offset, weights=w, log=log,
                                     prior_count=prior_count)

    if dispersion is None:
        dispersion = 0.05

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
