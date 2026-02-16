# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
Core data classes for edgePython.

Port of edgeR's S4 classes (DGEList, DGEExact, DGEGLM, DGELRT, TopTags)
as Python dataclasses with dict-like access, subsetting, and display.
"""

import numpy as np
import pandas as pd
from copy import deepcopy


class _EdgeRBase(dict):
    """Base class providing dict-like access, subsetting, and display."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def shape(self):
        if 'counts' in self:
            return self['counts'].shape
        return None

    def __repr__(self):
        cls = type(self).__name__
        components = list(self.keys())
        s = self.shape
        if s is not None:
            return f"{cls} with {s[0]} rows and {s[1]} columns\nComponents: {', '.join(components)}"
        return f"{cls}\nComponents: {', '.join(components)}"

    def _copy(self):
        """Deep copy of the object."""
        return deepcopy(self)

    def head(self, n=5):
        """Show first n rows."""
        if 'table' in self:
            return self['table'].head(n)
        if 'counts' in self:
            return pd.DataFrame(
                self['counts'][:n],
                index=_get_rownames(self)[:n] if _get_rownames(self) is not None else None,
                columns=_get_colnames(self) if _get_colnames(self) is not None else None
            )
        return None

    def tail(self, n=5):
        """Show last n rows."""
        if 'table' in self:
            return self['table'].tail(n)
        if 'counts' in self:
            return pd.DataFrame(
                self['counts'][-n:],
                index=_get_rownames(self)[-n:] if _get_rownames(self) is not None else None,
                columns=_get_colnames(self) if _get_colnames(self) is not None else None
            )
        return None


def _get_rownames(obj):
    """Get row names from genes or counts."""
    if 'genes' in obj and obj['genes'] is not None:
        return list(obj['genes'].index)
    if 'counts' in obj and obj['counts'] is not None:
        c = obj['counts']
        if hasattr(c, 'index'):
            return list(c.index)
    return None


def _get_colnames(obj):
    """Get column names from samples or counts."""
    if 'samples' in obj and obj['samples'] is not None:
        return list(obj['samples'].index)
    return None


def _subset_matrix_or_df(x, i=None, j=None):
    """Subset a matrix, DataFrame, or vector by row (i) and/or column (j)."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        if i is not None and j is not None:
            return x.iloc[i, j]
        elif i is not None:
            return x.iloc[i]
        elif j is not None:
            return x.iloc[:, j]
        return x
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            if i is not None and j is not None:
                return x[np.ix_(np.atleast_1d(i), np.atleast_1d(j))] if not isinstance(i, slice) else x[i, :][:, j]
            elif i is not None:
                return x[i] if isinstance(i, slice) else x[np.atleast_1d(i)]
            elif j is not None:
                return x[:, j] if isinstance(j, slice) else x[:, np.atleast_1d(j)]
        elif x.ndim == 1:
            if i is not None:
                return x[i] if isinstance(i, slice) else x[np.atleast_1d(i)]
        return x
    return x


def _resolve_index(idx, names):
    """Resolve index to integer array. Supports bool, int, str, slice."""
    if idx is None:
        return None
    if isinstance(idx, slice):
        return idx
    idx = np.atleast_1d(idx)
    if idx.dtype == bool:
        return np.where(idx)[0]
    if idx.dtype.kind in ('U', 'S', 'O') and names is not None:
        names_arr = np.asarray(names)
        result = []
        for name in idx:
            matches = np.where(names_arr == name)[0]
            if len(matches) == 0:
                raise KeyError(f"Name '{name}' not found")
            result.append(matches[0])
        return np.array(result)
    return idx.astype(int)


class DGEList(_EdgeRBase):
    """Digital Gene Expression data list.

    Attributes
    ----------
    counts : ndarray
        Matrix of counts (genes x samples).
    samples : DataFrame
        Sample information with columns group, lib.size, norm.factors.
    genes : DataFrame or None
        Gene annotation.
    common.dispersion : float or None
    trended.dispersion : ndarray or None
    tagwise.dispersion : ndarray or None
    AveLogCPM : ndarray or None
    offset : ndarray or None
    weights : ndarray or None
    """

    _IJ = {'counts', 'pseudo.counts', 'offset', 'weights'}
    _IX = {'genes'}
    _JX = {'samples'}
    _I = {'AveLogCPM', 'trended.dispersion', 'tagwise.dispersion', 'prior.n', 'prior.df'}

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
            else:
                raise IndexError("Two subscripts required")
        else:
            raise IndexError("Two subscripts required")

        rownames = _get_rownames(self)
        colnames = _get_colnames(self)
        i_idx = _resolve_index(i, rownames)
        j_idx = _resolve_index(j, colnames)

        out = self._copy()

        for k in self._IJ:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx, j_idx)
        for k in self._IX:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)
        for k in self._JX:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], j=j_idx)
        for k in self._I:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)

        # Drop empty group levels after column subsetting
        if j_idx is not None and 'samples' in out and 'group' in out['samples'].columns:
            out['samples']['group'] = out['samples']['group'].cat.remove_unused_categories() if hasattr(out['samples']['group'], 'cat') else out['samples']['group']

        return out

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    @property
    def nrow(self):
        if 'counts' in self:
            return self['counts'].shape[0]
        return 0

    @property
    def ncol(self):
        if 'counts' in self:
            return self['counts'].shape[1]
        return 0

    def __len__(self):
        return self.nrow

    def dim(self):
        return self.shape

    def dimnames(self):
        return (_get_rownames(self), _get_colnames(self))

    def to_dataframe(self):
        """Convert counts to DataFrame."""
        return pd.DataFrame(
            self['counts'],
            index=_get_rownames(self),
            columns=_get_colnames(self)
        )


class DGEExact(_EdgeRBase):
    """Results of exact test for differential expression.

    Attributes
    ----------
    table : DataFrame
        With columns logFC, logCPM, PValue.
    comparison : list
        Two group names being compared.
    genes : DataFrame or None
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
            else:
                raise IndexError("Two subscripts required")
        else:
            raise IndexError("Two subscripts required (rows, columns)")

        if j is not None:
            raise IndexError("Subsetting columns not allowed for DGEExact objects.")

        out = self._copy()
        rownames = list(out['table'].index) if out.get('table') is not None else None
        i_idx = _resolve_index(i, rownames)

        if 'table' in out and out['table'] is not None:
            out['table'] = _subset_matrix_or_df(out['table'], i_idx)
        if 'genes' in out and out['genes'] is not None:
            out['genes'] = _subset_matrix_or_df(out['genes'], i_idx)
        return out

    def __repr__(self):
        out = ""
        if 'comparison' in self and self['comparison'] is not None:
            out += f"Comparison of groups: {self['comparison'][1]}-{self['comparison'][0]}\n"
        if 'table' in self and self['table'] is not None:
            out += str(self['table'])
        return out

    @property
    def shape(self):
        if 'table' in self:
            return self['table'].shape
        return None


class DGEGLM(_EdgeRBase):
    """Fitted GLM object for DGE data.

    Attributes
    ----------
    coefficients : ndarray
        Matrix of coefficients.
    fitted.values : ndarray
    deviance : ndarray
    counts : ndarray
    offset : ndarray or CompressedMatrix
    weights : ndarray or None
    design : ndarray
    dispersion : float or ndarray
    df.residual : ndarray
    samples : DataFrame
    genes : DataFrame or None
    AveLogCPM : ndarray or None
    """

    _IX = {'counts', 'offset', 'weights', 'genes', 'coefficients', 'fitted.values',
           'unshrunk.coefficients', 'leverage', 'unit.deviance.adj', 'unit.df.adj'}
    _I = {'AveLogCPM', 'dispersion', 'prior.n', 'prior.df', 's2.post', 's2.prior',
          'df.prior', 'df.residual', 'df.residual.zeros', 'df.residual.adj',
          'deviance', 'deviance.adj', 'iter', 'failed'}

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
            else:
                raise IndexError("Two subscripts required")
        else:
            raise IndexError("Two subscripts required")

        if j is not None:
            raise IndexError("Subsetting columns not allowed for DGEGLM object.")

        out = self._copy()
        rownames = _get_rownames(self)
        i_idx = _resolve_index(i, rownames)

        for k in self._IX:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)
        for k in self._I:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)
        return out

    @property
    def nrow(self):
        if 'coefficients' in self:
            return self['coefficients'].shape[0]
        if 'counts' in self:
            return self['counts'].shape[0]
        return 0

    @property
    def ncol(self):
        if 'counts' in self:
            return self['counts'].shape[1]
        return 0


class DGELRT(_EdgeRBase):
    """Likelihood ratio test results for DGE GLMs.

    Attributes
    ----------
    table : DataFrame
        With columns logFC, logCPM, LR (or F), PValue.
    comparison : str
        Name of coefficient tested.
    coefficients, fitted.values, etc. : inherited from DGEGLM fit.
    """

    _IX = {'counts', 'offset', 'weights', 'genes', 'coefficients', 'fitted.values',
           'table', 'unshrunk.coefficients', 'leverage', 'unit.deviance.adj', 'unit.df.adj'}
    _I = {'AveLogCPM', 'dispersion', 'prior.n', 'prior.df', 's2.post', 's2.prior',
          'df.prior', 'df.residual', 'df.residual.zeros', 'df.residual.adj',
          'deviance', 'deviance.adj', 'iter', 'failed', 'df.test', 'df.total'}

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
            else:
                raise IndexError("Two subscripts required")
        else:
            raise IndexError("Two subscripts required")

        if j is not None:
            raise IndexError("Subsetting columns not allowed for DGELRT object.")

        out = self._copy()
        rownames = _get_rownames(self)
        if rownames is None and 'table' in self:
            rownames = list(self['table'].index)
        i_idx = _resolve_index(i, rownames)

        for k in self._IX:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)
        for k in self._I:
            if k in out and out[k] is not None:
                out[k] = _subset_matrix_or_df(out[k], i_idx)
        return out

    def __repr__(self):
        out = ""
        if 'comparison' in self:
            out += f"Coefficient: {self['comparison']}\n"
        if 'table' in self:
            out += str(self['table'])
        return out

    @property
    def shape(self):
        if 'table' in self:
            return self['table'].shape
        if 'coefficients' in self:
            return self['coefficients'].shape
        return None


class TopTags(_EdgeRBase):
    """Top differentially expressed genes.

    Attributes
    ----------
    table : DataFrame
        Sorted table of top genes.
    adjust_method : str
    comparison : str or list
    test : str
        Either 'exact' or 'glm'.
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
            else:
                raise IndexError("Two subscripts required")
        else:
            raise IndexError("Two subscripts required")

        out = self._copy()
        if 'table' in out:
            if i is not None or j is not None:
                out['table'] = _subset_matrix_or_df(out['table'], i, j)
        return out

    def __repr__(self):
        out = ""
        if self.get('test') == 'exact':
            comp = self.get('comparison', [])
            if len(comp) >= 2:
                out += f"Comparison of groups: {comp[1]}-{comp[0]}\n"
        else:
            out += f"Coefficient: {self.get('comparison', '')}\n"
        if 'table' in self:
            out += str(self['table'])
        return out

    @property
    def shape(self):
        if 'table' in self:
            return self['table'].shape
        return None


def cbind_dgelist(*objects):
    """Column-bind (combine samples) DGEList objects.

    Port of edgeR's cbind.DGEList.
    """
    if len(objects) == 1:
        return objects[0]

    out = objects[0]._copy()
    for obj in objects[1:]:
        # Check gene compatibility
        if 'genes' in out and out['genes'] is not None:
            if not out['genes'].equals(obj.get('genes')):
                raise ValueError("DGEList objects have different genes")

        out['counts'] = np.hstack([out['counts'], obj['counts']])
        out['samples'] = pd.concat([out['samples'], obj['samples']], ignore_index=False)

        for key in ['offset', 'weights', 'pseudo.counts']:
            if key in out and out[key] is not None and key in obj and obj[key] is not None:
                out[key] = np.hstack([out[key], obj[key]])

    # Clear dispersions
    for key in ['common.dispersion', 'trended.dispersion', 'tagwise.dispersion']:
        if key in out:
            del out[key]

    return out


def rbind_dgelist(*objects):
    """Row-bind (combine genes) DGEList objects.

    Port of edgeR's rbind.DGEList.
    """
    if len(objects) == 1:
        return objects[0]

    out = objects[0]._copy()
    for obj in objects[1:]:
        out['counts'] = np.vstack([out['counts'], obj['counts']])
        if 'genes' in out and out['genes'] is not None and 'genes' in obj and obj['genes'] is not None:
            out['genes'] = pd.concat([out['genes'], obj['genes']], ignore_index=False)
        for key in ['offset', 'weights', 'pseudo.counts']:
            if key in out and out[key] is not None and key in obj and obj[key] is not None:
                out[key] = np.vstack([out[key], obj[key]])

    # Clear dispersions
    for key in ['common.dispersion', 'trended.dispersion', 'tagwise.dispersion', 'AveLogCPM']:
        if key in out:
            del out[key]

    return out
