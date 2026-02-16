# This code was written by Claude (Anthropic). The project was directed by Lior Pachter.
"""
CompressedMatrix class for memory-efficient matrix storage.

Port of edgeR's CompressedMatrix (makeCompressedMatrix.R).
Stores scalars, row vectors, or column vectors with flags indicating
which dimensions should be repeated when expanding to a full matrix.
"""

import numpy as np


class CompressedMatrix:
    """Memory-efficient matrix that stores repeated rows/columns compactly.

    A CompressedMatrix stores a scalar, row vector, column vector, or full
    matrix along with flags indicating which dimensions are repeated.
    This avoids materializing large matrices when the same values are
    repeated across rows or columns.

    Parameters
    ----------
    x : scalar, 1-D array, or 2-D array
        The data to store.
    dims : tuple of (int, int), optional
        The logical dimensions (nrow, ncol) of the full matrix.
    byrow : bool
        If True (default), a 1-D vector is treated as a row to be repeated
        down rows. If False, treated as a column to be repeated across columns.
    """

    def __init__(self, x, dims=None, byrow=True):
        x = np.asarray(x, dtype=np.float64)
        self.repeat_row = False
        self.repeat_col = False

        if x.ndim == 2:
            if dims is not None:
                xr, xc = x.shape
                if xr == 1 and xc == 1:
                    self.repeat_row = True
                    self.repeat_col = True
                    self._data = x.reshape(1, 1)
                elif xr == 1 and xc >= 2:
                    if xc != dims[1]:
                        raise ValueError("dims[1] should equal length of row vector x")
                    if byrow:
                        self.repeat_row = True
                        self._data = x.reshape(1, xc)
                    else:
                        self._data = x.reshape(1, xc)
                        dims = (xr, xc)
                elif xr >= 2 and xc == 1:
                    if xr != dims[0]:
                        raise ValueError("dims[0] should equal length of column vector x")
                    if not byrow:
                        self.repeat_col = True
                        self._data = x.reshape(xr, 1)
                    else:
                        self._data = x.reshape(xr, 1)
                        dims = (xr, xc)
                else:
                    self._data = x
                    dims = x.shape
            else:
                self._data = x
                dims = x.shape
        elif x.ndim <= 1:
            x = x.ravel()
            if x.size == 1:
                self.repeat_row = True
                self.repeat_col = True
                self._data = x.reshape(1, 1)
                if dims is None:
                    dims = (1, 1)
            else:
                if dims is None:
                    raise ValueError("dims must be provided for vector input")
                if not byrow:
                    if dims[0] != x.size:
                        raise ValueError("dims[0] should equal length of x")
                    self._data = x.reshape(-1, 1)
                    self.repeat_col = True
                else:
                    if dims[1] != x.size:
                        raise ValueError("dims[1] should equal length of x")
                    self._data = x.reshape(1, -1)
                    self.repeat_row = True
        else:
            raise ValueError("x must be scalar, 1-D, or 2-D")

        self._dims = (int(dims[0]), int(dims[1]))

    @property
    def shape(self):
        """Logical dimensions of the full matrix."""
        return self._dims

    @property
    def nrow(self):
        return self._dims[0]

    @property
    def ncol(self):
        return self._dims[1]

    def __len__(self):
        return self._dims[0] * self._dims[1]

    def as_matrix(self):
        """Expand to a full numpy matrix."""
        nr, nc = self._dims
        if self.repeat_row and self.repeat_col:
            return np.tile(self._data, (nr, nc))[:nr, :nc]
        elif self.repeat_row:
            return np.tile(self._data, (nr, 1))[:nr, :nc]
        elif self.repeat_col:
            return np.tile(self._data, (1, nc))[:nr, :nc]
        else:
            return self._data.copy()

    def __array__(self, dtype=None):
        result = self.as_matrix()
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                i, j = key
                raw = self._data.copy()

                if not self.repeat_row and i is not None:
                    if isinstance(i, slice):
                        raw = raw[i, :]
                    else:
                        i_idx = np.arange(self._dims[0])[i]
                        i_idx = np.atleast_1d(i_idx)
                        raw = raw[i_idx, :]
                if not self.repeat_col and j is not None:
                    if isinstance(j, slice):
                        raw = raw[:, j]
                    else:
                        j_idx = np.arange(self._dims[1])[j]
                        j_idx = np.atleast_1d(j_idx)
                        raw = raw[:, j_idx]

                # Compute new dims
                nr = self._dims[0]
                if i is not None:
                    ref = np.arange(nr)
                    nr = len(ref[i]) if not np.isscalar(ref[i]) else 1
                nc = self._dims[1]
                if j is not None:
                    ref = np.arange(nc)
                    nc = len(ref[j]) if not np.isscalar(ref[j]) else 1

                result = CompressedMatrix.__new__(CompressedMatrix)
                result._data = raw
                result._dims = (nr, nc)
                result.repeat_row = self.repeat_row
                result.repeat_col = self.repeat_col

                # Drop to vector if single row or column
                i_scalar = (isinstance(i, (int, np.integer)) or
                            (hasattr(i, '__len__') and len(i) == 1))
                j_scalar = (isinstance(j, (int, np.integer)) or
                            (hasattr(j, '__len__') and len(j) == 1))
                if i_scalar or j_scalar:
                    return result.as_matrix().ravel()
                return result
            else:
                return self.as_matrix()[key]
        else:
            return self.as_matrix().ravel()[key]

    def __setitem__(self, key, value):
        full = self.as_matrix()
        if isinstance(value, CompressedMatrix):
            value = value.as_matrix()
        if isinstance(key, tuple):
            full[key] = value
        else:
            full.ravel()[key] = value
        new = CompressedMatrix(full)
        self._data = new._data
        self._dims = new._dims
        self.repeat_row = new.repeat_row
        self.repeat_col = new.repeat_col

    def _binary_op(self, other, op):
        if isinstance(other, CompressedMatrix):
            if self._dims != other._dims:
                raise ValueError("CompressedMatrix dimensions should be equal for binary operations")
            row_rep = self.repeat_row and other.repeat_row
            col_rep = self.repeat_col and other.repeat_col
            if row_rep or col_rep:
                e1 = self._data.ravel()
                e2 = other._data.ravel()
                outcome = op(e1, e2)
                return CompressedMatrix(outcome, self._dims, byrow=row_rep)
            else:
                return CompressedMatrix(op(self.as_matrix(), other.as_matrix()))
        else:
            other_arr = np.asarray(other, dtype=np.float64)
            if other_arr.ndim <= 1 and other_arr.size == 1:
                other_cm = CompressedMatrix(other_arr, self._dims, byrow=False)
            elif other_arr.ndim == 1:
                other_cm = CompressedMatrix(other_arr, self._dims, byrow=False)
            else:
                other_cm = CompressedMatrix(other_arr, self._dims, byrow=False)
            return self._binary_op(other_cm, op)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: np.add(b, a))

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: np.subtract(b, a))

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: np.multiply(b, a))

    def __truediv__(self, other):
        return self._binary_op(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: np.true_divide(b, a))

    def __pow__(self, other):
        return self._binary_op(other, np.power)

    def __neg__(self):
        result = CompressedMatrix.__new__(CompressedMatrix)
        result._data = -self._data
        result._dims = self._dims
        result.repeat_row = self.repeat_row
        result.repeat_col = self.repeat_col
        return result

    def __repr__(self):
        return (f"CompressedMatrix(shape={self._dims}, "
                f"repeat_row={self.repeat_row}, repeat_col={self.repeat_col}, "
                f"stored_shape={self._data.shape})")

    @staticmethod
    def rbind(*matrices):
        """Row-bind CompressedMatrix objects."""
        if len(matrices) == 1:
            return matrices[0]
        all_nr = sum(m.nrow for m in matrices)
        col_rep = [m.repeat_col for m in matrices]
        row_rep = [m.repeat_row for m in matrices]

        if all(col_rep):
            all_nc = matrices[0].ncol
            collected = []
            for m in matrices:
                if m.ncol != all_nc:
                    raise ValueError("cannot combine CompressedMatrix objects with different number of columns")
                collected.append(np.tile(m._data.ravel(), max(1, m.nrow // max(1, m._data.shape[0])))[:m.nrow])
            return CompressedMatrix(np.concatenate(collected), dims=(all_nr, all_nc), byrow=False)

        if all(row_rep):
            ref = matrices[0]._data
            ok = all(np.allclose(m._data, ref) for m in matrices[1:])
            if ok:
                result = CompressedMatrix.__new__(CompressedMatrix)
                result._data = matrices[0]._data.copy()
                result._dims = (all_nr, matrices[0].ncol)
                result.repeat_row = True
                result.repeat_col = matrices[0].repeat_col
                return result

        expanded = [m.as_matrix() for m in matrices]
        return CompressedMatrix(np.vstack(expanded))

    @staticmethod
    def cbind(*matrices):
        """Column-bind CompressedMatrix objects."""
        if len(matrices) == 1:
            return matrices[0]
        all_nc = sum(m.ncol for m in matrices)
        col_rep = [m.repeat_col for m in matrices]
        row_rep = [m.repeat_row for m in matrices]

        if all(row_rep):
            all_nr = matrices[0].nrow
            collected = []
            for m in matrices:
                if m.nrow != all_nr:
                    raise ValueError("cannot combine CompressedMatrix objects with different number of rows")
                collected.append(np.tile(m._data.ravel(), max(1, m.ncol // max(1, m._data.shape[1])))[:m.ncol])
            return CompressedMatrix(np.concatenate(collected), dims=(all_nr, all_nc), byrow=True)

        if all(col_rep):
            ref = matrices[0]._data
            ok = all(np.allclose(m._data, ref) for m in matrices[1:])
            if ok:
                result = CompressedMatrix.__new__(CompressedMatrix)
                result._data = matrices[0]._data.copy()
                result._dims = (matrices[0].nrow, all_nc)
                result.repeat_row = matrices[0].repeat_row
                result.repeat_col = True
                return result

        expanded = [m.as_matrix() for m in matrices]
        return CompressedMatrix(np.hstack(expanded))


def compress_offsets(y, offset=None, lib_size=None):
    """Compress offsets into a CompressedMatrix.

    Port of edgeR's .compressOffsets.
    """
    if isinstance(offset, CompressedMatrix):
        return offset
    dims = y.shape if hasattr(y, 'shape') else (y._dims if isinstance(y, CompressedMatrix) else None)
    if offset is None:
        if lib_size is None:
            if isinstance(y, np.ndarray):
                lib_size = y.sum(axis=0)
            else:
                lib_size = np.asarray(y).sum(axis=0)
        offset = np.log(lib_size)
    offset = np.asarray(offset, dtype=np.float64)
    if not np.all(np.isfinite(offset)):
        raise ValueError("offsets must be finite values")
    return CompressedMatrix(offset, dims, byrow=True)


def compress_weights(y, weights=None):
    """Compress weights into a CompressedMatrix.

    Port of edgeR's .compressWeights.
    """
    if isinstance(weights, CompressedMatrix):
        return weights
    dims = y.shape if hasattr(y, 'shape') else (y._dims if isinstance(y, CompressedMatrix) else None)
    if weights is None:
        weights = 1.0
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(np.isnan(weights)):
        raise ValueError("NA weights not allowed")
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive")
    return CompressedMatrix(weights, dims, byrow=True)


def compress_prior(y, prior_count):
    """Compress prior counts into a CompressedMatrix.

    Port of edgeR's .compressPrior.
    """
    if isinstance(prior_count, CompressedMatrix):
        return prior_count
    dims = y.shape if hasattr(y, 'shape') else (y._dims if isinstance(y, CompressedMatrix) else None)
    prior_count = np.asarray(prior_count, dtype=np.float64)
    if np.any(np.isnan(prior_count)):
        raise ValueError("NA prior counts not allowed")
    if np.any(prior_count < 0):
        raise ValueError("Negative prior counts not allowed")
    return CompressedMatrix(prior_count, dims, byrow=False)


def compress_dispersions(y, dispersion):
    """Compress dispersions into a CompressedMatrix.

    Port of edgeR's .compressDispersions.
    """
    if isinstance(dispersion, CompressedMatrix):
        return dispersion
    dims = y.shape if hasattr(y, 'shape') else (y._dims if isinstance(y, CompressedMatrix) else None)
    dispersion = np.asarray(dispersion, dtype=np.float64)
    if np.any(np.isnan(dispersion)):
        raise ValueError("NA dispersions not allowed")
    if np.any(dispersion < 0):
        raise ValueError("Negative dispersions not allowed")
    return CompressedMatrix(dispersion, dims, byrow=False)
