"""SparseMatrixBuilder — accumulates sparse (i, j, v) triples for assembly.

Provides a thin Python interface over scipy.sparse COO format, with
efficient incremental construction and format conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


class SparseMatrixBuilder:
    """Accumulates sparse (row, col, val) triples and produces scipy sparse matrices.

    Usage::

        builder = SparseMatrixBuilder()
        builder.set(0, 1, 3.0)
        builder.set(1, 2, -1.0)
        A = builder.to_scipy(format="csr")
    """

    __slots__ = ("_rows", "_cols", "_vals", "_nrows", "_ncols")

    def __init__(
        self,
        nrows: int = 0,
        ncols: int = 0,
    ) -> None:
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self._nrows = nrows
        self._ncols = ncols

    # -- dimension management ------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def ensure_shape(self, nrows: int, ncols: int) -> None:
        """Set or expand matrix shape."""
        self._nrows = max(self._nrows, nrows)
        self._ncols = max(self._ncols, ncols)

    @property
    def shape(self) -> tuple[int, int]:
        self._update_shape()
        return (self._nrows, self._ncols)

    # -- entry operations ----------------------------------------------------

    def set(self, row: int, col: int, val: float) -> None:
        """Set entry (overwrites any previous value at (row, col))."""
        self._rows.append(row)
        self._cols.append(col)
        self._vals.append(val)

    def add(self, row: int, col: int, val: float) -> None:
        """Add *val* to entry (row, col); creates if not present."""
        self._rows.append(row)
        self._cols.append(col)
        self._vals.append(val)

    def extend(
        self, rows: list[int], cols: list[int], vals: list[float]
    ) -> None:
        """Batch-append triplets (all lists must have equal length)."""
        if not rows:
            return
        if len(rows) != len(cols) or len(rows) != len(vals):
            raise ValueError(
                f"extend: mismatched lengths {len(rows)}, {len(cols)}, {len(vals)}"
            )
        self._rows.extend(rows)
        self._cols.extend(cols)
        self._vals.extend(vals)

    # -- shape tracking from triplets ----------------------------------------

    def _update_shape(self) -> None:
        if self._rows:
            self._nrows = max(self._nrows, max(self._rows) + 1)
            self._ncols = max(self._ncols, max(self._cols) + 1)

    # -- output formats ------------------------------------------------------

    def to_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (row_indices, col_indices, values) as NumPy arrays."""
        self._update_shape()
        import numpy as np

        return (
            np.array(self._rows, dtype=np.int32),
            np.array(self._cols, dtype=np.int32),
            np.array(self._vals, dtype=np.float64),
        )

    def to_scipy(self, format: str = "coo") -> sp.spmatrix:
        """Return a scipy sparse matrix in the requested format."""
        import numpy as np
        import scipy.sparse as sp

        rows, cols, vals = self.to_coo()
        return sp.coo_matrix(
            (vals, (rows, cols)), shape=(self._nrows, self._ncols)
        ).asformat(format)

    def to_scipy_dense(self) -> np.ndarray:
        """Return a dense NumPy array (C-contiguous, row-major)."""
        return self.to_scipy("csr").toarray()

    def to_triplets(self) -> list[tuple[int, int, float]]:
        """Return raw (row, col, val) Python tuples."""
        return list(zip(self._rows, self._cols, self._vals))

    # -- algebraic operations ------------------------------------------------

    def __add__(self, other: SparseMatrixBuilder) -> SparseMatrixBuilder:
        """Element-wise addition (triplets concatenated; duplicates summed by scipy)."""
        out = SparseMatrixBuilder(max(self._nrows, other._nrows), max(self._ncols, other._ncols))
        out._rows = self._rows + other._rows
        out._cols = self._cols + other._cols
        out._vals = self._vals + other._vals
        return out

    def __mul__(self, scalar: float) -> SparseMatrixBuilder:
        """Scalar multiplication."""
        out = SparseMatrixBuilder(self._nrows, self._ncols)
        out._rows = list(self._rows)
        out._cols = list(self._cols)
        out._vals = [v * scalar for v in self._vals]
        return out

    def __rmul__(self, scalar: float) -> SparseMatrixBuilder:
        return self * scalar

    def __neg__(self) -> SparseMatrixBuilder:
        out = SparseMatrixBuilder(self._nrows, self._ncols)
        out._rows = list(self._rows)
        out._cols = list(self._cols)
        out._vals = [-v for v in self._vals]
        return out

    def __repr__(self) -> str:
        return f"SparseMatrixBuilder(shape={self.shape}, nnz={len(self)})"


def build_matrix_from_expr(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    nrows: int,
    ncols: int,
) -> sp.coo_matrix:
    """Convenience: build a scipy COO matrix from COO-style arrays.

    Coalesces duplicate entries during construction.
    """
    import numpy as np
    import scipy.sparse as sp

    vals_arr = np.asarray(vals, dtype=np.float64)
    rows_arr = np.asarray(rows, dtype=np.int32)
    cols_arr = np.asarray(cols, dtype=np.int32)
    return sp.coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(nrows, ncols))


def build_symmetric_P(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    n: int,
) -> sp.coo_matrix:
    """Build a symmetric P matrix from upper-triangular entries.

    Given (i, j, v) with i <= j, produces both (i,j,v) and (j,i,v).
    Diagonal entries are doubled so that ``0.5 * x^T P x`` reproduces the
    original expression (each solver backend expects P in that convention;
    a diagonal term only appears once in x^T P x's expansion, so it must
    carry 2x the written coefficient to survive the leading 0.5).
    """
    rows_a = list(rows)
    cols_a = list(cols)
    vals_a = [2.0 * v if i == j else v for i, j, v in zip(rows, cols, vals)]

    for i, j, v in zip(rows, cols, vals):
        if i != j:
            rows_a.append(j)
            cols_a.append(i)
            vals_a.append(v)

    import numpy as np
    import scipy.sparse as sp

    return sp.coo_matrix(
        (np.array(vals_a, dtype=np.float64),
         (np.array(rows_a, dtype=np.int32),
          np.array(cols_a, dtype=np.int32))),
        shape=(n, n),
    )


def build_constraints_matrix(
    constraints,
    n_vars: int,
) -> tuple[sp.coo_matrix, np.ndarray, np.ndarray, np.ndarray, list]:
    """Build constraint matrix A and bounds from a list of Constraint objects.

    Returns:
        (A_coo, l, u, sense, row_constraints) where:
        - A_coo: sparse constraint matrix
        - l: lower bounds on Ax
        - u: upper bounds on Ax
        - sense: sense vector for IPM (=1, >=-1, <=0)
        - row_constraints: the Constraint object backing each row of A, in
          row order (constraints with an all-zero body are skipped and so
          do not appear here — row i's Constraint is row_constraints[i]).
    """
    import numpy as np
    import scipy.sparse as sp

    all_rows: list[int] = []
    all_cols: list[int] = []
    all_vals: list[float] = []
    l_parts: list[float] = []
    u_parts: list[float] = []
    sense_parts: list[float] = []
    row_constraints: list = []

    row_idx = 0
    for c in constraints:
        rows, cols, vals = c.body.to_linear_arrays(n_vars)
        if not rows:
            continue

        all_rows.extend([row_idx] * len(cols))
        all_cols.extend(cols)
        all_vals.extend(vals)
        row_idx += 1
        row_constraints.append(c)

        bound = c.effective_bound
        if c.sense == "<=":
            l_parts.append(-np.inf)
            u_parts.append(bound)
            sense_parts.append(0.0)  # IPM: <=
        elif c.sense == ">=":
            l_parts.append(bound)
            u_parts.append(np.inf)
            sense_parts.append(-1.0)  # IPM: >=
        elif c.sense == "==":
            l_parts.append(bound)
            u_parts.append(bound)
            sense_parts.append(1.0)  # IPM: ==

    A = sp.coo_matrix(
        (np.array(all_vals, dtype=np.float64),
         (np.array(all_rows, dtype=np.int32), np.array(all_cols, dtype=np.int32))),
        shape=(row_idx, n_vars),
    )
    l = np.array(l_parts, dtype=np.float64)
    u = np.array(u_parts, dtype=np.float64)
    sense = np.array(sense_parts, dtype=np.float64)

    return A, l, u, sense, row_constraints


def split_constraints_for_piqp(
    constraints,
    n_vars: int,
) -> tuple[sp.coo_matrix | None, np.ndarray | None, sp.coo_matrix | None, np.ndarray | None]:
    """Split constraints into equality (A, b) and inequality (G, h) for PIQP.

    Returns:
        (A_eq, b_eq, G_ineq, h_ineq)
    """
    import numpy as np
    import scipy.sparse as sp

    eq_rows: list[tuple[int, int, float]] = []
    ineq_rows: list[tuple[int, int, float]] = []
    b_eq: list[float] = []
    h_ineq: list[float] = []

    for c in constraints:
        rows, cols, vals = c.body.to_linear_arrays(n_vars)
        if not rows:
            continue

        if c.sense == "==":
            row_idx = len(b_eq)
            for col, val in zip(cols, vals):
                eq_rows.append((row_idx, col, val))
            b_eq.append(c.effective_bound)
        elif c.sense == "<=":
            row_idx = len(h_ineq)
            for col, val in zip(cols, vals):
                ineq_rows.append((row_idx, col, val))
            h_ineq.append(c.effective_bound)
        elif c.sense == ">=":
            row_idx = len(h_ineq)
            for col, val in zip(cols, vals):
                ineq_rows.append((row_idx, col, -val))
            h_ineq.append(-c.effective_bound)

    if eq_rows:
        data_eq = np.array([v for _, _, v in eq_rows])
        rows_eq = np.array([r for r, _, _ in eq_rows])
        cols_eq = np.array([c for _, c, _ in eq_rows])
        A_eq = sp.coo_matrix(
            (data_eq, (rows_eq, cols_eq)),
            shape=(len(b_eq), n_vars),
        )
    else:
        A_eq = None
        b_eq = None

    if ineq_rows:
        data_ineq = np.array([v for _, _, v in ineq_rows])
        rows_ineq = np.array([r for r, _, _ in ineq_rows])
        cols_ineq = np.array([c for _, c, _ in ineq_rows])
        G_ineq = sp.coo_matrix(
            (data_ineq, (rows_ineq, cols_ineq)),
            shape=(len(h_ineq), n_vars),
        )
    else:
        G_ineq = None
        h_ineq = None

    b_eq_out = np.array(b_eq, dtype=np.float64) if b_eq is not None else None
    h_ineq_out = np.array(h_ineq, dtype=np.float64) if h_ineq is not None else None
    return A_eq, b_eq_out, G_ineq, h_ineq_out


def split_constraints_for_proxqp(
    constraints,
    n_vars: int,
) -> tuple[sp.coo_matrix | None, np.ndarray | None, sp.coo_matrix | None, np.ndarray | None, np.ndarray | None]:
    """Split constraints into equality (A, b) and inequality (C, l, u) for ProxQP.

    ProxQP problem form:
        minimize  ½ xᵀHx + gᵀx
        s.t.      Ax = b
                  l ≤ Cx ≤ u

    Returns:
        (A_eq, b_eq, C_ineq, l_ineq, u_ineq)
    """
    import numpy as np
    import scipy.sparse as sp

    eq_rows: list[tuple[int, int, float]] = []
    ineq_rows: list[tuple[int, int, float]] = []
    b_eq: list[float] = []
    l_ineq: list[float] = []
    u_ineq: list[float] = []

    for c in constraints:
        rows, cols, vals = c.body.to_linear_arrays(n_vars)
        if not rows:
            continue

        if c.sense == "==":
            row_idx = len(b_eq)
            for col, val in zip(cols, vals):
                eq_rows.append((row_idx, col, val))
            b_eq.append(c.effective_bound)
        elif c.sense == "<=":
            row_idx = len(l_ineq)
            for col, val in zip(cols, vals):
                ineq_rows.append((row_idx, col, val))
            l_ineq.append(-np.inf)
            u_ineq.append(c.effective_bound)
        elif c.sense == ">=":
            row_idx = len(l_ineq)
            for col, val in zip(cols, vals):
                ineq_rows.append((row_idx, col, val))
            l_ineq.append(c.effective_bound)
            u_ineq.append(np.inf)

    if eq_rows:
        data_eq = np.array([v for _, _, v in eq_rows])
        rows_eq = np.array([r for r, _, _ in eq_rows])
        cols_eq = np.array([c for _, c, _ in eq_rows])
        A_eq = sp.coo_matrix(
            (data_eq, (rows_eq, cols_eq)),
            shape=(len(b_eq), n_vars),
        )
    else:
        A_eq = None
        b_eq = np.array([], dtype=np.float64)

    if ineq_rows:
        data_ineq = np.array([v for _, _, v in ineq_rows])
        rows_ineq = np.array([r for r, _, _ in ineq_rows])
        cols_ineq = np.array([c for _, c, _ in ineq_rows])
        C_ineq = sp.coo_matrix(
            (data_ineq, (rows_ineq, cols_ineq)),
            shape=(len(l_ineq), n_vars),
        )
    else:
        C_ineq = None
        l_ineq = None
        u_ineq = None

    b_eq_out = np.array(b_eq, dtype=np.float64) if A_eq is not None else None
    l_ineq_out = np.array(l_ineq, dtype=np.float64) if l_ineq is not None else None
    u_ineq_out = np.array(u_ineq, dtype=np.float64) if u_ineq is not None else None
    return A_eq, b_eq_out, C_ineq, l_ineq_out, u_ineq_out
