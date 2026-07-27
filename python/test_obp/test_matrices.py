"""Tests for SparseMatrixBuilder."""

import numpy as np
import pytest
import scipy.sparse as sp

from solvers.obp.matrices import SparseMatrixBuilder, build_matrix_from_expr, build_symmetric_P


class TestSparseMatrixBuilder:
    def test_basic_set(self):
        b = SparseMatrixBuilder()
        b.set(0, 1, 3.0)
        b.set(1, 2, -1.0)
        assert len(b) == 2
        assert b.shape == (2, 3)

    def test_add_entry(self):
        b = SparseMatrixBuilder()
        b.add(0, 0, 1.0)
        b.add(0, 0, 2.0)  # duplicate
        assert len(b) == 2
        rows, cols, vals = b.to_coo()
        assert np.allclose(vals, [1.0, 2.0])

    def test_extend(self):
        b = SparseMatrixBuilder()
        b.extend([0, 1], [1, 2], [3.0, 4.0])
        assert len(b) == 2
        rows, cols, vals = b.to_coo()
        assert np.array_equal(rows, [0, 1])
        assert np.array_equal(cols, [1, 2])
        assert np.allclose(vals, [3.0, 4.0])

    def test_extend_empty(self):
        b = SparseMatrixBuilder()
        b.extend([], [], [])
        assert len(b) == 0

    def test_extend_length_mismatch(self):
        b = SparseMatrixBuilder()
        with pytest.raises(ValueError):
            b.extend([0, 1], [1], [3.0, 4.0])

    def test_to_scipy(self):
        b = SparseMatrixBuilder()
        b.set(0, 1, 3.0)
        b.set(1, 2, -1.0)
        M = b.to_scipy("csr")
        assert M.shape == (2, 3)
        assert M[0, 1] == 3.0
        assert M[1, 2] == -1.0

    def test_to_scipy_dense(self):
        b = SparseMatrixBuilder()
        b.set(0, 0, 1.0)
        b.set(1, 1, 2.0)
        M = b.to_scipy_dense()
        assert M.shape == (2, 2)
        np.testing.assert_array_equal(M, np.eye(2) * [1, 2])

    def test_add_matrices(self):
        b1 = SparseMatrixBuilder()
        b1.set(0, 0, 1.0)
        b2 = SparseMatrixBuilder()
        b2.set(0, 0, 2.0)
        b3 = b1 + b2
        M = b3.to_scipy_dense()
        assert M[0, 0] == 3.0

    def test_mul_scalar(self):
        b = SparseMatrixBuilder()
        b.set(0, 0, 1.0)
        b2 = b * 5.0
        M = b2.to_scipy_dense()
        assert M[0, 0] == 5.0

    def test_neg(self):
        b = SparseMatrixBuilder()
        b.set(0, 0, 1.0)
        b2 = -b
        M = b2.to_scipy_dense()
        assert M[0, 0] == -1.0

    def test_shape_tracking(self):
        b = SparseMatrixBuilder()
        b.set(10, 20, 1.0)
        assert b.shape == (11, 21)

    def test_bool(self):
        b = SparseMatrixBuilder()
        assert not b
        b.set(0, 0, 1.0)
        assert b


class TestBuildFunctions:
    def test_build_matrix_from_expr(self):
        M = build_matrix_from_expr(
            rows=[0, 1], cols=[1, 2], vals=[3.0, 4.0],
            nrows=3, ncols=3,
        )
        assert isinstance(M, sp.coo_matrix)
        dense = M.toarray()
        assert dense[0, 1] == 3.0
        assert dense[1, 2] == 4.0

    def test_build_symmetric_P(self):
        # Upper-triangular input: (0,0,1), (0,1,2), (1,1,3)
        M = build_symmetric_P(
            rows=[0, 0, 1],
            cols=[0, 1, 1],
            vals=[1.0, 2.0, 3.0],
            n=2,
        )
        # Should be symmetric with doubled diagonal: [[2, 2], [2, 6]]
        # (0.5 * x^T P x must reproduce 1*x0^2 + 2*x0*x1 + 3*x1^2)
        dense = M.toarray()
        assert dense[0, 0] == 2.0
        assert dense[0, 1] == 2.0
        assert dense[1, 0] == 2.0
        assert dense[1, 1] == 6.0

    def test_build_symmetric_P_diagonal(self):
        M = build_symmetric_P(
            rows=[0, 1],
            cols=[0, 1],
            vals=[1.0, 2.0],
            n=2,
        )
        # Diagonal entries are doubled so 0.5*x^T P x reproduces 1*x0^2 + 2*x1^2
        dense = M.toarray()
        assert dense[0, 0] == 2.0
        assert dense[1, 1] == 4.0
        assert dense[0, 1] == 0.0
        assert dense[1, 0] == 0.0
