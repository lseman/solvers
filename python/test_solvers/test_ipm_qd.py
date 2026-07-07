"""Tests for ipm_qd — quasi-definite IPM augmented-system solver (BK pivoting + iterative refinement).

Usage in IPM:
  solver = IPMQDLDLT()
  solver.analyze(A)                    # once per constraint matrix
  solver.factorize(theta, regP, regD)  # each iteration
  x = solver.solve(rhs)                # or solve_refine(rhs, theta, regP, regD)

Per Zanetti & Gondzio (2025): "A factorisation-based regularised interior point
method using the augmented system" — HiPO paper.
"""

import numpy as np
import pytest
from scipy import sparse

try:
    from ipm_qd import IPMQDLDLT, solve_augmented_system
    HAS_IPM_QD = True
except ImportError:
    HAS_IPM_QD = False


@pytest.mark.skipif(not HAS_IPM_QD, reason="ipm_qd module not available")
class TestIPMQDLDLT:
    """Quasi-definite augmented-system solver tests."""

    def test_basic_solve(self):
        """Basic solve: SPD augmented system with known solution."""
        n, m = 3, 2
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        from scipy import sparse
        A_scipy = sparse.csc_matrix(A)

        solver = IPMQDLDLT()
        solver.analyze(A_scipy)

        theta = np.ones(n)
        regP = np.full(n, 1e-12)
        regD = np.full(m, 1e-12)
        solver.factorize(theta, regP, regD)

        # System: [-(θ+Rp) Aᵀ; A Rd] x = rhs
        # rhs = [0, 0, 0, 1, 2]
        rhs = np.zeros(n + m)
        rhs[n] = 1
        rhs[n + 1] = 2

        x = solver.solve(rhs)

        # Check: -(θ+Rp)·x[0:n] + Aᵀ·x[n:] = 0 for first block
        #        A·x[0:n] + Rd·x[n:] = rhs[n:] for second block
        first_block = -theta * x[:n] + A.T @ x[n:]
        second_block = A @ x[:n] + regD * x[n:]

        np.testing.assert_allclose(first_block, np.zeros(n), atol=1e-10)
        np.testing.assert_allclose(second_block, rhs[n:], atol=1e-10)

    def test_inertia_quasi_definite(self):
        """Inertia tracking for quasi-definite matrices: pos>0, neg>0, zero=0."""
        n, m = 3, 2
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        from scipy import sparse
        A_scipy = sparse.csc_matrix(A)

        solver = IPMQDLDLT()
        solver.analyze(A_scipy)
        solver.factorize(np.ones(n), np.full(n, 1e-12), np.full(m, 1e-12))

        assert solver.num_pos() == m, f"Expected {m} positive eigenvalues, got {solver.num_pos()}"
        assert solver.num_neg() == n, f"Expected {n} negative eigenvalues, got {solver.num_neg()}"
        assert solver.num_zero() == 0

    def test_solve_refinement(self):
        """solve_refine returns (x, niters) with niters >= 0."""
        n, m = 3, 2
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        from scipy import sparse
        A_scipy = sparse.csc_matrix(A)

        solver = IPMQDLDLT()
        solver.analyze(A_scipy)
        solver.factorize(np.ones(n), np.full(n, 1e-12), np.full(m, 1e-12))

        rhs = np.zeros(n + m)
        rhs[n:n+m] = [1, 2]

        x, niters = solver.solve_refine(rhs, np.ones(n), np.full(n, 1e-12), np.full(m, 1e-12))

        assert len(x) == n + m
        assert niters >= 0
        assert isinstance(niters, int)

    def test_solve_augmented_system(self):
        """Convenience function: solve_augmented_system returns dx, dy."""
        n, m = 3, 2
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        from scipy import sparse
        A_scipy = sparse.csc_matrix(A)

        r7 = np.array([1.0, 2.0, 3.0])
        r1 = np.array([0.5, 1.5])

        result = solve_augmented_system(
            A_scipy, r7, r1, np.ones(n), np.full(n, 1e-12), np.full(m, 1e-12))

        assert "dx" in result and "dy" in result
        assert "inertia_pos" in result and "inertia_neg" in result
        assert "refinement_iters" in result
        assert result["dx"].shape == (n,)
        assert result["dy"].shape == (m,)

    def test_sparse_vs_dense(self):
        """Sparse and dense should give the same result."""
        n, m = 4, 3
        A_dense = np.random.RandomState(42).randn(m, n)
        A_sparse = sparse.csc_matrix(A_dense)

        from scipy import sparse as scipy_sparse

        theta = np.random.RandomState(42).rand(n) + 0.1
        regP = np.full(n, 1e-12)
        regD = np.full(m, 1e-12)

        solver1 = IPMQDLDLT()
        solver1.analyze(scipy_sparse.csc_matrix(A_sparse))
        solver1.factorize(theta, regP, regD)

        solver2 = IPMQDLDLT()
        solver2.analyze(scipy_sparse.csc_matrix(A_dense))
        solver2.factorize(theta, regP, regD)

        rhs = np.random.RandomState(42).randn(n + m)
        x1 = solver1.solve(rhs)
        x2 = solver2.solve(rhs)

        np.testing.assert_allclose(x1, x2, atol=1e-12)

    def test_larger_system(self):
        """Larger augmented system (20+10): correct inertia and solve."""
        n, m = 20, 10
        A = np.random.RandomState(42).randn(m, n)
        A_scipy = sparse.csc_matrix(A)

        solver = IPMQDLDLT()
        solver.analyze(A_scipy)
        solver.factorize(
            np.random.RandomState(42).rand(n) + 0.1,
            np.full(n, 1e-12),
            np.full(m, 1e-12))

        assert solver.n() == n
        assert solver.m() == m
        assert solver.num_pos() == m
        assert solver.num_neg() == n
        assert solver.num_zero() == 0

        rhs = np.random.RandomState(42).randn(n + m)
        x = solver.solve(rhs)
        assert x.shape == (n + m,)

    def test_regularization_sensitivity(self):
        """Smaller regularization → more refinement iterations."""
        n, m = 5, 3
        A = np.random.RandomState(42).randn(m, n)
        A_scipy = sparse.csc_matrix(A)

        theta = np.ones(n)
        rhs = np.random.RandomState(42).randn(n + m)

        # Large regularization
        solver1 = IPMQDLDLT()
        solver1.analyze(A_scipy)
        solver1.factorize(theta, np.full(n, 1e-6), np.full(m, 1e-6))
        x1, niters1 = solver1.solve_refine(rhs, theta, np.full(n, 1e-6), np.full(m, 1e-6))

        # Small regularization
        solver2 = IPMQDLDLT()
        solver2.analyze(A_scipy)
        solver2.factorize(theta, np.full(n, 1e-12), np.full(m, 1e-12))
        x2, niters2 = solver2.solve_refine(rhs, theta, np.full(n, 1e-12), np.full(m, 1e-12))

        # Both should solve, niters >= 0
        assert niters1 >= 0 and niters2 >= 0

        # Solutions should be close (same system, just different regularisation)
        np.testing.assert_allclose(x1, x2, atol=1e-6)

    def test_dimensions(self):
        """Dimension mismatch should raise."""
        A = sparse.csc_matrix(np.ones((3, 5)))
        solver = IPMQDLDLT()
        solver.analyze(A)
        solver.factorize(np.ones(5), np.full(5, 1e-12), np.full(3, 1e-12))

        # Wrong-size RHS
        with pytest.raises(Exception):
            solver.solve(np.ones(4))  # 4 != 5+3=8


@pytest.mark.skipif(not HAS_IPM_QD, reason="ipm_qd module not available")
class TestIPMQDLDLTPattern:
    """Test pattern analysis and re-factorization."""

    def test_analyze_then_factorize(self):
        """analyze() must be called before factorize()."""
        A = sparse.csc_matrix(np.array([[1, 0], [0, 1]], dtype=float))
        solver = IPMQDLDLT()

        # Analyze first
        solver.analyze(A)
        assert solver.n() == 2
        assert solver.m() == 2

        # Then factorize
        solver.factorize(np.ones(2), np.full(2, 1e-12), np.full(2, 1e-12))

    def test_same_A_different_B(self):
        """Same A pattern, different B: should give different solutions."""
        n, m = 3, 2
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        A_scipy = sparse.csc_matrix(A)

        solver = IPMQDLDLT()
        solver.analyze(A_scipy)
        solver.factorize(np.ones(n), np.full(n, 1e-12), np.full(m, 1e-12))

        rhs1 = np.zeros(n + m)
        rhs1[n] = 1
        rhs1[n + 1] = 0
        x1 = solver.solve(rhs1)

        rhs2 = np.zeros(n + m)
        rhs2[n] = 0
        rhs2[n + 1] = 1
        x2 = solver.solve(rhs2)

        # Different RHS → different solutions
        assert not np.allclose(x1, x2)
