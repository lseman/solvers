"""
Test KLU-style sparse LU solver against scipy reference.

The KLU solver is a custom LU-style sparse solver (not SuiteSparse KLU)
designed for circuit simulation matrices with block-triangular structure.

Problem form:  Ax = b  (square, sparse, nonsingular)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pytest
from klu import Solver, solve as klu_solve

from .test_matrix_generators import rand_spd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _scipy_solve(A):
    """Solve Ax=b via scipy and return (x, b, relative_residual)."""
    n = A.shape[0]
    b = np.random.default_rng(42).standard_normal(n)
    As = sp.csc_matrix(A)
    x_scipy = spla.spsolve(As, b)
    x_arr = np.asarray(x_scipy).ravel()
    res = A @ x_arr - b
    err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
    return x_arr, b, err


# ── Module-level solve() ─────────────────────────────────────────────────────

class TestKLUSolve:
    """Tests for the convenience solve() function."""

    @pytest.mark.parametrize("n", [3, 10, 30, 50, 100, 200])
    def test_random_spd(self, n):
        """Random SPD matrices across sizes; full solution must match scipy."""
        A = rand_spd(n, density=max(0.1, 5.0 / n), seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        x_scipy, _, _ = _scipy_solve(A)

        r = klu_solve(A, b)
        x_klu = np.asarray(r["x"]).ravel()
        err = float(np.linalg.norm(x_klu - x_scipy))
        assert err < 1e-5, f"KLU n={n} diff: {err}"

    def test_structured_sparse(self):
        """Structured sparse matrix (banded)."""
        n = 50
        A = rand_spd(n, density=0.1, seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        x_scipy, _, _ = _scipy_solve(A)

        r = klu_solve(A, b)
        x_klu = np.asarray(r["x"]).ravel()
        assert float(np.linalg.norm(x_klu - x_scipy)) < 1e-5

    def test_result_dict_structure(self):
        """Verify result dict contains all expected keys."""
        n = 30
        A = rand_spd(n, seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        r = klu_solve(A, b)
        expected_keys = {"x", "n", "nblocks", "nnz_L", "nnz_U", "factorized"}
        assert expected_keys.issubset(r.keys())
        assert isinstance(r["n"], int)
        assert isinstance(r["nblocks"], int)
        assert isinstance(r["nnz_L"], int)
        assert isinstance(r["nnz_U"], int)

    def test_scipy_sparse_input(self):
        """Input via scipy sparse matrix."""
        n = 30
        A_dense = rand_spd(n, seed=42)
        A_sparse = sp.csc_matrix(A_dense)
        b = np.random.default_rng(42).standard_normal(n)
        x_scipy, _, _ = _scipy_solve(A_dense)

        r = klu_solve(A_sparse, b)
        x_klu = np.asarray(r["x"]).ravel()
        assert float(np.linalg.norm(x_klu - x_scipy)) < 1e-5

    def test_block_structure(self):
        """KLU should detect block structure in block-triangular matrices."""
        n = 30
        b1 = 10
        rng = np.random.default_rng(42)
        A_block1 = rng.standard_normal((b1, b1))
        A_block1 = A_block1 @ A_block1.T + b1 * np.eye(b1)
        A_block2 = rng.standard_normal((n - b1, n - b1))
        A_block2 = A_block2 @ A_block2.T + (n - b1) * np.eye(n - b1)
        A_upper = rng.standard_normal((b1, n - b1)) * 0.1
        A = np.block([
            [A_block1, A_upper],
            [np.zeros((n - b1, b1)), A_block2],
        ])
        # Use same b for both klu and scipy
        b = rng.standard_normal(n)
        As = sp.csc_matrix(A)
        x_scipy = spla.spsolve(As, b)

        r = klu_solve(A, b)
        assert r["nblocks"] >= 2, f"Expected >=2 blocks, got {r['nblocks']}"
        x_klu = np.asarray(r["x"]).ravel()
        assert float(np.linalg.norm(x_klu - x_scipy)) < 1e-5

    def test_many_rhs(self):
        """Repeated solves with fresh RHS vectors must all be exact."""
        n = 20
        A = rand_spd(n, density=0.2, seed=42)
        As = sp.csc_matrix(A)
        rng = np.random.default_rng(7)
        for _ in range(20):
            b = rng.standard_normal(n)
            x_klu = np.asarray(klu_solve(A, b)["x"]).ravel()
            x_scipy = np.asarray(spla.spsolve(As, b)).ravel()
            assert float(np.linalg.norm(x_klu - x_scipy)) < 1e-6


# ── Solver class API ─────────────────────────────────────────────────────────

class TestKLUSolverClass:
    """Tests for the Solver class (analyze/factorize/solve pattern).

    NOTE: The Solver class methods expect klu::SparseCSC<int> which is not
    directly constructible from Python. The module-level solve() handles
    the conversion. This class tests what can be accessed from Python.
    """

    def test_solver_construction(self):
        """Solver can be constructed."""
        s = Solver()
        assert s is not None

    def test_solver_methods(self):
        """Solver has expected methods."""
        s = Solver()
        methods = ["analyze", "factorize", "solve", "tsolve",
                    "set_pivot_tolerance", "set_row_scaling"]
        for m in methods:
            assert hasattr(s, m), f"Missing method: {m}"

    def test_pivot_tolerance(self):
        """Pivot tolerance setter should work."""
        s = Solver()
        s.set_pivot_tolerance(0.001)

    def test_row_scaling(self):
        """Row scaling setter should work."""
        s = Solver()
        s.set_row_scaling(0)
        s.set_row_scaling(1)
        s.set_row_scaling(2)

    def test_transpose_solve(self):
        """Test tsolve (transpose solve) exists and is callable."""
        s = Solver()
        assert callable(s.tsolve)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestKLUEdgeCases:
    """Edge cases and special scenarios."""

    def test_diagonal_matrix(self):
        """Diagonal matrix should factorize trivially."""
        n = 30
        A = np.diag(np.arange(1.0, n + 1))
        b = np.random.default_rng(42).standard_normal(n)
        x_scipy, _, _ = _scipy_solve(A)

        r = klu_solve(A, b)
        x_klu = np.asarray(r["x"]).ravel()
        assert float(np.linalg.norm(x_klu - x_scipy)) < 1e-10

    def test_near_singular(self):
        """Near-singular matrix should raise or return poor solution."""
        n = 30
        A = rand_spd(n, cond=1e14, seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        try:
            r = klu_solve(A, b)
            assert r["factorized"] is True
        except Exception:
            pass  # Acceptable to fail

    def test_nnz_counts(self):
        """nnz_L and nnz_U should be positive for non-trivial matrices."""
        n = 30
        A = rand_spd(n, density=0.1, seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        r = klu_solve(A, b)
        assert r["nnz_L"] > 0
        assert r["nnz_U"] > 0

    def test_n_blocks(self):
        """nblocks should be >= 1 for any square matrix."""
        n = 30
        A = rand_spd(n, seed=42)
        b = np.random.default_rng(42).standard_normal(n)
        r = klu_solve(A, b)
        assert r["nblocks"] >= 1
