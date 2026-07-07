"""
Test Bunch-Kaufman LDLT (indefinite-capable) against scipy/numpy.
BK handles symmetric indefinite matrices where standard LDLT fails.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pytest
from ldlt_bk import BunchKaufmanLDLT

from .test_matrix_generators import rand_sym_indef, rand_spd, tridiag


def _scipy_solve(A):
    n = A.shape[0]
    b = np.random.default_rng(42).standard_normal(n)
    As = sp.csc_matrix(A)
    x_scipy = spla.spsolve(As, b)
    x_arr = np.asarray(x_scipy).ravel()
    res = A @ x_arr - b
    err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
    return x_arr, err


class TestBunchKaufmanLDLT:
    """Bunch-Kaufman correctness tests."""

    def test_indefinite(self):
        """BK should solve symmetric indefinite systems."""
        A = rand_sym_indef(50, seed=42)
        b = np.random.default_rng(42).standard_normal(50)

        solver = BunchKaufmanLDLT()
        solver.compute(A)
        x_custom = solver.solve(b)
        x_arr = np.asarray(x_custom).ravel()
        res = A @ x_arr - b
        err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
        assert err < 1e-4, f"BK indefinite residual: {err}"

    def test_spd_fallback(self):
        """BK should also handle SPD matrices."""
        A = rand_spd(50, seed=42)
        x_scipy, _ = _scipy_solve(A)

        solver = BunchKaufmanLDLT()
        solver.compute(A)
        info = solver.info()
        assert info["factorized"] == True
        assert info["n"] == 50

        b = np.random.default_rng(42).standard_normal(50)
        x_custom = solver.solve(b)
        x_arr = np.asarray(x_custom).ravel()
        res = A @ x_arr - b
        err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
        assert err < 1e-4, f"BK SPD residual: {err}"

    def test_tridiag(self):
        A = tridiag(50)
        solver = BunchKaufmanLDLT()
        solver.compute(A)
        b = np.random.default_rng(42).standard_normal(50)
        x_custom = solver.solve(b)
        x_arr = np.asarray(x_custom).ravel()
        res = A @ x_arr - b
        err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
        assert err < 1e-8, f"BK tridiag residual: {err}"

    def test_inertia_tracking(self):
        """BK should correctly count positive/negative eigenvalues."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 30))
        A = (A + A.T) / 2
        # Force known inertia: 15 pos, 15 neg
        eigvals = rng.standard_normal(30)
        Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
        A = Q @ np.diag(eigvals) @ Q.T
        A = (A + A.T) / 2

        solver = BunchKaufmanLDLT()
        solver.compute(A)
        info = solver.info()
        # Count eigenvalues of A directly
        pos_count = int(np.sum(eigvals > 0))
        neg_count = int(np.sum(eigvals < 0))
        assert info["num_pos"] == pos_count, \
            f"Expected {pos_count} pos, got {info['num_pos']}"
        assert info["num_neg"] == neg_count, \
            f"Expected {neg_count} neg, got {info['num_neg']}"

    def test_pivot_tolerance(self):
        """Test pivot tolerance setting."""
        A = rand_spd(20, seed=42)
        solver = BunchKaufmanLDLT()
        solver.set_pivot_tolerance(1e-10)
        solver.compute(A)
        b = np.random.default_rng(42).standard_normal(20)
        x = solver.solve(b)
        x_arr = np.asarray(x).ravel()
        assert not np.any(np.isnan(x_arr))
        assert not np.any(np.isinf(x_arr))

    def test_solution_vs_scipy(self):
        """Compare BK solution against scipy for SPD."""
        A = rand_spd(40, seed=42)
        x_scipy, _ = _scipy_solve(A)

        solver = BunchKaufmanLDLT()
        solver.compute(A)
        b = np.random.default_rng(42).standard_normal(40)
        x_custom = solver.solve(b)
        diff = float(np.linalg.norm(np.asarray(x_custom).ravel() - x_scipy))
        assert diff < 1e-6, f"BK vs scipy diff: {diff}"
