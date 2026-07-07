"""
Test QDLDL solver against scipy.sparse.linalg.spsolve / numpy.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pytest
from qdldl import solve as solve_qdldl
from .test_matrix_generators import rand_spd, tridiag, laplacian_2d, banded_matrix


def _scipy_solve(A):
    n = A.shape[0]
    b = np.random.default_rng(42).standard_normal(n)
    As = sp.csc_matrix(A)
    x_scipy = spla.spsolve(As, b)
    x_arr = np.asarray(x_scipy).ravel()
    res = A @ x_arr - b
    err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
    return x_arr, err


def _custom_solve(A):
    n = A.shape[0]
    b = np.random.default_rng(42).standard_normal(n)
    result = solve_qdldl(A, b)
    x_custom = np.asarray(result["x"]).ravel()
    res = A @ x_custom - b
    err = float(np.linalg.norm(res)) / max(float(np.linalg.norm(b)), 1e-300)
    return x_custom, err


class TestQDLDL:
    """QDLDL correctness tests."""

    @pytest.mark.parametrize("n,density", [(20, 0.3), (50, 0.1), (100, 0.05)])
    def test_vs_scipy(self, n: int, density: float):
        A = rand_spd(n, density=density, seed=42)
        _, err_scipy = _scipy_solve(A)
        _, err_custom = _custom_solve(A)
        # Both should achieve similar accuracy
        assert err_scipy < 1e-10, f"scipy residual: {err_scipy}"
        assert err_custom < 1e-6, f"QDLDL residual: {err_custom}"

    def test_tridiag(self):
        A = tridiag(50)
        _, err_scipy = _scipy_solve(A)
        _, err_custom = _custom_solve(A)
        assert err_custom < 1e-8, f"QDLDL tridiag residual: {err_custom}"

    def test_laplacian_2d(self):
        A = laplacian_2d(5)  # 25x25
        _, err_scipy = _scipy_solve(A)
        _, err_custom = _custom_solve(A)
        assert err_custom < 1e-6, f"QDLDL Laplacian residual: {err_custom}"

    def test_banded(self):
        A = banded_matrix(50, bandwidth=5)
        _, err_scipy = _scipy_solve(A)
        _, err_custom = _custom_solve(A)
        assert err_custom < 1e-6, f"QDLDL banded residual: {err_custom}"

    def test_solution_consistency(self):
        """Both solvers should produce similar solutions."""
        A = rand_spd(30, seed=42)
        _, err_scipy = _scipy_solve(A)
        x_custom, err_custom = _custom_solve(A)
        # Re-solve to get scipy x
        n = A.shape[0]
        b = np.random.default_rng(42).standard_normal(n)
        As = sp.csc_matrix(A)
        x_scipy = spla.spsolve(As, b)
        x_scipy_arr = np.asarray(x_scipy).ravel()
        diff = float(np.linalg.norm(x_custom - x_scipy_arr)) / max(float(np.linalg.norm(x_scipy_arr)), 1e-300)
        assert diff < 1e-6, f"Solution diff: {diff}"
