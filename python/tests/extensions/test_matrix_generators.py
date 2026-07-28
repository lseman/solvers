"""
Matrix generators for solver testing.
Creates SPD, symmetric indefinite, diagonally dominant, banded, and
structured matrices (tridiagonal, 5-point Laplacian, KKT systems).
"""

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


def rand_spd(n: int, density: float = 0.1, cond: float = 10.0,
             min_diag: float = 1.0, seed: int = 42) -> NDArray[np.float64]:
    """Random symmetric positive-definite matrix.
    
    A = B B^T + diag(s) to guarantee SPD.
    """
    rng = np.random.default_rng(seed)
    num_nnz = int(n * n * density) if n * n * density > n else n
    B = np.zeros((n, num_nnz))
    mask = rng.random((n, num_nnz)) < 0.5
    B[mask] = rng.standard_normal(mask.sum())
    A = B @ B.T
    # Condition number control: spread eigenvalues
    eigvals = rng.uniform(1.0, cond, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    A = (A + A.T) / 2
    A += np.diag(np.full(n, min_diag))
    return A


def rand_sym_indef(n: int, density: float = 0.1, seed: int = 42) -> NDArray[np.float64]:
    """Random symmetric indefinite matrix (mixed +/− eigenvalues)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2
    # Flip signs of half eigenvalues
    eigvals = rng.standard_normal(n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    A = (A + A.T) / 2
    return A


def tridiag(n: int, lower: float = -1.0, diag: float = 2.0, upper: float = -1.0,
            seed: int = 42) -> NDArray[np.float64]:
    """1D Laplacian-like tridiagonal (SPD if diag > |lower| + |upper|)."""
    rng = np.random.default_rng(seed)
    A = np.diag(np.full(n - 1, lower), -1) + np.diag(np.full(n, diag), 0) + np.diag(np.full(n - 1, upper), 1)
    return A


def laplacian_2d(n: int, bc: str = "dirichlet") -> NDArray[np.float64]:
    """5-point Laplacian on n×n grid (SPD, N=n²)."""
    N = n * n
    d = np.full(N, 4.0)
    off1 = np.full(N - 1, -1.0)
    offN = np.full(N - n, -1.0)
    # Remove right-edge links (row i has no right neighbor if i % n == n-1)
    right_edge = np.arange(n - 1, N - 1, n)  # indices in off1
    off1[right_edge] = 0.0
    A = sp.diags([offN, off1, d, off1, offN], (-n, -1, 0, 1, n), format="csr")
    return A.toarray()


def kkt_system(P: NDArray[np.float64], A_mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """KKT system [P A^T; A 0] for equality-constrained QP."""
    n = P.shape[0]
    m = A_mat.shape[0]
    top = np.hstack([P, A_mat.T])
    bot = np.hstack([A_mat, np.zeros((m, m))])
    return np.vstack([top, bot])


def banded_matrix(n: int, bandwidth: int = 3, density: float = 0.8,
                  seed: int = 42) -> NDArray[np.float64]:
    """Dense band within `bandwidth` of diagonal, zero elsewhere."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(max(0, j - bandwidth), min(n, j + bandwidth + 1)):
            if rng.random() < density or i == j:
                A[i, j] = rng.standard_normal()
    A = (A + A.T) / 2
    A += np.diag(np.full(n, 1.0))  # ensure SPD-ish
    return A


def sparse_eye(n: int, density: float = 0.1, seed: int = 42) -> sp.csc_matrix:
    """Sparse identity + random sparse perturbations (SPD)."""
    rng = np.random.default_rng(seed)
    num_nnz = int(n * n * density) if n * n * density > n else n
    rows = rng.integers(0, n, num_nnz)
    cols = rng.integers(0, n, num_nnz)
    vals = rng.standard_normal(num_nnz)
    S = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    S = (S + S.T) / 2
    return S + sp.eye(n, format="csc") * 10.0


def test_all_generators():
    """Quick smoke test for all generators."""
    A1 = rand_spd(100)
    assert np.allclose(A1, A1.T)
    assert np.all(np.linalg.eigvalsh(A1) > 0), "rand_spd must be SPD"

    A2 = rand_sym_indef(100)
    assert np.allclose(A2, A2.T)
    eigvals = np.linalg.eigvalsh(A2)
    assert eigvals.min() < 0 and eigvals.max() > 0, "rand_sym_indef must have mixed inertia"

    A3 = tridiag(50)
    assert np.allclose(A3, A3.T)
    assert np.all(np.linalg.eigvalsh(A3) > 0), "tridiag must be SPD"

    A4 = laplacian_2d(10)
    assert np.allclose(A4, A4.T)
    assert np.all(np.linalg.eigvalsh(A4) > 0), "laplacian_2d must be SPD"

    P = rand_spd(10)
    A_mat = np.random.default_rng(0).standard_normal((3, 10))
    K = kkt_system(P, A_mat)
    expected_shape = (13, 13)
    assert K.shape == expected_shape, f"Expected {expected_shape}, got {K.shape}"
    assert np.allclose(K, K.T)

    A5 = banded_matrix(100, bandwidth=5)
    for i in range(100):
        for j in range(100):
            if abs(i - j) > 5 and i != j:
                assert A5[i, j] == 0.0

    A6 = sparse_eye(100)
    assert sp.isspmatrix_csc(A6)

    print("All generators pass.")


if __name__ == "__main__":
    test_all_generators()
