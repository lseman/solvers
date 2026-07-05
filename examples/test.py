# %% [markdown]
# # Custom LDLT vs NumPy/SciPy
#
# This notebook solves a symmetric linear system with the local `solvers.ldlt` module
# (wrapping `CustomSimplicialLDLT` from `linear_system/ldlt.h`), NumPy, and SciPy sparse linear algebra.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "build"))

import ldlt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# %%
# --- Test 1: small dense SPD system ---

rng = np.random.default_rng(42)
n = 8
M = rng.normal(size=(n, n))
A = M.T @ M + 1.0 * np.eye(n)  # SPD guarantee
b = rng.normal(size=n)

x_custom = ldlt.solve(A, b)["x"]
x_numpy = np.linalg.solve(A, b)
x_scipy = spla.spsolve(sp.csc_matrix(A), b)

print("Custom LDLT x:", x_custom)
print("NumPy x:      ", x_numpy)
print("SciPy x:      ", x_scipy)
print("Custom vs NumPy max error:", np.max(np.abs(x_custom - x_numpy)))
print("Custom vs SciPy max error:", np.max(np.abs(x_custom - x_scipy)))
print("Residual inf norm:", np.linalg.norm(A @ x_custom - b, ord=np.inf))

# %%
# --- Test 2: LDLTSolver class (reuse on multiple RHS) ---

rng = np.random.default_rng(7)
n = 16
M = rng.normal(size=(n, n))
A = M.T @ M + 0.5 * np.eye(n)

solver = ldlt.LDLTSolver(A)
info = solver.info()
print("Size:", info["size"])
print("Nonzeros in L:", info["nonzeros_l"])
print("Perturbed pivots:", info["perturbed_pivots"])
print("Min abs pivot:", info["min_abs_pivot"])
print("Factorized:", info["factorized"])

# Solve multiple RHS with same factorization
X_rhs = rng.normal(size=(n, 5))
X_custom = np.zeros_like(X_rhs)
X_numpy = np.zeros_like(X_rhs)

for i in range(5):
    b = X_rhs[:, i]
    X_custom[:, i] = solver.solve(b)
    X_numpy[:, i] = np.linalg.solve(A, b)

print("Multi-RHS max error:", np.max(np.abs(X_custom - X_numpy)))

# %%
# --- Test 3: sparse input ---

rng = np.random.default_rng(99)
n = 20
A_dense = rng.normal(size=(n, n))
A_sparse = sp.csc_matrix(A_dense.T @ A_dense + 0.3 * np.eye(n))
b = rng.normal(size=n)

x_sparse = ldlt.solve(A_sparse, b)["x"]
x_dense = np.linalg.solve(A_dense.T @ A_dense + 0.3 * np.eye(n), b)

print("Sparse input max error:", np.max(np.abs(x_sparse - x_dense)))
print("Residual inf norm:", np.linalg.norm(A_sparse @ x_sparse - b, ord=np.inf))

# %%
# --- Test 4: near-singular with regularization ---

n = 6
rng = np.random.default_rng(13)
M = rng.normal(size=(n, n))
A = M.T @ M  # rank-n, nearly singular without shift

# Without regularization: may fail or produce NaN
try:
    solver = ldlt.LDLTSolver(A)
    x = solver.solve(rng.normal(size=n))
    print("No reg: factorized, pivots =", solver.info()["perturbed_pivots"])
except RuntimeError as e:
    print("No reg: failed as expected:", e)

# With regularization: should succeed
solver = ldlt.LDLTSolver(A)
solver.set_regularization(1e-10)
solver.compute(A)
b = rng.normal(size=n)
x = solver.solve(b)
print("With reg: factorized, pivots =", solver.info()["perturbed_pivots"])
print("Residual:", np.linalg.norm(A @ x - b))

# %%
# --- Assertions ---

assert np.allclose(x_custom, x_numpy, rtol=1e-8, atol=1e-10), "Test 1 failed"
assert np.max(np.abs(X_custom - X_numpy)) < 1e-10, "Test 2 failed"
assert np.allclose(x_sparse, x_dense, rtol=1e-8, atol=1e-10), "Test 3 failed"
print("All assertions passed.")
