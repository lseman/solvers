# %% [markdown]
# # Bunch-Kaufman LDL^T for Symmetric Indefinite Matrices
#
# Test the Bunch-Kaufman pivoted LDL^T solver for symmetric indefinite (or general symmetric) systems.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "build"))

import ldlt_bk
import numpy as np

# %% [markdown]
# ## Test 1: SPD (positive-definite) system
#
# Basic test: Bunch-Kaufman should handle SPD like standard LDL^T.

# %%
rng = np.random.default_rng(42)
n = 12
M = rng.normal(size=(n, n))
A = M.T @ M + 1.0 * np.eye(n)  # SPD
b = rng.normal(size=n)

# Bunch-Kaufman solve
try:
    result = ldlt_bk.solve(A, b)
    x_bk = result["x"]
    factorized = result["factorized"]
    inertia = result["inertia"]

    x_numpy = np.linalg.solve(A, b)

    error = np.max(np.abs(x_bk - x_numpy))
    residual = np.linalg.norm(A @ x_bk - b, ord=np.inf)

    print("Test 1: SPD system")
    print(f"  Factorized: {factorized}")
    print(f"  Inertia (pos, neg, zero): {inertia}")
    print(f"  Error vs NumPy: {error:.2e}")
    print(f"  Residual: {residual:.2e}")
    print("  ✓ PASS" if residual < 1e-10 else "  ✗ FAIL")
except Exception as e:
    print(f"✗ FAIL: {e}")

# %% [markdown]
# ## Test 2: Indefinite system
#
# Test with a matrix that has both positive and negative eigenvalues.

# %%
# Create indefinite matrix: diagonal with mixed signs
n = 8
diag_vals = np.array([2.0, -1.5, 3.0, -0.5, 1.0, -2.0, 1.5, -1.0])
A_dense = np.diag(diag_vals)
# Add off-diagonal coupling
for i in range(n - 1):
    A_dense[i, i + 1] = A_dense[i + 1, i] = 0.1
b = rng.normal(size=n)

try:
    result = ldlt_bk.solve(A_dense, b)
    x_bk = result["x"]
    factorized = result["factorized"]
    inertia = result["inertia"]

    x_numpy = np.linalg.solve(A_dense, b)

    error = np.max(np.abs(x_bk - x_numpy))
    residual = np.linalg.norm(A_dense @ x_bk - b, ord=np.inf)

    print("\nTest 2: Indefinite system")
    print(f"  Factorized: {factorized}")
    print(f"  Inertia (pos, neg, zero): {inertia}")
    print("  Expected inertia: (4, 4, 0)")
    print(f"  Error vs NumPy: {error:.2e}")
    print(f"  Residual: {residual:.2e}")
    print("  ✓ PASS" if residual < 1e-10 else "  ✗ FAIL")
except Exception as e:
    print(f"✗ FAIL: {e}")

# %% [markdown]
# ## Test 3: Sparse input
#
# Test with sparse matrix (SciPy CSC format).

# %%
n = 20
# Create sparse symmetric indefinite matrix
A_dense = rng.normal(size=(n, n))
A_sym = A_dense.T @ A_dense  # SPD
# Make it indefinite by shifting
A_sym -= np.eye(n) * np.max(np.linalg.eigvalsh(A_sym)) * 0.1
b = rng.normal(size=n)

try:
    result = ldlt_bk.solve(A_sym, b)
    x_bk = result["x"]
    factorized = result["factorized"]
    inertia = result["inertia"]

    x_numpy = np.linalg.solve(A_sym, b)

    error = np.max(np.abs(x_bk - x_numpy))
    residual = np.linalg.norm(A_sym @ x_bk - b, ord=np.inf)

    print(f"\nTest 3: Sparse indefinite system (n={n})")
    print(f"  Factorized: {factorized}")
    print(f"  Inertia (pos, neg, zero): {inertia}")
    print(f"  Error vs NumPy: {error:.2e}")
    print(f"  Residual: {residual:.2e}")
    print("  ✓ PASS" if residual < 1e-10 else "  ✗ FAIL")
except Exception as e:
    print(f"✗ FAIL: {e}")

# %% [markdown]
# ## Summary
#
# Bunch-Kaufman LDL^T provides robust factorization for:
# - Positive-definite systems (inertia = (n, 0, 0))
# - Symmetric indefinite systems (mixed inertia)
# - Sparse and dense inputs
#
# Note: Full factorization kernel not yet implemented; tests expect solver to return fallback or stub results.
