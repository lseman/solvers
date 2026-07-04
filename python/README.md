# IPM Solver Python Module

Standalone Python bindings for the simplinho interior-point solver (IPM).

## Build

```bash
cd solvers/python
cmake -B build
cmake --build build
pip install build/ipm_solver.*.so
```

Or use `pip` directly:

```bash
pip install ./solvers/python
```

Requires: Python 3.8+, Eigen 3.4+, pybind11 2.6+

## Usage

```python
import ipm_solver
import numpy as np
import scipy.sparse as sp

# Dense LP
A = np.random.randn(20, 30)
b = A @ np.ones(30)
c = np.random.randn(30)
lb = np.zeros(30)
ub = np.full(30, np.inf)

solver = ipm_solver.IPSolver()
solution = solver.solve(A, b, c, lb, ub, sense=None, tol=1e-6)
print(f"Objective: {solution.objective}")
print(f"Status: {solution.status}")
print(f"Primals shape: {len(solution.x)}")

# Sparse LP
A_sparse = sp.csr_matrix(A)
solution = solver.solve(A_sparse, b, c, lb, ub, sense=None, tol=1e-6)
```

## Features

- Schur-complement frontal LDL factorization (BLAS3-optimized)
- Hybrid sparse/frontal solver auto-selection
- Adaptive diagonal regularization (HiPO-style)
- Iterative refinement + dense LU fallback
- No external solver dependencies (pure C++ + Eigen)

## References

See main [simplinho](../) documentation for theory and architecture.
