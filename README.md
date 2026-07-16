# <img src="assets/optblocks-logo.svg" alt="optBlocks Logo" width="400"/>

## optBlocks

High-performance sparse matrix solvers and optimization algorithms.

## Overview

**optBlocks** provides high-performance C++20/23 implementations of modern sparse solvers and conic optimization algorithms:

### Available Solvers

- **Linear Systems**:
  - `qdldl_ext`: QDLDL-style LDLᵀ factorization for symmetric indefinite systems
  - `ldlt_ext`: Custom simplicial LDLᵀ factorization
  - `ldlt_bk_ext`: LDLᵀ with Bunch-Kaufman pivoting + iterative refinement
  - `klu_ext`: KLU-style sparse LU factorization for circuit simulation matrices
  - `amd.h`: Approximate Minimum Degree reordering
  - `supernodes.h`: Supernode identification from symbolic analysis

- **Quadratic Programming**:
  - `osqp_ext`: OSQP solver (first-order ADMM-based QP)
  - `piqp_ext`: PIQP solver (interior-point primal-dual QP)

- **Conic Optimization**:
  - `ipm_ext`: HSDE (Homogeneous Self-Dual Embedding) interior-point method
    - Supports LP, QP, SOCP, and SDP cones
    - Exponential and power cone projections
    - Infeasibility detection via homogeneous embedding

All implementations emphasize numerical stability, cache-friendly memory layout, and SIMD optimizations.

## Features

### Linear System Solvers

- **qdldl.h**: Header-only LDLᵀ factorization for symmetric positive-definite systems
  - Cache-optimized numeric factorization
  - SIMD-accelerated solves (AVX-512, AVX2, SSE2)
  - Permutation support via orderings (e.g., AMD)
  - Iterative refinement for improved accuracy
  
- **klu.h**: KLU-style sparse LU factorization for circuit simulation matrices
  - Block Triangular Form (BTF) decomposition for SPICE-like matrices
  - AMD ordering within each diagonal block
  - Sparse LU factorization with partial pivoting
  - Designed for unsymmetric and structurally indefinite systems
  
- **amd.h**: Approximate Minimum Degree reordering
  - Array-based implementation with SoTA micro-architecture tweaks
  - Aggressive absorption and variable coalescence
  - Bandwidth and fill reduction for sparse factorization
  
- **supernodes.h**: Supernode identification from LDLᵀ symbolic analysis
  - Relaxed structural matching for modern sparse kernels
  - Postorder traversal of elimination trees
  - Permuted column-space supernode ranges

### Quadratic Programming

- **osqp.h**: Sparse OSQP solver (first-order ADMM)
  - Fast normal-equations path with cached AᵀA
  - KKT quasi-definite path for dense constraints
  - Ruiz scaling for numerical stability
  - Adaptive penalty (rho) updates
  - Active-set polishing for feasibility
  
- **piqp.h**: PIQP solver (interior-point primal-dual method)
  - Predictor-corrector algorithm
  - Proximal-center updates for robustness
  - Pattern-reuse LLT factorizations
  - Warm-start support for iterative refinement

## Quick Start

### Python Bindings

Build with CMake and install via pip:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
make -C build
pip install build/*.whl
```

### C++ Header-Only

Include directly for linear system solvers:

```cpp
#include "qdldl.h"
#include "supernodes.h"
#include "amd.h"
```

Requires: C++20 or later, Eigen 3.4+

### Example: KLU Sparse LU Factorization

```cpp
#include "linear_system/klu/klu.h"

klu::SparseCSC<int> csc;
csc.n = n; csc.Ap = std::move(Ap); csc.Ai = std::move(Ai); csc.Ax = std::move(Ax);

klu::Solver solver;
auto sym = solver.analyze(csc);
auto num = solver.factorize(csc, sym);

// Solve Ax = b
std::vector<double> x = solver.solve(num, b);
```

### Example: LDLᵀ Factorization

```cpp
#include "qdldl.h"
using namespace qdldl23;

// Assemble upper-triangular CSC matrix
SparseD32 A;
A.n = n; A.Ap = std::move(Ap); A.Ai = std::move(Ai); A.Ax = std::move(Ax);
finalize_upper_inplace(A);  // coalesce dupes, check upper+diag

// Symbolic analysis
auto S = analyze_fast(A);

// Numeric factorization
auto F = refactorize(A, S);

// Solve Ax = b
std::vector<double> x(b.begin(), b.end());
solve(F, x.data());
```

### Example: AMD Reordering

```cpp
#include "amd.h"

AMDReorderingArray amd(true);  // aggressive absorption
auto [perm, stats] = amd.compute_fill_reducing_permutation(A);

std::cout << "Bandwidth reduction: " << stats.bandwidth_reduction << "\n";
```

### Example: QP via OSQP

```cpp
#include "osqp.h"
using namespace sosqp;

sparse_osqp_solver solver(Settings{});
auto result = solver.solve(P, q, A, l, u);

if (result.status == "solved") {
    std::cout << "Objective: " << result.obj_val << "\n";
}
```

### Example: Conic Optimization via IPM

```cpp
#include "ipm/ip_solver.h"

// Initialize solver with problem data
ip_solver solver;
solver.initialize(A, P, q, b, cones);

// Solve
if (solver.solve()) {
    auto x = solver.get_x();
    auto y = solver.get_y();
    auto obj = solver.get_objective_value();
    std::cout << "Optimal objective: " << obj << "\n";
}
```

## Design Notes

- **No dynamic allocation in hot loops**: Temporary buffers are pre-allocated
- **Pattern reuse**: Factorization patterns cached across updates
- **SIMD-friendly**: Data layouts and kernels designed for modern CPUs
- **Numerically stable**: Exact-zero pivots, iterative refinement, scaling
- **Header-only**: No build system required; easy integration

## License

MIT / Apache-2.0 (dual-licensed)

## References

- QDLDL: [github.com/oxfordcontrol/qdldl](https://github.com/oxfordcontrol/qdldl)
- KLU: Davis, T. A. et al. "Algorithm 832: KLU, a direct sparse solver for circuit simulation matrices."
- AMD: Davis, T. A. (2004). Algorithm 8xx: AMD, an approximate minimum degree ordering algorithm.
- OSQP: Stellato et al. (2020). OSQP: An Operator Splitting Solver for Quadratic Programs.
- PIQP: [github.com/PREDICT-EPFL/piqp](https://github.com/PREDICT-EPFL/piqp)
