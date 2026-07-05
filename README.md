# Solvers

High-performance sparse matrix solvers and optimization algorithms.

## Overview

This project provides header-only C++20/23 implementations of modern sparse solvers and QP algorithms:

- **Linear Systems**: LDLᵀ factorization (QDLDL-style), AMD reordering, supernode identification
- **Quadratic Programming**: OSQP (first-order ADMM-based QP) and PIQP (interior-point QP)

All implementations emphasize numerical stability, cache-friendly memory layout, and SIMD optimizations.

## Features

### Linear System Solvers

- **qdldl.h**: Header-only LDLᵀ factorization for symmetric positive-definite systems
  - Cache-optimized numeric factorization
  - SIMD-accelerated solves (AVX-512, AVX2, SSE2)
  - Permutation support via orderings (e.g., AMD)
  - Iterative refinement for improved accuracy
  
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

### Linking

Header-only; include directly:

```cpp
#include "qdldl.h"
#include "supernodes.h"
#include "amd.h"
#include "osqp.h"
#include "piqp.h"
```

Requires: C++20 or later, Eigen 3.4+ (for QP solvers)

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

SparseOSQPSolver solver(Settings{});
auto result = solver.solve(P, q, A, l, u);

if (result.status == "solved") {
    std::cout << "Objective: " << result.obj_val << "\n";
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
- AMD: Davis, T. A. (2004). Algorithm 8xx: AMD, an approximate minimum degree ordering algorithm.
- OSQP: Stellato et al. (2020). OSQP: An Operator Splitting Solver for Quadratic Programs.
- PIQP: [github.com/PREDICT-EPFL/piqp](https://github.com/PREDICT-EPFL/piqp)
