# ipm/ — interior point method

Implements a factorization-based regularized IPM for the augmented system,
following Zanetti & Gondzio, *"A factorisation-based regularised interior
point method using the augmented system"* (2025),
[arXiv:2508.04370](https://arxiv.org/html/2508.04370v1) (HiPO, the HiGHS
interior point solver this repo's approach is modeled on).

## `hipo_ldlt.h` is the active paper implementation

`HiPOLDLT` (in `hipo_ldlt.h`) is where the active paper implementation lives —
everything else in this directory either feeds it or is a baseline/alternative
to compare against. Its solve path, end to end:

1. **Augmented system** `[-(Θ+Rp) Aᵀ; A Rd]` — not normal equations (§3, eq. 6).
2. **Block-quasi-definite AMD** (`hipo_ldlt.h::blockAmdOrdering`) — AMD run
   separately on the primal and dual blocks, concatenated, so no pivot ever
   crosses the negative-definite/positive-definite boundary. This is what
   makes Prop. 3's sign-preservation guarantee (§4.6) hold by construction
   instead of needing runtime sign monitoring.
3. **Multifrontal numeric factorization** implemented directly in
   `hipo_ldlt.h`: frontal matrices per supernode, Schur complements passed up
   the elimination tree, and dense kernels inside each frontal matrix.
4. **Regularization** — static floor + the Prop. 3 bound
   (`minSignPreservingPivotMagnitude`, §4.5–4.6), inlined in HiPO's BK kernel.
5. **Iterative refinement** with component-wise backward-error estimation
   (§4.5, ref [10]), in `HiPOLDLT::solveWithRefinement`.

HiPO uses Bunch-Kaufman 1×1/2×2 pivoting restricted to each frontal matrix's
pivot block. Remaining gaps versus the paper are Metis nested-dissection
ordering (we use AMD instead — paper §4.2), supernode
amalgamation/relaxation tuning (parameters exist in `supernodes.h` but aren't
wired up), parallel factorization.

## Files in this directory

| File | Role |
|---|---|
| `hipo_ldlt.h` | **Active paper implementation.** `HiPOLDLT` — augmented system, block-QD AMD, multifrontal BK factorization, Prop. 3 regularization, iterative refinement. |
| `ip_solver.h` / `../src/ipm/ip_solver.cpp` | Mehrotra predictor-corrector IPM loop: residuals, step sizing, standard-form conversion, Ruiz scaling. Calls into `sparse_solver.h` for every Newton system solve. |
| `sparse_solver.h` | `SparseSolver` — dispatches each IPM linear solve to one of the backends in the table below (`SolverType::{LDLT,SUPERNOODAL,FRONTAL,HIPO_LDLT,NORMAL_EQ,AUTO}`). |
| `normal_eq_ldlt.h` | Normal-equations alternative (`NormalEqLDLT`): eliminates `Δx` to get the SPD system `(A D⁻¹ Aᵀ + Rd) Δy = ...`, half the size of the augmented system. Also has `estimate_fill()`, the structural nnz heuristic `SolverType::AUTO` uses to pick augmented-vs-normal-eq once per `analyzePattern` (paper §4.2 discusses this tradeoff, but the paper's HiPO solver itself is augmented-system-only — this normal-eq path is our own addition, not from the paper). |

## What `hipo_ldlt.h` pulls from `linear_system/` and why

HiPO implements its multifrontal and BK kernels directly while reusing these
general-purpose structural and solve components:

| Component | Location | Paper section | Also used standalone by |
|---|---|---|---|
| Supernode detection | `linear_system/supernodes.h` | §4.1 (supernode partition) | `python/solvers/supernodes_ext.cpp` |
| AMD fill-reducing ordering | `linear_system/common/amd.h`, `common/ordering.h` | §4.2 (ordering — paper uses Metis nested dissection; we use AMD) | `linear_system/ldlt/ldlt.h`, `supernodal_ldlt.h` |
| Sign-preserving pivot regularization | inlined in `hipo_ldlt.h::bkFactorizeFrontal` | §4.5–4.6 (static+dynamic regularization, Prop. 3 bound) | HiPO only |
| CSC storage / triangular solves | `linear_system/common/sparse_csc.h`, `common/trisolve.h` | — (plumbing) | every solver in `linear_system/ldlt/` |

## Baselines/alternatives (not the paper)

| `SolverType` | What it is | Backing code |
|---|---|---|
| `LDLT` | Plain simplicial LDLᵀ on the full augmented system. | `linear_system/ldlt/ldlt.h` (also `python/ldlt_ext.cpp`) |
| `SUPERNOODAL` | Augmented system, dense-BLAS-on-supernodes, natural (non-block) AMD, flat pivot regularization, and no refinement. | `linear_system/ldlt/supernodal_ldlt.h` |
| `FRONTAL` | Schur-complement frontal method, a different multifrontal formulation than the paper's. **ipm-only.** | `linear_system/ldlt/schur_frontal_ldlt.h` + `linear_system/eigen_interop/schur_frontal_eigen_interop.h` |
| `NORMAL_EQ` | Eliminate to the SPD normal equations instead of the augmented system. | `normal_eq_ldlt.h` |
| `HIPO_LDLT` | HiPO augmented-system factorization. | `hipo_ldlt.h` |
| `AUTO` | Picks `HIPO_LDLT` vs `NORMAL_EQ` per problem via `estimate_fill()`. | `sparse_solver.h` |

The separate dense Bunch-Kaufman LDLᵀ implementation in
`linear_system/ldlt/ldlt_bk.h` remains reachable through
`python/ldlt_bk_ext.cpp`; HiPO has its own frontal BK kernel.
