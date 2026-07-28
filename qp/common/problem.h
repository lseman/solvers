/// qp/common/problem.h — Shared QP problem/result types
///
/// Common representation used by all solvers in qp/ (osqp, piqp, proxqp):
///
///   minimize  ½ xᵀPx + qᵀx
///   s.t.      Ax = b        (equality, A may have 0 rows)
///             l ≤ Cx ≤ u    (boxed inequality, C may have 0 rows)
///
/// Each solver keeps its own native internal formulation and algorithm;
/// this header only standardizes the input/output shape so callers
/// (bindings, benchmarks, obp backends) don't need solver-specific glue.

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <limits>
#include <string>

namespace qp_common {

using Scalar = double;
using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

/// Problem data: ½xᵀPx + qᵀx s.t. Ax = b, l ≤ Cx ≤ u.
struct QPProblem {
    SpMat P;  /// n x n, symmetric (only one triangle need be filled)
    Vec q;    /// n

    SpMat A;  /// n_eq x n (0 rows if no equality constraints)
    Vec b;    /// n_eq

    SpMat C;  /// n_in x n (0 rows if no inequality constraints)
    Vec l;    /// n_in (use -inf for one-sided)
    Vec u;    /// n_in (use +inf for one-sided)

    int n() const { return static_cast<int>(P.rows()); }
    int n_eq() const { return static_cast<int>(A.rows()); }
    int n_in() const { return static_cast<int>(C.rows()); }
};

/// Common status codes across solvers (each solver still exposes its own
/// richer/native status internally; this is the normalized view).
enum class QPStatus {
    Solved,
    MaxIterReached,
    PrimalInfeasible,
    DualInfeasible,
    FactorizationFailed,
};

inline const char* to_string(QPStatus status) {
    switch (status) {
        case QPStatus::Solved: return "solved";
        case QPStatus::MaxIterReached: return "max_iter_reached";
        case QPStatus::PrimalInfeasible: return "primal_infeasible";
        case QPStatus::DualInfeasible: return "dual_infeasible";
        case QPStatus::FactorizationFailed: return "factorization_failed";
    }
    return "unknown";
}

/// Common solution shape returned by each solver's solve_common() adapter.
struct QPResult {
    QPStatus status = QPStatus::MaxIterReached;
    int iters = 0;
    Scalar obj_val = std::numeric_limits<Scalar>::quiet_NaN();

    Vec x;  /// primal solution (n)
    Vec y;  /// equality multipliers (n_eq)
    Vec z;  /// inequality multipliers (n_in)

    Scalar pri_res = 0.0;  /// primal residual (inf-norm)
    Scalar dua_res = 0.0;  /// dual residual (inf-norm)
};

}  // namespace qp_common
