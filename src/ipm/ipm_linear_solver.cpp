// ipm_linear_solver.cpp — Hybrid linear solver implementations for IPM

#include "ipm/ipm_linear_solver.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cmath>
#include <sstream>

namespace ipm {

// ===== SparseIPMSolver (existing sparse behavior) =====

void SparseIPMSolver::factorize(const Eigen::SparseMatrix<double>& A) {
    factorized_ = false;
    use_dense_fallback_ = false;
    if (solver_type_ == LinearSolverType::kSparseLU) {
        lu_.compute(A);
        factorized_ = (lu_.info() == Eigen::Success);
        if (!factorized_) {
            dense_cod_.compute(Eigen::MatrixXd(A));
            use_dense_fallback_ = true;
            factorized_ = use_dense_fallback_;
        }
        return;
    }

    ldlt_.compute(A);
    if (ldlt_.info() == Eigen::Success) {
        factorized_ = true;
        return;
    }

    lu_.compute(A);
    if (lu_.info() == Eigen::Success) {
        solver_type_ = LinearSolverType::kSparseLU;
        factorized_ = true;
        return;
    }

    dense_cod_.compute(Eigen::MatrixXd(A));
    use_dense_fallback_ = true;
    factorized_ = use_dense_fallback_;
}

Eigen::VectorXd SparseIPMSolver::solve(const Eigen::VectorXd& rhs) {
    if (!factorized_) {
        throw std::runtime_error("SparseIPMSolver: not factorized");
    }
    if (use_dense_fallback_) {
        return dense_cod_.solve(rhs);
    }
    if (solver_type_ == LinearSolverType::kSparseLU) {
        return lu_.solve(rhs);
    }
    return ldlt_.solve(rhs);
}

std::string SparseIPMSolver::info() const {
    std::ostringstream oss;
    oss << "SparseIPMSolver(";
    switch (solver_type_) {
        case LinearSolverType::kSparseLDLT:
            oss << "LDLT";
            break;
        case LinearSolverType::kSparseLU:
            oss << "LU";
            break;
        case LinearSolverType::kFrontal:
            oss << "Frontal";
            break;
    }
    oss << ", factorized=" << factorized_ << ")";
    return oss.str();
}

// ===== FrontalIPMSolver (SOTA Schur-complement) =====

void FrontalIPMSolver::factorize(const Eigen::SparseMatrix<double>& A) {
    // TODO: Implement frontal factorization
    // Need to:
    // 1. Permute A to AMDOrdered form
    // 2. Identify supernodes
    // 3. Call build_and_factor_frontal()
    ldlt_.factorized = true;
}

Eigen::VectorXd FrontalIPMSolver::solve(const Eigen::VectorXd& rhs) {
    if (!ldlt_.factorized) {
        throw std::runtime_error("FrontalIPMSolver: not factorized");
    }
    // Convert Eigen vector to std::vector, solve, convert back
    std::vector<double> rhs_std(rhs.data(), rhs.data() + rhs.size());
    auto sol_std = schur_frontal::solve(ldlt_, rhs_std);
    return Eigen::VectorXd::Map(sol_std.data(), sol_std.size());
}

std::string FrontalIPMSolver::info() const {
    std::ostringstream oss;
    oss << "FrontalIPMSolver(";
    oss << "fronts=" << ldlt_.fronts.size() << ", ";
    oss << "factorized=" << ldlt_.factorized << ")";
    return oss.str();
}

// ===== HybridIPMSolver (auto-switching) =====

bool HybridIPMSolver::should_use_frontal(
    const Eigen::SparseMatrix<double>& A) const {
    (void) A;
    return false;

    // Simple heuristic: use frontal if matrix is dense enough
    // and has good supernode structure (i.e., column count close to row count)
    // For IPM KKT systems, typically m << n, so this is problem-dependent.

    const int m = A.rows();
    const int n = A.cols();
    const int nnz = A.nonZeros();

    // Density threshold
    const double density = double(nnz) / (double(m) * double(n) + 1e-12);

    // Use frontal if moderately dense (>5%) and not too rectangular
    if (density > 0.05 && std::abs(m - n) < 0.5 * m) {
        return true;
    }
    return false;
}

void HybridIPMSolver::factorize(const Eigen::SparseMatrix<double>& A) {
    LinearSolverType chosen = LinearSolverType::kSparseLDLT;

    if (should_use_frontal(A)) {
        chosen = LinearSolverType::kFrontal;
        solver_ = std::make_unique<FrontalIPMSolver>();
    } else {
        chosen = LinearSolverType::kSparseLU;
        solver_ = std::make_unique<SparseIPMSolver>(
            LinearSolverType::kSparseLU);
    }

    active_type_ = chosen;
    solver_->factorize(A);
}

Eigen::VectorXd HybridIPMSolver::solve(const Eigen::VectorXd& rhs) {
    if (!solver_ || !solver_->is_factorized()) {
        throw std::runtime_error("HybridIPMSolver: not factorized");
    }
    return solver_->solve(rhs);
}

}  // namespace ipm
