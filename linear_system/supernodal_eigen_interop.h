// supernodal_eigen_interop.h
// Eigen interoperability for supernodal supernodal LDLᵀ factorization.
//
// Provides conversion functions and templated wrapper for using Eigen sparse
// matrices with the standalone supernodal::SupernodalLDLT algorithm.
//
// Include this header ONLY if Eigen is available. supernodal_ldlt.h core
// itself is Eigen-free.
//
// Usage:
//   #include <Eigen/Sparse>
//   #include "linear_system/supernodal_eigen_interop.h"
//   supernodal::SupernodalLDLT<double,int> solver;
//   supernodal::factorizeEigen(solver, eigen_sparse_matrix);
//   auto x = supernodal::solveEigen(solver, eigen_vector);

#pragma once

#ifndef SUPERSONAL_EIGEN_INTEROP_H
#define SUPERSONAL_EIGEN_INTEROP_H

#include "supernodal_ldlt.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace supernodal {

// ===== Eigen <-> CSC conversion ===========================================

/// Convert Eigen::SparseMatrix (column-major) to standalone SparseCSC.
template < typename Scalar, typename Index >
inline SparseCSC< Scalar, Index > eigen_to_csc(
    const Eigen::SparseMatrix< Scalar, Eigen::ColMajor, Index >& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("supernodal: Matrix must be square");
    }

    Index n = static_cast< Index >(A.rows());
    SparseCSC< Scalar, Index > csc(n);

    csc.Ap.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    for (auto& p : csc.Ap) p = static_cast< Index >(p);

    csc.Ai.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    for (auto& i : csc.Ai) i = static_cast< Index >(i);

    csc.Ax.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());

    return csc;
}

/// Convert standalone SparseCSC to Eigen::SparseMatrix (column-major).
template < typename Scalar, typename Index >
inline Eigen::SparseMatrix< Scalar, Eigen::ColMajor, Index >
csc_to_eigen(const SparseCSC< Scalar, Index >& csc) {
    Eigen::SparseMatrix< Scalar, Eigen::ColMajor, Index > A(csc.n, csc.n);

    std::vector< Eigen::Triplet< Scalar, Index > > trips;
    trips.reserve(static_cast< size_t >(csc.nnz()));

    for (Index j = 0; j < csc.n; ++j) {
        for (Index p = csc.Ap[static_cast< size_t >(j)];
             p < csc.Ap[static_cast< size_t >(j) + 1]; ++p) {
            Index i = csc.Ai[static_cast< size_t >(p)];
            Scalar v = csc.Ax[static_cast< size_t >(p)];
            trips.emplace_back(i, j, v);
        }
    }

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// ===== Eigen solve wrapper =================================================

/// Solve Ax = b using standalone SupernodalLDLT with Eigen vectors.
template < typename Scalar, typename Index >
inline Eigen::VectorXd solveEigen(
    const SupernodalLDLT< Scalar, Index >& solver,
    const Eigen::VectorXd& b) {
    if (static_cast< Index >(b.size()) != solver.size()) {
        throw std::invalid_argument("supernodal: rhs size mismatch");
    }

    std::vector< Scalar > bv(static_cast< size_t >(b.size()));
    for (int i = 0; i < b.size(); ++i) {
        bv[static_cast< size_t >(i)] = static_cast< Scalar >(b[i]);
    }

    auto x = solver.solve(bv);

    Eigen::VectorXd result(b.size());
    for (int i = 0; i < b.size(); ++i) {
        result[i] = static_cast< Scalar >(x[static_cast< size_t >(i)]);
    }
    return result;
}

// ===== Convenience factorize from Eigen ===================================

/// Factorize an Eigen::SparseMatrix directly.
template < typename Scalar, typename Index >
inline void factorizeEigen(SupernodalLDLT< Scalar, Index >& solver,
                           const Eigen::SparseMatrix< Scalar, Eigen::ColMajor, Index >& A) {
    auto csc = eigen_to_csc< Scalar, Index >(A);
    solver.factorizeMatrix(csc);
}

// ===== Convenience: compute from Eigen ====================================

/// Compute (analyze + factorize) an Eigen::SparseMatrix directly.
template < typename Scalar, typename Index >
inline void computeEigen(SupernodalLDLT< Scalar, Index >& solver,
                         const Eigen::SparseMatrix< Scalar, Eigen::ColMajor, Index >& A) {
    auto csc = eigen_to_csc< Scalar, Index >(A);
    solver.compute(csc);
}

} // namespace supernodal

#endif // SUPERSONAL_EIGEN_INTEROP_H
