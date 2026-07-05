// schur_frontal_eigen_interop.h
// Interop layer between Eigen and Schur-frontal standalone LDLT.
//
// Provides conversions and templated wrapper for using Eigen sparse matrices
// with the standalone Schur-frontal algorithm.

#pragma once

#include "schur_frontal_ldlt.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace schur_frontal {

// Convert Eigen::SparseMatrix to standalone CSC
template <typename Scalar, typename StorageIndex>
inline SparseCSC<Scalar, int32_t> eigen_to_csc(
    const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex> &A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Matrix must be square");
  }

  Int n = static_cast<Int>(A.rows());
  SparseCSC<Scalar, int32_t> csc(n);

  csc.Ap.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
  for (auto &p : csc.Ap) p = static_cast<int32_t>(p);

  csc.Ai.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
  for (auto &i : csc.Ai) i = static_cast<int32_t>(i);

  csc.Ax.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());

  return csc;
}

// Convert standalone CSC to Eigen::SparseMatrix
template <typename Scalar>
inline Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int32_t> csc_to_eigen(
    const SparseCSC<Scalar, int32_t> &csc) {
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int32_t> A(csc.n, csc.n);

  std::vector<Eigen::Triplet<Scalar, int32_t>> trips;
  trips.reserve(csc.nnz());

  for (int32_t j = 0; j < csc.n; ++j) {
    for (int32_t p = csc.Ap[static_cast<size_t>(j)];
         p < csc.Ap[static_cast<size_t>(j + 1)]; ++p) {
      int32_t i = csc.Ai[static_cast<size_t>(p)];
      Scalar v = csc.Ax[static_cast<size_t>(p)];
      trips.emplace_back(i, j, v);
    }
  }

  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

// Wrapper to use Eigen matrices with standalone algorithm
template <typename Scalar, typename StorageIndex>
inline FrontalLDLT factor_eigen(
    const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex> &A,
    const std::vector<std::pair<Int, Int>> &supernode_ranges,
    const std::vector<Int> &etree, Scalar pivot_tolerance = 1e-14) {
  // Extract pattern only (standalone doesn't handle Eigen types)
  // For now, just convert and call generic interface
  auto csc = eigen_to_csc(A);
  // TODO: implement factorization with CSC matrix
  throw std::runtime_error("Eigen wrapper not yet implemented");
}

} // namespace schur_frontal

#endif
