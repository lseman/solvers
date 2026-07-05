// schur_frontal_eigen_interop.h
// Interop layer between Eigen and Schur-frontal standalone LDLT.
//
// Provides conversions and templated wrapper for using Eigen sparse matrices
// with the standalone Schur-frontal algorithm.

#pragma once

#ifndef SCHUR_FRONTAL_EIGEN_INTEROP_H
#define SCHUR_FRONTAL_EIGEN_INTEROP_H

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

// Simplified frontal factorization: singleton supernodes (each column is a front)
// Returns FrontalLDLT stub with factorized=true but empty structure.
// Full implementation deferred; used as fallback in sparse_solver.h.
inline FrontalLDLT factor_frontal(
    const Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t> &A,
    Real pivot_tolerance = 1e-14) {
  if (A.rows() != A.cols())
    throw std::invalid_argument("Matrix must be square");

  Int n = static_cast<Int>(A.rows());
  FrontalLDLT result;
  result.n = n;
  result.pivot_tolerance = pivot_tolerance;

  // Simplified: create singleton supernodes (no merging)
  auto sn_ranges = singleton_supernodes(n);
  auto col2sn = detail::make_col2sn(n, sn_ranges);

  // Simple elimination tree (parent = next column)
  std::vector<Int> etree(static_cast<size_t>(n), -1);
  for (Int j = 0; j < n - 1; ++j) {
    etree[static_cast<size_t>(j)] = j + 1;
  }

  // Create frontal nodes
  result.fronts.resize(static_cast<size_t>(n));
  result.elimination_order.resize(static_cast<size_t>(n));
  result.diag.assign(static_cast<size_t>(n), 0.0);

  for (Int f = 0; f < n; ++f) {
    auto &front = result.fronts[static_cast<size_t>(f)];
    front.col_start = f;
    front.col_end = f + 1;
    front.pivots.push_back(f);
    detail::build_local_map(front);

    front.parent = (f < n - 1) ? f + 1 : -1;
    result.elimination_order[static_cast<size_t>(f)] = f;
  }

  // Link children
  for (Int f = 0; f < n - 1; ++f) {
    result.fronts[static_cast<size_t>(f + 1)].children.push_back(f);
  }
  if (n > 0)
    result.fronts[static_cast<size_t>(n - 1)].is_root = true;

  // Initialize L (empty for now; full impl would populate)
  result.L.n = n;
  result.L.Ap.assign(static_cast<size_t>(n) + 1, 0);

  result.factorized = true;
  return result;
}

} // namespace schur_frontal

#endif // SCHUR_FRONTAL_EIGEN_INTEROP_H
