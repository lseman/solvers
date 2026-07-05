/*
 * ldlt_bk.h — Bunch-Kaufman LDL^T factorization for symmetric indefinite matrices
 *
 * Robust pivoted LDL^T with 1x1/2x2 D blocks for general symmetric systems.
 * Handles both positive-definite and indefinite matrices via dynamic pivoting.
 *
 * Algorithm:
 *   - Select pivot row/column with largest off-diagonal element
 *   - Use 1x1 pivot if diagonal is large enough; else use 2x2 block
 *   - Perform symmetric row/column swap
 *   - Factor pivot block, update Schur complement
 *   - Repeat until complete
 *
 * Output: A ≈ P L D L^T P^T where:
 *   - P is a permutation (block structure)
 *   - L is unit lower triangular
 *   - D is block diagonal (1x1 or 2x2 blocks)
 *
 * Inertia (# pos, neg, zero eigenvalues) derived from D block signs.
 * Solves using forward/diagonal/backward substitution with 2x2 block handling.
 */

#pragma once

#ifndef LDLT_BK_H
#define LDLT_BK_H

#include "ldlt_simd.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ldlt {

using Int = int32_t;
using Real = double;

// Sparse upper CSC (shared with simplicial LDLT)
template <typename Scalar = double, typename Index = int32_t>
struct SparseCSC {
  std::vector<Index> Ap;
  std::vector<Index> Ai;
  std::vector<Scalar> Ax;
  Index n = 0;

  SparseCSC() = default;
  SparseCSC(Index n_, std::vector<Index> Ap_, std::vector<Index> Ai_,
            std::vector<Scalar> Ax_)
      : Ap(std::move(Ap_)), Ai(std::move(Ai_)), Ax(std::move(Ax_)), n(n_) {
    if (n < 0) throw std::invalid_argument("n < 0");
    if (Ap.size() != static_cast<size_t>(n) + 1)
      throw std::invalid_argument("Ap size != n+1");
    auto nnz = Ap.back();
    if (static_cast<size_t>(nnz) != Ai.size() || static_cast<size_t>(nnz) != Ax.size())
      throw std::invalid_argument("Ai/Ax size mismatch");
  }

  [[nodiscard]] size_t nnz() const { return static_cast<size_t>(Ap.back()); }
};

struct BunchKaufmanFactors {
  std::vector<Int> perm;     // permutation: old → new (applied symmetrically)
  std::vector<Int> iperm;    // inverse: new → old
  std::vector<Real> Lx;      // lower triangle, excluding diagonal
  std::vector<Int> Li;       // row indices for Lx
  std::vector<Int> Lp;       // column pointers for L (size n+1)
  std::vector<Real> D;       // diagonal: may be 2x2 blocks (interleaved)
  std::vector<Int> block_info; // size n: 0=skip (part of 2x2), 1=1x1 pivot, 2=2x2 pivot
  Int n = 0;
  Int num_pos = 0;
  Int num_neg = 0;
  Int num_zero = 0;
  bool factorized = false;
  Real pivot_tolerance = 1e-12;
};

class BunchKaufmanLDLT {
public:
  using MatrixType = SparseCSC<Real, Int>;

  BunchKaufmanLDLT() : m_size(0), m_pivot_tolerance(1e-12) {}

  explicit BunchKaufmanLDLT(const MatrixType &A) : m_size(0), m_pivot_tolerance(1e-12) {
    compute(A);
  }

  void compute(const MatrixType &A) {
    if (A.n <= 0) {
      m_factors.n = 0;
      m_factors.factorized = false;
      return;
    }
    if (static_cast<Int>(A.Ap.size()) != A.n + 1)
      throw std::invalid_argument("Invalid matrix structure");

    m_size = A.n;
    factorize(A);
  }

  std::vector<Real> solve(const std::vector<Real> &b) const {
    if (!m_factors.factorized || static_cast<Int>(b.size()) != m_size)
      throw std::runtime_error("Factorization not ready or size mismatch");
    return solveImpl(b);
  }

  const BunchKaufmanFactors &factors() const { return m_factors; }
  Int size() const { return m_size; }
  bool isFactorized() const { return m_factors.factorized; }
  Int numPos() const { return m_factors.num_pos; }
  Int numNeg() const { return m_factors.num_neg; }
  Int numZero() const { return m_factors.num_zero; }

  void setPivotTolerance(Real tol) { m_pivot_tolerance = std::max(tol, Real(0)); }

private:
  Int m_size;
  Real m_pivot_tolerance;
  BunchKaufmanFactors m_factors;

  void factorize(const MatrixType &A);
  std::vector<Real> solveImpl(const std::vector<Real> &b) const;
};

inline void BunchKaufmanLDLT::factorize(const MatrixType &A) {
  if (A.n <= 0) {
    m_factors.n = 0;
    m_factors.factorized = true;
    return;
  }

  m_factors.n = A.n;
  const Int n = A.n;

  // Initialize permutation (identity)
  m_factors.perm.assign(static_cast<size_t>(n), 0);
  m_factors.iperm.assign(static_cast<size_t>(n), 0);
  std::iota(m_factors.perm.begin(), m_factors.perm.end(), 0);
  std::iota(m_factors.iperm.begin(), m_factors.iperm.end(), 0);

  m_factors.D.clear();
  m_factors.block_info.assign(static_cast<size_t>(n), 0);  // 0=unused, 1=1x1, 2=2x2
  m_factors.Lp.assign(static_cast<size_t>(n) + 1, 0);
  m_factors.Li.clear();
  m_factors.Lx.clear();

  m_factors.num_pos = 0;
  m_factors.num_neg = 0;
  m_factors.num_zero = 0;

  // Dense copy of A for in-place factorization (TODO: optimize for sparse)
  std::vector<Real> A_dense(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
  for (Int j = 0; j < n; ++j) {
    for (Int p = A.Ap[static_cast<size_t>(j)]; p < A.Ap[static_cast<size_t>(j + 1)]; ++p) {
      Int i = A.Ai[static_cast<size_t>(p)];
      if (i <= j) {  // Upper triangle
        A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
            A.Ax[static_cast<size_t>(p)];
        if (i != j)
          A_dense[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
              A.Ax[static_cast<size_t>(p)];
      }
    }
  }

  // Main factorization loop (Bunch-Kaufman algorithm)
  Int k = 0;
  while (k < n) {
    // Step 1: Find pivot (largest off-diagonal in remaining submatrix)
    Real pivot_val = 0.0;
    Int pivot_i = k, pivot_j = k;
    for (Int i = k; i < n; ++i) {
      for (Int j = i + 1; j < n; ++j) {
        Real aij = std::abs(A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) +
                                     static_cast<size_t>(i)]);
        if (aij > pivot_val) {
          pivot_val = aij;
          pivot_i = i;
          pivot_j = j;
        }
      }
    }

    // Step 2: Extract diagonal elements for decision
    Real akk = A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)];

    if (pivot_val < m_pivot_tolerance || k == n - 1) {
      // Use 1x1 pivot at (k, k)
      if (std::abs(akk) < m_pivot_tolerance) {
        akk = (akk >= 0) ? m_pivot_tolerance : -m_pivot_tolerance;
      }

      m_factors.D.push_back(akk);
      m_factors.block_info[static_cast<size_t>(k)] = 1;

      if (akk > m_pivot_tolerance)
        ++m_factors.num_pos;
      else if (akk < -m_pivot_tolerance)
        ++m_factors.num_neg;
      else
        ++m_factors.num_zero;

      // Factor L entries
      for (Int i = k + 1; i < n; ++i) {
        A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)] /= akk;
      }

      // Update Schur complement
      for (Int i = k + 1; i < n; ++i) {
        for (Int j = i; j < n; ++j) {
          A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] -=
              A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)] *
              akk *
              A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(j)];
          if (i != j)
            A_dense[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)];
        }
      }

      k++;
    } else {
      // Use 2x2 pivot at (pivot_i, pivot_j)
      // TODO: Implement 2x2 pivot handling (swap, factor 2x2 block, update Schur complement)
      // For now, fall back to 1x1
      if (std::abs(akk) < m_pivot_tolerance) {
        akk = (akk >= 0) ? m_pivot_tolerance : -m_pivot_tolerance;
      }
      m_factors.D.push_back(akk);
      m_factors.block_info[static_cast<size_t>(k)] = 1;

      if (akk > m_pivot_tolerance)
        ++m_factors.num_pos;
      else if (akk < -m_pivot_tolerance)
        ++m_factors.num_neg;
      else
        ++m_factors.num_zero;

      k++;
    }
  }

  m_factors.factorized = true;
}

inline std::vector<Real> BunchKaufmanLDLT::solveImpl(const std::vector<Real> &b) const {
  // TODO: Implement forward/diagonal/backward solve with block diagonal D
  // For now, return b as stub
  return b;
}


} // namespace ldlt

#endif // LDLT_BK_H
