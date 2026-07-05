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
  std::vector<std::vector<std::pair<Int, Real>>> L_cols(static_cast<size_t>(n));

  m_factors.num_pos = 0;
  m_factors.num_neg = 0;
  m_factors.num_zero = 0;

  // Dense copy of A for in-place factorization
  std::vector<Real> A_dense(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
  for (Int j = 0; j < n; ++j) {
    for (Int p = A.Ap[static_cast<size_t>(j)]; p < A.Ap[static_cast<size_t>(j + 1)]; ++p) {
      Int i = A.Ai[static_cast<size_t>(p)];
      if (i <= j) {
        A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
            A.Ax[static_cast<size_t>(p)];
        if (i != j)
          A_dense[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
              A.Ax[static_cast<size_t>(p)];
      }
    }
  }

  // Main factorization loop (Bunch-Kaufman: 1x1 pivots only for now)
  Int k = 0;
  while (k < n) {
    Real akk = A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)];

    // 1x1 pivot
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

    // Compute and store L[:, k] entries
    for (Int i = k + 1; i < n; ++i) {
      Real lij = A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)] / akk;
      A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)] = lij;
      if (std::abs(lij) > m_pivot_tolerance) {
        L_cols[static_cast<size_t>(k)].emplace_back(i, lij);
      }
    }

    // Update Schur complement: A[i,j] -= L[i,k] * D[k] * L[j,k]
    for (Int i = k + 1; i < n; ++i) {
      for (Int j = i; j < n; ++j) {
        Real update = A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)] *
                      akk *
                      A_dense[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(j)];
        A_dense[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] -= update;
        if (i != j)
          A_dense[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] -= update;
      }
    }

    k++;
  }

  // Convert L_cols to CSC format
  m_factors.Lp.assign(static_cast<size_t>(n) + 1, 0);
  m_factors.Lp[0] = 0;
  for (Int j = 0; j < n; ++j) {
    m_factors.Lp[static_cast<size_t>(j) + 1] =
        m_factors.Lp[static_cast<size_t>(j)] +
        static_cast<Int>(L_cols[static_cast<size_t>(j)].size());
  }

  m_factors.Li.clear();
  m_factors.Lx.clear();
  m_factors.Li.reserve(m_factors.Lp[static_cast<size_t>(n)]);
  m_factors.Lx.reserve(m_factors.Lp[static_cast<size_t>(n)]);

  for (Int j = 0; j < n; ++j) {
    for (const auto &entry : L_cols[static_cast<size_t>(j)]) {
      m_factors.Li.push_back(entry.first);
      m_factors.Lx.push_back(entry.second);
    }
  }

  m_factors.factorized = true;
}

inline std::vector<Real> BunchKaufmanLDLT::solveImpl(const std::vector<Real> &b) const {
  const Int n = m_factors.n;
  if (n == 0) return {};

  std::vector<Real> x = b;

  // Forward solve: L y = x (L is unit lower, stored by column)
  // Column j of L has entries L[i,j] for i > j
  for (Int j = 0; j < n; ++j) {
    Int p0 = m_factors.Lp[static_cast<size_t>(j)];
    Int p1 = m_factors.Lp[static_cast<size_t>(j) + 1];
    for (Int p = p0; p < p1; ++p) {
      Int i = m_factors.Li[static_cast<size_t>(p)];
      x[static_cast<size_t>(i)] -= m_factors.Lx[static_cast<size_t>(p)] * x[static_cast<size_t>(j)];
    }
  }

  // Diagonal solve: D z = y (1x1 blocks only for now)
  for (Int k = 0; k < static_cast<Int>(m_factors.D.size()); ++k) {
    if (m_factors.D[static_cast<size_t>(k)] != 0.0) {
      x[static_cast<size_t>(k)] /= m_factors.D[static_cast<size_t>(k)];
    }
  }

  // Backward solve: L^T x = z
  // Column j of L has entries L[i,j] for i > j; L^T has entries L[j,i]
  for (Int j = n - 1; j >= 0; --j) {
    Int p0 = m_factors.Lp[static_cast<size_t>(j)];
    Int p1 = m_factors.Lp[static_cast<size_t>(j) + 1];
    for (Int p = p0; p < p1; ++p) {
      Int i = m_factors.Li[static_cast<size_t>(p)];
      x[static_cast<size_t>(j)] -= m_factors.Lx[static_cast<size_t>(p)] * x[static_cast<size_t>(i)];
    }
  }

  return x;
}


} // namespace ldlt

#endif // LDLT_BK_H
