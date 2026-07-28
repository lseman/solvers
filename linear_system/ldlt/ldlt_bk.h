/*
 * ldlt_bk.h — Bunch-Kaufman LDL^T factorization for symmetric indefinite matrices
 *
 * Pivoted LDL^T with 1x1/2x2 D blocks (Bunch-Kaufman partial pivoting,
 * alpha = (1+sqrt(17))/8).
 * Factorization: A = P^T L D L^T P  (equivalently  P A P^T = L D L^T)
 * where P is the accumulated permutation from pivoting.
 */

#pragma once

#ifndef LDLT_BK_H
#define LDLT_BK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../common/sparse_csc.h"
#include "../common/trisolve.h"
#include "../simd/ldlt_simd.h"
#include "../common/dense_bk.h"

namespace ldlt {

using Int = int32_t;
using Real = double;

using linsys::SparseCSC;

struct BunchKaufmanFactors {
    std::vector< Int > perm; // perm[i] = original row now at position i
    std::vector< Real > Lx;  // L column values, grouped by column
    std::vector< Int > Li;   // L column indices (permuted space)
    std::vector< Int > Lp;   // column pointers for L (size n+1)
    // D blocks, interleaved: 1x1 block -> 1 value [d];
    // 2x2 block -> 3 values [d11, d21, d22] (symmetric block, d21 = off-diagonal)
    std::vector< Real > D;
    std::vector< Int > block_info; // block_info[k]: 1=1x1, 2=start of 2x2, 0=second col of 2x2
    Int n = 0;
    Int num_pos = 0;
    Int num_neg = 0;
    Int num_zero = 0;
    bool factorized = false;
    Real pivot_tolerance = 1e-12;
};

class BunchKaufmanLDLT {
  public:
    using MatrixType = SparseCSC< Real, Int >;

    BunchKaufmanLDLT() : m_size(0), m_pivot_tolerance(1e-12) {
    }
    explicit BunchKaufmanLDLT(const MatrixType& A) : m_size(0), m_pivot_tolerance(1e-12) {
        compute(A);
    }

    void compute(const MatrixType& A) {
        if (A.n <= 0) {
            m_factors = BunchKaufmanFactors{};
            m_factors.factorized = true;
            m_size = 0;
            return;
        }
        if (static_cast< Int >(A.Ap.size()) != A.n + 1)
            throw std::invalid_argument("Invalid matrix structure");

        m_size = A.n;
        factorize(A);
    }

    std::vector< Real > solve(const std::vector< Real >& b) const {
        if (!m_factors.factorized || static_cast< Int >(b.size()) != m_size)
            throw std::runtime_error("Factorization not ready or size mismatch");
        return solveImpl(b);
    }

    const BunchKaufmanFactors& factors() const {
        return m_factors;
    }
    Int size() const {
        return m_size;
    }
    bool isFactorized() const {
        return m_factors.factorized;
    }
    Int numPos() const {
        return m_factors.num_pos;
    }
    Int numNeg() const {
        return m_factors.num_neg;
    }
    Int numZero() const {
        return m_factors.num_zero;
    }
    void setPivotTolerance(Real tol) {
        m_pivot_tolerance = std::max(tol, Real(0));
    }

  private:
    Int m_size;
    Real m_pivot_tolerance;
    BunchKaufmanFactors m_factors;

    void factorize(const MatrixType& A);
    std::vector< Real > solveImpl(const std::vector< Real >& b) const;
};

// ── Factorization ────────────────────────────────────────────────────────────
inline void BunchKaufmanLDLT::factorize(const MatrixType& A) {
    const Int n = A.n;
    m_factors = BunchKaufmanFactors{};
    m_factors.n = n;
    m_factors.pivot_tolerance = m_pivot_tolerance;
    if (n <= 0) {
        m_factors.factorized = true;
        return;
    }

    const size_t N = static_cast< size_t >(n);
    linsys::DenseMatrix< Real > front(n, n);
    for (Int j = 0; j < n; ++j) {
        for (Int p = static_cast< Int >(A.Ap[static_cast< size_t >(j)]);
             p < static_cast< Int >(A.Ap[static_cast< size_t >(j + 1)]); ++p) {
            const Int i = A.Ai[static_cast< size_t >(p)];
            const Real v = A.Ax[static_cast< size_t >(p)];
            front(i, j) = v;
            front(j, i) = v;
        }
    }

    detail::DenseBunchKaufmanOptions< Real, Int > options;
    options.inertia_tolerance = m_pivot_tolerance;
    const auto dense_factors =
        detail::denseBunchKaufman(front, n, options);

    m_factors.num_pos = dense_factors.positive_inertia;
    m_factors.num_neg = dense_factors.negative_inertia;
    m_factors.num_zero = dense_factors.zero_inertia;
    m_factors.D.clear();
    m_factors.block_info.assign(N, 0);
    for (Int k = 0; k < n;) {
        const int8_t block =
            dense_factors.blocks[static_cast< size_t >(k)];
        m_factors.block_info[static_cast< size_t >(k)] = block;
        m_factors.D.push_back(
            dense_factors.diagonal[static_cast< size_t >(k)]);
        if (block == int8_t{2}) {
            m_factors.block_info[static_cast< size_t >(k + 1)] = 0;
            m_factors.D.push_back(
                dense_factors.subdiagonal[static_cast< size_t >(k)]);
            m_factors.D.push_back(
                dense_factors.diagonal[static_cast< size_t >(k + 1)]);
            k += 2;
        } else {
            ++k;
        }
    }

    // Extract L (strict lower triangle of the dense front) into CSC format.
    // For a 2x2 block starting at column k, L(k+1,k) = 0 (the pivot-block
    // entry left in w there is d21, not a multiplier), so start at k+2.
    m_factors.Lp.assign(N + 1, 0);
    m_factors.Li.clear();
    m_factors.Lx.clear();
    for (Int j = 0; j < n; ++j) {
        const Int first = (m_factors.block_info[static_cast< size_t >(j)] == 2) ? j + 2 : j + 1;
        for (Int i = first; i < n; ++i) {
            const Real v = front(i, j);
            if (v != Real(0)) {
                m_factors.Li.push_back(i);
                m_factors.Lx.push_back(v);
            }
        }
        m_factors.Lp[static_cast< size_t >(j + 1)] = static_cast< Int >(m_factors.Li.size());
    }

    m_factors.perm = dense_factors.permutation;
    m_factors.factorized = true;
}

// ── Solve ────────────────────────────────────────────────────────────────────
inline std::vector< Real > BunchKaufmanLDLT::solveImpl(const std::vector< Real >& b) const {
    const Int n = m_factors.n;
    if (n == 0)
        return {};

    // Permute rhs: solve (P A P^T) y = P b, with (P b)[i] = b[perm[i]]
    std::vector< Real > x(static_cast< size_t >(n));
    linsys::permute_gather(n, m_factors.perm.data(), b.data(), x.data());

    // Forward substitution: L y = P b  (plain column sweep; L is unit lower
    // and 2x2 blocks contribute no entry at (k+1,k), so blocks need no special case)
    linsys::lsolve_unit(n, m_factors.Lp.data(), m_factors.Li.data(), m_factors.Lx.data(), x.data());

    // Block-diagonal solve: D z = y
    Int ptr = 0;
    for (Int k = 0; k < n;) {
        if (m_factors.block_info[static_cast< size_t >(k)] == 1) {
            const Real d = m_factors.D[static_cast< size_t >(ptr)];
            x[static_cast< size_t >(k)] =
                (d != Real(0)) ? x[static_cast< size_t >(k)] / d : Real(0);
            ptr += 1;
            k += 1;
        } else {
            // 2x2 block [d11 d21; d21 d22]
            const Real d11 = m_factors.D[static_cast< size_t >(ptr)];
            const Real d21 = m_factors.D[static_cast< size_t >(ptr + 1)];
            const Real d22 = m_factors.D[static_cast< size_t >(ptr + 2)];
            const Real det = d11 * d22 - d21 * d21;
            const Real b0 = x[static_cast< size_t >(k)];
            const Real b1 = x[static_cast< size_t >(k + 1)];
            if (det != Real(0)) {
                x[static_cast< size_t >(k)] = (d22 * b0 - d21 * b1) / det;
                x[static_cast< size_t >(k + 1)] = (d11 * b1 - d21 * b0) / det;
            } else {
                x[static_cast< size_t >(k)] = (d11 != Real(0)) ? b0 / d11 : Real(0);
                x[static_cast< size_t >(k + 1)] = (d22 != Real(0)) ? b1 / d22 : Real(0);
            }
            ptr += 3;
            k += 2;
        }
    }

    // Backward substitution: L^T x = z
    linsys::ltsolve_unit(n, m_factors.Lp.data(), m_factors.Li.data(), m_factors.Lx.data(),
                         x.data());

    // Un-permute: y = P x_true  →  x_true[perm[i]] = y[i]
    std::vector< Real > res(static_cast< size_t >(n));
    linsys::permute_scatter(n, m_factors.perm.data(), x.data(), res.data());
    return res;
}

} // namespace ldlt

#endif // LDLT_BK_H
