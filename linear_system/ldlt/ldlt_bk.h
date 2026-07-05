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
#include "ldlt_simd.h"

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

    // Dense symmetric working copy of A (row-major, both triangles kept in sync)
    std::vector< Real > w(N * N, 0.0);
    auto W = [&w, N](Int i, Int j) -> Real& {
        return w[static_cast< size_t >(i) * N + static_cast< size_t >(j)];
    };
    for (Int j = 0; j < n; ++j) {
        for (Int p = static_cast< Int >(A.Ap[static_cast< size_t >(j)]);
             p < static_cast< Int >(A.Ap[static_cast< size_t >(j + 1)]); ++p) {
            const Int i = A.Ai[static_cast< size_t >(p)];
            const Real v = A.Ax[static_cast< size_t >(p)];
            W(i, j) = v;
            W(j, i) = v;
        }
    }

    // Permutation: perm[i] = original row index now at position i
    std::vector< Int > perm(N);
    std::iota(perm.begin(), perm.end(), 0);

    // Symmetric row+col swap in w, tracked in perm. Swapping full rows also
    // permutes the rows of already-computed L multipliers stored in the strict
    // lower triangle of factored columns (LAPACK-style), keeping L consistent
    // with the final permutation.
    auto symswap = [&](Int a, Int b) {
        if (a == b)
            return;
        for (Int j = 0; j < n; ++j)
            std::swap(W(a, j), W(b, j));
        for (Int i = 0; i < n; ++i)
            std::swap(W(i, a), W(i, b));
        std::swap(perm[static_cast< size_t >(a)], perm[static_cast< size_t >(b)]);
    };

    m_factors.D.clear();
    m_factors.block_info.assign(N, 0);

    const Real alpha = (1.0 + std::sqrt(17.0)) / 8.0; // Bunch-Kaufman constant
    const Real tol = m_pivot_tolerance;

    // Scratch: saved pivot column(s) and 2x2 multipliers
    std::vector< Real > c0(N), c1(N), mx(N), my(N);

    Int k = 0;
    while (k < n) {
        // lambda = max |W(i,k)| below the diagonal, r = its row
        Real lambda = 0.0;
        Int r = k;
        for (Int i = k + 1; i < n; ++i) {
            const Real v = std::abs(W(i, k));
            if (v > lambda) {
                lambda = v;
                r = i;
            }
        }
        const Real absakk = std::abs(W(k, k));

        // Bunch-Kaufman pivot selection
        bool two_by_two = false;
        Int swap_row = k;
        if (lambda > 0.0 && absakk < alpha * lambda) {
            // sigma = max off-diagonal magnitude in column r of the trailing submatrix
            Real sigma = 0.0;
            for (Int i = k; i < n; ++i) {
                if (i != r)
                    sigma = std::max(sigma, std::abs(W(i, r)));
            }
            if (absakk * sigma >= alpha * lambda * lambda) {
                // 1x1 pivot with W(k,k)
            } else if (std::abs(W(r, r)) >= alpha * sigma) {
                swap_row = r; // 1x1 pivot with W(r,r)
            } else {
                two_by_two = true; // 2x2 pivot with rows k and r
            }
        }

        if (!two_by_two) {
            // ─── 1x1 pivot ──────────────────────────────────────────────────────
            symswap(k, swap_row);

            const Real d = W(k, k);
            m_factors.D.push_back(d);
            m_factors.block_info[static_cast< size_t >(k)] = 1;
            if (d > tol)
                ++m_factors.num_pos;
            else if (d < -tol)
                ++m_factors.num_neg;
            else
                ++m_factors.num_zero;

            // Save pivot column, overwrite it with multipliers L(i,k) = W(i,k) / d
            const Real dinv = (d != Real(0)) ? Real(1) / d : Real(0);
            for (Int i = k + 1; i < n; ++i) {
                c0[static_cast< size_t >(i)] = W(i, k);
                W(i, k) = c0[static_cast< size_t >(i)] * dinv;
            }

            // Schur complement: W(i,j) -= c0[i] * c0[j] / d, both triangles
            for (Int j = k + 1; j < n; ++j) {
                const Real s = c0[static_cast< size_t >(j)] * dinv;
                if (s == Real(0))
                    continue;
                for (Int i = j; i < n; ++i) {
                    const Real upd = c0[static_cast< size_t >(i)] * s;
                    W(i, j) -= upd;
                    if (i != j)
                        W(j, i) -= upd;
                }
            }

            ++k;
        } else {
            // ─── 2x2 pivot ──────────────────────────────────────────────────────
            symswap(k + 1, r);

            const Real d11 = W(k, k);
            const Real d21 = W(k + 1, k);
            const Real d22 = W(k + 1, k + 1);
            const Real det = d11 * d22 - d21 * d21;

            m_factors.D.push_back(d11);
            m_factors.D.push_back(d21);
            m_factors.D.push_back(d22);
            m_factors.block_info[static_cast< size_t >(k)] = 2;
            m_factors.block_info[static_cast< size_t >(k + 1)] = 0;

            // Inertia of the 2x2 block (BK 2x2 pivots have det < 0 by construction)
            if (det < 0) {
                ++m_factors.num_pos;
                ++m_factors.num_neg;
            } else if (det > 0) {
                if (d11 + d22 > 0)
                    m_factors.num_pos += 2;
                else
                    m_factors.num_neg += 2;
            } else {
                const Real trace = d11 + d22;
                if (trace > tol) {
                    ++m_factors.num_pos;
                    ++m_factors.num_zero;
                } else if (trace < -tol) {
                    ++m_factors.num_neg;
                    ++m_factors.num_zero;
                } else {
                    m_factors.num_zero += 2;
                }
            }

            // Save pivot columns, overwrite them with multipliers
            // [mx;my] = inv([d11 d21; d21 d22]) * [c0; c1]
            const Real detinv = (det != Real(0)) ? Real(1) / det : Real(0);
            for (Int i = k + 2; i < n; ++i) {
                const Real b0 = W(i, k);
                const Real b1 = W(i, k + 1);
                c0[static_cast< size_t >(i)] = b0;
                c1[static_cast< size_t >(i)] = b1;
                const Real x = (b0 * d22 - b1 * d21) * detinv;
                const Real y = (b1 * d11 - b0 * d21) * detinv;
                mx[static_cast< size_t >(i)] = x;
                my[static_cast< size_t >(i)] = y;
                W(i, k) = x;
                W(i, k + 1) = y;
            }

            // Schur complement: W(i,j) -= [mx_i my_i] * B * [mx_j; my_j]
            //                          = mx_i * c0[j] + my_i * c1[j]   (B * m_j = c_j)
            for (Int j = k + 2; j < n; ++j) {
                for (Int i = j; i < n; ++i) {
                    const Real upd = mx[static_cast< size_t >(i)] * c0[static_cast< size_t >(j)] +
                                     my[static_cast< size_t >(i)] * c1[static_cast< size_t >(j)];
                    W(i, j) -= upd;
                    if (i != j)
                        W(j, i) -= upd;
                }
            }

            k += 2;
        }
    }

    // Extract L (strict lower triangle of w, multipliers) into CSC format.
    // For a 2x2 block starting at column k, L(k+1,k) = 0 (the pivot-block
    // entry left in w there is d21, not a multiplier), so start at k+2.
    m_factors.Lp.assign(N + 1, 0);
    m_factors.Li.clear();
    m_factors.Lx.clear();
    for (Int j = 0; j < n; ++j) {
        const Int first = (m_factors.block_info[static_cast< size_t >(j)] == 2) ? j + 2 : j + 1;
        for (Int i = first; i < n; ++i) {
            const Real v = W(i, j);
            if (v != Real(0)) {
                m_factors.Li.push_back(i);
                m_factors.Lx.push_back(v);
            }
        }
        m_factors.Lp[static_cast< size_t >(j + 1)] = static_cast< Int >(m_factors.Li.size());
    }

    m_factors.perm = std::move(perm);
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
    linsys::lsolve_unit(n, m_factors.Lp.data(), m_factors.Li.data(), m_factors.Lx.data(),
                        x.data());

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
