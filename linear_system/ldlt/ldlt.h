/*
 * ldlt_standalone.h — Eigen-free sparse simplicial LDLᵀ factorization
 *
 * A custom sparse LDL^T solver using only std::vector + std algorithms.
 * Same algorithm as the Eigen-backed CustomSimplicialLDLT, but with zero
 * external dependencies.
 *
 * API mirrors the Eigen version for drop-in replacement:
 *   SparseCSC<Scalar,Index>  — compressed sparse column (upper+diag)
 *   Ordering<int>            — perm/iperm
 *   SimplicialLDLT<Scalar,Index> — solver class
 *
 * Supports:
 *   - Left-looking sparse LDL^T factorization
 *   - AMD or natural ordering
 *   - Diagonal regularization for near-singular pivots
 *   - Diagonal shifting for indefinite systems
 */

#ifndef LDLT_STANDALONE_H
#define LDLT_STANDALONE_H

// GCC 16 <cstring> bug fix: <string.h> must be the very first include so
// that ::memchr, ::memcpy etc. are in the global namespace before <cstring>
// does 'using ::memchr;'.
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../common/ordering.h"
#include "../common/sparse_csc.h"
#include "../common/trisolve.h"
#include "ldlt_simd.h"

namespace ldlt {

// Shared components (see linear_system/common/):
//   SparseCSC — column-compressed sparse storage
//   Ordering  — perm/iperm pair; linsys::amd_ordering for AMD
using linsys::Ordering;
using linsys::SparseCSC;

// ===== SimplicialLDLT solver =============================================

template < typename Scalar = double, typename Index = int32_t > struct LDLFactors {
    std::vector< Index > Lp;  // size n+1
    std::vector< Index > Li;  // size nnz(L)
    std::vector< Scalar > Lx; // size nnz(L)
    std::vector< Scalar > D;  // size n
    Index n = 0;
    Index perturbed_pivots = 0;
    Scalar min_abs_pivot = Scalar{0};
    bool factorized = false;
    enum Info { Success = 0, NumericalIssue = 1, NotInitialized = 2 };
    Info info_val = NotInitialized;
};

/// Left-looking sparse simplicial LDL^T with AMD or natural ordering.
///
/// Usage:
///   SimplicialLDLT<double,int> solver;
///   solver.compute(A_csc);
///   auto x = solver.solve(b);
template < typename Scalar = double, typename Index = int32_t > class SimplicialLDLT {
  public:
    using MatrixType = SparseCSC< Scalar, Index >;
    using RealScalar = double;

    SimplicialLDLT() : m_size(0), m_info(LDLFactors< Scalar, Index >::NotInitialized) {
        m_regularization = RealScalar(1e-12);
    }

    /// Construct and compute immediately.
    explicit SimplicialLDLT(const MatrixType& a)
        : m_size(0), m_info(LDLFactors< Scalar, Index >::NotInitialized) {
        m_regularization = RealScalar(1e-12);
        compute(a);
    }

    void reset() {
        m_size = 0;
        m_factors.Lp.clear();
        m_factors.Li.clear();
        m_factors.Lx.clear();
        m_factors.D.clear();
        m_ordering.perm.clear();
        m_ordering.iperm.clear();
        m_info = LDLFactors< Scalar, Index >::NotInitialized;
        m_numPerturbedPivots = 0;
        m_minAbsPivot = RealScalar(0);
        m_regularization = RealScalar(1e-12);
        m_shiftOffset = Scalar(0);
        m_shiftScale = Scalar(1);
        m_factorized = false;
        m_patternAnalyzed = false;
    }

    void compute(const MatrixType& a) {
        analyzePattern(a);
        factorize(a);
    }

    void factorizeMatrix(const MatrixType& a) {
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n)) {
            analyzePattern(a);
        }
        factorize(a);
    }

    /// Solve Ax = b given already computed factorization.
    std::vector< Scalar > solve(const std::vector< Scalar >& b) const {
        if (!m_factorized) {
            throw std::runtime_error("ldlt: solver is not factorized");
        }
        if (static_cast< Index >(b.size()) != m_size) {
            throw std::invalid_argument("ldlt: rhs size mismatch");
        }
        return solveImpl(b);
    }

    const LDLFactors< Scalar, Index >& factors() const {
        return m_factors;
    }
    Index size() const {
        return m_size;
    }
    Index info() const {
        return m_info;
    }
    bool isFactorized() const {
        return m_factorized;
    }
    Index nonZerosL() const {
        return static_cast< Index >(m_factors.Li.size());
    }
    Index perturbedPivots() const {
        return m_numPerturbedPivots;
    }
    RealScalar minAbsPivot() const {
        return m_minAbsPivot;
    }

    void setRegularization(RealScalar eps) {
        m_regularization = std::max(eps, RealScalar(0));
    }

    void setShift(Scalar offset, Scalar scale = Scalar(1)) {
        m_shiftOffset = offset;
        m_shiftScale = scale;
    }

    const Ordering< Index >& permutation() const {
        return m_ordering;
    }

  private:
    // ===== Pattern analysis: compute ordering =====

    void analyzePattern(const MatrixType& a) {
        if (a.n <= 0) {
            m_info = LDLFactors< Scalar, Index >::NotInitialized;
            m_patternAnalyzed = false;
            m_factorized = false;
            return;
        }
        if (a.Ap.size() != static_cast< size_t >(a.n) + 1)
            throw std::invalid_argument("ldlt: Ap size mismatch");

        m_size = static_cast< Index >(a.n);

        // Build symmetric adjacency edges (structural pattern).
        std::vector< std::pair< Index, Index > > edges;
        edges.reserve(static_cast< size_t >(a.nnz()) * 2 + 1);
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                Index i = a.Ai[static_cast< size_t >(p)];
                Index r = std::min(i, j);
                Index c = std::max(i, j);
                if (r == c)
                    continue;
                edges.emplace_back(r, c);
            }
        }

        // Compute permutation: AMD if n > threshold, else natural.
        if (m_size > 20) {
            m_ordering = Ordering< Index >::from_perm(linsys::amd_ordering(m_size, edges));
        } else {
            m_ordering = Ordering< Index >::identity(m_size);
        }

        m_patternAnalyzed = true;
        m_factorized = false;
        m_info = LDLFactors< Scalar, Index >::Success;
    }

    // ===== Numeric factorization: left-looking sparse LDL^T =

    void factorize(const MatrixType& a) {
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n)) {
            analyzePattern(a);
        }
        if (!m_patternAnalyzed)
            return;

        // Build permuted lower columns: sorted (row, value) pairs per column.
        auto Acols = buildPermutedLowerColumns(a);

        // Factorization storage.
        m_factors.n = m_size;
        m_factors.D.assign(static_cast< size_t >(m_size), Scalar(0));
        m_factors.perturbed_pivots = 0;
        m_factors.min_abs_pivot = RealScalar(0);
        m_factors.factorized = false;
        m_factors.info_val = LDLFactors< Scalar, Index >::Success;

        // L column storage during factorization.
        std::vector< std::vector< std::pair< Index, Scalar > > > Lcols(
            static_cast< size_t >(m_size));

        // rowToPreviousColumns[i] = columns j < i such that L(i,j) != 0.
        std::vector< std::vector< Index > > rowToPreviousColumns(static_cast< size_t >(m_size));

        // Reusable accumulators (same type as Acols elements).
        std::vector< ColVec > w;
        w.reserve(static_cast< size_t >(m_size));
        std::vector< ColVec > merge_scratch;
        merge_scratch.reserve(static_cast< size_t >(m_size));

        for (Index k = 0; k < m_size; ++k) {
            // w = Acols[k] — copy of sorted vector.
            w = Acols[static_cast< size_t >(k)];

            // Left-looking updates from previous columns touching row k.
            const auto& contributors = rowToPreviousColumns[static_cast< size_t >(k)];
            for (Index j : contributors) {
                const auto& Lj = Lcols[static_cast< size_t >(j)];
                const Scalar L_kj = lookupInSortedColumn(Lj, k);
                if (isEffectivelyZero(L_kj))
                    continue;

                const Scalar alpha = m_factors.D[static_cast< size_t >(j)] * L_kj;

                auto start = std::lower_bound(
                    Lj.begin(), Lj.end(), k,
                    [](const std::pair< Index, Scalar >& a, Index v) { return a.first < v; });

                // Merge w and Lj[start..end) sorted by index.
                // Subtract contribution: w -= Lj * alpha where alpha = D[j] * L[k,j]
                merge_scratch.clear();
                auto wi = w.begin();
                auto li = start;
                while (wi != w.end() && li != Lj.end()) {
                    if (wi->row < li->first) {
                        merge_scratch.push_back(*wi++);
                    } else if (wi->row > li->first) {
                        merge_scratch.emplace_back(li->first, li->second * alpha);
                        ++li;
                    } else {
                        wi->val -= li->second * alpha;
                        merge_scratch.push_back(*wi++);
                        ++li;
                    }
                }
                merge_scratch.insert(merge_scratch.end(), wi, w.end());
                while (li != Lj.end()) {
                    merge_scratch.emplace_back(li->first, li->second * alpha);
                    ++li;
                }
                w = std::move(merge_scratch);
            }

            // Extract diagonal, erase from w.
            Scalar d = Scalar(0);
            auto diagIt = std::lower_bound(w.begin(), w.end(), k,
                                           [](const ColVec& a, Index v) { return a.row < v; });
            if (diagIt != w.end() && diagIt->row == k) {
                d = diagIt->val;
                w.erase(diagIt);
            }

            // Apply optional diagonal shift.
            d = d * m_shiftScale + m_shiftOffset;

            // Regularize small pivots.
            d = regularizedPivot(d);
            m_factors.D[static_cast< size_t >(k)] = d;

            if (isEffectivelyZero(d) || !std::isfinite(std::abs(d))) {
                m_factors.info_val = LDLFactors< Scalar, Index >::NumericalIssue;
                m_info = m_factors.info_val;
                return;
            }

            // Extract L entries: rows i > k.
            auto& Lk = Lcols[static_cast< size_t >(k)];
            Lk.reserve(w.size());
            for (const auto& entry : w) {
                const Index i = entry.row;
                if (i <= k)
                    continue;
                Scalar lij = entry.val / d;
                if (isEffectivelyZero(lij))
                    continue;
                Lk.emplace_back(i, lij);
                rowToPreviousColumns[static_cast< size_t >(i)].push_back(k);
            }
            w.clear();
        }

        // Build L in compressed column (CSC) format from Lcols.
        // Each Lj contains (row, value) for rows > j.
        std::vector< Index > Lp_count(static_cast< size_t >(m_size) + 1, 0);
        for (Index j = 0; j < m_size; ++j) {
            Lp_count[static_cast< size_t >(j) + 1] =
                static_cast< Index >(Lcols[static_cast< size_t >(j)].size());
        }
        for (Index j = 0; j < m_size; ++j) {
            Lp_count[static_cast< size_t >(j) + 1] += Lp_count[static_cast< size_t >(j)];
        }

        Index nnzL = Lp_count[static_cast< size_t >(m_size)];
        m_factors.Lp.resize(static_cast< size_t >(m_size) + 1);
        m_factors.Li.resize(static_cast< size_t >(nnzL));
        m_factors.Lx.resize(static_cast< size_t >(nnzL));
        m_factors.Lp[0] = 0;
        for (Index j = 0; j <= m_size; ++j) {
            m_factors.Lp[static_cast< size_t >(j)] = Lp_count[static_cast< size_t >(j)];
        }

        std::vector< Index > pos(static_cast< size_t >(m_size) + 1, 0);
        for (Index j = 0; j < m_size; ++j) {
            const auto& Lj = Lcols[static_cast< size_t >(j)];
            for (const auto& e : Lj) {
                Index p = m_factors.Lp[static_cast< size_t >(j)] + pos[static_cast< size_t >(j)]++;
                m_factors.Li[static_cast< size_t >(p)] = e.first;
                m_factors.Lx[static_cast< size_t >(p)] = e.second;
            }
        }

        m_factors.n = m_size;
        m_factors.factorized = true;
        m_factorized = true;
        m_info = m_factors.info_val;
    }

    // ===== Solve: forward + diagonal + backward + un-permute =============

    std::vector< Scalar > solveImpl(const std::vector< Scalar >& b) const {
        std::vector< Scalar > x = b;

        // Permute: x_perm[new] = b[old] = b[iperm[new]].
        if (!m_ordering.iperm.empty()) {
            std::vector< Scalar > y(static_cast< size_t >(m_size));
            linsys::permute_gather(m_size, m_ordering.iperm.data(), x.data(), y.data());
            x = std::move(y);
        }

        // Forward solve: L y = x, L is unit lower.
        linsys::lsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(),
                            m_factors.Lx.data(), x.data());

        // Diagonal solve: D z = y (scale by 1/D, use SIMD if available).
        std::vector< Scalar > d_inv(static_cast< size_t >(m_size));
        for (Index k = 0; k < m_size; ++k) {
            d_inv[static_cast< size_t >(k)] = Scalar(1) / m_factors.D[static_cast< size_t >(k)];
        }
        // Use SIMD-accelerated scale if Scalar is double
        if constexpr (std::is_same_v< Scalar, double >) {
            simd_scale_array(reinterpret_cast< double* >(x.data()),
                             reinterpret_cast< const double* >(d_inv.data()),
                             static_cast< size_t >(m_size));
        } else {
            // Parallel scale for non-double types (e.g., float)
            par_for_n(static_cast< size_t >(m_size), [&](std::size_t k) { x[k] *= d_inv[k]; });
        }

        // Backward solve: L^T x = z.
        linsys::ltsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(),
                             m_factors.Lx.data(), x.data());

        // Un-permute: x_original = P^T x_permuted.
        // P[old] = new, so x_old = x_perm[P[old]].
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        if (!m_ordering.perm.empty()) {
            linsys::permute_gather(m_size, m_ordering.perm.data(), x.data(), result.data());
        } else {
            result = std::move(x);
        }

        return result;
    }

    // ===== Helpers =======================================================

    static bool isEffectivelyZero(const Scalar& x) {
        return x == Scalar(0);
    }

    Scalar regularizedPivot(Scalar d) {
        RealScalar absd = std::abs(d);
        if (m_minAbsPivot == RealScalar(0) || absd < m_minAbsPivot) {
            m_minAbsPivot = absd;
        }
        if (absd >= m_regularization || m_regularization == RealScalar(0)) {
            return d;
        }
        m_numPerturbedPivots++;
        if (d < RealScalar(0)) {
            return Scalar(-m_regularization);
        }
        return Scalar(m_regularization);
    }

    struct ColVec {
        Index row;
        Scalar val;
        bool operator<(const ColVec& other) const {
            return row < other.row;
        }
    };

    std::vector< std::vector< ColVec > > buildPermutedLowerColumns(const MatrixType& a) const {
        const auto& pinv = m_ordering.iperm;

        // Count entries per column for reserve.
        std::vector< size_t > colCounts(static_cast< size_t >(m_size), 0);
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                Index old_r = a.Ai[static_cast< size_t >(p)];
                Index nr = static_cast< Index >(pinv[static_cast< size_t >(old_r)]);
                Index nc = static_cast< Index >(pinv[static_cast< size_t >(j)]);
                if (nr >= nc) {
                    colCounts[static_cast< size_t >(nc)]++;
                } else {
                    colCounts[static_cast< size_t >(nr)]++;
                }
            }
        }

        std::vector< std::vector< ColVec > > Acols(static_cast< size_t >(m_size));
        for (Index k = 0; k < m_size; ++k) {
            Acols[static_cast< size_t >(k)].reserve(colCounts[static_cast< size_t >(k)]);
        }

        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                Index old_r = a.Ai[static_cast< size_t >(p)];
                Scalar v = a.Ax[static_cast< size_t >(p)];

                Index nr = static_cast< Index >(pinv[static_cast< size_t >(old_r)]);
                Index nc = static_cast< Index >(pinv[static_cast< size_t >(j)]);

                // Only process lower-triangular entries (nr >= nc) to avoid double-counting
                // when both (i,j) and (j,i) are present in the CSC.
                if (nr < nc)
                    continue;
                Acols[static_cast< size_t >(nc)].push_back({nr, v});
            }
        }

        // Sort and dedup each column.
        for (Index k = 0; k < m_size; ++k) {
            auto& col = Acols[static_cast< size_t >(k)];
            if (col.empty())
                continue;
            std::sort(col.begin(), col.end());
            size_t w = 1;
            for (size_t i = 1; i < col.size(); ++i) {
                if (col[static_cast< size_t >(i)].row == col[static_cast< size_t >(w - 1)].row) {
                    col[static_cast< size_t >(w - 1)].val += col[static_cast< size_t >(i)].val;
                } else {
                    col[static_cast< size_t >(w++)] = col[static_cast< size_t >(i)];
                }
            }
            col.resize(static_cast< size_t >(w));
        }

        return Acols;
    }

    static Scalar lookupInSortedColumn(const std::vector< std::pair< Index, Scalar > >& col,
                                       Index row) {
        auto it = std::lower_bound(
            col.begin(), col.end(), row,
            [](const std::pair< Index, Scalar >& a, Index r) { return a.first < r; });
        if (it != col.end() && it->first == row)
            return it->second;
        return Scalar(0);
    }

    // ===== State ========================================================

    Index m_size = 0;
    LDLFactors< Scalar, Index > m_factors;
    Ordering< Index > m_ordering;
    bool m_patternAnalyzed = false;
    bool m_factorized = false;
    Index m_info = LDLFactors< Scalar, Index >::NotInitialized;

    Index m_numPerturbedPivots = 0;
    RealScalar m_minAbsPivot = RealScalar(0);
    RealScalar m_regularization = RealScalar(1e-12);
    Scalar m_shiftOffset = Scalar(0);
    Scalar m_shiftScale = Scalar(1);
};

} // namespace ldlt

#endif // LDLT_STANDALONE_H
