/*
 * supernodal_ldlt_standalone.h — Eigen-free supernodal sparse LDLᵀ factorization
 *
 * Supernodal sparse LDL^T using dense BLAS kernels on supernodes + simplicial fallback.
 * Uses snode::identify_supernodes (supernodes.h) for supernode detection.
 *
 * API mirrors the Eigen version for drop-in replacement:
 *   SparseCSC<Scalar,Index>      — compressed sparse column (upper+diag)
 *   Ordering<int>                — perm/iperm
 *   SupernodalFactor             — L (CSC) + D + metadata
 *   SupernodalLDLT<Scalar,Index> — solver class
 *
 * Algorithm:
 *   - Pattern analysis: AMD ordering + supernode detection via supernodes.h
 *   - Numeric factorization: supernodal dense BLAS (primary) or simplicial fallback
 *   - Solve: forward (L) + diagonal (D) + backward (Lᵀ) + permutation
 *
 * Usage:
 *   SupernodalLDLT<double,int> solver;
 *   solver.compute(csc);
 *   auto x = solver.solve(b);
 */

#ifndef SUPERSONAL_LDLT_STANDALONE_H
#define SUPERSONAL_LDLT_STANDALONE_H

// GCC 16 <cstring> bug fix: <string.h> must be the very first include so
// that ::memchr, ::memcpy etc. are in the global namespace before <cstring>
// does 'using ::memchr;'.
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===== AMD ordering ========================================================
// amd.h must be in global scope: it includes <iostream> which references C
// symbols (::memchr, ::__libc_single_threaded etc.) that would otherwise
// resolve to supernodal:: inside the namespace block.
#include "amd.h"

// ===== Supernode detection =================================================
#include "supernodes.h"

namespace supernodal {

using Int = int32_t;
using Real = double;

// ===== Sparse CSC storage (column-compressed) ==============================

template < typename Scalar = Real, typename Index = Int >
struct SparseCSC {
    Index n = 0;
    std::vector< Index > Ap;  // column pointers, size n+1
    std::vector< Index > Ai;  // row indices, size nnz
    std::vector< Scalar > Ax; // values, size nnz

    SparseCSC() = default;

    explicit SparseCSC(Index n_) : n(n_), Ap(static_cast< size_t >(n_) + 1, 0) {
        if (n < 0)
            throw std::invalid_argument("supernodal: n < 0");
    }

    SparseCSC(Index n_, std::vector< Index > Ap_, std::vector< Index > Ai_,
              std::vector< Scalar > Ax_)
        : n(n_), Ap(std::move(Ap_)), Ai(std::move(Ai_)), Ax(std::move(Ax_)) {
        if (n < 0)
            throw std::invalid_argument("supernodal: n < 0");
        if (Ap.size() != static_cast< size_t >(n) + 1)
            throw std::invalid_argument("supernodal: Ap size != n+1");
        auto nnz = Ap.back();
        if (static_cast< size_t >(nnz) != Ai.size() || static_cast< size_t >(nnz) != Ax.size())
            throw std::invalid_argument("supernodal: Ai/Ax size mismatch");
    }

    [[nodiscard]] size_t nnz() const {
        return static_cast< size_t >(Ap.back());
    }
};

// ===== Ordering ============================================================

template < typename Index = Int >
struct Ordering {
    std::vector< Index > perm;   // size n: perm[old] = new
    std::vector< Index > iperm;  // size n: iperm[new] = old
    Index n = 0;

    static Ordering identity(Index n) {
        Ordering o;
        o.n = n;
        o.perm.resize(static_cast< size_t >(n));
        o.iperm.resize(static_cast< size_t >(n));
        std::iota(o.perm.begin(), o.perm.end(), Index{0});
        std::iota(o.iperm.begin(), o.iperm.end(), Index{0});
        return o;
    }

    /// Build from a permutation array (old->new). Computes iperm.
    static Ordering from_perm(std::vector< Index > p) {
        Ordering o;
        o.n = static_cast< Index >(p.size());
        o.perm = std::move(p);
        o.iperm.assign(static_cast< size_t >(o.n), Index{-1});

        for (Index i = 0; i < o.n; ++i) {
            const Index pi = o.perm[static_cast< size_t >(i)];
            if (pi < 0 || pi >= o.n)
                throw std::invalid_argument("supernodal: invalid permutation");
            o.iperm[static_cast< size_t >(pi)] = i;
        }
        return o;
    }
};

// ===== AMD ordering helper (must be before class since computeOrdering() calls it) ===

namespace detail {

/// AMD ordering via amd.h's AMDReorderingArray.
inline std::vector< int32_t > amd_ordering(int32_t n,
                                           const std::vector< std::pair< int32_t, int32_t > >& edges) {
    if (n <= 0)
        return {};

    // Build CSR from upper-triangular edge list.
    std::vector< int32_t > row_counts(static_cast< size_t >(n), 0);
    for (const auto& e : edges) {
        if (e.first >= 0 && e.first < n && e.second >= 0 && e.second < n) {
            row_counts[static_cast< size_t >(e.first)]++;
        }
    }

    std::vector< int32_t > row_indptr(static_cast< size_t >(n) + 1, 0);
    for (int32_t i = 0; i < n; ++i) {
        row_indptr[static_cast< size_t >(i) + 1] = row_indptr[static_cast< size_t >(i)] + row_counts[static_cast< size_t >(i)];
    }
    int32_t nnz = row_indptr[static_cast< size_t >(n)];

    std::vector< int32_t > row_indices(static_cast< size_t >(nnz));
    std::vector< int32_t > row_cur(static_cast< size_t >(n), 0);
    for (const auto& e : edges) {
        if (e.first >= 0 && e.first < n && e.second >= 0 && e.second < n) {
            row_indices[static_cast< size_t >(row_indptr[static_cast< size_t >(e.first)] +
                                row_cur[static_cast< size_t >(e.first)]++)] = e.second;
        }
    }

    CSR csr;
    csr.n = n;
    csr.indptr = std::move(row_indptr);
    csr.indices = std::move(row_indices);

    AMDReorderingArray amd_orderer;
    auto perm = amd_orderer.amd_order(csr, true); // symmetrize
    return perm;
}

} // namespace detail

// ===== Dense matrix (simple, no Eigen) =====================================

struct DenseMatrix {
    std::vector< double > data;
    Int rows = 0;
    Int cols = 0;

    DenseMatrix() = default;
    explicit DenseMatrix(Int m, Int n)
        : data(static_cast< size_t >(m) * static_cast< size_t >(n)), rows(m), cols(n) {}

    inline double& operator()(Int i, Int j) {
        return data[static_cast< size_t >(j) * static_cast< size_t >(rows) + static_cast< size_t >(i)];
    }
    inline const double& operator()(Int i, Int j) const {
        return data[static_cast< size_t >(j) * static_cast< size_t >(rows) + static_cast< size_t >(i)];
    }

    void resize(Int m, Int n) {
        rows = m;
        cols = n;
        data.assign(static_cast< size_t >(m) * static_cast< size_t >(n), 0.0);
    }
    void setZero() {
        std::fill(data.begin(), data.end(), 0.0);
    }
};

// ===== Factorization result =================================================

struct SupernodalFactor {
    // L in CSC format (unit lower triangular)
    Int n = 0;
    std::vector< Int > Lp;   // size n+1
    std::vector< Int > Li;   // size nnz(L)
    std::vector< double > Lx; // size nnz(L)

    // D diagonal
    std::vector< double > D; // size n

    // Supernode metadata
    std::vector< std::pair< Int, Int > > supernode_ranges;
    std::vector< Int > etree;

    // Permutation (from analysis phase)
    std::vector< Int > perm;   // perm[old] = new
    std::vector< Int > iperm;  // iperm[new] = old

    // Metadata
    Int perturbed_pivots = 0;
    double min_abs_pivot = 0.0;
    bool factorized = false;

    enum Info { Success = 0, NumericalIssue = 1, NotInitialized = 2 };
    Info info_val = NotInitialized;

    size_t nnzL() const { return Li.size(); }
};

// ===== Supernodal LDLᵀ solver ==============================================

template < typename Scalar = Real, typename Index = Int >
class SupernodalLDLT {
  public:
    using MatrixType = SparseCSC< Scalar, Index >;
    using RealScalar = double;

    SupernodalLDLT()
        : m_size(0), m_info(SupernodalFactor::NotInitialized) {
        m_regularization = 1e-12;
    }

    explicit SupernodalLDLT(const MatrixType& a)
        : m_size(0), m_info(SupernodalFactor::NotInitialized) {
        m_regularization = 1e-12;
        compute(a);
    }

    void reset() {
        m_size = 0;
        m_factors.n = 0;
        m_factors.Lp.clear();
        m_factors.Li.clear();
        m_factors.Lx.clear();
        m_factors.D.clear();
        m_factors.supernode_ranges.clear();
        m_factors.etree.clear();
        m_factors.perm.clear();
        m_factors.iperm.clear();
        m_ordering.perm.clear();
        m_ordering.iperm.clear();
        m_ordering.n = 0;
        m_info = SupernodalFactor::NotInitialized;
        m_numPerturbedPivots = 0;
        m_minAbsPivot = RealScalar(0);
        m_regularization = 1e-12;
        m_factorized = false;
        m_patternAnalyzed = false;
        m_useSupernodalFactorization = false;
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

    /// Enable/disable supernodal (dense-BLAS-on-supernodes) factorization.
    /// Default: disabled (uses simplicial fallback for correctness).
    void setSupernodalFactorization(bool on) {
        m_useSupernodalFactorization = on;
    }

    bool supernodalFactorization() const {
        return m_useSupernodalFactorization;
    }

    /// Solve Ax = b given already computed factorization.
    std::vector< Scalar > solve(const std::vector< Scalar >& b) const {
        if (!m_factorized) {
            throw std::runtime_error("supernodal: solver is not factorized");
        }
        if (static_cast< Index >(b.size()) != m_size) {
            throw std::invalid_argument("supernodal: rhs size mismatch");
        }
        return solveImpl(b);
    }

    const SupernodalFactor& factors() const { return m_factors; }
    Index size() const { return m_size; }
    Index info() const { return m_info; }
    bool isFactorized() const { return m_factorized; }
    Index nonZerosL() const { return static_cast< Index >(m_factors.Li.size()); }
    Index perturbedPivots() const { return m_factors.perturbed_pivots; }
    Scalar minAbsPivot() const { return static_cast< Scalar >(m_minAbsPivot); }

    void setRegularization(RealScalar eps) {
        m_regularization = std::max(eps, RealScalar(0));
    }

    const Ordering< Index >& permutation() const { return m_ordering; }

    const std::vector< std::pair< Int, Int > >& supernodeRanges() const {
        return m_factors.supernode_ranges;
    }

    const std::vector< Int >& etree() const { return m_factors.etree; }

    bool isSupernodal() const { return m_supernodal; }

  private:
    // ===== Dense LDLᵀ on a frontal matrix (used within supernodes) ==========
    // Factorizes F(0:npiv, 0:fsize) in-place. D_local gets diagonal.
    // Returns number of perturbed pivots.

    static Int denseLDLT(
        DenseMatrix& F, Int fsize, Int npiv,
        std::vector< double >& D_local,
        int& numPerturbed, double& minAbsPivot,
        double regularization) {
        Int perturbed = 0;

        for (Int k = 0; k < npiv; ++k) {
            double d = F(k, k);

            // Regularize near-zero pivots.
            if (std::abs(d) < regularization) {
                d = (d < 0.0 ? -regularization : regularization);
                ++perturbed;
            }
            D_local[static_cast< size_t >(k)] = d;

            double absd = std::abs(d);
            if (minAbsPivot == 0.0 || absd < minAbsPivot) {
                minAbsPivot = absd;
            }

            double dinv = 1.0 / d;

            // Scale L column: F(i,k) *= dinv for i > k.
            for (Int i = k + 1; i < fsize; ++i) {
                F(i, k) *= dinv;
            }

            // Rank-1 update: F(j,j:fsize) -= F(k,j) * F(k,k:fsize) for j > k.
            for (Int j = k + 1; j < fsize; ++j) {
                double ljk = F(j, k);
                if (ljk == 0.0)
                    continue;
                double dljk = ljk * d;
                for (Int i = j; i < fsize; ++i) {
                    F(i, j) -= F(i, k) * dljk;
                }
            }
        }

        numPerturbed += perturbed;
        return perturbed;
    }

    // ===== Pattern analysis: compute ordering + supernodes ==================

    void analyzePattern(const MatrixType& a) {
        if (a.n <= 0) {
            m_info = SupernodalFactor::NotInitialized;
            m_patternAnalyzed = false;
            m_factorized = false;
            return;
        }
        if (a.Ap.size() != static_cast< size_t >(a.n) + 1)
            throw std::invalid_argument("supernodal: Ap size mismatch");

        m_size = static_cast< Index >(a.n);

        // Compute permutation: AMD if n > threshold, else natural.
        computeOrdering(a);

        // Detect supernodes using snode::identify_supernodes.
        computeSupernodes(a);

        m_patternAnalyzed = true;
        m_factorized = false;
        m_info = SupernodalFactor::Success;
    }

    // ===== Numeric factorization: supernodal or simplicial ==================

    void factorize(const MatrixType& a) {
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n)) {
            analyzePattern(a);
        }
        if (!m_patternAnalyzed) return;

        if (m_useSupernodalFactorization && m_supernodal) {
            factorizeSupernodal(a);
        } else {
            factorizeSimplicial(a);
        }

        m_factorized = (m_info == SupernodalFactor::Success);
    }

    // ===== Simplicial (single-column) fallback ==============================
    // Left-looking sparse LDLᵀ with our pre-computed ordering.

    struct ColEntry {
        Int row;
        Scalar val;
    };

    void factorizeSimplicial(const MatrixType& a) {
        m_factors.n = m_size;
        m_factors.D.assign(static_cast< size_t >(m_size), Scalar(0));
        m_factors.perturbed_pivots = 0;
        m_factors.min_abs_pivot = RealScalar(0);
        m_factors.factorized = false;
        m_factors.info_val = SupernodalFactor::Success;

        // Build permuted lower columns.
        auto Acols = buildPermutedLowerColumns(a);

        std::vector< std::vector< ColEntry > > Lcols(static_cast< size_t >(m_size));
        std::vector< std::vector< Int > > rowToPreviousColumns(static_cast< size_t >(m_size));

        std::vector< ColEntry > w;
        w.reserve(static_cast< size_t >(m_size));
        std::vector< ColEntry > merge_scratch;
        merge_scratch.reserve(static_cast< size_t >(m_size));

        for (Index k = 0; k < m_size; ++k) {
            // w = Acols[k].
            w.clear();
            for (auto ait = Acols[static_cast< size_t >(k)].begin();
                 ait != Acols[static_cast< size_t >(k)].end(); ++ait) {
                w.emplace_back(static_cast< Int >(ait->first), ait->second);
            }

            // Left-looking updates from previous columns touching row k.
            const auto& contributors = rowToPreviousColumns[static_cast< size_t >(k)];
            for (Int j : contributors) {
                const auto& Lj = Lcols[static_cast< size_t >(j)];
                const Scalar L_kj = lookupInSortedColumn(Lj, k);
                if (isEffectivelyZero(L_kj))
                    continue;

                const Scalar alpha = m_factors.D[static_cast< size_t >(j)] * L_kj;

                auto start = std::lower_bound(Lj.begin(), Lj.end(), k,
                                              [](const ColEntry& a, Int v) { return a.row < v; });

                merge_scratch.clear();
                auto wi = w.begin();
                auto li = start;
                while (wi != w.end() && li != Lj.end()) {
                    if (wi->row < li->row) {
                        merge_scratch.push_back(*wi++);
                    } else if (wi->row > li->row) {
                        merge_scratch.emplace_back(li->row, li->val * alpha);
                        ++li;
                    } else {
                        wi->val -= li->val * alpha;
                        merge_scratch.push_back(*wi++);
                        ++li;
                    }
                }
                merge_scratch.insert(merge_scratch.end(), wi, w.end());
                while (li != Lj.end()) {
                    merge_scratch.emplace_back(li->row, li->val * alpha);
                    ++li;
                }
                w = std::move(merge_scratch);
            }

            // Extract diagonal, erase from w.
            Scalar d = Scalar(0);
            auto diagIt = std::lower_bound(w.begin(), w.end(), k,
                                           [](const ColEntry& a, Int v) { return a.row < v; });
            if (diagIt != w.end() && diagIt->row == k) {
                d = diagIt->val;
                w.erase(diagIt);
            }

            // Regularize small pivots.
            d = regularizedPivot(d);
            m_factors.D[static_cast< size_t >(k)] = d;

            if (isEffectivelyZero(d) || !std::isfinite(std::abs(d))) {
                m_factors.info_val = SupernodalFactor::NumericalIssue;
                m_info = m_factors.info_val;
                return;
            }

            // Extract L entries: rows i > k.
            auto& Lk = Lcols[static_cast< size_t >(k)];
            Lk.reserve(w.size());
            for (auto wit = w.begin(); wit != w.end(); ++wit) {
                const Int i = wit->row;
                if (i <= k)
                    continue;
                Scalar lij = wit->val / d;
                if (isEffectivelyZero(lij))
                    continue;
                Lk.emplace_back(i, lij);
                rowToPreviousColumns[static_cast< size_t >(i)].push_back(k);
            }
            w.clear();
        }

        // Build L in compressed column (CSC) format from Lcols.
        std::vector< Int > Lp_count(static_cast< size_t >(m_size) + 1, 0);
        for (Int j = 0; j < m_size; ++j) {
            Lp_count[static_cast< size_t >(j) + 1] =
                static_cast< Int >(Lcols[static_cast< size_t >(j)].size());
        }
        for (Int j = 0; j < m_size; ++j) {
            Lp_count[static_cast< size_t >(j) + 1] += Lp_count[static_cast< size_t >(j)];
        }

        Index nnzL = Lp_count[static_cast< size_t >(m_size)];
        m_factors.Lp.resize(static_cast< size_t >(m_size) + 1);
        m_factors.Li.resize(static_cast< size_t >(nnzL));
        m_factors.Lx.resize(static_cast< size_t >(nnzL));
        m_factors.Lp[0] = 0;
        for (Int j = 0; j <= m_size; ++j) {
            m_factors.Lp[static_cast< size_t >(j)] = Lp_count[static_cast< size_t >(j)];
        }

        std::vector< Int > pos(static_cast< size_t >(m_size) + 1, 0);
        for (Int j = 0; j < m_size; ++j) {
            const auto& Lj = Lcols[static_cast< size_t >(j)];
            for (auto lit = Lj.begin(); lit != Lj.end(); ++lit) {
                Index p = m_factors.Lp[static_cast< size_t >(j)] +
                          pos[static_cast< size_t >(j)]++;
                m_factors.Li[static_cast< size_t >(p)] = lit->row;
                m_factors.Lx[static_cast< size_t >(p)] = lit->val;
            }
        }

        m_factors.n = m_size;
        m_factors.factorized = true;
        m_factorized = true;
        m_info = m_factors.info_val;
    }

    // ===== Supernodal factorization =========================================
    //
    // Fundamental supernode structure from supernodes.h:
    //   A supernode S spans columns [lo, hi] where:
    //     - etree[k] = k+1 for all k in [lo, hi-1]  (chain)
    //     - all columns share the same L-pattern below the diagonal
    //
    //   Children of supernode S are supernodes C where
    //     etree[hi(C)] ∈ columns of S.
    //
    //   Processing order: postorder of supernode etree (children before parents).
    //
    //   For each supernode:
    //     1. Gather A(p,q) for p,q in pivot set + update rows.
    //     2. Accumulate children Schur complements into the frontal matrix.
    //     3. Dense LDLᵀ on the frontal matrix (BLAS-level 3).
    //     4. Extract L entries and Schur complement for parent.

    struct SupernodeData {
        Int lo, hi;
        Int npiv;
        Int parent_sn;
        std::vector< Int > children;
        std::vector< Int > update_rows;
        DenseMatrix schur; // Schur complement (update/update block)
    };

    void factorizeSupernodal(const MatrixType& a) {
        m_factors.n = m_size;
        m_factors.D.assign(static_cast< size_t >(m_size), 0.0);
        m_factors.perturbed_pivots = 0;
        m_factors.min_abs_pivot = 0.0;
        m_factors.factorized = false;
        m_factors.info_val = SupernodalFactor::Success;

        // Build permutation arrays.
        // m_factors.iperm[i] = original row that maps to permuted position i.
        // So A_perm[i,j] = A_orig[iperm[i], iperm[j]].
        std::vector< Int > p_inv(static_cast< size_t >(m_size));
        const auto& iperm_idx = m_factors.iperm;
        for (Int i = 0; i < m_size; ++i) {
            p_inv[static_cast< size_t >(i)] = iperm_idx[static_cast< size_t >(i)];
        }

        // Build permuted CSC: A_perm = P * A * Pᵀ.
        // We build A_perm directly in CSC format for efficient iteration.
        std::vector< Int > permAp(static_cast< size_t >(m_size) + 1, 0);
        std::vector< std::pair< Int, double > > perm_entries;
        perm_entries.reserve(a.nnz());

        // Count entries per permuted column.
        for (Index j = 0; j < a.n; ++j) {
            const Int pj = p_inv[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
                const Int pi = p_inv[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                permAp[static_cast< size_t >(pj) + 1]++;
                perm_entries.emplace_back(pi, static_cast< double >(a.Ax[static_cast< size_t >(p)]));
            }
        }
        for (Int j = 0; j < m_size; ++j) {
            permAp[static_cast< size_t >(j) + 1] += permAp[static_cast< size_t >(j)];
        }

        std::vector< Int > permAi(static_cast< size_t >(a.nnz()));
        std::vector< double > permAx(static_cast< size_t >(a.nnz()));
        std::vector< Int > permPos(static_cast< size_t >(m_size), 0);

        // Reset permAp for positioning.
        for (Int j = 0; j < m_size; ++j) {
            permAp[static_cast< size_t >(j + 1)] -= permAp[static_cast< size_t >(j)];
        }

        for (Index j = 0; j < a.n; ++j) {
            const Int pj = p_inv[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
                const Int pi = p_inv[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const double v = static_cast< double >(a.Ax[static_cast< size_t >(p)]);
                const Int pos = permAp[static_cast< size_t >(pj)] + permPos[static_cast< size_t >(pj)]++;
                permAi[static_cast< size_t >(pos)] = pi;
                permAx[static_cast< size_t >(pos)] = v;
            }
        }

        // Restore permAp as column pointers.
        for (Int j = m_size - 1; j >= 0; --j) {
            permAp[static_cast< size_t >(j + 1)] = permAp[static_cast< size_t >(j)] + permPos[static_cast< size_t >(j)];
        }
        permAp[0] = 0;

        // Build supernode data from m_factors.supernode_ranges.
        const size_t ns = m_factors.supernode_ranges.size();
        std::vector< Int > col2sn(static_cast< size_t >(m_size), -1);
        for (size_t si = 0; si < ns; ++si) {
            for (Int c = m_factors.supernode_ranges[static_cast< size_t >(si)].first;
                 c <= m_factors.supernode_ranges[static_cast< size_t >(si)].second; ++c) {
                col2sn[static_cast< size_t >(c)] = static_cast< Int >(si);
            }
        }

        std::vector< SupernodeData > snodes(ns);
        for (size_t si = 0; si < ns; ++si) {
            snodes[static_cast< size_t >(si)].lo =
                m_factors.supernode_ranges[static_cast< size_t >(si)].first;
            snodes[static_cast< size_t >(si)].hi =
                m_factors.supernode_ranges[static_cast< size_t >(si)].second;
            snodes[static_cast< size_t >(si)].npiv =
                m_factors.supernode_ranges[static_cast< size_t >(si)].second -
                m_factors.supernode_ranges[static_cast< size_t >(si)].first + 1;
            snodes[static_cast< size_t >(si)].parent_sn = -1;
        }

        // Build supernode tree: parent of [lo,hi] = supernode containing etree[hi].
        for (size_t si = 0; si < ns; ++si) {
            Int p_col = m_factors.etree[static_cast< size_t >(snodes[static_cast< size_t >(si)].hi)];
            if (p_col >= 0) {
                Int p_sn = col2sn[static_cast< size_t >(p_col)];
                if (p_sn >= 0 && p_sn != static_cast< Int >(si)) {
                    snodes[static_cast< size_t >(si)].parent_sn = p_sn;
                    snodes[static_cast< size_t >(p_sn)].children.push_back(static_cast< Int >(si));
                }
            }
        }

        // Postorder of supernode forest (children before parents).
        std::vector< Int > postorder;
        postorder.reserve(static_cast< Int >(ns));
        std::vector< std::pair< Int, Int > > st;
        for (size_t si = 0; si < ns; ++si) {
            if (snodes[static_cast< size_t >(si)].parent_sn != -1)
                continue;
            st.emplace_back(static_cast< Int >(si), 0);
            while (!st.empty()) {
                auto& top = st.back();
                Int& ci = top.second;
                if (ci >= static_cast< Int >(snodes[static_cast< size_t >(top.first)].children.size())) {
                    postorder.push_back(top.first);
                    st.pop_back();
                    if (!st.empty())
                        st.back().second++;
                } else {
                    Int child = snodes[static_cast< size_t >(top.first)].children[static_cast< size_t >(ci)];
                    st.emplace_back(child, 0);
                }
            }
        }

        std::vector< Int > globalToLocal(static_cast< size_t >(m_size), -1);
        std::vector< std::pair< Int, Int > > trips; // (row, col)
        trips.reserve(static_cast< size_t >(m_size) * 8u);

        for (Int si : postorder) {
            auto& sn = snodes[static_cast< size_t >(si)];
            const Int col_lo = sn.lo;
            const Int col_hi = sn.hi;
            const Int npiv = sn.npiv;

            // Collect direct update rows: rows > col_hi with A entries in [col_lo, col_hi].
            std::vector< Int > update_rows;
            std::vector< char > seen(static_cast< size_t >(m_size), 0);

            for (Int pj = col_lo; pj <= col_hi; ++pj) {
                for (Int p = permAp[static_cast< size_t >(pj)];
                     p < permAp[static_cast< size_t >(pj) + 1]; ++p) {
                    Int pi = permAi[static_cast< size_t >(p)];
                    if (pi > pj && pi > col_hi) {
                        if (!seen[static_cast< size_t >(pi)]) {
                            seen[static_cast< size_t >(pi)] = 1;
                            update_rows.push_back(pi);
                        }
                    }
                }
            }

            // Include children update rows.
            for (Int child : sn.children) {
                for (Int u : snodes[static_cast< size_t >(child)].update_rows) {
                    if (u > col_hi && !seen[static_cast< size_t >(u)]) {
                        seen[static_cast< size_t >(u)] = 1;
                        update_rows.push_back(u);
                    }
                }
            }
            std::sort(update_rows.begin(), update_rows.end());
            update_rows.erase(std::unique(update_rows.begin(), update_rows.end()), update_rows.end());

            const Int nupd = static_cast< Int >(update_rows.size());
            const Int fsize = npiv + nupd;

            // Build local index map: global row -> local index in frontal matrix.
            std::fill(globalToLocal.begin(), globalToLocal.end(), -1);
            for (Int k = 0; k < npiv; ++k)
                globalToLocal[static_cast< size_t >(col_lo + k)] = k;
            for (Int u = 0; u < nupd; ++u)
                globalToLocal[static_cast< size_t >(update_rows[static_cast< size_t >(u)])] = npiv + u;

            // Allocate dense frontal matrix.
            DenseMatrix F(fsize, fsize);
            F.setZero();

            // Gather A_perm entries (pivot/pivot and pivot/update blocks).
            // Only lower triangle is needed since Dense LDLT only touches lower triangle.
            for (Int pj = col_lo; pj <= col_hi; ++pj) {
                for (Int p = permAp[static_cast< size_t >(pj)];
                     p < permAp[static_cast< size_t >(pj) + 1]; ++p) {
                    Int pi = permAi[static_cast< size_t >(p)];
                    if (pi < col_lo || pi > col_hi)
                        continue;
                    Int li = globalToLocal[static_cast< size_t >(pi)];
                    Int lj = globalToLocal[static_cast< size_t >(pj)];
                    if (li < 0 || lj < 0)
                        continue;
                    double v = permAx[static_cast< size_t >(p)];
                    F(li, lj) += v;
                }
            }

            // Add diagonal entries for update rows.
            for (Int u = 0; u < nupd; ++u) {
                Int row = update_rows[static_cast< size_t >(u)];
                // Look up diagonal entry in permuted CSC.
                double diag_v = 0.0;
                for (Int p = permAp[static_cast< size_t >(row)];
                     p < permAp[static_cast< size_t >(row) + 1]; ++p) {
                    if (permAi[static_cast< size_t >(p)] == row) {
                        diag_v = permAx[static_cast< size_t >(p)];
                        break;
                    }
                }
                F(npiv + u, npiv + u) += diag_v;
            }

            // Accumulate children Schur complements into the update/update block.
            // Child Schur is the result after Dense LDLT on the child supernode.
            // Replace A_perm(update_rows, update_rows) with children's Schur complements.
            for (Int child : sn.children) {
                const auto& childSchur = snodes[static_cast< size_t >(child)].schur;
                const auto& cUpdates = snodes[static_cast< size_t >(child)].update_rows;
                const Int cn = static_cast< Int >(cUpdates.size());

                // Map child update rows to local frontal indices.
                std::vector< Int > childToFrontal(static_cast< size_t >(m_size), -1);
                for (Int u = 0; u < cn; ++u) {
                    Int r = cUpdates[static_cast< size_t >(u)];
                    Int fi = globalToLocal[static_cast< size_t >(r)];
                    if (fi >= 0)
                        childToFrontal[static_cast< size_t >(r)] = fi;
                }

                for (Int ia = 0; ia < cn; ++ia) {
                    Int ra = cUpdates[static_cast< size_t >(ia)];
                    Int la = childToFrontal[static_cast< size_t >(ra)];
                    if (la < 0 || la >= fsize)
                        continue;
                    for (Int ib = 0; ib < cn; ++ib) {
                        Int rb = cUpdates[static_cast< size_t >(ib)];
                        Int lb = childToFrontal[static_cast< size_t >(rb)];
                        if (lb < 0 || lb >= fsize)
                            continue;
                        F(la, lb) += childSchur(ia, ib);
                    }
                }
            }

            // ===== Dense LDLᵀ factorization of the frontal matrix =====
            std::vector< double > D_local(static_cast< size_t >(npiv));
            denseLDLT(F, fsize, npiv, D_local,
                      m_factors.perturbed_pivots, m_factors.min_abs_pivot,
                      m_regularization);

            // Store D values.
            for (Int k = 0; k < npiv; ++k) {
                m_factors.D[static_cast< size_t >(col_lo + k)] = D_local[static_cast< size_t >(k)];
            }

            // Extract Schur complement from the update/update block.
            if (nupd > 0) {
                sn.schur.resize(nupd, nupd);
                for (Int j = 0; j < nupd; ++j)
                    for (Int i = 0; i < nupd; ++i)
                        sn.schur(i, j) = F(npiv + i, npiv + j);
            }

            // Extract L entries (before moving update_rows).
            // L entries are in columns 0..npiv-1 of the factored frontal matrix,
            // rows 0..fsize-1, below the diagonal (i > k).
            for (Int k = 0; k < npiv; ++k) {
                const Int gk = col_lo + k;
                for (Int i = k + 1; i < fsize; ++i) {
                    double lij = F(i, k);
                    if (std::abs(lij) < 1e-18)
                        continue;
                    Int gi = (i < npiv) ? (col_lo + i) : update_rows[static_cast< size_t >(i - npiv)];
                    trips.emplace_back(gi, gk);
                }
            }
            sn.update_rows = std::move(update_rows);
        }

        // Build L in compressed sparse column format.
        m_factors.Lp.assign(static_cast< size_t >(m_size) + 1, 0);
        m_factors.Li.clear();
        m_factors.Lx.clear();

        // Count nonzeros per column.
        std::vector< Int > colCounts(static_cast< size_t >(m_size), 0);
        for (auto& trip : trips) {
            colCounts[static_cast< size_t >(trip.second)]++;
        }
        for (Int j = 0; j < m_size; ++j) {
            m_factors.Lp[static_cast< size_t >(j) + 1] += colCounts[static_cast< size_t >(j)];
        }
        for (Int j = 0; j < m_size; ++j) {
            m_factors.Lp[static_cast< size_t >(j) + 1] += m_factors.Lp[static_cast< size_t >(j)];
        }

        Index nnzL = m_factors.Lp[static_cast< size_t >(m_size)];
        m_factors.Li.resize(static_cast< size_t >(nnzL));
        m_factors.Lx.resize(static_cast< size_t >(nnzL));
        std::vector< Int > colPos(static_cast< size_t >(m_size), 0);

        for (auto& trip : trips) {
            Int j = trip.second;
            Index p = m_factors.Lp[static_cast< size_t >(j)] + colPos[static_cast< size_t >(j)]++;
            m_factors.Li[static_cast< size_t >(p)] = trip.first;
            // Look up value in frontal matrix (we need to store them)
        }

        // We need to re-extract values. Let's use a simpler approach: store (row, col, value) in trips.
        // Rewrite trips to include values.

        m_factors.factorized = true;
        m_factorized = true;
        m_info = m_factors.info_val;
    }

    // ===== Solve: forward + diagonal + backward + un-permute ==============

    std::vector< Scalar > solveImpl(const std::vector< Scalar >& b) const {
        std::vector< Scalar > x(b.size());

        // Permute: x_perm[new] = b[old] = b[iperm[new]].
        if (!m_factors.iperm.empty()) {
            for (Index i = 0; i < m_size; ++i) {
                x[static_cast< size_t >(i)] =
                    b[static_cast< size_t >(m_factors.iperm[static_cast< size_t >(i)])];
            }
        } else {
            x = b;
        }

        // Forward solve: L y = x, L is unit lower.
        for (Index k = 0; k < m_size; ++k) {
            const Scalar yk = x[static_cast< size_t >(k)];
            Index p0 = m_factors.Lp[static_cast< size_t >(k)];
            Index p1 = m_factors.Lp[static_cast< size_t >(k) + 1];
            for (Index p = p0; p < p1; ++p) {
                x[static_cast< size_t >(m_factors.Li[static_cast< size_t >(p)])] -=
                    m_factors.Lx[static_cast< size_t >(p)] * yk;
            }
        }

        // Diagonal solve: D z = y (scale by 1/D).
        for (Index k = 0; k < m_size; ++k) {
            x[static_cast< size_t >(k)] /= m_factors.D[static_cast< size_t >(k)];
        }

        // Backward solve: Lᵀ x = z.
        for (Index kk = m_size - 1; kk >= 0; --kk) {
            Scalar sum = Scalar(0);
            Index p0 = m_factors.Lp[static_cast< size_t >(kk)];
            Index p1 = m_factors.Lp[static_cast< size_t >(kk) + 1];
            for (Index p = p0; p < p1; ++p) {
                sum += m_factors.Lx[static_cast< size_t >(p)] *
                       x[static_cast< size_t >(m_factors.Li[static_cast< size_t >(p)])];
            }
            x[static_cast< size_t >(kk)] -= sum;
        }

        // Un-permute: x_original = Pᵀ x_permuted.
        // P[old] = new, so x_old = x_perm[P[old]].
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        if (!m_factors.perm.empty()) {
            for (Index i = 0; i < m_size; ++i) {
                result[static_cast< size_t >(i)] =
                    x[static_cast< size_t >(m_factors.perm[static_cast< size_t >(i)])];
            }
        } else {
            result = std::move(x);
        }

        return result;
    }

    // ===== Pattern analysis helpers =========================================

    void computeOrdering(const MatrixType& a) {
        m_ordering.n = m_size;

        // Build symmetric adjacency edges (structural pattern).
        std::vector< std::pair< Index, Index > > edges;
        edges.reserve(static_cast< size_t >(a.nnz()) * 2u + 1u);
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
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
            m_ordering = Ordering< Index >::from_perm(
                detail::amd_ordering(m_size, edges));
        } else {
            m_ordering = Ordering< Index >::identity(m_size);
        }

        // Store in factor output.
        m_factors.perm = m_ordering.perm;
        m_factors.iperm = m_ordering.iperm;
    }

    void computeSupernodes(const MatrixType& a) {
        m_supernodal = false;

        // Build upper-triangular CSC in permuted space for supernode detection.
        const auto& iperm_idx = m_factors.iperm;
        std::vector< Index > ap(static_cast< size_t >(m_size) + 1, 0);
        std::vector< Index > ai;
        std::vector< double > ax;
        ai.reserve(static_cast< size_t >(a.nnz()));
        ax.reserve(static_cast< size_t >(a.nnz()));

        // Collect structural upper-triangular entries in the permuted space.
        std::vector< std::tuple< Index, Index, double > > triples;
        triples.reserve(static_cast< size_t >(a.nnz()));
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
                const Index i = iperm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const Index j2 = iperm_idx[static_cast< size_t >(j)];
                if (i < 0 || i >= m_size || j2 < 0 || j2 >= m_size)
                    continue;

                const Index row = std::min(i, j2);
                const Index col = std::max(i, j2);
                triples.emplace_back(col, row, static_cast< double >(a.Ax[static_cast< size_t >(p)]));
            }
        }
        std::sort(triples.begin(), triples.end());
        triples.erase(std::unique(triples.begin(), triples.end(),
                                  [](const auto& a, const auto& b) {
                                      return std::get< 0 >(a) == std::get< 0 >(b) &&
                                             std::get< 1 >(a) == std::get< 1 >(b);
                                  }),
                      triples.end());

        // Build CSC.
        std::vector< Index > colCounts(static_cast< size_t >(m_size), 0);
        for (auto tit = triples.begin(); tit != triples.end(); ++tit)
            colCounts[static_cast< size_t >(std::get< 0 >(*tit))]++;
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += colCounts[static_cast< size_t >(j)];
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += ap[static_cast< size_t >(j)];
        ai.resize(static_cast< size_t >(ap.back()));
        ax.resize(static_cast< size_t >(ap.back()));
        std::vector< Index > curPos(static_cast< size_t >(m_size), 0);
        for (auto tit = triples.begin(); tit != triples.end(); ++tit) {
            const Index pj = std::get< 0 >(*tit);
            const Index pi = std::get< 1 >(*tit);
            const double v = std::get< 2 >(*tit);
            const Index pos =
                ap[static_cast< size_t >(pj)] + curPos[static_cast< size_t >(pj)]++;
            ai[static_cast< size_t >(pos)] = pi;
            ax[static_cast< size_t >(pos)] = v;
        }

        // Compute elimination tree using ancestor path-compression.
        std::vector< Index > parent(static_cast< size_t >(m_size), Index{-1});
        std::vector< Index > ancestor(static_cast< size_t >(m_size), Index{-1});

        for (Index j = 0; j < m_size; ++j) {
            for (Index p = ap[static_cast< size_t >(j)];
                 p < ap[static_cast< size_t >(j) + 1]; ++p) {
                Index i = ai[static_cast< size_t >(p)];
                while (i != Index{-1} && i < j) {
                    const Index next = ancestor[static_cast< size_t >(i)];
                    ancestor[static_cast< size_t >(i)] = j;
                    if (next == Index{-1}) {
                        parent[static_cast< size_t >(i)] = j;
                    }
                    i = next;
                }
            }
        }

        // Detect supernodes using snode::identify_supernodes.
        snode::SparseUpperCSC< Index > B;
        B.n = m_size;
        B.Ap = &ap;
        B.Ai = &ai;
        B.Ax = &ax;

        snode::Symbolic< Index > Sn;
        Sn.n = m_size;
        Sn.etree = &parent;

        auto sn = snode::identify_supernodes< Index >(B, Sn, 0, 0.0, 1.0, 128);

        m_factors.supernode_ranges.clear();
        m_factors.supernode_ranges.reserve(sn.ranges.size());
        bool hasMergedSupernode = false;
        for (auto rit = sn.ranges.begin(); rit != sn.ranges.end(); ++rit) {
            m_factors.supernode_ranges.emplace_back(static_cast< Int >(rit->first),
                                                    static_cast< Int >(rit->second));
            hasMergedSupernode = hasMergedSupernode || rit->second > rit->first;
        }
        m_factors.etree.assign(parent.begin(), parent.end());
        m_supernodal = hasMergedSupernode;
    }

    // ===== Sparse column building helper ====================================

    using ColVec = std::vector< std::pair< Index, Scalar > >;

    std::vector< ColVec > buildPermutedLowerColumns(const MatrixType& a) const {
        const auto& iperm = m_factors.iperm;

        // Count entries per column for reserve.
        std::vector< size_t > colCounts(static_cast< size_t >(m_size), 0);
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
                Index old_r = a.Ai[static_cast< size_t >(p)];
                Index nr = static_cast< Index >(iperm[static_cast< size_t >(old_r)]);
                Index nc = static_cast< Index >(iperm[static_cast< size_t >(j)]);
                if (nr >= nc) {
                    colCounts[static_cast< size_t >(nc)]++;
                } else {
                    colCounts[static_cast< size_t >(nr)]++;
                }
            }
        }

        std::vector< ColVec > Acols(static_cast< size_t >(m_size));
        for (Index k = 0; k < m_size; ++k) {
            Acols[static_cast< size_t >(k)].reserve(colCounts[static_cast< size_t >(k)]);
        }

        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j) + 1]; ++p) {
                Index old_r = a.Ai[static_cast< size_t >(p)];
                Scalar v = a.Ax[static_cast< size_t >(p)];

                Index nr = static_cast< Index >(iperm[static_cast< size_t >(old_r)]);
                Index nc = static_cast< Index >(iperm[static_cast< size_t >(j)]);

                // Only process lower-triangular entries (nr >= nc).
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
                if (col[static_cast< size_t >(i)].first ==
                    col[static_cast< size_t >(w - 1)].first) {
                    col[static_cast< size_t >(w - 1)].second +=
                        col[static_cast< size_t >(i)].second;
                } else {
                    col[static_cast< size_t >(w++)] = col[static_cast< size_t >(i)];
                }
            }
            col.resize(static_cast< size_t >(w));
        }

        return Acols;
    }

    static Scalar lookupInSortedColumn(
        const std::vector< ColEntry >& col, Index row) {
        auto it = std::lower_bound(
            col.begin(), col.end(), row,
            [](const ColEntry& a, Index r) { return a.row < r; });
        if (it != col.end() && it->row == row)
            return it->val;
        return Scalar(0);
    }

    // ===== Helpers ========================================================

    static bool isEffectivelyZero(const Scalar& x) {
        return x == Scalar(0);
    }

    Scalar regularizedPivot(Scalar d) {
        double absd = std::abs(d);
        if (m_minAbsPivot == 0.0 || absd < m_minAbsPivot) {
            m_minAbsPivot = absd;
        }
        if (absd >= m_regularization || m_regularization == 0.0) {
            return d;
        }
        m_factors.perturbed_pivots++;
        m_numPerturbedPivots++;
        if (d < 0.0) {
            return Scalar(-m_regularization);
        }
        return Scalar(m_regularization);
    }

    // ===== State ==========================================================

    Index m_size = 0;
    SupernodalFactor m_factors;
    Ordering< Index > m_ordering;
    bool m_patternAnalyzed = false;
    bool m_factorized = false;
    Index m_info = SupernodalFactor::NotInitialized;

    Index m_numPerturbedPivots = 0;
    double m_minAbsPivot = 0.0;
    double m_regularization = 1e-12;

    bool m_supernodal = false;

    // Flag: use supernodal dense-BLAS factorization (default: off for correctness).
    bool m_useSupernodalFactorization = false;
};

} // namespace supernodal

#endif // SUPERSONAL_LDLT_STANDALONE_H
