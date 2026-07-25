/*
 * supernodal_ldlt_standalone.h — Eigen-free supernodal sparse LDLᵀ factorization
 *
 * Supernodal sparse LDL^T using dense BLAS kernels on supernodes (no
 * simplicial fallback — always dense-BLAS, correct for singleton supernodes
 * too).
 * Uses snode::identify_supernodes (supernodes.h) for supernode detection.
 *
 * API mirrors the Eigen version for drop-in replacement:
 *   SparseCSC<Scalar,Index>      — compressed sparse column (upper+diag)
 *   Ordering<int>                — perm/iperm
 *   SupernodalFactor             — L (CSC) + D + metadata
 *   SupernodalLDLT<Scalar,Index> — solver class
 *
 * Algorithm:
 *   - Pattern analysis: AMD ordering (or externally supplied, see
 *     setExternalOrdering) + supernode detection via supernodes.h
 *   - Numeric factorization: supernodal dense BLAS, unconditionally
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
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(LINSYS_HAS_BLAS)
#include <cblas.h>
#endif

// ===== AMD ordering ========================================================
// amd.h must be in global scope: it includes <iostream> which references C
// symbols (::memchr, ::__libc_single_threaded etc.) that would otherwise
// resolve to supernodal:: inside the namespace block.
#include "../common/amd.h"

// ===== Supernode detection =================================================
#include "../supernodes.h"

#include "../common/dense_matrix.h"
#include "../common/ordering.h"
#include "../common/sparse_csc.h"
#include "../common/trisolve.h"

// ===== Shared types (see linear_system/common/) ============================

namespace supernodal {

using Int = int32_t;
using Real = double;

using linsys::amd_ordering;
using linsys::lsolve_unit;
using linsys::ltsolve_unit;
using linsys::Ordering;
using linsys::permute_gather;
using linsys::SparseCSC;

// Dense matrix from shared linsys::
using linsys::DenseMatrix;

// ===== Factorization result =================================================

struct SupernodalFactor {
    // L in CSC format (unit lower triangular)
    Int n = 0;
    std::vector< Int > Lp;    // size n+1
    std::vector< Int > Li;    // size nnz(L)
    std::vector< double > Lx; // size nnz(L)

    // D diagonal
    std::vector< double > D; // size n

    // Supernode metadata
    std::vector< std::pair< Int, Int > > supernode_ranges;
    std::vector< Int > etree;
    std::vector< Int > supernode_parent;
    std::vector< Int > supernode_post;
    std::vector< Int > supernode_row_ptr;
    std::vector< Int > supernode_rows;

    // Permutation (from analysis phase)
    std::vector< Int > perm;  // perm[old] = new
    std::vector< Int > iperm; // iperm[new] = old

    // Metadata
    Int perturbed_pivots = 0;
    double min_abs_pivot = 0.0;
    bool factorized = false;

    enum Info { Success = 0, NumericalIssue = 1, NotInitialized = 2 };
    Info info_val = NotInitialized;

    size_t nnzL() const {
        return Li.size();
    }
};

// ===== Supernodal LDLᵀ solver ==============================================

template < typename Scalar = Real, typename Index = Int > class SupernodalLDLT {
  public:
    using MatrixType = SparseCSC< Scalar, Index >;
    using RealScalar = double;

    SupernodalLDLT() = default;

    explicit SupernodalLDLT(const MatrixType& a) {
        compute(a);
    }

    void reset() {
        m_size = 0;
        m_factors = SupernodalFactor{};
        m_ordering = Ordering< Index >{};
        m_externalOrdering = Ordering< Index >{};
        m_regularization = 1e-12;
        m_patternAnalyzed = false;
        m_useExternalOrdering = false;
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
        if (!m_factors.factorized) {
            throw std::runtime_error("supernodal: solver is not factorized");
        }
        if (static_cast< Index >(b.size()) != m_size) {
            throw std::invalid_argument("supernodal: rhs size mismatch");
        }
        return solveImpl(b);
    }

    const SupernodalFactor& factors() const {
        return m_factors;
    }
    Index size() const {
        return m_size;
    }
    Index info() const {
        return static_cast< Index >(m_factors.info_val);
    }
    bool isFactorized() const {
        return m_factors.factorized;
    }
    Index nonZerosL() const {
        return static_cast< Index >(m_factors.Li.size());
    }
    Index perturbedPivots() const {
        return m_factors.perturbed_pivots;
    }
    Scalar minAbsPivot() const {
        return static_cast< Scalar >(m_factors.min_abs_pivot);
    }

    void setRegularization(RealScalar eps) {
        m_regularization = std::max(eps, RealScalar(0));
    }

    /// Supply a precomputed permutation instead of letting analyzePattern
    /// run full-matrix AMD. Useful when the caller has structural knowledge
    /// AMD can't see (e.g. block-quasi-definite systems where pivots must
    /// not cross block boundaries). Must be set before compute()/factorizeMatrix().
    void setExternalOrdering(Ordering< Index > ordering) {
        m_externalOrdering = std::move(ordering);
        m_useExternalOrdering = true;
    }

    const Ordering< Index >& permutation() const {
        return m_ordering;
    }

    const std::vector< std::pair< Int, Int > >& supernodeRanges() const {
        return m_factors.supernode_ranges;
    }

    const std::vector< Int >& etree() const {
        return m_factors.etree;
    }

    bool isSupernodal() const {
        return std::any_of(
            m_factors.supernode_ranges.begin(), m_factors.supernode_ranges.end(),
            [](const auto& range) { return range.second > range.first; });
    }

  private:
    // ===== Dense LDLᵀ on a frontal matrix (used within supernodes) ==========
    // Factorizes F(0:npiv, 0:fsize) in-place. D_local gets the diagonal.

    static void updateTrailingBlock(DenseMatrix< Real >& F, Int first, Int panel_begin,
                                    Int panel_end, const std::vector< double >& D_local,
                                    std::vector< double >& scaled_panel,
                                    std::vector< double >& product) {
        const Int trailing_size = F.rows - first;
        const Int panel_size = panel_end - panel_begin;
        if (trailing_size <= 0 || panel_size <= 0)
            return;

#if defined(LINSYS_HAS_BLAS)
        scaled_panel.resize(static_cast< size_t >(trailing_size) *
                            static_cast< size_t >(panel_size));
        product.resize(static_cast< size_t >(trailing_size) *
                       static_cast< size_t >(trailing_size));

        for (Int k = 0; k < panel_size; ++k) {
            const double d = D_local[static_cast< size_t >(panel_begin + k)];
            for (Int i = 0; i < trailing_size; ++i) {
                scaled_panel[static_cast< size_t >(k) * static_cast< size_t >(trailing_size) +
                             static_cast< size_t >(i)] = F(first + i, panel_begin + k) * d;
            }
        }

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, trailing_size, trailing_size,
                    panel_size, 1.0, scaled_panel.data(), trailing_size,
                    &F(first, panel_begin), F.rows, 0.0, product.data(), trailing_size);

        for (Int j = 0; j < trailing_size; ++j) {
            for (Int i = j; i < trailing_size; ++i) {
                F(first + i, first + j) -=
                    product[static_cast< size_t >(j) * static_cast< size_t >(trailing_size) +
                            static_cast< size_t >(i)];
            }
        }
#else
        (void)scaled_panel;
        (void)product;
        for (Int j = first; j < F.rows; ++j) {
            for (Int i = j; i < F.rows; ++i) {
                double update = 0.0;
                for (Int k = panel_begin; k < panel_end; ++k) {
                    update += F(i, k) * D_local[static_cast< size_t >(k)] * F(j, k);
                }
                F(i, j) -= update;
            }
        }
#endif
    }

    static void denseLDLT(DenseMatrix< Real >& F, Int fsize, Int npiv,
                          std::vector< double >& D_local, Int& perturbed_pivots,
                          double& min_abs_pivot, double regularization) {
        constexpr Int block_size = 32;
        std::vector< double > scaled_panel;
        std::vector< double > product;

        for (Int panel_begin = 0; panel_begin < npiv; panel_begin += block_size) {
            const Int panel_end = std::min(npiv, panel_begin + block_size);

            // Factor the diagonal panel and compute the rectangular L panel.
            for (Int k = panel_begin; k < panel_end; ++k) {
                double d = F(k, k);

                if (std::abs(d) < regularization) {
                    d = (d < 0.0 ? -regularization : regularization);
                    ++perturbed_pivots;
                }
                D_local[static_cast< size_t >(k)] = d;

                const double absd = std::abs(d);
                if (min_abs_pivot == 0.0 || absd < min_abs_pivot) {
                    min_abs_pivot = absd;
                }

                const double dinv = 1.0 / d;
                for (Int i = k + 1; i < fsize; ++i) {
                    F(i, k) *= dinv;
                }

                // Only update columns still inside the panel. The entire
                // trailing block is updated by DGEMM after the panel closes.
                for (Int j = k + 1; j < panel_end; ++j) {
                    const double dljk = F(j, k) * d;
                    for (Int i = j; i < fsize; ++i) {
                        F(i, j) -= F(i, k) * dljk;
                    }
                }
            }

            updateTrailingBlock(F, panel_end, panel_begin, panel_end, D_local,
                                scaled_panel, product);
        }
    }

    // ===== Pattern analysis: compute ordering + supernodes ==================

    static void validateMatrix(const MatrixType& a) {
        if (a.n < 0)
            throw std::invalid_argument("supernodal: matrix dimension must be nonnegative");
        if (a.Ap.size() != static_cast< size_t >(a.n) + 1)
            throw std::invalid_argument("supernodal: Ap size mismatch");
        if (a.Ap.empty() || a.Ap.front() != Index{0})
            throw std::invalid_argument("supernodal: Ap must start at zero");
        if (a.Ai.size() != a.Ax.size())
            throw std::invalid_argument("supernodal: Ai/Ax size mismatch");

        for (Index col = 0; col < a.n; ++col) {
            const Index begin = a.Ap[static_cast< size_t >(col)];
            const Index end = a.Ap[static_cast< size_t >(col) + 1];
            if (begin < 0 || begin > end || static_cast< size_t >(end) > a.Ai.size())
                throw std::invalid_argument("supernodal: invalid CSC column pointers");
            for (Index p = begin; p < end; ++p) {
                const Index row = a.Ai[static_cast< size_t >(p)];
                if (row < 0 || row >= a.n)
                    throw std::invalid_argument("supernodal: row index out of range");
            }
        }
        if (static_cast< size_t >(a.Ap.back()) != a.Ai.size())
            throw std::invalid_argument("supernodal: Ap.back() must equal Ai/Ax size");
    }

    void analyzePattern(const MatrixType& a) {
        validateMatrix(a);
        if (a.n <= 0) {
            m_factors.info_val = SupernodalFactor::NotInitialized;
            m_factors.factorized = false;
            m_patternAnalyzed = false;
            return;
        }
        m_size = static_cast< Index >(a.n);

        // Compute permutation: AMD if n > threshold, else natural.
        computeOrdering(a);

        // Detect supernodes using snode::identify_supernodes.
        computeSupernodes(a);

        m_patternAnalyzed = true;
        m_factors.factorized = false;
        m_factors.info_val = SupernodalFactor::Success;
    }

    // ===== Numeric factorization: always dense-BLAS-on-supernodes ==========
    // (No simplicial fallback: factorizeSupernodal is correct and used
    // unconditionally, whether or not supernode merging found any blocks —
    // singleton supernodes are just fsize==1 frontal matrices.)

    void factorize(const MatrixType& a) {
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n)) {
            analyzePattern(a);
        }
        if (!m_patternAnalyzed)
            return;

        factorizeSupernodal(a);
        m_factors.factorized = (m_factors.info_val == SupernodalFactor::Success);
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
        std::vector< Int > children;
        std::vector< Int > update_rows;
        DenseMatrix< Real > schur; // Schur complement (update/update block)
    };

    void factorizeSupernodal(const MatrixType& a) {
        m_factors.n = m_size;
        m_factors.D.assign(static_cast< size_t >(m_size), 0.0);
        m_factors.perturbed_pivots = 0;
        m_factors.min_abs_pivot = 0.0;
        m_factors.factorized = false;
        m_factors.info_val = SupernodalFactor::Success;

        // m_factors.perm[old] = new, so it maps input coordinates directly
        // into A_perm.
        const auto& perm_idx = m_factors.perm;

        // Build permuted CSC: A_perm = P * A * Pᵀ.
        // We build A_perm directly in CSC format for efficient iteration.
        std::vector< Int > permAp(static_cast< size_t >(m_size) + 1, 0);

        // Count entries per permuted column.
        for (Index j = 0; j < a.n; ++j) {
            const Int pj = perm_idx[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                permAp[static_cast< size_t >(pj) + 1]++;
            }
        }
        for (Int j = 0; j < m_size; ++j) {
            permAp[static_cast< size_t >(j) + 1] += permAp[static_cast< size_t >(j)];
        }

        std::vector< Int > permAi(static_cast< size_t >(a.nnz()));
        std::vector< double > permAx(static_cast< size_t >(a.nnz()));
        std::vector< Int > permPos(static_cast< size_t >(m_size), 0);

        // permAp already holds cumulative column-start offsets (prefix-summed
        // above); permPos tracks the running write position within each
        // column separately, so permAp itself must not be touched here.
        for (Index j = 0; j < a.n; ++j) {
            const Int pj = perm_idx[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Int pi = perm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const double v = static_cast< double >(a.Ax[static_cast< size_t >(p)]);
                const Int pos =
                    permAp[static_cast< size_t >(pj)] + permPos[static_cast< size_t >(pj)]++;
                permAi[static_cast< size_t >(pos)] = pi;
                permAx[static_cast< size_t >(pos)] = v;
            }
        }

        // Build numeric supernode state from the compact symbolic analysis.
        const size_t ns = m_factors.supernode_ranges.size();
        std::vector< SupernodeData > snodes(ns);
        for (size_t si = 0; si < ns; ++si) {
            snodes[static_cast< size_t >(si)].lo =
                m_factors.supernode_ranges[static_cast< size_t >(si)].first;
            snodes[static_cast< size_t >(si)].hi =
                m_factors.supernode_ranges[static_cast< size_t >(si)].second;
            snodes[static_cast< size_t >(si)].npiv =
                m_factors.supernode_ranges[static_cast< size_t >(si)].second -
                m_factors.supernode_ranges[static_cast< size_t >(si)].first + 1;
            const Int row_begin =
                m_factors.supernode_row_ptr[static_cast< size_t >(si)] +
                snodes[static_cast< size_t >(si)].npiv;
            const Int row_end =
                m_factors.supernode_row_ptr[static_cast< size_t >(si) + 1];
            snodes[static_cast< size_t >(si)].update_rows.assign(
                m_factors.supernode_rows.begin() + row_begin,
                m_factors.supernode_rows.begin() + row_end);
            const Int parent =
                m_factors.supernode_parent[static_cast< size_t >(si)];
            if (parent != Int{-1})
                snodes[static_cast< size_t >(parent)].children.push_back(
                    static_cast< Int >(si));
        }

        std::vector< Int > globalToLocal(static_cast< size_t >(m_size), -1);
        // (row, col, value) triples for L entries
        struct Ltrip {
            Int row, col;
            double val;
        };
        std::vector< Ltrip > trips;
        trips.reserve(static_cast< size_t >(m_size) * 8u);

        for (Int si : m_factors.supernode_post) {
            auto& sn = snodes[static_cast< size_t >(si)];
            const Int col_lo = sn.lo;
            const Int col_hi = sn.hi;
            const Int npiv = sn.npiv;
            const auto& update_rows = sn.update_rows;
            const Int nupd = static_cast< Int >(sn.update_rows.size());
            const Int fsize = npiv + nupd;

            // Build local index map: global row -> local index in frontal matrix.
            std::fill(globalToLocal.begin(), globalToLocal.end(), -1);
            for (Int k = 0; k < npiv; ++k)
                globalToLocal[static_cast< size_t >(col_lo + k)] = k;
            for (Int u = 0; u < nupd; ++u)
                globalToLocal[static_cast< size_t >(update_rows[static_cast< size_t >(u)])] =
                    npiv + u;

            // Allocate dense frontal matrix.
            DenseMatrix< Real > F(fsize, fsize);
            F.setZero();

            // Gather A_perm entries into the pivot/pivot and update/pivot
            // blocks of the frontal matrix. Only lower triangle is needed
            // since Dense LDLT only touches lower triangle; rows are either
            // within the pivot block [col_lo, col_hi] or one of this
            // supernode's update_rows (rows > col_hi with an entry in a
            // pivot column) — both map via globalToLocal.
            for (Int pj = col_lo; pj <= col_hi; ++pj) {
                for (Int p = permAp[static_cast< size_t >(pj)];
                     p < permAp[static_cast< size_t >(pj) + 1]; ++p) {
                    Int pi = permAi[static_cast< size_t >(p)];
                    if (pi < pj)
                        continue; // strictly upper entries: covered by the symmetric (pj,pi) pass
                    Int li = globalToLocal[static_cast< size_t >(pi)];
                    Int lj = globalToLocal[static_cast< size_t >(pj)];
                    if (li < 0 || lj < 0)
                        continue;
                    double v = permAx[static_cast< size_t >(p)];
                    F(li, lj) += v;
                }
            }

            // Note: update rows' own diagonals are intentionally NOT seeded
            // here from A. Each row's true diagonal is added exactly once,
            // by the main gather loop above, at the single supernode where
            // that row is itself a pivot column. Seeding it here too would
            // double-count it: this update-row block can be shared by
            // several sibling supernodes (all children of the same
            // ancestor), and each would independently re-add A(row,row) into
            // its own Schur complement, which then all get summed into the
            // ancestor's frontal matrix.

            // Accumulate children Schur complements into the update/update block.
            // Child Schur is the result after Dense LDLT on the child supernode.
            // Replace A_perm(update_rows, update_rows) with children's Schur complements.
            for (Int child : sn.children) {
                const auto& childSchur = snodes[static_cast< size_t >(child)].schur;
                const auto& cUpdates = snodes[static_cast< size_t >(child)].update_rows;
                const Int cn = static_cast< Int >(cUpdates.size());

                for (Int ia = 0; ia < cn; ++ia) {
                    Int ra = cUpdates[static_cast< size_t >(ia)];
                    Int la = globalToLocal[static_cast< size_t >(ra)];
                    if (la < 0 || la >= fsize)
                        continue;
                    for (Int ib = 0; ib < cn; ++ib) {
                        Int rb = cUpdates[static_cast< size_t >(ib)];
                        Int lb = globalToLocal[static_cast< size_t >(rb)];
                        if (lb < 0 || lb >= fsize)
                            continue;
                        F(la, lb) += childSchur(ia, ib);
                    }
                }
            }

            // ===== Dense LDLᵀ factorization of the frontal matrix =====
            std::vector< double > D_local(static_cast< size_t >(npiv));
            denseLDLT(F, fsize, npiv, D_local, m_factors.perturbed_pivots, m_factors.min_abs_pivot,
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
                    Int gi =
                        (i < npiv) ? (col_lo + i) : update_rows[static_cast< size_t >(i - npiv)];
                    trips.emplace_back(Ltrip{gi, gk, lij});
                }
            }
        }

        // Build L in compressed sparse column format.
        m_factors.Lp.assign(static_cast< size_t >(m_size) + 1, 0);
        m_factors.Li.clear();
        m_factors.Lx.clear();

        // Count nonzeros per column.
        std::vector< Int > colCounts(static_cast< size_t >(m_size), 0);
        for (auto& trip : trips) {
            colCounts[static_cast< size_t >(trip.col)]++;
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
            Index j = trip.col;
            Index p = m_factors.Lp[static_cast< size_t >(j)] + colPos[static_cast< size_t >(j)]++;
            m_factors.Li[static_cast< size_t >(p)] = trip.row;
            m_factors.Lx[static_cast< size_t >(p)] = static_cast< Scalar >(trip.val);
        }

    }

    // ===== Solve: forward + diagonal + backward + un-permute ==============

    std::vector< Scalar > solveImpl(const std::vector< Scalar >& b) const {
        std::vector< Scalar > x(b.size());

        // Permute: x_perm[new] = b[iperm[new]].
        if (!m_factors.iperm.empty()) {
            std::vector< Scalar > y(static_cast< size_t >(m_size));
            permute_gather(m_size, m_factors.iperm.data(), b.data(), y.data());
            x = std::move(y);
        } else {
            x = b;
        }

        // Forward solve: L y = x, L is unit lower.
        lsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(), m_factors.Lx.data(),
                    x.data());

        // Diagonal solve: D z = y (scale by 1/D).
        for (Index k = 0; k < m_size; ++k) {
            x[static_cast< size_t >(k)] /= m_factors.D[static_cast< size_t >(k)];
        }

        // Backward solve: Lᵀ x = z.
        ltsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(), m_factors.Lx.data(),
                     x.data());

        // Un-permute: x_old[old] = x_perm[perm[old]].
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        if (!m_factors.perm.empty()) {
            permute_gather(m_size, m_factors.perm.data(), x.data(), result.data());
        } else {
            result = std::move(x);
        }

        return result;
    }

    // ===== Pattern analysis helpers =========================================

    void computeOrdering(const MatrixType& a) {
        if (m_useExternalOrdering) {
            if (m_externalOrdering.n != m_size ||
                m_externalOrdering.perm.size() != static_cast< size_t >(m_size))
                throw std::invalid_argument("supernodal: external ordering size mismatch");
            m_ordering = Ordering< Index >::from_perm(m_externalOrdering.perm);
            if (!m_externalOrdering.iperm.empty() &&
                m_externalOrdering.iperm != m_ordering.iperm)
                throw std::invalid_argument("supernodal: inconsistent external inverse permutation");
            m_factors.perm = m_ordering.perm;
            m_factors.iperm = m_ordering.iperm;
            return;
        }

        m_ordering.n = m_size;

        // Build symmetric adjacency edges (structural pattern).
        std::vector< std::pair< Index, Index > > edges;
        edges.reserve(static_cast< size_t >(a.nnz()) * 2u + 1u);
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
            m_ordering = Ordering< Index >::from_perm(amd_ordering(m_size, edges));
        } else {
            m_ordering = Ordering< Index >::identity(m_size);
        }

        // Store in factor output.
        m_factors.perm = m_ordering.perm;
        m_factors.iperm = m_ordering.iperm;
    }

    void computeSupernodes(const MatrixType& a) {
        // Build upper-triangular CSC in permuted space for supernode detection.
        const auto& perm_idx = m_factors.perm;
        std::vector< Index > ap(static_cast< size_t >(m_size) + 1, 0);
        std::vector< Index > ai;
        ai.reserve(static_cast< size_t >(a.nnz()));

        // Collect structural upper-triangular entries in the permuted space.
        std::vector< std::pair< Index, Index > > entries;
        entries.reserve(static_cast< size_t >(a.nnz()));
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Index i = perm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const Index j2 = perm_idx[static_cast< size_t >(j)];
                if (i < 0 || i >= m_size || j2 < 0 || j2 >= m_size)
                    continue;

                const Index row = std::min(i, j2);
                const Index col = std::max(i, j2);
                entries.emplace_back(col, row);
            }
        }
        std::sort(entries.begin(), entries.end());
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());

        // Build CSC.
        std::vector< Index > colCounts(static_cast< size_t >(m_size), 0);
        for (const auto& entry : entries)
            ++colCounts[static_cast< size_t >(entry.first)];
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += colCounts[static_cast< size_t >(j)];
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += ap[static_cast< size_t >(j)];
        ai.resize(static_cast< size_t >(ap.back()));
        std::vector< Index > curPos(static_cast< size_t >(m_size), 0);
        for (const auto& entry : entries) {
            const Index pj = entry.first;
            const Index pos = ap[static_cast< size_t >(pj)] + curPos[static_cast< size_t >(pj)]++;
            ai[static_cast< size_t >(pos)] = entry.second;
        }

        // Compute elimination tree using ancestor path-compression.
        std::vector< Index > parent(static_cast< size_t >(m_size), Index{-1});
        std::vector< Index > ancestor(static_cast< size_t >(m_size), Index{-1});

        for (Index j = 0; j < m_size; ++j) {
            for (Index p = ap[static_cast< size_t >(j)]; p < ap[static_cast< size_t >(j) + 1];
                 ++p) {
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

        snode::Symbolic< Index > Sn;
        Sn.n = m_size;
        Sn.etree = &parent;

        // Relaxed amalgamation: merge columns whose L-patterns differ by up to
        // 8 rows (or 10%), not just exact (fundamental-only) matches. Bigger
        // supernodes -> better arithmetic intensity in denseLDLT, at the cost
        // of a few explicit zero entries in the frontal matrix (already
        // zero-filled via F.setZero() before gather, so this is safe).
        auto sn = snode::identify_supernodes< Index >(B, Sn, 8, 0.1, 0.9, 128);

        m_factors.supernode_ranges.clear();
        m_factors.supernode_ranges.reserve(sn.ranges.size());
        for (auto rit = sn.ranges.begin(); rit != sn.ranges.end(); ++rit) {
            m_factors.supernode_ranges.emplace_back(static_cast< Int >(rit->first),
                                                    static_cast< Int >(rit->second));
        }
        m_factors.etree.assign(parent.begin(), parent.end());
        m_factors.supernode_parent.assign(sn.parent.begin(), sn.parent.end());
        m_factors.supernode_post.assign(sn.supernode_post.begin(),
                                        sn.supernode_post.end());
        m_factors.supernode_row_ptr.assign(sn.row_ptr.begin(), sn.row_ptr.end());
        m_factors.supernode_rows.assign(sn.rows.begin(), sn.rows.end());
    }


    // ===== State ==========================================================

    Index m_size = 0;
    SupernodalFactor m_factors;
    Ordering< Index > m_ordering;
    Ordering< Index > m_externalOrdering;
    bool m_useExternalOrdering = false;
    bool m_patternAnalyzed = false;

    double m_regularization = 1e-12;

};

} // namespace supernodal

#endif // SUPERSONAL_LDLT_STANDALONE_H
