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
using linsys::permute_scatter;
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

    SupernodalLDLT() : m_size(0), m_info(SupernodalFactor::NotInitialized) {
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
        m_externalOrdering.perm.clear();
        m_externalOrdering.iperm.clear();
        m_externalOrdering.n = 0;
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

    /// Retained for API compatibility. Dense-BLAS-on-supernodes factorization
    /// is now always used (no simplicial fallback) — this is a no-op.
    [[deprecated("factorizeSimplicial fallback removed; dense-BLAS supernodal "
                "factorization is always used now")]] void
    setSupernodalFactorization(bool /*on*/) {
    }

    bool supernodalFactorization() const {
        return true;
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

    const SupernodalFactor& factors() const {
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
        return m_factors.perturbed_pivots;
    }
    Scalar minAbsPivot() const {
        return static_cast< Scalar >(m_minAbsPivot);
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
        return m_supernodal;
    }

  private:
    // ===== Dense LDLᵀ on a frontal matrix (used within supernodes) ==========
    // Factorizes F(0:npiv, 0:fsize) in-place. D_local gets diagonal.
    // Returns number of perturbed pivots.

    static Int denseLDLT(DenseMatrix< Real >& F, Int fsize, Int npiv,
                         std::vector< double >& D_local, int& numPerturbed, double& minAbsPivot,
                         double regularization) {
        Int perturbed = 0;

        for (Int k = 0; k < npiv; ++k) {
            double d = F(k, k);

            // Regularize near-zero pivots with the configured flat threshold.
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

        m_factorized = (m_info == SupernodalFactor::Success);
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
        DenseMatrix< Real > schur; // Schur complement (update/update block)
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
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Int pi = p_inv[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                permAp[static_cast< size_t >(pj) + 1]++;
                perm_entries.emplace_back(pi,
                                          static_cast< double >(a.Ax[static_cast< size_t >(p)]));
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
            const Int pj = p_inv[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Int pi = p_inv[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const double v = static_cast< double >(a.Ax[static_cast< size_t >(p)]);
                const Int pos =
                    permAp[static_cast< size_t >(pj)] + permPos[static_cast< size_t >(pj)]++;
                permAi[static_cast< size_t >(pos)] = pi;
                permAx[static_cast< size_t >(pos)] = v;
            }
        }

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
            Int p_col =
                m_factors.etree[static_cast< size_t >(snodes[static_cast< size_t >(si)].hi)];
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
                if (ci >=
                    static_cast< Int >(snodes[static_cast< size_t >(top.first)].children.size())) {
                    postorder.push_back(top.first);
                    st.pop_back();
                    if (!st.empty())
                        st.back().second++;
                } else {
                    Int child = snodes[static_cast< size_t >(top.first)]
                                    .children[static_cast< size_t >(ci)];
                    st.emplace_back(child, 0);
                }
            }
        }

        std::vector< Int > globalToLocal(static_cast< size_t >(m_size), -1);
        // (row, col, value) triples for L entries
        struct Ltrip {
            Int row, col;
            double val;
        };
        std::vector< Ltrip > trips;
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
            update_rows.erase(std::unique(update_rows.begin(), update_rows.end()),
                              update_rows.end());

            const Int nupd = static_cast< Int >(update_rows.size());
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
            sn.update_rows = std::move(update_rows);
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

        m_factors.factorized = true;
        m_factorized = true;
        m_info = m_factors.info_val;
    }

    // ===== Solve: forward + diagonal + backward + un-permute ==============

    std::vector< Scalar > solveImpl(const std::vector< Scalar >& b) const {
        std::vector< Scalar > x(b.size());

        // Permute: x_perm[new] = b[perm[new]] = b[old]. m_factors.perm holds
        // perm[new] = old (see Ordering::from_perm / linsys::amd_ordering),
        // so the forward gather must index with perm, not iperm.
        if (!m_factors.perm.empty()) {
            std::vector< Scalar > y(static_cast< size_t >(m_size));
            permute_gather(m_size, m_factors.perm.data(), b.data(), y.data());
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

        // Un-permute: x_old[old] = x_perm[iperm[old]]. m_factors.iperm holds
        // iperm[old] = new, the inverse of perm, so the backward gather must
        // index with iperm here.
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        if (!m_factors.iperm.empty()) {
            permute_gather(m_size, m_factors.iperm.data(), x.data(), result.data());
        } else {
            result = std::move(x);
        }

        return result;
    }

    // ===== Pattern analysis helpers =========================================

    void computeOrdering(const MatrixType& a) {
        if (m_useExternalOrdering) {
            m_ordering = m_externalOrdering;
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
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Index i = iperm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const Index j2 = iperm_idx[static_cast< size_t >(j)];
                if (i < 0 || i >= m_size || j2 < 0 || j2 >= m_size)
                    continue;

                const Index row = std::min(i, j2);
                const Index col = std::max(i, j2);
                triples.emplace_back(col, row,
                                     static_cast< double >(a.Ax[static_cast< size_t >(p)]));
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
            const Index pos = ap[static_cast< size_t >(pj)] + curPos[static_cast< size_t >(pj)]++;
            ai[static_cast< size_t >(pos)] = pi;
            ax[static_cast< size_t >(pos)] = v;
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


    // ===== State ==========================================================

    Index m_size = 0;
    SupernodalFactor m_factors;
    Ordering< Index > m_ordering;
    Ordering< Index > m_externalOrdering;
    bool m_useExternalOrdering = false;
    bool m_patternAnalyzed = false;
    bool m_factorized = false;
    Index m_info = SupernodalFactor::NotInitialized;

    Index m_numPerturbedPivots = 0;
    double m_minAbsPivot = 0.0;
    double m_regularization = 1e-12;

    bool m_supernodal = false;
};

} // namespace supernodal

#endif // SUPERSONAL_LDLT_STANDALONE_H
