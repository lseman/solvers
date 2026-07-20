/*
 * hipo_ldlt.h — Self-contained multifrontal augmented-system solver,
 * following Zanetti & Gondzio, "A factorisation-based regularised interior
 * point method using the augmented system" (2025), arXiv:2508.04370 (HiPO,
 * the HiGHS interior point solver). AMD is used in place of the paper's
 * Metis nested dissection (§4.2) — everything else follows the paper
 * directly rather than through the shared supernodal_ldlt.h path:
 *
 *   §3.2      static + dynamic regularisation of the augmented system
 *   §4.1      multifrontal factorisation (elimination tree, frontal
 *             matrices, Schur complements passed up the tree)
 *   §4.2      ordering (AMD here, not Metis) computed once and reused
 *   §4.3      elimination tree / supernode tree driving the assembly order
 *   §4.5      Bunch-Kaufman 1x1/2x2 pivoting, restricted to each frontal
 *             matrix's own pivot-block columns (BD in the paper's Fig. 4),
 *             plus perturbation of pivots that are too small
 *   §4.6      Proposition 3's sign-preservation bound on the dynamic
 *             perturbation, so the Schur complement's diagonal entries keep
 *             their pre-elimination sign
 *
 * The augmented system solved is
 *
 *   [ -(Theta + Rp)   A^T ] [dx]   [xi_d]
 *   [       A          Rd ] [dy] = [xi_p]
 *
 * which is quasi-definite once Rp, Rd > 0 (§3.2, eq. 8): the (1,1) block is
 * negative definite, the (2,2) block positive definite. Ordering is done
 * per-block (blockAmdOrdering below) so no pivot ever
 * crosses the block boundary — this is what lets BK's pivot search stay
 * within a supernode's own columns and still guarantee the Schur complement
 * remains quasi-definite (§4.6).
 *
 * Reused as generic building blocks (not paper-specific, also used by the
 * standalone python solver bindings):
 *   linsys::SparseCSC, DenseMatrix, Ordering, amd_ordering   (common/)
 *   linsys::lsolve_unit/ltsolve_unit/permute_gather/scatter  (trisolve.h)
 *   snode::identify_supernodes, postorder_etree              (supernodes.h)
 *   blockAmdOrdering                                         (below)
 *
 * Not reused: the multifrontal tree walk, frontal matrix assembly, dense
 * kernel (BK + regularisation), and iterative refinement are implemented
 * here directly rather than going through supernodal_ldlt.h, so this file
 * is a faithful single-file rendering of §4 rather than a thin wrapper.
 *
 * Usage:
 *   HiPOLDLT solver;
 *   solver.analyzePattern(A);                    // once (sparsity invariant)
 *   solver.factorize(theta, regP, regD);          // each IPM iteration
 *   auto [x, niters] = solver.solveWithRefinement(rhs, theta, regP, regD);
 */

#pragma once

#ifndef IPM_HIPO_LDLT_H
#define IPM_HIPO_LDLT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Sparse>
#include <limits>
#include <utility>
#include <vector>

#include "../linear_system/common/dense_matrix.h"
#include "../linear_system/common/ordering.h"
#include "../linear_system/common/sparse_csc.h"
#include "../linear_system/common/trisolve.h"
#include "../linear_system/supernodes.h"

namespace ipm {

using HReal = double;
using HIndex = int32_t;

/** Block-preserving AMD ordering for the quasi-definite augmented system. */
inline linsys::Ordering< HIndex >
blockAmdOrdering(HIndex n, HIndex m, const std::vector< std::vector< HIndex > >& A_cols) {
    std::vector< std::pair< HIndex, HIndex > > primal_edges;
    std::vector< std::pair< HIndex, HIndex > > dual_edges;
    std::vector< std::vector< HIndex > > row_cols(static_cast< size_t >(m));

    for (HIndex j = 0; j < n; ++j) {
        for (HIndex i : A_cols[static_cast< size_t >(j)]) {
            if (i >= 0 && i < m)
                row_cols[static_cast< size_t >(i)].push_back(j);
        }
    }
    for (HIndex i = 0; i < m; ++i) {
        const auto& cols = row_cols[static_cast< size_t >(i)];
        for (size_t a = 0; a < cols.size(); ++a)
            for (size_t b = a + 1; b < cols.size(); ++b)
                primal_edges.emplace_back(cols[a], cols[b]);
    }
    for (HIndex j = 0; j < n; ++j) {
        const auto& rows = A_cols[static_cast< size_t >(j)];
        for (size_t a = 0; a < rows.size(); ++a)
            for (size_t b = a + 1; b < rows.size(); ++b)
                dual_edges.emplace_back(rows[a], rows[b]);
    }

    std::vector< HIndex > primal_perm =
        n > 0 ? linsys::amd_ordering(n, primal_edges) : std::vector< HIndex >{};
    std::vector< HIndex > dual_perm =
        m > 0 ? linsys::amd_ordering(m, dual_edges) : std::vector< HIndex >{};
    std::vector< HIndex > perm(static_cast< size_t >(n + m));
    for (HIndex old_j = 0; old_j < n; ++old_j)
        perm[static_cast< size_t >(old_j)] = primal_perm[static_cast< size_t >(old_j)];
    for (HIndex old_i = 0; old_i < m; ++old_i)
        perm[static_cast< size_t >(n + old_i)] = n + dual_perm[static_cast< size_t >(old_i)];
    return linsys::Ordering< HIndex >::from_perm(std::move(perm));
}

// §3.2 defaults: static regularisation is a small uniform floor applied to
// every pivot; dynamic regularisation (via Proposition 3) only kicks in when
// a pivot is smaller than this after the static floor is already included
// in Rp/Rd by the caller (ip_solver.cpp decays regP/regD toward this each
// iteration — see ipm/README.md).
static const HReal HIPO_STATIC_REG = 1e-10;
static const HReal HIPO_PIVOT_TOL = 1e-11;
static const int HIPO_MAX_REFINEMENT = 2;
static const HReal HIPO_REFINEMENT_TOL = 1e-10;

/// L (CSC, unit lower triangular) + block-diagonal D, in permuted space.
struct HiPOFactor {
    HIndex n = 0;
    std::vector< HIndex > Lp, Li;
    std::vector< HReal > Lx;
    // D blocks, interleaved: 1x1 -> [d]; 2x2 -> [d11, d21, d22].
    std::vector< HReal > D;
    std::vector< HIndex > block_info; // 1=1x1, 2=start of 2x2, 0=second col of 2x2
    HIndex perturbed_pivots = 0;
    HIndex num_2x2 = 0;
    HReal min_abs_pivot = 0.0;
    bool factorized = false;
};

/**
 * Multifrontal quasi-definite augmented-system solver (§4 of the paper).
 */
class HiPOLDLT {
  public:
    HiPOLDLT() : m_n(0), m_m(0) {
    }

    /** Analyze pattern: block-AMD order + supernode detection. Sparsity is
     * IPM-invariant, so this runs once before the IPM loop (§4.2). */
    void analyzePattern(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& A) {
        m_A = A;
        m_n = A.cols();
        m_m = A.rows();
        const HIndex N = m_n + m_m;

        std::vector< std::vector< HIndex > > A_cols(static_cast< size_t >(m_n));
        for (int j = 0; j < m_n; ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                A_cols[static_cast< size_t >(j)].push_back(static_cast< HIndex >(it.row()));
            }
        }
        m_ordering = blockAmdOrdering(static_cast< HIndex >(m_n), static_cast< HIndex >(m_m), A_cols);

        // Structural (symbolic) augmented-system pattern in permuted space,
        // upper-triangular, to drive etree + supernode detection (§4.1, §4.3).
        buildSymbolicPattern();

        std::vector< HIndex > parent = computeEtree();
        snode::SparseUpperCSC< HIndex > B{N, &m_symAp, &m_symAi, nullptr};
        snode::Symbolic< HIndex > S{N, &parent, nullptr, nullptr};
        // relax_abs=0, relax_rel=0, tau=1: fundamental supernodes only, no
        // amalgamation (paper §4.1 mentions relaxed partitions as a possible
        // efficiency improvement; not enabled here — see ipm/README.md).
        m_supernodes = snode::identify_supernodes< HIndex >(B, S, 0, 0.0, 1.0,
                                                             std::numeric_limits< HIndex >::max());
        m_etree = std::move(parent);

        buildSupernodeTree();
        m_patternAnalyzed = true;
    }

    /** Numeric factorization: multifrontal LDL^T with BK pivoting +
     * regularisation (§4.1, §4.5, §4.6). theta/regP/regD carry the static
     * regularisation floor (decayed by the IPM loop each iteration; see
     * ipm/README.md); dynamic regularisation is applied here per-pivot. */
    void factorize(const std::vector< HReal >& theta, const std::vector< HReal >& regP,
                   const std::vector< HReal >& regD) {
        if (!m_patternAnalyzed)
            analyzePattern(m_A);
        if (m_n == 0 || m_m == 0)
            return;

        const HIndex N = m_n + m_m;
        buildPermutedAugmentedMatrix(theta, regP, regD);

        m_factor = HiPOFactor{};
        m_factor.n = N;
        m_factor.D.clear();
        m_factor.block_info.assign(static_cast< size_t >(N), 0);
        m_factor.perturbed_pivots = 0;
        m_factor.num_2x2 = 0;
        m_factor.min_abs_pivot = 0.0;
        m_DByColumn.assign(static_cast< size_t >(N), {});

        std::vector< std::vector< std::pair< HIndex, HReal > > > Ltrips(static_cast< size_t >(N));
        // Per-supernode Schur complement, keyed by supernode id, passed to
        // the parent when the child is processed (§4.1: "Schur complements
        // are stored only for the time needed to pass them from one frontal
        // matrix to another").
        std::vector< linsys::DenseMatrix< HReal > > childSchur(m_snodes.size());
        std::vector< std::vector< HIndex > > childUpdateRows(m_snodes.size());

        // BK pivoting swaps columns within each supernode's own pivot block
        // (§4.5), so the final elimination-order-to-original-variable
        // mapping is AMD's ordering composed with those swaps. Track it
        // here: m_pivotOrderPreBK starts as AMD's iperm (position -> pre-AMD
        // variable); m_pivotOrder accumulates the composed result as each
        // supernode is processed, then overwrites m_ordering.iperm/perm.
        m_pivotOrderPreBK = m_ordering.iperm;
        m_pivotOrder = m_ordering.iperm;

        for (HIndex si : m_snodePostorder) {
            factorizeSupernode(si, Ltrips, childSchur, childUpdateRows);
        }

        // Supernodes are factorized in tree postorder, which is not
        // necessarily global column order. solve() walks D by elimination
        // column, so flatten the staged blocks in that order.
        for (HIndex k = 0; k < N; ++k) {
            const auto& block = m_DByColumn[static_cast< size_t >(k)];
            m_factor.D.insert(m_factor.D.end(), block.begin(), block.end());
        }
        m_DByColumn.clear();

        // Final solve-time ordering: AMD composed with this factorize()
        // call's BK swaps. Kept separate from m_ordering (which stays
        // AMD-only, reused by analyzePattern across IPM iterations).
        m_finalIperm = m_pivotOrder;
        m_finalPerm.assign(static_cast< size_t >(N), HIndex{-1});
        for (HIndex pos = 0; pos < N; ++pos)
            m_finalPerm[static_cast< size_t >(m_finalIperm[static_cast< size_t >(pos)])] = pos;

        assembleL(Ltrips);
        m_factor.factorized = true;
    }

    /** Solve without refinement. */
    std::vector< HReal > solve(const std::vector< HReal >& rhs) const {
        if (!m_factor.factorized)
            return {};
        const HIndex N = m_factor.n;

        // x_perm[new] = rhs[old] = rhs[iperm[new]]. m_finalIperm[new] = old
        // is AMD composed with this factorization's BK within-block swaps
        // (see factorize()) — the forward gather needs its inverse role
        // relative to m_finalPerm, i.e. iperm itself here.
        std::vector< HReal > x(static_cast< size_t >(N));
        linsys::permute_gather< HReal, HIndex >(N, m_finalIperm.data(), rhs.data(), x.data());

        linsys::lsolve_unit< HReal, HIndex >(N, m_factor.Lp.data(), m_factor.Li.data(),
                                             m_factor.Lx.data(), x.data());

        HIndex ptr = 0;
        for (HIndex k = 0; k < N;) {
            if (m_factor.block_info[static_cast< size_t >(k)] == 1) {
                const HReal d = m_factor.D[static_cast< size_t >(ptr)];
                x[static_cast< size_t >(k)] = (d != 0.0) ? x[static_cast< size_t >(k)] / d : 0.0;
                ptr += 1;
                k += 1;
            } else {
                const HReal d11 = m_factor.D[static_cast< size_t >(ptr)];
                const HReal d21 = m_factor.D[static_cast< size_t >(ptr + 1)];
                const HReal d22 = m_factor.D[static_cast< size_t >(ptr + 2)];
                const HReal det = d11 * d22 - d21 * d21;
                const HReal b0 = x[static_cast< size_t >(k)];
                const HReal b1 = x[static_cast< size_t >(k + 1)];
                if (det != 0.0) {
                    x[static_cast< size_t >(k)] = (d22 * b0 - d21 * b1) / det;
                    x[static_cast< size_t >(k + 1)] = (d11 * b1 - d21 * b0) / det;
                } else {
                    x[static_cast< size_t >(k)] = (d11 != 0.0) ? b0 / d11 : 0.0;
                    x[static_cast< size_t >(k + 1)] = (d22 != 0.0) ? b1 / d22 : 0.0;
                }
                ptr += 3;
                k += 2;
            }
        }

        linsys::ltsolve_unit< HReal, HIndex >(N, m_factor.Lp.data(), m_factor.Li.data(),
                                              m_factor.Lx.data(), x.data());

        // result[old] = x_perm[perm[old]]. m_finalPerm[old] = new is the
        // composed forward mapping, exactly what the backward gather needs.
        std::vector< HReal > result(static_cast< size_t >(N));
        linsys::permute_gather< HReal, HIndex >(N, m_finalPerm.data(), x.data(), result.data());
        return result;
    }

    /** Solve with component-wise backward-error iterative refinement (§4.5,
     * ref [10]). */
    std::pair< std::vector< HReal >, int >
    solveWithRefinement(const std::vector< HReal >& rhs, const std::vector< HReal >& theta,
                        const std::vector< HReal >& regP, const std::vector< HReal >& regD) const {
        auto x = solve(rhs);
        int niters = 0;

        for (int iter = 0; iter < HIPO_MAX_REFINEMENT; ++iter) {
            std::vector< HReal > r = rhs;
            applyAugmentedMatVec(x, r, theta, regP, regD);

            const HReal max_err = computeBackwardError(r, x, theta, regP, regD);
            if (max_err < HIPO_REFINEMENT_TOL)
                break;

            auto dx = solve(r);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] += dx[i];
            niters = iter + 1;
        }

        return {std::move(x), niters};
    }

    int n() const {
        return m_n;
    }
    int m() const {
        return m_m;
    }
    HIndex perturbedPivots() const {
        return m_factor.perturbed_pivots;
    }
    HIndex num2x2Pivots() const {
        return m_factor.num_2x2;
    }
    HReal minAbsPivot() const {
        return m_factor.min_abs_pivot;
    }

    // DEBUG (temporary)
    const linsys::Ordering< HIndex >& debugOrdering() const { return m_ordering; }
    const std::vector< HIndex >& debugNumAp() const { return m_numAp; }
    const std::vector< HIndex >& debugNumAi() const { return m_numAi; }
    const std::vector< HReal >& debugNumAx() const { return m_numAx; }
    const HiPOFactor& debugFactor() const { return m_factor; }
    const std::vector< HIndex >& debugFinalPerm() const { return m_finalPerm; }
    const std::vector< HIndex >& debugFinalIperm() const { return m_finalIperm; }

  private:
    int m_n = 0, m_m = 0;
    Eigen::SparseMatrix< double, Eigen::ColMajor, int > m_A;
    linsys::Ordering< HIndex > m_ordering; // AMD-only; stable across factorize() calls

    // BK-composed ordering (AMD + this factorize() call's within-block
    // pivot swaps), rebuilt every factorize(); used by solve().
    std::vector< HIndex > m_pivotOrderPreBK; // scratch: AMD iperm at factorize() start
    std::vector< HIndex > m_pivotOrder;      // scratch: accumulates composed iperm
    std::vector< HIndex > m_finalPerm, m_finalIperm;
    std::vector< std::vector< HReal > > m_DByColumn;

    // Symbolic pattern (permuted, upper-triangular) of the augmented system.
    std::vector< HIndex > m_symAp, m_symAi;
    std::vector< HIndex > m_etree;
    snode::SupernodeInfo< HIndex > m_supernodes;
    std::vector< HIndex > m_snodePostorder;

    struct SnodeMeta {
        HIndex lo = 0, hi = 0, npiv = 0;
        HIndex parent_sn = -1;
        std::vector< HIndex > children;
    };
    std::vector< SnodeMeta > m_snodes;

    // Numeric, permuted augmented matrix (full symmetric CSC, both
    // triangles) for the current factorize() call.
    std::vector< HIndex > m_numAp, m_numAi;
    std::vector< HReal > m_numAx;

    HiPOFactor m_factor;
    bool m_patternAnalyzed = false;

    // ===== Pattern analysis helpers =========================================

    void buildSymbolicPattern() {
        const HIndex N = m_n + m_m;
        std::vector< std::vector< HIndex > > cols(static_cast< size_t >(N));

        // (1,1) diagonal.
        for (HIndex j = 0; j < m_n; ++j)
            cols[static_cast< size_t >(j)].push_back(j);
        // (1,2) block: A^T entries land at (row=j, col=n+i) in upper form,
        // i.e. column j (primal) gets a row at n+i (dual) since perm keeps
        // primal < dual always (blockAmdOrdering's guarantee) so j < n+i.
        for (int j = 0; j < m_n; ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                cols[static_cast< size_t >(j)].push_back(static_cast< HIndex >(m_n + it.row()));
            }
        }
        // (2,2) diagonal.
        for (HIndex i = 0; i < m_m; ++i)
            cols[static_cast< size_t >(m_n + i)].push_back(m_n + i);

        // Permute columns/rows into new-space (perm[old]=new) and re-bucket.
        std::vector< std::vector< HIndex > > permCols(static_cast< size_t >(N));
        for (HIndex old_j = 0; old_j < N; ++old_j) {
            const HIndex new_j = m_ordering.perm[static_cast< size_t >(old_j)];
            for (HIndex old_i : cols[static_cast< size_t >(old_j)]) {
                const HIndex new_i = m_ordering.perm[static_cast< size_t >(old_i)];
                const HIndex r = std::min(new_i, new_j);
                const HIndex c = std::max(new_i, new_j);
                permCols[static_cast< size_t >(c)].push_back(r);
            }
        }

        m_symAp.assign(static_cast< size_t >(N) + 1, 0);
        m_symAi.clear();
        for (HIndex j = 0; j < N; ++j) {
            auto& col = permCols[static_cast< size_t >(j)];
            std::sort(col.begin(), col.end());
            col.erase(std::unique(col.begin(), col.end()), col.end());
            m_symAi.insert(m_symAi.end(), col.begin(), col.end());
            m_symAp[static_cast< size_t >(j) + 1] = static_cast< HIndex >(m_symAi.size());
        }
    }

    // Elimination tree from the (permuted, upper-triangular) symbolic
    // pattern via ancestor path-compression (§4.3).
    std::vector< HIndex > computeEtree() const {
        const HIndex N = m_n + m_m;
        std::vector< HIndex > parent(static_cast< size_t >(N), HIndex{-1});
        std::vector< HIndex > ancestor(static_cast< size_t >(N), HIndex{-1});

        for (HIndex j = 0; j < N; ++j) {
            for (HIndex p = m_symAp[static_cast< size_t >(j)]; p < m_symAp[static_cast< size_t >(j) + 1];
                 ++p) {
                HIndex i = m_symAi[static_cast< size_t >(p)];
                while (i != HIndex{-1} && i < j) {
                    const HIndex next = ancestor[static_cast< size_t >(i)];
                    ancestor[static_cast< size_t >(i)] = j;
                    if (next == HIndex{-1})
                        parent[static_cast< size_t >(i)] = j;
                    i = next;
                }
            }
        }
        return parent;
    }

    void buildSupernodeTree() {
        const size_t ns = m_supernodes.ranges.size();
        m_snodes.assign(ns, SnodeMeta{});
        for (size_t si = 0; si < ns; ++si) {
            m_snodes[si].lo = m_supernodes.ranges[si].first;
            m_snodes[si].hi = m_supernodes.ranges[si].second;
            m_snodes[si].npiv = m_snodes[si].hi - m_snodes[si].lo + 1;
        }
        for (size_t si = 0; si < ns; ++si) {
            const HIndex p_col = m_etree[static_cast< size_t >(m_snodes[si].hi)];
            if (p_col < 0)
                continue;
            const HIndex p_sn = m_supernodes.col2sn[static_cast< size_t >(p_col)];
            if (p_sn >= 0 && static_cast< size_t >(p_sn) != si) {
                m_snodes[si].parent_sn = p_sn;
                m_snodes[static_cast< size_t >(p_sn)].children.push_back(static_cast< HIndex >(si));
            }
        }

        // Postorder of the supernode forest (children before parents) —
        // determines the multifrontal assembly order (§4.3).
        m_snodePostorder.clear();
        m_snodePostorder.reserve(ns);
        std::vector< std::pair< HIndex, HIndex > > st;
        for (size_t si = 0; si < ns; ++si) {
            if (m_snodes[si].parent_sn != -1)
                continue;
            st.emplace_back(static_cast< HIndex >(si), HIndex{0});
            while (!st.empty()) {
                auto& top = st.back();
                HIndex& ci = top.second;
                if (ci >= static_cast< HIndex >(m_snodes[static_cast< size_t >(top.first)].children.size())) {
                    m_snodePostorder.push_back(top.first);
                    st.pop_back();
                    if (!st.empty())
                        ++st.back().second;
                } else {
                    const HIndex child = m_snodes[static_cast< size_t >(top.first)]
                                              .children[static_cast< size_t >(ci)];
                    st.emplace_back(child, HIndex{0});
                }
            }
        }
    }

    // ===== Numeric factorization helpers ====================================

    void buildPermutedAugmentedMatrix(const std::vector< HReal >& theta,
                                      const std::vector< HReal >& regP,
                                      const std::vector< HReal >& regD) {
        const HIndex N = m_n + m_m;
        std::vector< std::vector< std::pair< HIndex, HReal > > > cols(static_cast< size_t >(N));

        for (int j = 0; j < m_n; ++j) {
            const HReal v = -(theta[static_cast< size_t >(j)] + regP[static_cast< size_t >(j)]);
            const HIndex new_j = m_ordering.perm[static_cast< size_t >(j)];
            cols[static_cast< size_t >(new_j)].emplace_back(new_j, v);
        }
        // Emit both triangles explicitly (full symmetric storage, matching
        // the convention every solver in linear_system/ expects).
        for (int j = 0; j < m_A.cols(); ++j) {
            const HIndex new_j = m_ordering.perm[static_cast< size_t >(j)];
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                const HIndex new_i = m_ordering.perm[static_cast< size_t >(m_n + it.row())];
                const HReal v = it.value();
                cols[static_cast< size_t >(std::max(new_i, new_j))].emplace_back(
                    std::min(new_i, new_j), v);
                cols[static_cast< size_t >(std::min(new_i, new_j))].emplace_back(
                    std::max(new_i, new_j), v);
            }
        }
        for (int i = 0; i < m_m; ++i) {
            const HIndex new_i = m_ordering.perm[static_cast< size_t >(m_n + i)];
            cols[static_cast< size_t >(new_i)].emplace_back(new_i, regD[static_cast< size_t >(i)]);
        }

        m_numAp.assign(static_cast< size_t >(N) + 1, 0);
        m_numAi.clear();
        m_numAx.clear();
        for (HIndex j = 0; j < N; ++j) {
            auto& col = cols[static_cast< size_t >(j)];
            std::sort(col.begin(), col.end());
            for (size_t p = 0; p < col.size(); ++p) {
                if (p > 0 && col[p].first == col[p - 1].first) {
                    m_numAx.back() += col[p].second;
                } else {
                    m_numAi.push_back(col[p].first);
                    m_numAx.push_back(col[p].second);
                }
            }
            m_numAp[static_cast< size_t >(j) + 1] = static_cast< HIndex >(m_numAi.size());
        }
    }

    HReal lookupNum(HIndex row, HIndex col) const {
        for (HIndex p = m_numAp[static_cast< size_t >(col)]; p < m_numAp[static_cast< size_t >(col) + 1];
             ++p) {
            if (m_numAi[static_cast< size_t >(p)] == row)
                return m_numAx[static_cast< size_t >(p)];
        }
        return 0.0;
    }

    // Frontal matrix assembly + BK factorization for one supernode (§4.1,
    // §4.4, §4.5). Mirrors supernodal_ldlt.h's factorizeSupernodal loop
    // structurally, but with a BK-capable dense kernel instead of plain
    // diagonal pivots.
    void factorizeSupernode(HIndex si, std::vector< std::vector< std::pair< HIndex, HReal > > >& Ltrips,
                            std::vector< linsys::DenseMatrix< HReal > >& childSchur,
                            std::vector< std::vector< HIndex > >& childUpdateRows) {
        auto& sn = m_snodes[static_cast< size_t >(si)];
        const HIndex col_lo = sn.lo, col_hi = sn.hi, npiv = sn.npiv;

        std::vector< HIndex > update_rows;
        std::vector< char > seen(static_cast< size_t >(m_n + m_m), 0);
        for (HIndex pj = col_lo; pj <= col_hi; ++pj) {
            for (HIndex p = m_numAp[static_cast< size_t >(pj)]; p < m_numAp[static_cast< size_t >(pj) + 1];
                 ++p) {
                const HIndex pi = m_numAi[static_cast< size_t >(p)];
                if (pi > col_hi && !seen[static_cast< size_t >(pi)]) {
                    seen[static_cast< size_t >(pi)] = 1;
                    update_rows.push_back(pi);
                }
            }
        }
        for (HIndex child : sn.children) {
            for (HIndex u : childUpdateRows[static_cast< size_t >(child)]) {
                if (u > col_hi && !seen[static_cast< size_t >(u)]) {
                    seen[static_cast< size_t >(u)] = 1;
                    update_rows.push_back(u);
                }
            }
        }
        std::sort(update_rows.begin(), update_rows.end());

        const HIndex nupd = static_cast< HIndex >(update_rows.size());
        const HIndex fsize = npiv + nupd;

        std::vector< HIndex > globalToLocal(static_cast< size_t >(m_n + m_m), -1);
        for (HIndex k = 0; k < npiv; ++k)
            globalToLocal[static_cast< size_t >(col_lo + k)] = k;
        for (HIndex u = 0; u < nupd; ++u)
            globalToLocal[static_cast< size_t >(update_rows[static_cast< size_t >(u)])] = npiv + u;

        linsys::DenseMatrix< HReal > F(fsize, fsize);
        F.setZero();

        // Gather this supernode's own entries (lower triangle in permuted
        // space: pivot/pivot and update/pivot blocks).
        for (HIndex pj = col_lo; pj <= col_hi; ++pj) {
            for (HIndex p = m_numAp[static_cast< size_t >(pj)]; p < m_numAp[static_cast< size_t >(pj) + 1];
                 ++p) {
                const HIndex pi = m_numAi[static_cast< size_t >(p)];
                if (pi < pj)
                    continue;
                const HIndex li = globalToLocal[static_cast< size_t >(pi)];
                const HIndex lj = globalToLocal[static_cast< size_t >(pj)];
                if (li < 0 || lj < 0)
                    continue;
                F(li, lj) += m_numAx[static_cast< size_t >(p)];
            }
        }

        // Accumulate children's Schur complements (§4.1: pass Schur
        // complement from one frontal matrix to the next up the tree).
        for (HIndex child : sn.children) {
            const auto& cSchur = childSchur[static_cast< size_t >(child)];
            const auto& cUpdates = childUpdateRows[static_cast< size_t >(child)];
            const HIndex cn = static_cast< HIndex >(cUpdates.size());
            for (HIndex ia = 0; ia < cn; ++ia) {
                const HIndex la = globalToLocal[static_cast< size_t >(cUpdates[static_cast< size_t >(ia)])];
                if (la < 0 || la >= fsize)
                    continue;
                for (HIndex ib = 0; ib < cn; ++ib) {
                    const HIndex lb =
                        globalToLocal[static_cast< size_t >(cUpdates[static_cast< size_t >(ib)])];
                    if (lb < 0 || lb >= fsize)
                        continue;
                    F(la, lb) += cSchur(ia, ib);
                }
            }
            // Free the child's Schur complement now that it's been folded in.
            childSchur[static_cast< size_t >(child)] = linsys::DenseMatrix< HReal >();
        }

        // Own-column assembly above fills the lower triangle. BK pivoting
        // performs full symmetric row/column swaps, so materialize the upper
        // triangle before entering the dense kernel.
        for (HIndex j = 0; j < fsize; ++j)
            for (HIndex i = j + 1; i < fsize; ++i)
                F(j, i) = F(i, j);

        std::vector< HReal > D_local;
        std::vector< HIndex > block_local;
        std::vector< HIndex > localPerm;
        bkFactorizeFrontal(F, fsize, npiv, D_local, block_local, localPerm);

        // localPerm[k] tells us which original local column now sits at
        // position k. Record the composed permutation: final elimination
        // position col_lo+k actually holds the pre-BK-swap column
        // col_lo+localPerm[k], whose original (pre-AMD) identity is
        // m_pivotOrderPreBK[col_lo + localPerm[k]].
        for (HIndex k = 0; k < npiv; ++k) {
            m_pivotOrder[static_cast< size_t >(col_lo + k)] =
                m_pivotOrderPreBK[static_cast< size_t >(col_lo + localPerm[static_cast< size_t >(k)])];
        }

        // A BK interchange in this pivot block also permutes these variables'
        // rows in every L column that was already assembled by descendants.
        // localPerm[new_local] = old_local, so invert it to remap each stored
        // pre-swap row position to its final elimination position.
        std::vector< HIndex > inverseLocalPerm(static_cast< size_t >(npiv));
        for (HIndex new_local = 0; new_local < npiv; ++new_local)
            inverseLocalPerm[static_cast< size_t >(localPerm[static_cast< size_t >(new_local)])] = new_local;
        for (HIndex j = 0; j < col_lo; ++j) {
            for (auto& entry : Ltrips[static_cast< size_t >(j)]) {
                if (entry.first >= col_lo && entry.first <= col_hi) {
                    const HIndex old_local = entry.first - col_lo;
                    entry.first = col_lo + inverseLocalPerm[static_cast< size_t >(old_local)];
                }
            }
        }

        // Store D + block_info for this supernode's pivot columns.
        for (HIndex k = 0; k < npiv; ++k)
            m_factor.block_info[static_cast< size_t >(col_lo + k)] = block_local[static_cast< size_t >(k)];
        HIndex dpos = 0;
        for (HIndex k = 0; k < npiv;) {
            const HIndex block_size = block_local[static_cast< size_t >(k)] == 2 ? 3 : 1;
            auto& block = m_DByColumn[static_cast< size_t >(col_lo + k)];
            block.assign(D_local.begin() + dpos, D_local.begin() + dpos + block_size);
            dpos += block_size;
            k += block_local[static_cast< size_t >(k)] == 2 ? 2 : 1;
        }

        // Extract Schur complement (update/update block) for the parent.
        linsys::DenseMatrix< HReal > schur;
        if (nupd > 0) {
            schur.resize(nupd, nupd);
            for (HIndex j = 0; j < nupd; ++j)
                for (HIndex i = 0; i < nupd; ++i)
                    schur(i, j) = F(npiv + i, npiv + j);
        }
        childSchur[static_cast< size_t >(si)] = std::move(schur);
        childUpdateRows[static_cast< size_t >(si)] = update_rows;

        // Extract L entries (columns 0..npiv-1, rows below the pivot's own
        // block-diagonal entry — for a 2x2 block starting at local col k,
        // L(k+1,k) is not a multiplier, so skip it).
        for (HIndex k = 0; k < npiv; ++k) {
            const HIndex gk = col_lo + k;
            const HIndex first = (block_local[static_cast< size_t >(k)] == 2) ? k + 2 : k + 1;
            for (HIndex i = first; i < fsize; ++i) {
                const HReal lij = F(i, k);
                if (lij == 0.0)
                    continue;
                const HIndex gi = (i < npiv) ? (col_lo + i) : update_rows[static_cast< size_t >(i - npiv)];
                Ltrips[static_cast< size_t >(gk)].emplace_back(gi, lij);
            }
        }
    }

    // Proposition 3 (§4.6): minimum |p| such that every resulting Schur
    // diagonal entry (M11)_jj +/- (q1)_j^2/|p| keeps its sign. j ranges over
    // ALL rows the pivot updates (pivot-block + update rows) even though the
    // pivot search itself (below) is restricted to the pivot block.
    static HReal minSignPreservingPivotMagnitude(const linsys::DenseMatrix< HReal >& F, HIndex k,
                                                 HIndex fsize) {
        HReal bound = 0.0;
        for (HIndex j = k + 1; j < fsize; ++j) {
            const HReal q1j = F(j, k);
            if (q1j == 0.0)
                continue;
            const HReal diag_j = std::abs(F(j, j));
            if (diag_j <= 0.0)
                continue;
            const HReal needed = (q1j * q1j) / diag_j;
            if (needed > bound)
                bound = needed;
        }
        return bound;
    }

    // Bunch-Kaufman 1x1/2x2 pivoting restricted to the frontal matrix's own
    // pivot-block columns [0, npiv) (§4.5: "only the portion of pivotal
    // columns within BD are considered, while entries in BP are not used
    // during pivoting"), with the Schur update applied across the full
    // [0, fsize) frontal matrix. Dynamic regularisation uses the
    // Proposition 3 bound (§4.6) once a pivot is already below the static
    // floor.
    // localPerm[k] = the local pivot-block index (into [0, npiv)) whose
    // original column now sits at position k after BK's within-block swaps.
    // Callers need this because L-extraction and the D/block_info storage
    // must reference the *original* column identity, not the post-swap
    // position (§4.5's swaps happen only within BD but still permute which
    // original column ends up where).
    void bkFactorizeFrontal(linsys::DenseMatrix< HReal >& F, HIndex fsize, HIndex npiv,
                            std::vector< HReal >& D_local, std::vector< HIndex >& block_local,
                            std::vector< HIndex >& localPerm) {
        D_local.clear();
        block_local.assign(static_cast< size_t >(npiv), 0);
        localPerm.resize(static_cast< size_t >(npiv));
        for (HIndex i = 0; i < npiv; ++i)
            localPerm[static_cast< size_t >(i)] = i;

        const HReal alpha = (1.0 + std::sqrt(17.0)) / 8.0; // Bunch-Kaufman constant
        std::vector< HReal > c0(static_cast< size_t >(fsize)), c1(static_cast< size_t >(fsize));
        std::vector< HReal > mx(static_cast< size_t >(fsize)), my(static_cast< size_t >(fsize));

        auto symswap = [&](HIndex a, HIndex b, HIndex active_start) {
            if (a == b)
                return;
            // Swap the already-computed L rows, but do not interchange prior
            // pivot columns/diagonals. Apply the symmetric permutation only
            // to the active trailing matrix.
            for (HIndex j = 0; j < active_start; ++j)
                std::swap(F(a, j), F(b, j));
            for (HIndex j = active_start; j < fsize; ++j)
                std::swap(F(a, j), F(b, j));
            for (HIndex i = active_start; i < fsize; ++i)
                std::swap(F(i, a), F(i, b));
            if (a < npiv && b < npiv)
                std::swap(localPerm[static_cast< size_t >(a)], localPerm[static_cast< size_t >(b)]);
        };

        HIndex k = 0;
        while (k < npiv) {
            // Pivot search restricted to [k, npiv) — the frontal matrix's
            // own pivot-block columns, per §4.5.
            HReal lambda = 0.0;
            HIndex r = k;
            for (HIndex i = k + 1; i < npiv; ++i) {
                const HReal v = std::abs(F(i, k));
                if (v > lambda) {
                    lambda = v;
                    r = i;
                }
            }
            const HReal absakk = std::abs(F(k, k));

            bool two_by_two = false;
            HIndex swap_row = k;
            if (lambda > 0.0 && absakk < alpha * lambda) {
                HReal sigma = 0.0;
                for (HIndex i = k; i < npiv; ++i) {
                    if (i != r)
                        sigma = std::max(sigma, std::abs(F(i, r)));
                }
                if (absakk * sigma >= alpha * lambda * lambda) {
                    // 1x1 pivot with F(k,k)
                } else if (std::abs(F(r, r)) >= alpha * sigma) {
                    swap_row = r; // 1x1 pivot with F(r,r)
                } else {
                    two_by_two = true;
                }
            }

            if (!two_by_two) {
                symswap(k, swap_row, k);
                HReal d = F(k, k);

                if (std::abs(d) < HIPO_PIVOT_TOL) {
                    const HReal propBound = minSignPreservingPivotMagnitude(F, k, fsize);
                    const HReal floor = std::max(HIPO_STATIC_REG, propBound);
                    d = (d < 0.0 ? -floor : floor);
                    ++m_factor.perturbed_pivots;
                }
                D_local.push_back(d);
                block_local[static_cast< size_t >(k)] = 1;

                const HReal absd = std::abs(d);
                if (m_factor.min_abs_pivot == 0.0 || absd < m_factor.min_abs_pivot)
                    m_factor.min_abs_pivot = absd;

                const HReal dinv = (d != 0.0) ? 1.0 / d : 0.0;
                for (HIndex i = k + 1; i < fsize; ++i) {
                    c0[static_cast< size_t >(i)] = F(i, k);
                    F(i, k) = c0[static_cast< size_t >(i)] * dinv;
                }
                for (HIndex j = k + 1; j < fsize; ++j) {
                    const HReal s = c0[static_cast< size_t >(j)] * dinv;
                    if (s == 0.0)
                        continue;
                    for (HIndex i = j; i < fsize; ++i) {
                        const HReal upd = c0[static_cast< size_t >(i)] * s;
                        F(i, j) -= upd;
                        if (i != j)
                            F(j, i) -= upd;
                    }
                }
                ++k;
            } else {
                symswap(k + 1, r, k);
                const HReal d11 = F(k, k), d21 = F(k + 1, k), d22 = F(k + 1, k + 1);
                HReal det = d11 * d22 - d21 * d21;

                if (std::abs(det) < HIPO_PIVOT_TOL * HIPO_PIVOT_TOL) {
                    const HReal floor = HIPO_STATIC_REG;
                    det = (det < 0.0 ? -floor : floor);
                    ++m_factor.perturbed_pivots;
                }
                D_local.push_back(d11);
                D_local.push_back(d21);
                D_local.push_back(d22);
                block_local[static_cast< size_t >(k)] = 2;
                block_local[static_cast< size_t >(k + 1)] = 0;
                ++m_factor.num_2x2;

                const HReal detinv = (det != 0.0) ? 1.0 / det : 0.0;
                for (HIndex i = k + 2; i < fsize; ++i) {
                    const HReal b0 = F(i, k), b1 = F(i, k + 1);
                    c0[static_cast< size_t >(i)] = b0;
                    c1[static_cast< size_t >(i)] = b1;
                    const HReal x = (b0 * d22 - b1 * d21) * detinv;
                    const HReal y = (b1 * d11 - b0 * d21) * detinv;
                    mx[static_cast< size_t >(i)] = x;
                    my[static_cast< size_t >(i)] = y;
                    F(i, k) = x;
                    F(i, k + 1) = y;
                }
                for (HIndex j = k + 2; j < fsize; ++j) {
                    for (HIndex i = j; i < fsize; ++i) {
                        const HReal upd = mx[static_cast< size_t >(i)] * c0[static_cast< size_t >(j)] +
                                          my[static_cast< size_t >(i)] * c1[static_cast< size_t >(j)];
                        F(i, j) -= upd;
                        if (i != j)
                            F(j, i) -= upd;
                    }
                }
                k += 2;
            }
        }
    }

    void assembleL(std::vector< std::vector< std::pair< HIndex, HReal > > >& Ltrips) {
        const HIndex N = m_n + m_m;
        m_factor.Lp.assign(static_cast< size_t >(N) + 1, 0);
        m_factor.Li.clear();
        m_factor.Lx.clear();
        for (HIndex j = 0; j < N; ++j) {
            auto& col = Ltrips[static_cast< size_t >(j)];
            std::sort(col.begin(), col.end());
            for (auto& e : col) {
                m_factor.Li.push_back(e.first);
                m_factor.Lx.push_back(e.second);
            }
            m_factor.Lp[static_cast< size_t >(j) + 1] = static_cast< HIndex >(m_factor.Li.size());
        }
    }

    // ===== Refinement helpers ===============================================

    void applyAugmentedMatVec(const std::vector< HReal >& x, std::vector< HReal >& out,
                              const std::vector< HReal >& theta, const std::vector< HReal >& regP,
                              const std::vector< HReal >& regD) const {
        const int N = m_n + m_m;
        if (static_cast< int >(x.size()) != N || static_cast< int >(out.size()) != N)
            return;
        for (int i = 0; i < m_n; ++i)
            out[static_cast< size_t >(i)] -=
                -(theta[static_cast< size_t >(i)] + regP[static_cast< size_t >(i)]) *
                x[static_cast< size_t >(i)];
        for (int i = 0; i < m_m; ++i)
            out[static_cast< size_t >(m_n + i)] -=
                regD[static_cast< size_t >(i)] * x[static_cast< size_t >(m_n + i)];
        for (int j = 0; j < m_A.cols(); ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                out[static_cast< size_t >(j)] -=
                    it.value() * x[static_cast< size_t >(m_n + it.row())];
                out[static_cast< size_t >(m_n + it.row())] -=
                    it.value() * x[static_cast< size_t >(j)];
            }
        }
    }

    HReal computeBackwardError(const std::vector< HReal >& r, const std::vector< HReal >& x,
                               const std::vector< HReal >& theta, const std::vector< HReal >& regP,
                               const std::vector< HReal >& regD) const {
        HReal x_inf = 0.0;
        for (HReal xi : x)
            x_inf = std::max(x_inf, std::abs(xi));

        std::vector< HReal > abs_row_sum(static_cast< size_t >(m_n + m_m), 0.0);
        for (int i = 0; i < m_n; ++i)
            abs_row_sum[static_cast< size_t >(i)] =
                theta[static_cast< size_t >(i)] + regP[static_cast< size_t >(i)];
        for (int i = 0; i < m_m; ++i)
            abs_row_sum[static_cast< size_t >(m_n + i)] = regD[static_cast< size_t >(i)];
        for (int j = 0; j < m_A.cols(); ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                abs_row_sum[static_cast< size_t >(j)] += std::abs(it.value());
                abs_row_sum[static_cast< size_t >(m_n + it.row())] += std::abs(it.value());
            }
        }

        HReal max_err = 0.0;
        for (size_t i = 0; i < r.size(); ++i) {
            HReal denom = abs_row_sum[i] * (x_inf + std::abs(x[i]));
            if (denom < 1e-300)
                denom = 1e-300;
            max_err = std::max(max_err, std::abs(r[i]) / denom);
        }
        return max_err;
    }
};

} // namespace ipm

#endif // IPM_HIPO_LDLT_H
