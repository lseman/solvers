/// supernodes.h — Supernode identification for sparse LDLᵀ factorization
///
/// Identifies supernodes (groups of consecutive columns with identical lower
/// L-patterns) from the permuted upper CSC matrix and symbolic analysis.
/// Supernodes enable BLAS-3 kernels for better cache utilization and throughput.
///
/// Usage:
///   #include "qdldl.h"
///   #include "supernodes.h"
///
///   using namespace qdldl23;
///   SparseD32 B = /* permuted upper CSC */;
///   Symb32 S = analyze_fast(B);
///   auto sn = identify_supernodes_qdldl(B, S,
///       relax_abs = 2, relax_rel = 0.10, tau = 0.70, max_size = 128);
///
/// The returned SupernodeInfo contains:
///   - ranges: inclusive [lo, hi] column pairs for each supernode
///   - col2sn: column-to-supernode mapping
///   - etree: elimination tree (copy)
///   - post: postorder traversal of etree
///
/// Columns are indexed in the *permuted* space (after P A Pᵀ).
///
/// © 2025 MIT / Apache-2.0

#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "qdldl.h"

namespace snode {

// ===== types =====
using i32 = int32_t;

/// Supernode metadata from symbolic analysis.
/// All indices (ranges, col2sn, etree) are in the *permuted* column space.
template <typename IntT = i32>
struct SupernodeInfo {
    std::vector<std::pair<IntT, IntT>>
        ranges;                // inclusive [lo, hi] for each supernode
    std::vector<IntT> col2sn;  // size n; maps col -> supernode id
    std::vector<IntT> etree;   // elimination tree parent (copy of S.etree)
    std::vector<IntT> post;    // postorder of etree
};

/// Compute postorder traversal of an elimination tree forest (iterative DFS).
/// Handles forests gracefully (trees with -1 roots); visits all roots.
template <typename IntT = i32>
static std::vector<IntT> postorder_etree_safe(const std::vector<IntT>& parent) {
    const IntT n = static_cast<IntT>(parent.size());
    std::vector<IntT> head(n, -1), next(n, -1), roots;
    roots.reserve(n);
    for (IntT v = 0; v < n; ++v) {
        const IntT p = parent[(size_t)v];
        if (p == -1)
            roots.push_back(v);
        else {
            next[(size_t)v] = head[(size_t)p];
            head[(size_t)p] = v;
        }
    }

    std::vector<IntT> post;
    post.reserve(n);
    std::vector<std::pair<IntT, IntT>> st;
    st.reserve(n);  // (node, child-iter index via head/next)
    for (IntT r : roots) {
        st.emplace_back(r, head[(size_t)r]);
        while (!st.empty()) {
            auto& top = st.back();
            IntT& u = top.first;
            IntT& it = top.second;
            if (it == -2) {  // finished: emit and pop
                post.push_back(u);
                st.pop_back();
                if (!st.empty()) {
                    // advance parent's iterator to next sibling
                    auto& par = st.back();
                    if (par.second != -1) par.second = next[(size_t)par.second];
                }
                continue;
            }
            if (it == -1) {  // no children
                it = -2;
                continue;
            }
            // descend first (current) child of u
            st.emplace_back(it, head[(size_t)it]);
        }
    }
    return post;
}

/// Compute symbolic L-factor row patterns from the upper CSC matrix and etree.
/// L-pattern[j] is the sorted set of descendants < j touched by the symbolic
/// reach of B(:,j). These patterns form the structural basis for supernode
/// detection: exact matching identifies fundamental supernodes; relaxed matching
/// merges additional columns with similar (but not identical) row patterns.
template <typename FloatT = double, typename IntT = int32_t>
static std::vector<std::vector<IntT>> symbolic_l_patterns(
    const qdldl23::SparseUpperCSC<FloatT, IntT>& B,
    const qdldl23::Symbolic<IntT>& S) {
    const IntT n = B.n;
    std::vector<std::vector<IntT>> patterns((size_t)n);
    std::vector<IntT> mark((size_t)n, IntT{-1});
    std::vector<IntT> stack;
    stack.reserve((size_t)n);

    for (IntT j = 0; j < n; ++j) {
        stack.clear();
        for (IntT p = B.Ap[(size_t)j]; p < B.Ap[(size_t)j + 1]; ++p) {
            IntT i = B.Ai[(size_t)p];
            if (i < 0 || i >= j) continue;
            while (i != -1 && i < j && mark[(size_t)i] != j) {
                mark[(size_t)i] = j;
                stack.push_back(i);
                i = S.etree[(size_t)i];
            }
        }
        for (IntT c : stack) patterns[(size_t)c].push_back(j);
    }

    for (auto& out : patterns) {
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
    return patterns;
}

/// Test relaxed structural match between two sorted row sets via two-pointer scan.
/// Columns j and k match if:
///   - |S(j) Δ S(k)| ≤ relax_abs, AND
///   - |S(j) Δ S(k)| / |S(j)| ≤ relax_rel (if |S(j)| > 0), AND
///   - Jaccard(S(j), S(k)) ≥ tau
/// Enables flexible supernode merging beyond exact structural match.
template <typename IntT = i32>
static inline bool relaxed_match_twoptr(const IntT* Aj, IntT lenj,
                                        const IntT* Ak, IntT lenk,
                                        IntT relax_abs, double relax_rel,
                                        double tau) {
    // exact fast-path
    if (relax_abs <= 0 && relax_rel <= 0.0) {
        if (lenj != lenk) return false;
        for (IntT t = 0; t < lenj; ++t)
            if (Aj[(size_t)t] != Ak[(size_t)t]) return false;
        return true;
    }

    // two-pointer intersection/union
    IntT inter = 0, a = 0, b = 0;
    while (a < lenj && b < lenk) {
        const IntT va = Aj[(size_t)a], vb = Ak[(size_t)b];
        if (va == vb) {
            ++inter;
            ++a;
            ++b;
        } else if (va < vb) {
            ++a;
        } else {
            ++b;
        }
    }
    const IntT uni = lenj + lenk - inter;
    const IntT symd = lenj + lenk - 2 * inter;

    if (symd > relax_abs) return false;
    if (lenj > 0) {
        const double rel = double(symd) / double(lenj);
        if (rel > relax_rel) return false;
    }
    const double jac = (uni == 0) ? 1.0 : double(inter) / double(uni);
    return jac >= tau;
}

/// Identify supernodes from permuted upper CSC and symbolic analysis.
/// Groups consecutive columns whose L-patterns satisfy relaxed matching criteria.
/// Params:
///   - relax_abs: max symmetric set difference (absolute count)
///   - relax_rel: max symmetric set difference (relative to |S(j)|)
///   - tau: minimum Jaccard similarity required
///   - max_size: cap on supernode width (prevents pathologically large blocks)
/// Returns SupernodeInfo with ranges, col2sn map, etree, and postorder.
template <typename FloatT = double, typename IntT = int32_t>
static SupernodeInfo<IntT> identify_supernodes_qdldl(
    const qdldl23::SparseUpperCSC<FloatT, IntT>& B,
    const qdldl23::Symbolic<IntT>& S, IntT relax_abs = 0,
    double relax_rel = 0.0, double tau = 1.0,
    IntT max_size = std::numeric_limits<IntT>::max()) {
    const IntT n = B.n;
    SupernodeInfo<IntT> out;
    out.col2sn.assign((size_t)n, IntT{-1});
    out.etree = S.etree;  // copy for convenience/visibility
    out.post = postorder_etree_safe<IntT>(out.etree);
    const auto lpat = symbolic_l_patterns<FloatT, IntT>(B, S);

    IntT j = 0, sid = 0;
    while (j < n) {
        IntT t = j;
        // Grow chain j..t while etree[t] == t+1 and symbolic L row sets match
        // (relaxed). This is a fundamental-supernode criterion with optional
        // relaxed amalgamation.
        while (t + 1 < n) {
            if (out.etree[(size_t)t] != t + 1) break;  // must be a chain

            const auto& Sj = lpat[(size_t)t];
            const auto& Sk = lpat[(size_t)t + 1];

            const bool ok =
                relaxed_match_twoptr<IntT>(Sj.data(), (IntT)Sj.size(),
                                           Sk.data(), (IntT)Sk.size(),
                                           relax_abs, relax_rel, tau);
            if (!ok) break;

            const IntT width = (t + 1) - j + 1;
            if (width >= max_size) break;

            ++t;
        }
        out.ranges.emplace_back(j, t);
        for (IntT c = j; c <= t; ++c) out.col2sn[(size_t)c] = sid;
        ++sid;
        j = t + 1;
    }
    return out;
}

/// Convenience overload for qdldl23's double/int32 type aliases.
inline SupernodeInfo<i32> identify_supernodes_qdldl(
    const qdldl23::SparseD32& B, const qdldl23::Symb32& S, i32 relax_abs = 0,
    double relax_rel = 0.0, double tau = 1.0,
    i32 max_size = std::numeric_limits<i32>::max()) {
    return identify_supernodes_qdldl<double, int32_t>(B, S, relax_abs,
                                                      relax_rel, tau, max_size);
}

}  // namespace snode
