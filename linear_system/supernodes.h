// supernodes.h
// SoTA-leaning supernode identification using qdldl23's *permuted upper CSC*
// and its symbolic analysis. Works directly on B = P A Pᵀ (upper+diag).
//
// Drop-in usage:
//   #include "qdldl.h"
//   #include "supernodes.h"
//
//   using namespace qdldl23;
//   SparseD32 B = /* your permuted upper CSC */;
//   Symb32    S = analyze_fast(B);        // etree + column counts
//   auto sn = identify_supernodes_qdldl(B, S,
//                                       /*relax_abs*/ 2,
//                                       /*relax_rel*/ 0.10,
//                                       /*tau*/       0.70,
//                                       /*max_size*/  128);
//
// Returns supernode ranges in *permuted* column space, plus etree and
// postorder.
//
// © 2025 MIT/Apache-2.0

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

// Supernode metadata (columns are in *permuted* space)
template <typename IntT = i32>
struct SupernodeInfo {
    std::vector<std::pair<IntT, IntT>>
        ranges;                // inclusive [lo, hi] for each supernode
    std::vector<IntT> col2sn;  // size n; maps col -> supernode id
    std::vector<IntT> etree;   // elimination tree parent (copy of S.etree)
    std::vector<IntT> post;    // postorder of etree
};

// ----- robust postorder over an etree forest (iterative DFS) -----
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

// ----- build symbolic lower-factor row patterns from B and etree -----
// L-pattern column j is the sorted set of descendants touched by the symbolic
// reach of B(:,j), with rows > j. This is the structural object used for
// supernode detection; comparing raw B columns misses fill and is too weak for
// modern sparse factorization.
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

// ----- relaxed structural match on sorted row sets (two-pointer) -----
// Compare factor-row sets S(j) and S(k). Accept if:
//   - symmetric difference <= relax_abs
//   - and symdiff / |S(j)| <= relax_rel      (if |S(j)| > 0)
//   - and Jaccard >= tau
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

// ----- main: identify supernodes from qdldl's permuted upper CSC & symbolics
// ----- Works for any FloatT/IntT supported by qdldl23.
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

// ----- convenience overload for qdldl double/int32 aliases -----
inline SupernodeInfo<i32> identify_supernodes_qdldl(
    const qdldl23::SparseD32& B, const qdldl23::Symb32& S, i32 relax_abs = 0,
    double relax_rel = 0.0, double tau = 1.0,
    i32 max_size = std::numeric_limits<i32>::max()) {
    return identify_supernodes_qdldl<double, int32_t>(B, S, relax_abs,
                                                      relax_rel, tau, max_size);
}

}  // namespace snode
