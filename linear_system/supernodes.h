#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace snode {

// ===== Minimal CSC interface =====

template <typename IntT = int32_t> struct SparseUpperCSC {
  IntT n;
  const std::vector<IntT> *Ap = nullptr;
  const std::vector<IntT> *Ai = nullptr;
  const std::vector<double> *Ax = nullptr;
};

template <typename IntT = int32_t> struct Symbolic {
  IntT n;
  const std::vector<IntT> *etree = nullptr;
  const std::vector<IntT> *Lnz = nullptr;
  const std::vector<IntT> *Lp = nullptr;
};

// ===== Supernode metadata =====

template <typename IntT = int32_t> struct SupernodeInfo {
  std::vector<std::pair<IntT, IntT>> ranges;
  std::vector<IntT> col2sn;
  std::vector<IntT> etree;
  std::vector<IntT> post;
};

// ===== Postorder traversal of elimination tree forest =====

template <typename IntT = int32_t>
static std::vector<IntT> postorder_etree(const std::vector<IntT> &parent) {
  const IntT n = static_cast<IntT>(parent.size());

  std::vector<IntT> head(static_cast<size_t>(n), IntT{-1});
  std::vector<IntT> next(static_cast<size_t>(n), IntT{-1});
  std::vector<IntT> roots;
  roots.reserve(static_cast<size_t>(n));

  for (IntT v = 0; v < n; ++v) {
    const IntT p = parent[static_cast<size_t>(v)];

    if (p == -1) {
      roots.push_back(v);
    } else if (p >= 0 && p < n) {
      next[static_cast<size_t>(v)] = head[static_cast<size_t>(p)];
      head[static_cast<size_t>(p)] = v;
    }
  }

  std::vector<IntT> post;
  post.reserve(static_cast<size_t>(n));

  std::vector<std::pair<IntT, IntT>> st;
  st.reserve(static_cast<size_t>(n));

  for (IntT r : roots) {
    st.emplace_back(r, head[static_cast<size_t>(r)]);

    while (!st.empty()) {
      auto &top = st.back();
      IntT u = top.first;
      IntT &child = top.second;

      if (child == -2) {
        post.push_back(u);
        st.pop_back();

        if (!st.empty()) {
          IntT &parent_child = st.back().second;
          if (parent_child != -1) {
            parent_child = next[static_cast<size_t>(parent_child)];
          }
        }

        continue;
      }

      if (child == -1) {
        child = -2;
        continue;
      }

      st.emplace_back(child, head[static_cast<size_t>(child)]);
    }
  }

  return post;
}

// ===== Symbolic L-patterns =====

/// Compute symbolic lower L row patterns from upper CSC and etree.
/// patterns[j] = sorted structural row indices i > j such that L(i,j) may be
/// nonzero.
///
/// B is assumed to be upper CSC, so entries in column j have row i <= j.
/// The symbolic reach climbs etree paths from each structural row i < j and
/// inserts column j into every reached L-column pattern.
template <typename IntT = int32_t>
static std::vector<std::vector<IntT>>
symbolic_l_patterns(const SparseUpperCSC<IntT> &B, const Symbolic<IntT> &S) {
  const IntT n = B.n;

  std::vector<std::vector<IntT>> patterns(static_cast<size_t>(n));
  std::vector<IntT> mark(static_cast<size_t>(n), IntT{-1});
  std::vector<IntT> stack;
  stack.reserve(static_cast<size_t>(n));

  for (IntT j = 0; j < n; ++j) {
    stack.clear();

    const IntT col_start = (*B.Ap)[static_cast<size_t>(j)];
    const IntT col_end = (*B.Ap)[static_cast<size_t>(j) + 1];

    for (IntT p = col_start; p < col_end; ++p) {
      IntT i = (*B.Ai)[static_cast<size_t>(p)];

      if (i < 0 || i >= j) {
        continue;
      }

      while (i != -1 && i < j && mark[static_cast<size_t>(i)] != j) {
        mark[static_cast<size_t>(i)] = j;
        stack.push_back(i);
        i = (*S.etree)[static_cast<size_t>(i)];
      }
    }

    for (IntT c : stack) {
      patterns[static_cast<size_t>(c)].push_back(j);
    }
  }

  for (auto &pat : patterns) {
    std::sort(pat.begin(), pat.end());
    pat.erase(std::unique(pat.begin(), pat.end()), pat.end());
  }

  return patterns;
}

// ===== Relaxed structural matching =====

/// Test relaxed structural match between two sorted row sets.
template <typename IntT = int32_t>
static inline bool relaxed_match(const IntT *Aj, IntT lenj, const IntT *Ak,
                                 IntT lenk, IntT relax_abs, double relax_rel,
                                 double tau) {
  IntT inter = 0;
  IntT a = 0;
  IntT b = 0;

  while (a < lenj && b < lenk) {
    const IntT va = Aj[static_cast<size_t>(a)];
    const IntT vb = Ak[static_cast<size_t>(b)];

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

  if (relax_abs >= 0 && symd > relax_abs) {
    return false;
  }

  if (relax_rel >= 0.0) {
    const IntT denom = std::max<IntT>(lenj, 1);
    if (double(symd) / double(denom) > relax_rel) {
      return false;
    }
  }

  const double jac = (uni == 0) ? 1.0 : double(inter) / double(uni);
  return jac >= tau;
}

// ===== Supernode identification =====

template <typename IntT = int32_t>
static SupernodeInfo<IntT>
identify_supernodes(const SparseUpperCSC<IntT> &B, const Symbolic<IntT> &S,
                    IntT relax_abs = 0, double relax_rel = 0.0,
                    double tau = 1.0,
                    IntT max_size = std::numeric_limits<IntT>::max()) {
  const IntT n = B.n;

  SupernodeInfo<IntT> out;
  out.col2sn.assign(static_cast<size_t>(n), IntT{-1});

  out.etree = *S.etree;
  out.post = postorder_etree<IntT>(out.etree);

  const auto lpat = symbolic_l_patterns<IntT>(B, S);

  IntT j = 0;
  IntT sid = 0;

  while (j < n) {
    IntT t = j;

    while (t + 1 < n) {
      if (max_size > 0 && static_cast<IntT>((t + 1) - j + 1) > max_size) {
        break;
      }

      // Check below-block L-pattern match between col t and t+1.
      const auto &Sj_full = lpat[static_cast<size_t>(t)];
      const auto &Sk_full = lpat[static_cast<size_t>(t + 1)];

      auto Sj_it = std::upper_bound(Sj_full.begin(), Sj_full.end(), t + 1);
      auto Sk_it = std::upper_bound(Sk_full.begin(), Sk_full.end(), t + 1);

      const IntT lenj = static_cast<IntT>(Sj_full.end() - Sj_it);
      const IntT lenk = static_cast<IntT>(Sk_full.end() - Sk_it);

      const IntT *Sj_ptr = (lenj > 0) ? &(*Sj_it) : nullptr;
      const IntT *Sk_ptr = (lenk > 0) ? &(*Sk_it) : nullptr;

      if (!relaxed_match<IntT>(Sj_ptr, lenj, Sk_ptr, lenk, relax_abs, relax_rel,
                               tau)) {
        break;
      }

      // For fundamental supernodes: require etree chain (etree[t] = t+1).
      // For non-fundamental: allow merge if etree parents are identical
      // (both columns have the same etree parent, so their Schur complements
      // flow to the same parent supernode).
      const IntT etree_t = (*S.etree)[static_cast<size_t>(t)];
      const IntT etree_t1 = (*S.etree)[static_cast<size_t>(t + 1)];
      bool chain_ok = (etree_t == static_cast<IntT>(t) + 1);
      bool same_parent = (etree_t >= 0 && etree_t == etree_t1);
      if (!chain_ok && !same_parent) {
        break;
      }

      ++t;
    }

    out.ranges.emplace_back(j, t);

    for (IntT c = j; c <= t; ++c) {
      out.col2sn[static_cast<size_t>(c)] = sid;
    }

    ++sid;
    j = t + 1;
  }

  return out;
}

} // namespace snode
