#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace snode {

template <typename IntT>
static void validate_etree(const std::vector<IntT> &parent) {
  const IntT n = static_cast<IntT>(parent.size());
  for (IntT v = 0; v < n; ++v) {
    const IntT p = parent[static_cast<size_t>(v)];
    if (p != IntT{-1} && (p <= v || p >= n)) {
      throw std::invalid_argument(
          "snode: etree parents must be -1 or greater valid column indices");
    }
  }
}

// ===== Minimal CSC interface =====

template <typename IntT = int32_t> struct SparseUpperCSC {
  IntT n;
  const std::vector<IntT> *Ap = nullptr;
  const std::vector<IntT> *Ai = nullptr;
};

template <typename IntT = int32_t> struct Symbolic {
  IntT n;
  const std::vector<IntT> *etree = nullptr;
};

template <typename IntT>
static void validate_symbolic_input(const SparseUpperCSC<IntT> &B,
                                    const Symbolic<IntT> &S) {
  if (B.n < 0 || S.n != B.n)
    throw std::invalid_argument("snode: inconsistent matrix dimensions");
  if (B.Ap == nullptr || B.Ai == nullptr || S.etree == nullptr)
    throw std::invalid_argument("snode: Ap, Ai, and etree are required");
  if (B.Ap->size() != static_cast<size_t>(B.n) + 1 ||
      S.etree->size() != static_cast<size_t>(B.n))
    throw std::invalid_argument("snode: invalid Ap or etree size");
  if (B.Ap->empty() || B.Ap->front() != IntT{0})
    throw std::invalid_argument("snode: Ap must start at zero");

  for (IntT col = 0; col < B.n; ++col) {
    const IntT begin = (*B.Ap)[static_cast<size_t>(col)];
    const IntT end = (*B.Ap)[static_cast<size_t>(col) + 1];
    if (begin < 0 || begin > end ||
        static_cast<size_t>(end) > B.Ai->size())
      throw std::invalid_argument("snode: invalid CSC column pointers");
    for (IntT p = begin; p < end; ++p) {
      const IntT row = (*B.Ai)[static_cast<size_t>(p)];
      if (row < 0 || row > col)
        throw std::invalid_argument("snode: matrix must be upper-triangular CSC");
    }
  }
  if (static_cast<size_t>(B.Ap->back()) != B.Ai->size())
    throw std::invalid_argument("snode: Ap.back() must equal Ai.size()");
  validate_etree(*S.etree);
}

// ===== Supernode metadata =====

template <typename IntT = int32_t> struct SupernodeInfo {
  std::vector<std::pair<IntT, IntT>> ranges;
  std::vector<IntT> col2sn;
  std::vector<IntT> etree;
  std::vector<IntT> post;
  std::vector<IntT> boundaries;
  std::vector<IntT> parent;
  std::vector<IntT> supernode_post;
  std::vector<IntT> row_ptr;
  std::vector<IntT> rows;
  std::vector<IntT> column_counts;
  size_t factor_storage = 0;
  IntT max_front_size = 0;
};

// ===== Postorder traversal of elimination tree forest =====

template <typename IntT>
static std::vector<IntT>
postorder_etree_unchecked(const std::vector<IntT> &parent) {
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

template <typename IntT = int32_t>
static std::vector<IntT> postorder_etree(const std::vector<IntT> &parent) {
  validate_etree(parent);
  return postorder_etree_unchecked(parent);
}

// ===== Compact symbolic analysis =====

/// Count structural entries in each column of L without materializing the
/// individual column patterns. The diagonal is included in every count.
template <typename IntT = int32_t>
static std::vector<IntT>
symbolic_column_counts(const SparseUpperCSC<IntT> &B,
                       const Symbolic<IntT> &S) {
  validate_symbolic_input(B, S);
  const IntT n = B.n;
  std::vector<IntT> counts(static_cast<size_t>(n), IntT{1});
  std::vector<IntT> mark(static_cast<size_t>(n), IntT{-1});

  for (IntT j = 0; j < n; ++j) {
    const IntT col_start = (*B.Ap)[static_cast<size_t>(j)];
    const IntT col_end = (*B.Ap)[static_cast<size_t>(j) + 1];
    for (IntT p = col_start; p < col_end; ++p) {
      IntT i = (*B.Ai)[static_cast<size_t>(p)];
      if (i >= j)
        continue;
      while (i != -1 && i < j && mark[static_cast<size_t>(i)] != j) {
        mark[static_cast<size_t>(i)] = j;
        ++counts[static_cast<size_t>(i)];
        i = (*S.etree)[static_cast<size_t>(i)];
      }
    }
  }
  return counts;
}

template <typename IntT>
static std::vector<std::vector<IntT>>
build_supernode_rows(const SparseUpperCSC<IntT> &B,
                     const std::vector<IntT> &etree,
                     const std::vector<std::pair<IntT, IntT>> &ranges,
                     const std::vector<IntT> &col2sn) {
  const IntT n = B.n;
  const IntT nsuper = static_cast<IntT>(ranges.size());
  std::vector<std::vector<IntT>> patterns(static_cast<size_t>(nsuper));
  for (IntT s = 0; s < nsuper; ++s) {
    const auto [lo, hi] = ranges[static_cast<size_t>(s)];
    auto &pattern = patterns[static_cast<size_t>(s)];
    pattern.reserve(static_cast<size_t>(hi - lo + 1));
    for (IntT col = lo; col <= hi; ++col)
      pattern.push_back(col);
  }

  std::vector<IntT> column_mark(static_cast<size_t>(n), IntT{-1});
  std::vector<IntT> supernode_mark(static_cast<size_t>(nsuper), IntT{-1});
  for (IntT row = 0; row < n; ++row) {
    const IntT begin = (*B.Ap)[static_cast<size_t>(row)];
    const IntT end = (*B.Ap)[static_cast<size_t>(row) + 1];
    for (IntT p = begin; p < end; ++p) {
      IntT col = (*B.Ai)[static_cast<size_t>(p)];
      if (col >= row)
        continue;
      while (col != IntT{-1} && col < row &&
             column_mark[static_cast<size_t>(col)] != row) {
        column_mark[static_cast<size_t>(col)] = row;
        const IntT s = col2sn[static_cast<size_t>(col)];
        if (row > ranges[static_cast<size_t>(s)].second &&
            supernode_mark[static_cast<size_t>(s)] != row) {
          supernode_mark[static_cast<size_t>(s)] = row;
          patterns[static_cast<size_t>(s)].push_back(row);
        }
        col = etree[static_cast<size_t>(col)];
      }
    }
  }
  return patterns;
}

// ===== Supernode identification =====

template <typename IntT = int32_t>
static SupernodeInfo<IntT>
identify_supernodes(const SparseUpperCSC<IntT> &B, const Symbolic<IntT> &S,
                    IntT relax_abs = 0, double relax_rel = 0.0,
                    double tau = 1.0,
                    IntT max_size = std::numeric_limits<IntT>::max(),
                    const std::vector<std::pair<IntT, double>>
                        *size_dependent_relaxation = nullptr) {
  const IntT n = B.n;
  if (relax_rel < 0.0 && relax_abs < 0 &&
      size_dependent_relaxation == nullptr)
    throw std::invalid_argument("snode: at least one relaxation limit is required");
  if (tau < 0.0 || tau > 1.0)
    throw std::invalid_argument("snode: tau must be in [0, 1]");

  SupernodeInfo<IntT> out;
  out.column_counts = symbolic_column_counts<IntT>(B, S);
  out.etree = *S.etree;
  out.post = postorder_etree_unchecked(out.etree);
  if (n == 0) {
    out.boundaries.push_back(0);
    out.row_ptr.push_back(0);
    return out;
  }

  // Fundamental supernodes from etree topology and exact L column counts.
  std::vector<IntT> child_count(static_cast<size_t>(n), IntT{0});
  for (IntT col = 0; col < n; ++col) {
    const IntT p = out.etree[static_cast<size_t>(col)];
    if (p != IntT{-1})
      ++child_count[static_cast<size_t>(p)];
  }

  std::vector<IntT> fundamental{IntT{0}};
  fundamental.reserve(static_cast<size_t>(n) + 1);
  for (IntT col = 1; col < n; ++col) {
    if (out.etree[static_cast<size_t>(col - 1)] != col ||
        out.column_counts[static_cast<size_t>(col - 1)] !=
            out.column_counts[static_cast<size_t>(col)] + 1 ||
        child_count[static_cast<size_t>(col)] > 1) {
      fundamental.push_back(col);
    }
  }
  fundamental.push_back(n);

  const IntT nf = static_cast<IntT>(fundamental.size() - 1);
  std::vector<IntT> column_to_fundamental(static_cast<size_t>(n), IntT{-1});
  for (IntT s = 0; s < nf; ++s)
    for (IntT col = fundamental[static_cast<size_t>(s)];
         col < fundamental[static_cast<size_t>(s + 1)]; ++col)
      column_to_fundamental[static_cast<size_t>(col)] = s;

  std::vector<IntT> fs_parent(static_cast<size_t>(nf), IntT{-1});
  std::vector<IntT> merged(static_cast<size_t>(nf), IntT{-1});
  std::vector<IntT> widths(static_cast<size_t>(nf));
  std::vector<IntT> leading_nnz(static_cast<size_t>(nf));
  std::vector<double> explicit_zeros(static_cast<size_t>(nf), 0.0);
  for (IntT s = 0; s < nf; ++s) {
    const IntT lo = fundamental[static_cast<size_t>(s)];
    const IntT hi = fundamental[static_cast<size_t>(s + 1)] - 1;
    widths[static_cast<size_t>(s)] = hi - lo + 1;
    leading_nnz[static_cast<size_t>(s)] =
        out.column_counts[static_cast<size_t>(lo)];
    const IntT parent_col = out.etree[static_cast<size_t>(hi)];
    if (parent_col != IntT{-1})
      fs_parent[static_cast<size_t>(s)] =
          column_to_fundamental[static_cast<size_t>(parent_col)];
  }

  auto live_representative = [&merged](IntT s) {
    while (s != IntT{-1} && merged[static_cast<size_t>(s)] != IntT{-1})
      s = merged[static_cast<size_t>(s)];
    return s;
  };

  // Relax only along supernodal-tree parent edges. The merge criteria measure
  // the actual explicit zeros introduced in the dense frontal storage.
  for (IntT s = nf - 2; s >= 0; --s) {
    const IntT parent =
        live_representative(fs_parent[static_cast<size_t>(s)]);
    if (parent != s + 1)
      continue;

    const IntT left_width = widths[static_cast<size_t>(s)];
    const IntT parent_width = widths[static_cast<size_t>(parent)];
    const IntT merged_width = left_width + parent_width;
    if (max_size > 0 && merged_width > max_size)
      continue;

    const double new_zeros = std::max(
        0.0, static_cast<double>(left_width) *
                 static_cast<double>(leading_nnz[static_cast<size_t>(parent)] +
                                     left_width -
                                     leading_nnz[static_cast<size_t>(s)]));
    const double total_zeros =
        explicit_zeros[static_cast<size_t>(parent)] + new_zeros;
    const double storage =
        static_cast<double>(merged_width) * (merged_width + 1) / 2.0 +
        static_cast<double>(merged_width) *
            (leading_nnz[static_cast<size_t>(parent)] - parent_width);
    const double zero_fraction = storage == 0.0 ? 0.0 : total_zeros / storage;
    const bool within_absolute =
        relax_abs < 0 || new_zeros <= static_cast<double>(relax_abs);
    const bool within_relative =
        relax_rel < 0.0 || zero_fraction <= relax_rel;
    const bool dense_enough = 1.0 - zero_fraction >= tau;
    bool within_size_dependent_relaxation = false;
    if (size_dependent_relaxation != nullptr) {
      for (const auto &[column_limit, zero_limit] :
           *size_dependent_relaxation) {
        if (merged_width <= column_limit && zero_fraction <= zero_limit) {
          within_size_dependent_relaxation = true;
          break;
        }
      }
    }

    if (new_zeros == 0.0 ||
        (size_dependent_relaxation != nullptr
             ? within_size_dependent_relaxation
             : (within_absolute && within_relative && dense_enough))) {
      merged[static_cast<size_t>(parent)] = s;
      explicit_zeros[static_cast<size_t>(s)] = total_zeros;
      leading_nnz[static_cast<size_t>(s)] =
          left_width + leading_nnz[static_cast<size_t>(parent)];
      widths[static_cast<size_t>(s)] = merged_width;
    }
  }

  for (IntT s = 0; s < nf; ++s)
    if (merged[static_cast<size_t>(s)] == IntT{-1})
      out.boundaries.push_back(fundamental[static_cast<size_t>(s)]);
  out.boundaries.push_back(n);

  const IntT nsuper = static_cast<IntT>(out.boundaries.size() - 1);
  out.col2sn.assign(static_cast<size_t>(n), IntT{-1});
  out.ranges.reserve(static_cast<size_t>(nsuper));
  for (IntT s = 0; s < nsuper; ++s) {
    const IntT lo = out.boundaries[static_cast<size_t>(s)];
    const IntT hi = out.boundaries[static_cast<size_t>(s + 1)] - 1;
    out.ranges.emplace_back(lo, hi);
    for (IntT col = lo; col <= hi; ++col)
      out.col2sn[static_cast<size_t>(col)] = s;
  }

  out.parent.assign(static_cast<size_t>(nsuper), IntT{-1});
  for (IntT s = 0; s < nsuper; ++s) {
    const IntT hi = out.ranges[static_cast<size_t>(s)].second;
    const IntT parent_col = out.etree[static_cast<size_t>(hi)];
    if (parent_col != IntT{-1})
      out.parent[static_cast<size_t>(s)] =
          out.col2sn[static_cast<size_t>(parent_col)];
  }
  out.supernode_post = postorder_etree_unchecked(out.parent);

  const auto patterns =
      build_supernode_rows(B, out.etree, out.ranges, out.col2sn);
  out.row_ptr.reserve(static_cast<size_t>(nsuper) + 1);
  out.row_ptr.push_back(0);
  for (IntT s = 0; s < nsuper; ++s) {
    const auto &pattern = patterns[static_cast<size_t>(s)];
    out.rows.insert(out.rows.end(), pattern.begin(), pattern.end());
    out.row_ptr.push_back(static_cast<IntT>(out.rows.size()));
    const IntT width = out.ranges[static_cast<size_t>(s)].second -
                       out.ranges[static_cast<size_t>(s)].first + 1;
    out.factor_storage +=
        static_cast<size_t>(width) * static_cast<size_t>(pattern.size());
    out.max_front_size =
        std::max(out.max_front_size, static_cast<IntT>(pattern.size()));
  }

  return out;
}

} // namespace snode
