// schur_frontal_ldlt_fixed.h
// Self-contained Schur-complement frontal LDL^T factorization prototype.
//
// Design goals:
//   - Correct and consistent frontal layout: [pivots | update rows].
//   - Children contribute only their update/update Schur complements.
//   - Each front eliminates its pivot block with a dense LDL^T kernel.
//   - The numeric factor is stored as global unit-lower sparse L plus diagonal
//   D,
//     making solves simple and reliable: A ≈ L D L^T.
//
// Notes:
//   - This is an unpivoted LDL^T implementation. It is appropriate for SPD or
//     sufficiently well-behaved symmetric/quasi-definite matrices. General
//     indefinite matrices require 1x1/2x2 pivoting and delayed pivots.
//   - The input matrix is assumed symmetric. It may contain lower, upper, or
//   both
//     triangles; values are symmetrized during frontal assembly.
//   - Supernode ranges are inclusive pairs [first,last].

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace schur_frontal {

using Int = int32_t;
using Real = double;

struct FrontalNode {
  // Inclusive/exclusive pivot interval: [col_start, col_end)
  Int col_start = 0;
  Int col_end = 0;

  // Local ordering is always:
  //   row_idx[0:npiv]      = pivot columns
  //   row_idx[npiv:fsize] = update rows
  std::vector<Int> pivots;
  std::vector<Int> updates;
  std::vector<Int> row_idx;
  std::unordered_map<Int, Int> local;

  Int npiv = 0;
  Int nupd = 0;
  Int fsize = 0;

  Int parent = -1;
  std::vector<Int> children;
  bool is_root = false;

  // Dense front, column-major, size fsize x fsize.
  // After factorization, the first npiv columns contain local L entries:
  //   F(i,k) = L(row_idx[i], pivots[k]) for i > k.
  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F;

  // Delayed contribution to parent: trailing update/update Schur complement.
  // Column-major, size nupd x nupd.
  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> schur;
};

struct FrontalLDLT {
  Int n = 0;
  bool factorized = false;
  Real pivot_tolerance = 1e-14;

  std::vector<FrontalNode> fronts;
  std::vector<Int> elimination_order;
  std::vector<Real> diag;

  // Explicit global factor. Unit diagonal is implicit.
  Eigen::SparseMatrix<Real, Eigen::ColMajor, Int> L;
};

namespace detail {

inline bool contains_sorted(const std::vector<Int> &xs, Int v) {
  return std::binary_search(xs.begin(), xs.end(), v);
}

inline void sort_unique(std::vector<Int> &xs) {
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
}

inline void build_local_map(FrontalNode &front) {
  front.row_idx.clear();
  front.row_idx.reserve(front.pivots.size() + front.updates.size());
  front.row_idx.insert(front.row_idx.end(), front.pivots.begin(),
                       front.pivots.end());
  front.row_idx.insert(front.row_idx.end(), front.updates.begin(),
                       front.updates.end());

  front.npiv = static_cast<Int>(front.pivots.size());
  front.nupd = static_cast<Int>(front.updates.size());
  front.fsize = static_cast<Int>(front.row_idx.size());

  front.local.clear();
  front.local.reserve(static_cast<size_t>(front.fsize) * 2u + 1u);
  for (Int i = 0; i < front.fsize; ++i) {
    front.local.emplace(front.row_idx[static_cast<size_t>(i)], i);
  }
}

inline Int find_front_containing_column(
    const std::vector<std::pair<Int, Int>> &supernode_ranges, Int col) {
  for (Int f = 0; f < static_cast<Int>(supernode_ranges.size()); ++f) {
    if (supernode_ranges[static_cast<size_t>(f)].first <= col &&
        col <= supernode_ranges[static_cast<size_t>(f)].second) {
      return f;
    }
  }
  return -1;
}

inline std::vector<Int>
make_col2sn(Int n, const std::vector<std::pair<Int, Int>> &supernode_ranges) {
  std::vector<Int> col2sn(static_cast<size_t>(n), -1);
  for (Int f = 0; f < static_cast<Int>(supernode_ranges.size()); ++f) {
    const Int a = supernode_ranges[static_cast<size_t>(f)].first;
    const Int b = supernode_ranges[static_cast<size_t>(f)].second;
    if (a < 0 || b < a || b >= n) {
      throw std::invalid_argument("invalid supernode range");
    }
    for (Int j = a; j <= b; ++j) {
      if (col2sn[static_cast<size_t>(j)] != -1) {
        throw std::invalid_argument("overlapping supernode ranges");
      }
      col2sn[static_cast<size_t>(j)] = f;
    }
  }
  for (Int j = 0; j < n; ++j) {
    if (col2sn[static_cast<size_t>(j)] == -1) {
      throw std::invalid_argument("supernode ranges do not cover all columns");
    }
  }
  return col2sn;
}

inline void add_symmetric_entry(
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &F,
    Int a, Int b, Real v) {
  F(a, b) += v;
  if (a != b)
    F(b, a) += v;
}

} // namespace detail

inline std::vector<std::pair<Int, Int>> singleton_supernodes(Int n) {
  std::vector<std::pair<Int, Int>> ranges;
  ranges.reserve(static_cast<size_t>(n));
  for (Int j = 0; j < n; ++j)
    ranges.emplace_back(j, j);
  return ranges;
}

// Build and factor using explicit supernodes, col2sn, and etree.
// supernode_ranges are inclusive: [first,last].
template <typename MatrixType>
FrontalLDLT build_and_factor_frontal(
    const MatrixType &A,
    const std::vector<std::pair<Int, Int>> &supernode_ranges,
    const std::vector<Int> &col2sn_in, const std::vector<Int> &etree,
    Real pivot_tolerance = 1e-14) {

  if (A.rows() != A.cols()) {
    throw std::invalid_argument("A must be square");
  }

  const Int n = static_cast<Int>(A.rows());
  if (static_cast<Int>(etree.size()) != n) {
    throw std::invalid_argument("etree size must equal A.rows()");
  }

  std::vector<Int> col2sn = col2sn_in;
  if (static_cast<Int>(col2sn.size()) != n) {
    col2sn = detail::make_col2sn(n, supernode_ranges);
  }

  FrontalLDLT ldlt;
  ldlt.n = n;
  ldlt.pivot_tolerance = pivot_tolerance;
  ldlt.elimination_order.resize(static_cast<size_t>(n));
  std::iota(ldlt.elimination_order.begin(), ldlt.elimination_order.end(), 0);
  ldlt.diag.assign(static_cast<size_t>(n), 0.0);

  const Int nfronts = static_cast<Int>(supernode_ranges.size());
  if (nfronts == 0 && n > 0) {
    throw std::invalid_argument("empty supernode list");
  }

  ldlt.fronts.resize(static_cast<size_t>(nfronts));

  // Initialize pivot sets and parent links.
  for (Int f = 0; f < nfronts; ++f) {
    auto &front = ldlt.fronts[static_cast<size_t>(f)];
    const Int first = supernode_ranges[static_cast<size_t>(f)].first;
    const Int last = supernode_ranges[static_cast<size_t>(f)].second;

    front.col_start = first;
    front.col_end = last + 1;
    front.pivots.reserve(static_cast<size_t>(front.col_end - front.col_start));
    for (Int j = front.col_start; j < front.col_end; ++j) {
      front.pivots.push_back(j);
    }

    const Int ep = etree[static_cast<size_t>(last)];
    front.parent = (ep < 0) ? -1 : col2sn[static_cast<size_t>(ep)];
    if (front.parent == f)
      front.parent = -1;
  }

  // Build children lists.
  for (Int f = 0; f < nfronts; ++f) {
    auto &front = ldlt.fronts[static_cast<size_t>(f)];
    if (front.parent >= 0) {
      ldlt.fronts[static_cast<size_t>(front.parent)].children.push_back(f);
    } else {
      front.is_root = true;
    }
  }

  std::vector<Eigen::Triplet<Real, Int>> L_triplets;
  L_triplets.reserve(static_cast<size_t>(std::max<Int>(n, 1)) * 8u);

  std::vector<char> done(static_cast<size_t>(nfronts), 0);

  std::function<void(Int)> factor_front = [&](Int f) {
    auto &front = ldlt.fronts[static_cast<size_t>(f)];

    // Postorder: children first.
    for (Int c : front.children) {
      if (!done[static_cast<size_t>(c)])
        factor_front(c);
    }

    // Determine update rows from original matrix entries touching this front's
    // pivots. This works whether A stores the lower triangle, upper triangle,
    // or both.
    std::vector<Int> updates;
    for (Int j = 0; j < n; ++j) {
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        const Int i = static_cast<Int>(it.row());
        const bool j_piv = (front.col_start <= j && j < front.col_end);
        const bool i_piv = (front.col_start <= i && i < front.col_end);

        if (j_piv && i >= front.col_end)
          updates.push_back(i);
        if (i_piv && j >= front.col_end)
          updates.push_back(j);
      }
    }

    // Add only child update rows. Some of them may be pivot rows of this front;
    // those must not be placed in updates because pivots already come first.
    for (Int c : front.children) {
      const auto &child = ldlt.fronts[static_cast<size_t>(c)];
      for (Int r : child.updates) {
        if (r >= front.col_end)
          updates.push_back(r);
      }
    }

    detail::sort_unique(updates);
    front.updates.clear();
    front.updates.reserve(updates.size());
    for (Int r : updates) {
      if (r >= front.col_end)
        front.updates.push_back(r);
    }

    detail::build_local_map(front);
    front.F.setZero(front.fsize, front.fsize);

    // Assemble original A entries whose at least one endpoint is a pivot in
    // this front. Entries are canonicalized so storing both triangles does
    // not double off-diagonal coefficients.
    struct Accum {
      Real sum = 0.0;
      Int count = 0;
    };
    std::unordered_map<std::uint64_t, Accum> acc;
    acc.reserve(static_cast<size_t>(front.fsize) *
                static_cast<size_t>(front.fsize));

    auto key_pair = [](Int a, Int b) -> std::uint64_t {
      const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
      const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
      return (static_cast<std::uint64_t>(hi) << 32) | lo;
    };

    for (Int j = 0; j < n; ++j) {
      for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
        const Int i = static_cast<Int>(it.row());
        auto li_it = front.local.find(i);
        auto lj_it = front.local.find(j);
        if (li_it == front.local.end() || lj_it == front.local.end())
          continue;

        const bool i_piv = (front.col_start <= i && i < front.col_end);
        const bool j_piv = (front.col_start <= j && j < front.col_end);
        if (!i_piv && !j_piv)
          continue;

        const Int li = li_it->second;
        const Int lj = lj_it->second;
        auto &slot = acc[key_pair(li, lj)];
        slot.sum += static_cast<Real>(it.value());
        slot.count += 1;
      }
    }

    for (const auto &kv : acc) {
      const std::uint64_t key = kv.first;
      const Int lo = static_cast<Int>(key & 0xffffffffu);
      const Int hi = static_cast<Int>(key >> 32);
      const Real v = kv.second.sum / static_cast<Real>(kv.second.count);
      detail::add_symmetric_entry(front.F, hi, lo, v);
    }

    // Assemble children Schur complements into this front.
    for (Int c : front.children) {
      const auto &child = ldlt.fronts[static_cast<size_t>(c)];
      for (Int a = 0; a < child.nupd; ++a) {
        const Int ga = child.updates[static_cast<size_t>(a)];
        auto la_it = front.local.find(ga);
        if (la_it == front.local.end())
          continue;

        for (Int b = 0; b < child.nupd; ++b) {
          const Int gb = child.updates[static_cast<size_t>(b)];
          auto lb_it = front.local.find(gb);
          if (lb_it == front.local.end())
            continue;
          front.F(la_it->second, lb_it->second) += child.schur(a, b);
        }
      }
    }

    // Dense unpivoted LDL^T elimination of the pivot columns only.
    // After each pivot k:
    //   d_k = F(k,k)
    //   L(i,k) = F(i,k) / d_k
    //   F(i,j) -= L(i,k) d_k L(j,k), i,j > k
    for (Int k = 0; k < front.npiv; ++k) {
      const Int gk = front.pivots[static_cast<size_t>(k)];
      Real d = front.F(k, k);

      if (std::abs(d) < pivot_tolerance) {
        d = (d < 0.0 ? -pivot_tolerance : pivot_tolerance);
      }

      ldlt.diag[static_cast<size_t>(gk)] = d;

      for (Int i = k + 1; i < front.fsize; ++i) {
        front.F(i, k) /= d;
      }

      for (Int j = k + 1; j < front.fsize; ++j) {
        const Real ljk = front.F(j, k);
        if (ljk == 0.0)
          continue;
        for (Int i = j; i < front.fsize; ++i) {
          front.F(i, j) -= front.F(i, k) * d * ljk;
        }
      }

      // Keep the explicitly accessed upper/lower triangle symmetric for
      // clarity.
      for (Int j = k + 1; j < front.fsize; ++j) {
        for (Int i = j + 1; i < front.fsize; ++i) {
          front.F(j, i) = front.F(i, j);
        }
      }

      // Export local L entries into global sparse L.
      for (Int i = k + 1; i < front.fsize; ++i) {
        const Int gi = front.row_idx[static_cast<size_t>(i)];
        const Real lij = front.F(i, k);
        if (lij != 0.0) {
          L_triplets.emplace_back(gi, gk, lij);
        }
      }
    }

    // Extract update/update Schur complement for parent.
    front.schur.setZero(front.nupd, front.nupd);
    for (Int j = 0; j < front.nupd; ++j) {
      for (Int i = 0; i < front.nupd; ++i) {
        front.schur(i, j) = front.F(front.npiv + i, front.npiv + j);
      }
    }

    done[static_cast<size_t>(f)] = 1;
  };

  for (Int f = 0; f < nfronts; ++f) {
    if (ldlt.fronts[static_cast<size_t>(f)].is_root &&
        !done[static_cast<size_t>(f)]) {
      factor_front(f);
    }
  }

  // In case the supplied tree was disconnected or malformed but fronts exist,
  // factor any remaining components rather than silently ignoring them.
  for (Int f = 0; f < nfronts; ++f) {
    if (!done[static_cast<size_t>(f)])
      factor_front(f);
  }

  ldlt.L.resize(n, n);
  ldlt.L.setFromTriplets(L_triplets.begin(), L_triplets.end(),
                         [](Real a, Real b) { return a + b; });
  ldlt.L.makeCompressed();

  ldlt.factorized = true;
  return ldlt;
}

// Convenience overload: compute col2sn from ranges.
template <typename MatrixType>
FrontalLDLT build_and_factor_frontal(
    const MatrixType &A,
    const std::vector<std::pair<Int, Int>> &supernode_ranges,
    const std::vector<Int> &etree, Real pivot_tolerance = 1e-14) {
  const Int n = static_cast<Int>(A.rows());
  auto col2sn = detail::make_col2sn(n, supernode_ranges);
  return build_and_factor_frontal(A, supernode_ranges, col2sn, etree,
                                  pivot_tolerance);
}

// Convenience overload: singleton fronts, no fill-reducing ordering.
// etree[j] should be the parent column of j or -1.
template <typename MatrixType>
FrontalLDLT build_and_factor_frontal(const MatrixType &A,
                                     const std::vector<Int> &etree,
                                     Real pivot_tolerance = 1e-14) {
  const Int n = static_cast<Int>(A.rows());
  auto ranges = singleton_supernodes(n);
  auto col2sn = detail::make_col2sn(n, ranges);
  return build_and_factor_frontal(A, ranges, col2sn, etree, pivot_tolerance);
}

inline std::vector<Real> solve(const FrontalLDLT &ldlt,
                               const std::vector<Real> &b) {
  if (!ldlt.factorized) {
    throw std::runtime_error("factorization handle is not factorized");
  }
  if (static_cast<Int>(b.size()) != ldlt.n) {
    throw std::invalid_argument("rhs size mismatch");
  }

  std::vector<Real> y = b;

  // Forward solve: L y = b, unit diagonal.
  for (Int col = 0; col < ldlt.n; ++col) {
    const Real ycol = y[static_cast<size_t>(col)];
    for (Eigen::SparseMatrix<Real, Eigen::ColMajor, Int>::InnerIterator it(
             ldlt.L, col);
         it; ++it) {
      const Int row = static_cast<Int>(it.row());
      y[static_cast<size_t>(row)] -= it.value() * ycol;
    }
  }

  // Diagonal solve: D z = y.
  std::vector<Real> z(static_cast<size_t>(ldlt.n), 0.0);
  for (Int i = 0; i < ldlt.n; ++i) {
    z[static_cast<size_t>(i)] =
        y[static_cast<size_t>(i)] / ldlt.diag[static_cast<size_t>(i)];
  }

  // Backward solve: L^T x = z.
  std::vector<Real> x = z;
  for (Int col = ldlt.n - 1; col >= 0; --col) {
    Real acc = x[static_cast<size_t>(col)];
    for (Eigen::SparseMatrix<Real, Eigen::ColMajor, Int>::InnerIterator it(
             ldlt.L, col);
         it; ++it) {
      const Int row = static_cast<Int>(it.row());
      acc -= it.value() * x[static_cast<size_t>(row)];
    }
    x[static_cast<size_t>(col)] = acc;
    if (col == 0)
      break;
  }

  return x;
}

inline Eigen::VectorXd solve(const FrontalLDLT &ldlt,
                             const Eigen::VectorXd &b) {
  if (b.size() != ldlt.n) {
    throw std::invalid_argument("rhs size mismatch");
  }
  std::vector<Real> bv(static_cast<size_t>(ldlt.n));
  for (Int i = 0; i < ldlt.n; ++i)
    bv[static_cast<size_t>(i)] = b[i];
  auto xv = solve(ldlt, bv);
  Eigen::VectorXd x(ldlt.n);
  for (Int i = 0; i < ldlt.n; ++i)
    x[i] = xv[static_cast<size_t>(i)];
  return x;
}

inline Eigen::SparseMatrix<Real, Eigen::ColMajor, Int>
reconstruct_factorized_matrix(const FrontalLDLT &ldlt) {
  if (!ldlt.factorized) {
    throw std::runtime_error("factorization handle is not factorized");
  }

  Eigen::SparseMatrix<Real, Eigen::ColMajor, Int> I(ldlt.n, ldlt.n);
  I.setIdentity();

  Eigen::SparseMatrix<Real, Eigen::ColMajor, Int> Lunit = I + ldlt.L;

  std::vector<Eigen::Triplet<Real, Int>> dtrip;
  dtrip.reserve(static_cast<size_t>(ldlt.n));
  for (Int i = 0; i < ldlt.n; ++i) {
    dtrip.emplace_back(i, i, ldlt.diag[static_cast<size_t>(i)]);
  }
  Eigen::SparseMatrix<Real, Eigen::ColMajor, Int> D(ldlt.n, ldlt.n);
  D.setFromTriplets(dtrip.begin(), dtrip.end());

  return Lunit * D * Lunit.transpose();
}

} // namespace schur_frontal
