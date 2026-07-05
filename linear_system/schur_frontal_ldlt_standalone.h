// schur_frontal_ldlt_standalone.h
// Eigen-free Schur-complement frontal LDL^T factorization.
//
// Core algorithm extracted from schur_frontal_ldlt.h, decoupled from Eigen.
// Uses simple vector-based dense matrices and CSC sparse format.

#pragma once

#ifndef SCHUR_FRONTAL_LDLT_STANDALONE_H
#define SCHUR_FRONTAL_LDLT_STANDALONE_H

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

// Simple dense matrix: column-major storage, wraps std::vector
template <typename Scalar = Real>
struct DenseMatrix {
  std::vector<Scalar> data;
  Int rows = 0;
  Int cols = 0;

  DenseMatrix() = default;
  DenseMatrix(Int m, Int n) : data(static_cast<size_t>(m) * static_cast<size_t>(n)), rows(m), cols(n) {}

  Scalar& operator()(Int i, Int j) {
    return data[static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i)];
  }
  const Scalar& operator()(Int i, Int j) const {
    return data[static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i)];
  }

  void resize(Int m, Int n) {
    rows = m;
    cols = n;
    data.assign(static_cast<size_t>(m) * static_cast<size_t>(n), Scalar(0));
  }
  void setZero() {
    std::fill(data.begin(), data.end(), Scalar(0));
  }
};

// CSC sparse matrix
template <typename Scalar = Real, typename Index = int32_t>
struct SparseCSC {
  std::vector<Index> Ap;
  std::vector<Index> Ai;
  std::vector<Scalar> Ax;
  Index n = 0;

  SparseCSC() = default;
  SparseCSC(Index n_) : n(n_), Ap(static_cast<size_t>(n_) + 1, 0) {}
  size_t nnz() const { return Ai.size(); }
};

struct FrontalNode {
  Int col_start = 0;
  Int col_end = 0;

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

  DenseMatrix<Real> F;
  DenseMatrix<Real> schur;
};

struct FrontalLDLT {
  Int n = 0;
  bool factorized = false;
  Real pivot_tolerance = 1e-14;

  std::vector<FrontalNode> fronts;
  std::vector<Int> elimination_order;
  std::vector<Real> diag;

  SparseCSC<Real, Int> L;
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
  front.row_idx.insert(front.row_idx.end(), front.pivots.begin(), front.pivots.end());
  front.row_idx.insert(front.row_idx.end(), front.updates.begin(), front.updates.end());

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

inline std::vector<Int> make_col2sn(Int n, const std::vector<std::pair<Int, Int>> &supernode_ranges) {
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

inline void add_symmetric_entry(DenseMatrix<Real> &F, Int a, Int b, Real v) {
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

} // namespace schur_frontal

#endif // SCHUR_FRONTAL_LDLT_STANDALONE_H
