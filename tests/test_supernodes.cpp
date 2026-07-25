#include "linear_system/ldlt/supernodal_ldlt.h"
#include "linear_system/eigen_interop/supernodal_eigen_interop.h"
#include "linear_system/supernodes.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Range = std::pair<int, int>;
using Ranges = std::vector<Range>;
using SpMat = Eigen::SparseMatrix<double>;

template <typename T>
std::string vectorToString(const std::vector<T> &values) {
  std::ostringstream out;
  out << "{";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << values[i];
  }
  out << "}";
  return out.str();
}

std::string rangesToString(const Ranges &ranges) {
  std::ostringstream out;
  out << "{";
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (i != 0) out << ", ";
    out << "[" << ranges[i].first << ", " << ranges[i].second << "]";
  }
  out << "}";
  return out.str();
}

void fail(const std::string &testName, const std::string &message) {
  std::cerr << "FAIL " << testName << ": " << message << "\n";
  std::exit(EXIT_FAILURE);
}

template <typename T>
void expectEqual(const std::string &testName, const std::vector<T> &actual,
                 const std::vector<T> &expected, const std::string &label) {
  if (actual != expected) {
    fail(testName, label + " expected " + vectorToString(expected) + ", got " +
                       vectorToString(actual));
  }
}

void expectRanges(const std::string &testName, const Ranges &actual,
                  const Ranges &expected) {
  if (actual != expected) {
    fail(testName, "ranges expected " + rangesToString(expected) + ", got " +
                       rangesToString(actual));
  }
}

std::vector<int> denseBlockEtree(int n, const Ranges &blocks) {
  std::vector<int> etree(static_cast<size_t>(n), -1);
  for (const auto &[lo, hi] : blocks) {
    for (int col = lo; col < hi; ++col) {
      etree[static_cast<size_t>(col)] = col + 1;
    }
  }
  return etree;
}

void addDenseBlockPattern(std::vector<std::set<int>> &columns, int lo,
                          int hi) {
  for (int col = lo; col <= hi; ++col) {
    for (int row = lo; row <= col; ++row) {
      columns[static_cast<size_t>(col)].insert(row);
    }
  }
}

void makeUpperCsc(int n, const Ranges &denseBlocks, std::vector<int> &ap,
                  std::vector<int> &ai) {
  std::vector<std::set<int>> columns(static_cast<size_t>(n));
  for (int col = 0; col < n; ++col) {
    columns[static_cast<size_t>(col)].insert(col);
  }

  for (const auto &[lo, hi] : denseBlocks) {
    addDenseBlockPattern(columns, lo, hi);
  }

  ap.assign(static_cast<size_t>(n) + 1, 0);
  ai.clear();
  for (int col = 0; col < n; ++col) {
    ap[static_cast<size_t>(col)] = static_cast<int>(ai.size());
    ai.insert(ai.end(), columns[static_cast<size_t>(col)].begin(),
              columns[static_cast<size_t>(col)].end());
  }
  ap[static_cast<size_t>(n)] = static_cast<int>(ai.size());
}

void expectPostorderValid(const std::string &testName,
                          const std::vector<int> &etree,
                          const std::vector<int> &post) {
  const int n = static_cast<int>(etree.size());
  if (static_cast<int>(post.size()) != n) {
    fail(testName, "postorder size expected " + std::to_string(n) + ", got " +
                       std::to_string(post.size()));
  }

  std::vector<int> position(static_cast<size_t>(n), -1);
  for (int k = 0; k < n; ++k) {
    const int col = post[static_cast<size_t>(k)];
    if (col < 0 || col >= n) {
      fail(testName, "postorder contains out-of-range column " +
                         std::to_string(col));
    }
    if (position[static_cast<size_t>(col)] != -1) {
      fail(testName, "postorder repeats column " + std::to_string(col));
    }
    position[static_cast<size_t>(col)] = k;
  }

  for (int col = 0; col < n; ++col) {
    const int parent = etree[static_cast<size_t>(col)];
    if (parent >= 0 && position[static_cast<size_t>(col)] >
                           position[static_cast<size_t>(parent)]) {
      fail(testName, "postorder visits parent " + std::to_string(parent) +
                         " before child " + std::to_string(col));
    }
  }
}

void expectCol2Sn(const std::string &testName, const std::vector<int> &col2sn,
                  const Ranges &ranges) {
  std::vector<int> expected(col2sn.size(), -1);
  for (int sid = 0; sid < static_cast<int>(ranges.size()); ++sid) {
    for (int col = ranges[static_cast<size_t>(sid)].first;
         col <= ranges[static_cast<size_t>(sid)].second; ++col) {
      expected[static_cast<size_t>(col)] = sid;
    }
  }
  expectEqual(testName, col2sn, expected, "col2sn");
}

void expectCompactMetadata(const std::string &testName,
                           const snode::SupernodeInfo<int> &info) {
  std::vector<int> expectedBoundaries;
  for (const auto &[lo, hi] : info.ranges) {
    expectedBoundaries.push_back(lo);
    if (hi < lo)
      fail(testName, "invalid supernode range");
  }
  expectedBoundaries.push_back(
      info.ranges.empty() ? 0 : info.ranges.back().second + 1);
  expectEqual(testName, info.boundaries, expectedBoundaries, "boundaries");

  if (info.row_ptr.size() != info.ranges.size() + 1 ||
      info.parent.size() != info.ranges.size() ||
      info.supernode_post.size() != info.ranges.size()) {
    fail(testName, "compact metadata sizes are inconsistent");
  }
  expectPostorderValid(testName + " supernodal", info.parent,
                       info.supernode_post);

  size_t expectedStorage = 0;
  int expectedMaxFront = 0;
  for (size_t sid = 0; sid < info.ranges.size(); ++sid) {
    const auto [lo, hi] = info.ranges[sid];
    const int begin = info.row_ptr[sid];
    const int end = info.row_ptr[sid + 1];
    if (begin < 0 || begin > end ||
        static_cast<size_t>(end) > info.rows.size()) {
      fail(testName, "invalid supernode row pointers");
    }
    const int width = hi - lo + 1;
    if (end - begin < width)
      fail(testName, "supernode row pattern is shorter than its pivot block");
    for (int k = 0; k < width; ++k)
      if (info.rows[static_cast<size_t>(begin + k)] != lo + k)
        fail(testName, "pivot columns do not lead the supernode row pattern");
    if (!std::is_sorted(info.rows.begin() + begin, info.rows.begin() + end) ||
        std::adjacent_find(info.rows.begin() + begin,
                           info.rows.begin() + end) !=
            info.rows.begin() + end) {
      fail(testName, "supernode row pattern is not sorted and unique");
    }
    expectedStorage +=
        static_cast<size_t>(width) * static_cast<size_t>(end - begin);
    expectedMaxFront = std::max(expectedMaxFront, end - begin);
  }
  if (info.factor_storage != expectedStorage ||
      info.max_front_size != expectedMaxFront) {
    fail(testName, "symbolic storage estimates are inconsistent");
  }
}

template <typename Fn>
void expectInvalidArgument(const std::string &testName, Fn &&fn) {
  try {
    fn();
  } catch (const std::invalid_argument &) {
    return;
  }
  fail(testName, "expected std::invalid_argument");
}

void testInvalidInputs() {
  expectInvalidArgument("invalid etree parent", [] {
    (void)snode::postorder_etree(std::vector<int>{2, -1});
  });

  const std::vector<int> etree{1, -1};
  const std::vector<int> badAp{0, 1, 2};
  const std::vector<int> lowerEntry{1, 1};
  const snode::SparseUpperCSC<int> badMatrix{2, &badAp, &lowerEntry};
  const snode::Symbolic<int> symbolic{2, &etree};
  expectInvalidArgument("lower triangular input", [&] {
    (void)snode::identify_supernodes(badMatrix, symbolic);
  });

  const snode::SparseUpperCSC<int> nullMatrix{2, nullptr, nullptr};
  expectInvalidArgument("null CSC arrays", [&] {
    (void)snode::identify_supernodes(nullMatrix, symbolic);
  });
}

void runStandaloneCase(const std::string &testName, int n,
                       const Ranges &denseBlocks,
                       const Ranges &expectedRanges) {
  std::vector<int> ap;
  std::vector<int> ai;
  makeUpperCsc(n, denseBlocks, ap, ai);

  const std::vector<int> etree = denseBlockEtree(n, denseBlocks);
  snode::SparseUpperCSC<int> B{n, &ap, &ai};
  snode::Symbolic<int> S{n, &etree};

  const auto info = snode::identify_supernodes<int>(B, S);

  expectRanges(testName, info.ranges, expectedRanges);
  expectEqual(testName, info.etree, etree, "etree");
  expectCol2Sn(testName, info.col2sn, expectedRanges);
  expectPostorderValid(testName, etree, info.post);
  expectCompactMetadata(testName, info);
}

void testParentEdgeAmalgamation() {
  const std::vector<int> ap{0, 1, 2, 5};
  const std::vector<int> ai{0, 1, 0, 1, 2};
  const std::vector<int> etree{2, 2, -1};
  const snode::SparseUpperCSC<int> matrix{3, &ap, &ai};
  const snode::Symbolic<int> symbolic{3, &etree};

  const auto exact =
      snode::identify_supernodes(matrix, symbolic, 0, 0.0, 1.0, 8);
  expectRanges("parent-edge amalgamation", exact.ranges, {{0, 0}, {1, 2}});
  expectCompactMetadata("parent-edge amalgamation", exact);
}

SpMat makeBlockSpd(int n, const Ranges &denseBlocks) {
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  dense.diagonal().array() = 8.0;

  for (const auto &[lo, hi] : denseBlocks) {
    for (int row = lo; row <= hi; ++row) {
      for (int col = lo; col <= hi; ++col) {
        if (row != col) {
          dense(row, col) = 0.25;
        }
      }
    }
  }

  return dense.sparseView(0.0, 0.0);
}

void runEigenCase(const std::string &testName, int n,
                  const Ranges &denseBlocks,
                  const Ranges &expectedRanges) {
  const SpMat A = makeBlockSpd(n, denseBlocks);
  supernodal::SupernodalLDLT<double, int> solver;
  supernodal::computeEigen(solver, A);

  expectRanges(testName, solver.supernodeRanges(), expectedRanges);
  expectEqual(testName, solver.etree(), denseBlockEtree(n, denseBlocks),
              "computed etree");

  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1.0, static_cast<double>(n));
  Eigen::VectorXd x = supernodal::solveEigen(solver, b);
  const double relativeResidual =
      (A * x - b).norm() / std::max(1.0, b.norm());
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-12) {
    fail(testName, "solve relative residual too large: " +
                       std::to_string(relativeResidual));
  }
}

void testExternalOrderingContract() {
  constexpr int n = 5;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  dense.diagonal().array() = 4.0;
  for (int i = 0; i + 1 < n; ++i) {
    dense(i, i + 1) = -1.0;
    dense(i + 1, i) = -1.0;
  }

  supernodal::SupernodalLDLT<double, int> solver;
  solver.setExternalOrdering(
      linsys::Ordering<int>::from_perm({1, 2, 3, 4, 0}));
  const SpMat sparse = dense.sparseView(0.0, 0.0);
  supernodal::computeEigen(solver, sparse);

  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1.0, n);
  const Eigen::VectorXd x = supernodal::solveEigen(solver, b);
  const double relativeResidual =
      (dense * x - b).norm() / std::max(1.0, b.norm());
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-12) {
    fail("external ordering contract",
         "solve relative residual too large: " +
             std::to_string(relativeResidual));
  }
  if (solver.minAbsPivot() != solver.factors().min_abs_pivot ||
      solver.minAbsPivot() <= 0.0) {
    fail("external ordering contract", "minimum-pivot metadata is inconsistent");
  }

  solver.reset();
  if (solver.isFactorized() || solver.factors().factorized ||
      solver.info() != supernodal::SupernodalFactor::NotInitialized) {
    fail("solver reset", "factorization metadata was not cleared");
  }
}

void testAmdOrderingSolve() {
  constexpr int n = 25;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  dense.diagonal().array() = 6.0;
  for (int i = 0; i < n; ++i) {
    for (int offset : {1, 4}) {
      if (i + offset < n) {
        dense(i, i + offset) = -0.4;
        dense(i + offset, i) = -0.4;
      }
    }
  }

  supernodal::SupernodalLDLT<double, int> solver;
  const SpMat sparse = dense.sparseView(0.0, 0.0);
  supernodal::computeEigen(solver, sparse);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);
  const Eigen::VectorXd x = supernodal::solveEigen(solver, b);
  const double relativeResidual =
      (dense * x - b).norm() / std::max(1.0, b.norm());
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-11) {
    fail("AMD ordering solve",
         "solve relative residual too large: " +
             std::to_string(relativeResidual));
  }
}

void testRandomSparseSpdSolves() {
  std::mt19937 generator(20260723);
  std::uniform_real_distribution<double> value(-0.4, 0.4);
  std::uniform_real_distribution<double> keep(0.0, 1.0);

  for (int trial = 0; trial < 12; ++trial) {
    const int n = 21 + trial;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < col; ++row) {
        if (keep(generator) < 0.12) {
          const double entry = value(generator);
          dense(row, col) = entry;
          dense(col, row) = entry;
          dense(row, row) += std::abs(entry);
          dense(col, col) += std::abs(entry);
        }
      }
      dense(col, col) += 1.0;
    }

    supernodal::SupernodalLDLT<double, int> solver;
    const SpMat sparse = dense.sparseView(0.0, 0.0);
    supernodal::computeEigen(solver, sparse);
    const Eigen::VectorXd b =
        Eigen::VectorXd::LinSpaced(n, -0.5, 1.5 + 0.1 * trial);
    const Eigen::VectorXd x = supernodal::solveEigen(solver, b);
    const double residual =
        (dense * x - b).norm() /
        (dense.norm() * x.norm() + b.norm() +
         std::numeric_limits<double>::epsilon());
    if (!std::isfinite(residual) || residual > 1e-12) {
      fail("random sparse SPD", "normalized solve residual too large: " +
                                    std::to_string(residual));
    }
  }
}

} // namespace

int main() {
  testInvalidInputs();
  testParentEdgeAmalgamation();
  testExternalOrderingContract();
  testAmdOrderingSolve();
  testRandomSparseSpdSolves();
  runStandaloneCase("standalone dense block at front", 7, {{0, 2}},
                    {{0, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
  runStandaloneCase("standalone dense block in middle", 7, {{2, 4}},
                    {{0, 0}, {1, 1}, {2, 4}, {5, 5}, {6, 6}});
  runStandaloneCase("standalone dense block at end", 7, {{4, 6}},
                    {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 6}});
  runStandaloneCase("standalone multiple dense blocks", 9, {{0, 1}, {3, 5}, {7, 8}},
                    {{0, 1}, {2, 2}, {3, 5}, {6, 6}, {7, 8}});

  runEigenCase("eigen dense block at front", 7, {{0, 2}},
               {{0, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
  runEigenCase("eigen dense block in middle", 7, {{2, 4}},
               {{0, 0}, {1, 1}, {2, 4}, {5, 5}, {6, 6}});
  runEigenCase("eigen dense block at end", 7, {{4, 6}},
               {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 6}});
  runEigenCase("eigen multiple dense blocks", 9, {{0, 1}, {3, 5}, {7, 8}},
               {{0, 1}, {2, 2}, {3, 5}, {6, 6}, {7, 8}});

  std::cout << "supernode tests passed\n";
  return EXIT_SUCCESS;
}
