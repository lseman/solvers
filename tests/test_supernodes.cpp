#include "linear_system/supernodal_ldlt.h"
#include "linear_system/supernodal_eigen_interop.h"
#include "linear_system/supernodes.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
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

void runStandaloneCase(const std::string &testName, int n,
                       const Ranges &denseBlocks,
                       const Ranges &expectedRanges) {
  std::vector<int> ap;
  std::vector<int> ai;
  makeUpperCsc(n, denseBlocks, ap, ai);

  const std::vector<int> etree = denseBlockEtree(n, denseBlocks);
  snode::SparseUpperCSC<int> B{n, &ap, &ai, nullptr};
  snode::Symbolic<int> S{n, &etree, nullptr, nullptr};

  const auto info = snode::identify_supernodes<int>(B, S);

  expectRanges(testName, info.ranges, expectedRanges);
  expectEqual(testName, info.etree, etree, "etree");
  expectCol2Sn(testName, info.col2sn, expectedRanges);
  expectPostorderValid(testName, etree, info.post);
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

} // namespace

int main() {
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
