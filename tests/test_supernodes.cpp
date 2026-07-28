#include "linear_system/ldlt/supernodal_ldlt.h"
#include "linear_system/eigen_interop/supernodal_eigen_interop.h"
#include "linear_system/eigen_interop/schur_frontal_eigen_interop.h"
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

template <typename Fn>
void expectDomainError(const std::string &testName, Fn &&fn) {
  try {
    fn();
  } catch (const std::domain_error &) {
    return;
  }
  fail(testName, "expected std::domain_error");
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
  solver.setBackendPolicy(
      supernodal::SupernodalLDLT<double, int>::BackendPolicy::ForceSupernodal);
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

void testPatternReuseContract() {
  constexpr int n = 8;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Identity(n, n) * 5.0;
  for (int i = 0; i + 1 < n; ++i) {
    dense(i, i + 1) = -0.5;
    dense(i + 1, i) = -0.5;
  }
  SpMat sparse = dense.sparseView(0.0, 0.0);

  supernodal::SupernodalLDLT<double, int> solver;
  solver.setBackendPolicy(
      supernodal::SupernodalLDLT<double, int>::BackendPolicy::ForceSupernodal);
  auto csc = supernodal::eigen_to_csc<double, int>(sparse);
  solver.compute(csc);
  const double *panelAddress = solver.factors().panel_values.data();

  for (double &value : csc.Ax) value *= 1.25;
  solver.refactorizeSamePattern(csc);
  if (solver.factors().panel_values.data() != panelAddress)
    fail("pattern reuse", "numeric panel storage was reallocated");

  Eigen::MatrixXd changed = dense;
  changed(0, 3) = changed(3, 0) = 0.2;
  auto changedCsc =
      supernodal::eigen_to_csc<double, int>(changed.sparseView(0.0, 0.0));
  expectInvalidArgument("pattern mismatch", [&] {
    solver.refactorizeSamePattern(changedCsc);
  });

  solver.factorizeMatrix(changedCsc);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
  const Eigen::VectorXd x = supernodal::solveEigen(solver, b);
  const double residual = (changed * x - b).norm() /
                          (changed.norm() * x.norm() + b.norm() +
                           std::numeric_limits<double>::epsilon());
  if (!std::isfinite(residual) || residual > 1e-12)
    fail("pattern reanalysis", "solve residual too large");
}

void testUpperTriangleInput() {
  constexpr int n = 9;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Identity(n, n) * 4.0;
  for (int i = 0; i + 1 < n; ++i)
    dense(i, i + 1) = dense(i + 1, i) = -0.75;
  const Eigen::MatrixXd upperDense =
      dense.triangularView<Eigen::Upper>().toDenseMatrix();
  const SpMat upper = upperDense.sparseView(0.0, 0.0);

  supernodal::SupernodalLDLT<double, int> solver;
  solver.setBackendPolicy(
      supernodal::SupernodalLDLT<double, int>::BackendPolicy::ForceSupernodal);
  supernodal::computeEigen(solver, upper);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 0.5, 2.5);
  const Eigen::VectorXd x = supernodal::solveEigen(solver, b);
  const double residual = (dense * x - b).norm() /
                          (dense.norm() * x.norm() + b.norm() +
                           std::numeric_limits<double>::epsilon());
  if (!std::isfinite(residual) || residual > 1e-12)
    fail("upper triangle input", "solve residual too large");
}

void testPivotPolicies() {
  supernodal::SparseCSC<double, int> zeroDiagonal;
  zeroDiagonal.n = 2;
  zeroDiagonal.Ap = {0, 1, 2};
  zeroDiagonal.Ai = {0, 1};
  zeroDiagonal.Ax = {0.0, 0.0};

  supernodal::SupernodalLDLT<double, int> signedSolver;
  signedSolver.setBackendPolicy(
      supernodal::SupernodalLDLT<double, int>::BackendPolicy::ForceSupernodal);
  signedSolver.setRegularization(1e-6);
  signedSolver.setExpectedPivotSigns({-1, 1});
  signedSolver.compute(zeroDiagonal);
  if (signedSolver.factors().D != std::vector<double>({-1e-6, 1e-6}) ||
      signedSolver.perturbedPivots() != 2)
    fail("signed regularization", "regularized pivot signs are incorrect");

  supernodal::SupernodalLDLT<double, int> strictSolver;
  strictSolver.setBackendPolicy(
      supernodal::SupernodalLDLT<double, int>::BackendPolicy::ForceSupernodal);
  strictSolver.setStrictPivots(true);
  expectDomainError("strict pivots", [&] { strictSolver.compute(zeroDiagonal); });

  Eigen::Matrix2d indefinite;
  indefinite << 0.0, 1.0, 1.0, 0.0;
  const SpMat indefiniteSparse = indefinite.sparseView(0.0, 0.0);
  supernodal::SupernodalLDLT<double, int> pivotedSolver;
  pivotedSolver.setPivotPolicy(
      supernodal::SupernodalLDLT<double, int>::PivotPolicy::BunchKaufman);
  supernodal::computeEigen(pivotedSolver, indefiniteSparse);
  const Eigen::Vector2d rhs(2.0, -3.0);
  const Eigen::VectorXd solution =
      supernodal::solveEigen(pivotedSolver, rhs);
  if ((indefinite * solution - rhs).norm() > 1e-13 ||
      pivotedSolver.factors().positive_inertia != 1 ||
      pivotedSolver.factors().negative_inertia != 1 ||
      !pivotedSolver.factors().pivoted)
    fail("Bunch-Kaufman pivoting", "2x2 pivot or inertia is incorrect");
}

void testAutomaticBackendSelection() {
  constexpr int sparseN = 32;
  SpMat diagonal(sparseN, sparseN);
  diagonal.setIdentity();
  supernodal::SupernodalLDLT<double, int> sparseSolver;
  supernodal::computeEigen(sparseSolver, diagonal);
  if (sparseSolver.backend() !=
      supernodal::SymbolicLDLT::Backend::Simplicial)
    fail("automatic backend", "diagonal matrix should select simplicial");

  constexpr int denseN = 160;
  Eigen::MatrixXd dense =
      Eigen::MatrixXd::Constant(denseN, denseN, 0.001);
  dense.diagonal().array() = 2.0;
  supernodal::SupernodalLDLT<double, int> denseSolver;
  const SpMat denseSparse = dense.sparseView(0.0, 0.0);
  supernodal::computeEigen(denseSolver, denseSparse);
  if (denseSolver.backend() !=
      supernodal::SymbolicLDLT::Backend::Supernodal)
    fail("automatic backend", "dense matrix should select supernodal");
}

void testSymmetricStorageContracts() {
  using Solver = supernodal::SupernodalLDLT<double, int>;
  using Storage = Solver::SymmetricStorage;
  const Eigen::Matrix2d dense =
      (Eigen::Matrix2d() << 4.0, 3.0, 3.0, 5.0).finished();
  const std::vector<double> rhs{1.0, -2.0};

  auto checkSolve = [&](const std::string &name,
                        supernodal::SparseCSC<double, int> matrix,
                        Storage storage) {
    Solver solver;
    solver.setBackendPolicy(Solver::BackendPolicy::ForceSupernodal);
    solver.setSymmetricStorage(storage);
    solver.compute(matrix);
    const auto solution = solver.solve(rhs);
    const Eigen::Vector2d x(solution[0], solution[1]);
    const double residual =
        (dense * x - Eigen::Vector2d(rhs[0], rhs[1])).norm();
    if (!std::isfinite(residual) || residual > 1e-13)
      fail(name, "duplicate entries were not summed correctly");
  };

  supernodal::SparseCSC<double, int> upper;
  upper.n = 2;
  upper.Ap = {0, 1, 4};
  upper.Ai = {0, 0, 0, 1};
  upper.Ax = {4.0, 1.0, 2.0, 5.0};
  checkSolve("upper duplicate sums", upper, Storage::Upper);
  checkSolve("auto-detected upper duplicate sums", upper, Storage::AutoDetect);

  supernodal::SparseCSC<double, int> lower;
  lower.n = 2;
  lower.Ap = {0, 3, 4};
  lower.Ai = {0, 1, 1, 1};
  lower.Ax = {4.0, 1.0, 2.0, 5.0};
  checkSolve("lower duplicate sums", lower, Storage::Lower);

  supernodal::SparseCSC<double, int> full;
  full.n = 2;
  full.Ap = {0, 3, 6};
  full.Ai = {0, 1, 1, 0, 0, 1};
  full.Ax = {4.0, 1.0, 2.0, 1.5, 1.5, 5.0};
  checkSolve("full mirrored duplicate sums", full, Storage::FullSymmetric);

  Solver wrongTriangle;
  wrongTriangle.setSymmetricStorage(Storage::Upper);
  expectInvalidArgument("explicit triangle mismatch",
                        [&] { wrongTriangle.compute(lower); });

  full.Ax[3] = 2.5;
  Solver inconsistentMirror;
  inconsistentMirror.setSymmetricStorage(Storage::FullSymmetric);
  expectInvalidArgument("inconsistent mirrored sums",
                        [&] { inconsistentMirror.compute(full); });

  full.Ap = {0, 2, 3};
  full.Ai = {0, 1, 1};
  full.Ax = {4.0, 3.0, 5.0};
  Solver missingMirror;
  missingMirror.setSymmetricStorage(Storage::FullSymmetric);
  expectInvalidArgument("missing mirrored entry",
                        [&] { missingMirror.compute(full); });
}

void testContributorAdjacencyAndLazyCsc() {
  using Solver = supernodal::SupernodalLDLT<double, int>;
  constexpr int n = 14;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  dense.diagonal().array() = 7.0;
  for (int col = 0; col < n; ++col) {
    for (int offset : {1, 3, 6}) {
      if (col + offset < n) {
        dense(col, col + offset) = -0.2;
        dense(col + offset, col) = -0.2;
      }
    }
  }

  Solver solver;
  solver.setBackendPolicy(Solver::BackendPolicy::ForceSupernodal);
  solver.setRelaxationThresholds({{1, 0.0}});
  const SpMat sparse = dense.sparseView(0.0, 0.0);
  supernodal::computeEigen(solver, sparse);

  if (solver.scalarCscMaterialized())
    fail("lazy scalar CSC", "CSC was built during numeric factorization");
  const auto &panels = solver.panelFactors();
  const auto symbolic = solver.symbolic();
  if (solver.scalarCscMaterialized())
    fail("lazy scalar CSC", "panel access unexpectedly built scalar CSC");
  if (solver.nonZerosL() == 0)
    fail("lazy scalar CSC", "symbolic nonzero count was not retained");
  if (symbolic->contributorPointers().size() !=
          symbolic->supernodeRanges().size() + 1 ||
      symbolic->contributorPointers().back() !=
          static_cast<int>(symbolic->contributors().size()))
    fail("contributor adjacency", "invalid compressed adjacency offsets");

  for (size_t target = 0; target < symbolic->supernodeRanges().size(); ++target) {
    const auto [lo, hi] = symbolic->supernodeRanges()[target];
    for (int p = symbolic->contributorPointers()[target];
         p < symbolic->contributorPointers()[target + 1]; ++p) {
      const int source = symbolic->contributors()[static_cast<size_t>(p)];
      if (source < 0 || static_cast<size_t>(source) >= target)
        fail("contributor adjacency", "contributor is not an earlier panel");
      bool intersects = false;
      for (int r = symbolic->supernodeRowPointers()[static_cast<size_t>(source)];
           r < symbolic->supernodeRowPointers()[static_cast<size_t>(source) + 1];
           ++r) {
        const int row = symbolic->supernodeRows()[static_cast<size_t>(r)];
        intersects = intersects || (row >= lo && row <= hi);
      }
      if (!intersects)
        fail("contributor adjacency", "listed panel does not reach target");
    }
  }
  if (symbolic->contributors().empty())
    fail("contributor adjacency", "test matrix produced no contributors");

  const auto &scalar = solver.factors();
  if (!solver.scalarCscMaterialized() ||
      scalar.Li.size() != static_cast<size_t>(solver.nonZerosL()) ||
      scalar.Lx.size() != scalar.Li.size())
    fail("lazy scalar CSC", "on-demand CSC materialization is inconsistent");
}

void testMultipleRhsPanelSolve() {
  using Solver = supernodal::SupernodalLDLT<double, int>;
  constexpr int n = 9;
  const SpMat sparse = makeBlockSpd(n, {{0, 4}, {5, 8}});
  const Eigen::MatrixXd dense(sparse);
  Solver solver;
  solver.setBackendPolicy(Solver::BackendPolicy::ForceSupernodal);
  solver.setExternalOrdering(linsys::Ordering<int>::identity(n));
  supernodal::computeEigen(solver, sparse);

  Eigen::MatrixXd rhs(n, 3);
  rhs.col(0) = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
  rhs.col(1) = Eigen::VectorXd::LinSpaced(n, 2.0, -0.5);
  rhs.col(2).setOnes();
  std::vector<double> packed(rhs.data(), rhs.data() + rhs.size());
  const auto solved = solver.solveMultiple(packed, rhs.cols());
  const Eigen::Map<const Eigen::MatrixXd> x(solved.data(), n, rhs.cols());
  const double residual =
      (dense * x - rhs).norm() / std::max(1.0, rhs.norm());
  if (!std::isfinite(residual) || residual > 1e-12)
    fail("multiple RHS panel solve", "relative residual too large");

  for (int col = 0; col < rhs.cols(); ++col) {
    const std::vector<double> b(rhs.col(col).data(),
                                rhs.col(col).data() + n);
    const auto single = solver.solve(b);
    if ((Eigen::Map<const Eigen::VectorXd>(single.data(), n) - x.col(col))
            .norm() > 1e-13)
      fail("multiple RHS panel solve", "batch and scalar solves disagree");
  }
}

void testSparseIntranodalPivoting() {
  using Solver = supernodal::SupernodalLDLT<double, int>;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(6, 6);
  dense.topLeftCorner<3, 3>() <<
      0.0, 1.0, 0.2, 1.0, 0.0, 0.1, 0.2, 0.1, 3.0;
  dense.bottomRightCorner<3, 3>() <<
      4.0, 0.3, 0.1, 0.3, 5.0, 0.2, 0.1, 0.2, 6.0;

  Solver solver;
  solver.setBackendPolicy(Solver::BackendPolicy::ForceSupernodal);
  solver.setPivotPolicy(Solver::PivotPolicy::IntranodalBunchKaufman);
  solver.setExternalOrdering(linsys::Ordering<int>::identity(6));
  const SpMat sparse = dense.sparseView(0.0, 0.0);
  supernodal::computeEigen(solver, sparse);
  const auto &f = solver.panelFactors();
  if (!f.intranodal_pivoted || f.pivoted)
    fail("sparse intranodal pivoting", "dense fallback was used");
  if (solver.symbolic()->supernodeRanges().size() < 2)
    fail("sparse intranodal pivoting", "sparse block structure was not retained");
  if (std::find(f.pivot_blocks.begin(), f.pivot_blocks.end(), int8_t{2}) ==
      f.pivot_blocks.end())
    fail("sparse intranodal pivoting", "expected a 2x2 pivot block");
  if (f.positive_inertia != 5 || f.negative_inertia != 1 ||
      f.zero_inertia != 0)
    fail("sparse intranodal pivoting", "inertia is incorrect");

  Eigen::MatrixXd rhs(6, 2);
  rhs << 1.0, -2.0, 0.5, 1.0, -1.5, 0.25,
      0.75, -0.25, 2.0, 1.5, -0.5, 0.8;
  const std::vector<double> packed(rhs.data(), rhs.data() + rhs.size());
  const auto solved = solver.solveMultiple(packed, rhs.cols());
  const Eigen::Map<const Eigen::MatrixXd> x(solved.data(), 6, rhs.cols());
  const double residual =
      (dense * x - rhs).norm() / std::max(1.0, rhs.norm());
  if (!std::isfinite(residual) || residual > 1e-12)
    fail("sparse intranodal pivoting", "solve residual too large");
}

void testSharedImmutableSymbolicAnalysis() {
  using Solver = supernodal::SupernodalLDLT<double, int>;
  constexpr int n = 8;
  Eigen::MatrixXd first = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd second = Eigen::MatrixXd::Zero(n, n);
  for (int col = 0; col < n; ++col) {
    first(col, col) = 5.0;
    second(col, col) = 7.0;
    for (int offset : {1, 3}) {
      if (col + offset < n) {
        first(col, col + offset) = first(col + offset, col) = -0.2;
        second(col, col + offset) = second(col + offset, col) = 0.15;
      }
    }
  }
  const SpMat firstSparse = first.sparseView(0.0, 0.0);
  const SpMat secondSparse = second.sparseView(0.0, 0.0);
  const auto firstCsc = supernodal::eigen_to_csc<double, int>(firstSparse);
  const auto secondCsc = supernodal::eigen_to_csc<double, int>(secondSparse);

  Solver analyzer;
  analyzer.setBackendPolicy(Solver::BackendPolicy::ForceSupernodal);
  const std::shared_ptr<const supernodal::SymbolicLDLT> symbolic =
      analyzer.analyzeSymbolic(firstCsc);
  if (analyzer.isFactorized())
    fail("shared symbolic analysis", "symbolic analysis performed numeric work");
  if (!symbolic || symbolic->size() != n ||
      symbolic->panelValuePointers().size() !=
          symbolic->supernodeRanges().size() + 1 ||
      symbolic->panelStorage() == 0 || symbolic->maxFrontSize() == 0)
    fail("shared symbolic analysis", "symbolic metadata is incomplete");

  Solver factor1;
  Solver factor2;
  factor1.factorizeWithSymbolic(firstCsc, symbolic);
  factor2.factorizeWithSymbolic(secondCsc, symbolic);
  if (factor1.symbolic().get() != symbolic.get() ||
      factor2.symbolic().get() != symbolic.get())
    fail("shared symbolic analysis", "numeric factors copied the symbolic object");
  if (factor1.panelFactors().panel_values.data() ==
      factor2.panelFactors().panel_values.data())
    fail("shared symbolic analysis", "numeric panel storage was shared");

  const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);
  const Eigen::VectorXd x1 = supernodal::solveEigen(factor1, rhs);
  const Eigen::VectorXd x2 = supernodal::solveEigen(factor2, rhs);
  if ((first * x1 - rhs).norm() / std::max(1.0, rhs.norm()) > 1e-12 ||
      (second * x2 - rhs).norm() / std::max(1.0, rhs.norm()) > 1e-12)
    fail("shared symbolic analysis", "shared-pattern solve residual is too large");

  Eigen::MatrixXd mismatch = second;
  mismatch(0, 4) = mismatch(4, 0) = 0.1;
  const SpMat mismatchSparse = mismatch.sparseView(0.0, 0.0);
  const auto mismatchCsc =
      supernodal::eigen_to_csc<double, int>(mismatchSparse);
  expectInvalidArgument("shared symbolic pattern mismatch", [&] {
    Solver rejected;
    rejected.factorizeWithSymbolic(mismatchCsc, symbolic);
  });
}

void testCommonSymbolicAcrossLdltImplementations() {
  constexpr int n = 25;
  Eigen::MatrixXd first = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd second = Eigen::MatrixXd::Zero(n, n);
  first.diagonal().array() = 5.0;
  second.diagonal().array() = 6.0;
  for (int col = 0; col + 1 < n; ++col) {
    first(col, col + 1) = first(col + 1, col) = -0.2;
    second(col, col + 1) = second(col + 1, col) = 0.1;
  }
  for (int col = 0; col + 5 < n; ++col) {
    first(col, col + 5) = first(col + 5, col) = -0.05;
    second(col, col + 5) = second(col + 5, col) = 0.04;
  }
  const SpMat firstSparse = first.sparseView(0.0, 0.0);
  const SpMat secondSparse = second.sparseView(0.0, 0.0);
  const auto firstCsc = supernodal::eigen_to_csc<double, int>(firstSparse);
  const auto secondCsc = supernodal::eigen_to_csc<double, int>(secondSparse);

  ldlt::SimplicialLDLT<double, int> analyzer;
  const auto symbolic = analyzer.analyzeSymbolic(firstCsc);
  if (!symbolic ||
      symbolic->backend() != linsys::SymbolicLDLT::Backend::Simplicial)
    fail("common symbolic LDLT", "simplicial analysis has wrong backend");

  ldlt::SimplicialLDLT<double, int> simplicial;
  simplicial.factorizeWithSymbolic(firstCsc, symbolic);
  supernodal::SupernodalLDLT<double, int> hybrid;
  hybrid.factorizeWithSymbolic(secondCsc, symbolic);
  if (simplicial.symbolic().get() != symbolic.get() ||
      hybrid.symbolic().get() != symbolic.get())
    fail("common symbolic LDLT", "LDLT implementations did not share analysis");

  const std::vector<double> rhs(static_cast<size_t>(n), 1.0);
  const auto x1 = simplicial.solve(rhs);
  const auto x2 = hybrid.solve(rhs);
  const Eigen::Map<const Eigen::VectorXd> x1Eigen(x1.data(), n);
  const Eigen::Map<const Eigen::VectorXd> x2Eigen(x2.data(), n);
  const Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  if ((first * x1Eigen - b).norm() / b.norm() > 1e-12 ||
      (second * x2Eigen - b).norm() / b.norm() > 1e-12)
    fail("common symbolic LDLT", "cross-implementation residual is too large");
}

void testSchurFrontalSharedSymbolic() {
  constexpr int n = 25;
  Eigen::MatrixXd first = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd second = Eigen::MatrixXd::Zero(n, n);
  first.diagonal().array() = 4.0;
  second.diagonal().array() = 5.0;
  for (int col = 0; col + 1 < n; ++col) {
    first(col, col + 1) = first(col + 1, col) = -0.15;
    second(col, col + 1) = second(col + 1, col) = 0.12;
  }
  for (int col = 0; col + 4 < n; ++col) {
    first(col, col + 4) = first(col + 4, col) = -0.04;
    second(col, col + 4) = second(col + 4, col) = 0.03;
  }
  const SpMat firstSparse = first.sparseView(0.0, 0.0);
  const SpMat secondSparse = second.sparseView(0.0, 0.0);
  const auto symbolic = schur_frontal::analyze_frontal(firstSparse);
  const auto factor1 =
      schur_frontal::factor_frontal(firstSparse, symbolic);
  const auto factor2 =
      schur_frontal::factor_frontal(secondSparse, symbolic);
  if (factor1.symbolic.get() != symbolic.get() ||
      factor2.symbolic.get() != symbolic.get())
    fail("Schur frontal shared symbolic", "analysis allocation was copied");

  const std::vector<double> rhs(static_cast<size_t>(n), 1.0);
  const auto x1 = schur_frontal::solve(factor1, rhs);
  const auto x2 = schur_frontal::solve(factor2, rhs);
  const Eigen::Map<const Eigen::VectorXd> x1Eigen(x1.data(), n);
  const Eigen::Map<const Eigen::VectorXd> x2Eigen(x2.data(), n);
  const Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
  if ((first * x1Eigen - b).norm() / b.norm() > 1e-12 ||
      (second * x2Eigen - b).norm() / b.norm() > 1e-12)
    fail("Schur frontal shared symbolic", "solve residual is too large");

  Eigen::MatrixXd mismatch = second;
  mismatch(0, 8) = mismatch(8, 0) = 0.02;
  const SpMat mismatchSparse = mismatch.sparseView(0.0, 0.0);
  expectInvalidArgument("Schur frontal symbolic mismatch", [&] {
    (void)schur_frontal::factor_frontal(mismatchSparse, symbolic);
  });
}

} // namespace

int main() {
  testInvalidInputs();
  testParentEdgeAmalgamation();
  testExternalOrderingContract();
  testAmdOrderingSolve();
  testRandomSparseSpdSolves();
  testPatternReuseContract();
  testUpperTriangleInput();
  testPivotPolicies();
  testAutomaticBackendSelection();
  testSymmetricStorageContracts();
  testContributorAdjacencyAndLazyCsc();
  testMultipleRhsPanelSolve();
  testSparseIntranodalPivoting();
  testSharedImmutableSymbolicAnalysis();
  testCommonSymbolicAcrossLdltImplementations();
  testSchurFrontalSharedSymbolic();
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
