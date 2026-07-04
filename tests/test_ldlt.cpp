#include "linear_system/eigen_addon/ldlt_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using SpMat = Eigen::SparseMatrix<double>;
using DenseMat = Eigen::MatrixXd;
using Solver = ldlt::SimplicialLDLT<double, int>;

void fail(const std::string &testName, const std::string &message) {
  std::cerr << "FAIL " << testName << ": " << message << "\n";
  std::exit(EXIT_FAILURE);
}

SpMat denseSpd(int n) {
  DenseMat M = DenseMat::Zero(n, n);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      M(r, c) = 0.1 * static_cast<double>((r + 2) * (c + 3) % 7);
    }
  }
  DenseMat A = M.transpose() * M +
                      static_cast<double>(n) * DenseMat::Identity(n, n);
  return A.sparseView(0.0, 0.0);
}

SpMat tridiagSpd(int n) {
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<size_t>(3 * n));
  for (int i = 0; i < n; ++i) {
    trips.emplace_back(i, i, 4.0);
  }
  for (int i = 0; i + 1 < n; ++i) {
    trips.emplace_back(i, i + 1, 1.0);
    trips.emplace_back(i + 1, i, 1.0);
  }
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
}

void expectAccurateSolve(const std::string &testName, const Solver &solver,
                         const SpMat &A, const Eigen::VectorXd &b) {
  std::vector<double> bv(static_cast<size_t>(b.size()));
  for (int i = 0; i < b.size(); ++i) bv[static_cast<size_t>(i)] = b[i];
  auto xv = solver.solve(bv);
  Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(xv.data(), xv.size());
  const double relativeResidual =
      (A * x - b).norm() / std::max(1.0, b.norm());
  if (!solver.factors().factorized) {
    fail(testName, "factorization failed");
  }
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-11) {
    fail(testName, "relative residual too large: " +
                       std::to_string(relativeResidual));
  }
}

void testAmdAccuracy() {
  for (int n = 1; n <= 30; ++n) {
    {
      SpMat A = denseSpd(n);
      ldlt::SparseCSC<double, int> csc = ldlt::fromEigenSparse<double, int>(A);
      Solver solver(csc);
      Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 1.0, static_cast<double>(A.rows()));
      expectAccurateSolve("amd dense SPD n=" + std::to_string(n), solver, A, b);
    }
    {
      SpMat A = tridiagSpd(n);
      ldlt::SparseCSC<double, int> csc = ldlt::fromEigenSparse<double, int>(A);
      Solver solver(csc);
      Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 1.0, static_cast<double>(A.rows()));
      expectAccurateSolve("amd tridiag SPD n=" + std::to_string(n), solver, A, b);
    }
  }
}

void testNaturalOrdering() {
  SpMat A = tridiagSpd(8);
  ldlt::SparseCSC<double, int> csc = ldlt::fromEigenSparse<double, int>(A);
  Solver solver(csc);
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 1.0, static_cast<double>(A.rows()));
  expectAccurateSolve("natural ordering", solver, A, b);
}

void testShift() {
  SpMat A = tridiagSpd(6);
  ldlt::SparseCSC<double, int> csc = ldlt::fromEigenSparse<double, int>(A);
  Solver solver;
  solver.setShift(2.0, 1.0);
  solver.compute(csc);

  SpMat shifted = A;
  shifted.diagonal().array() += 2.0;
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(A.rows(), 1.0, static_cast<double>(A.rows()));
  // Solve shifted system with shifted solver
  std::vector<double> bv(static_cast<size_t>(b.size()));
  for (int i = 0; i < b.size(); ++i) bv[static_cast<size_t>(i)] = b[i];
  auto xv = solver.solve(bv);
  Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(xv.data(), xv.size());
  const double relativeResidual =
      (shifted * x - b).norm() / std::max(1.0, b.norm());
  if (!std::isfinite(relativeResidual) || relativeResidual > 1e-11) {
    fail("diagonal shift", "relative residual too large: " +
                       std::to_string(relativeResidual));
  }
}

void testRegularizedZeroPivot() {
  SpMat A(2, 2);
  std::vector<Eigen::Triplet<double>> trips{{0, 0, 0.0}, {1, 1, 2.0}};
  A.setFromTriplets(trips.begin(), trips.end());

  Solver solver;
  ldlt::SparseCSC<double, int> csc = ldlt::fromEigenSparse<double, int>(A);
  solver.setRegularization(1e-6);
  solver.compute(csc);

  if (!solver.factors().factorized) {
    fail("regularized zero pivot", "factorization failed");
  }
  if (solver.factors().perturbed_pivots != 1) {
    fail("regularized zero pivot", "expected one perturbed pivot, got " +
                                          std::to_string(solver.factors().perturbed_pivots));
  }
}

} // namespace

int main() {
  testAmdAccuracy();
  testNaturalOrdering();
  testShift();
  testRegularizedZeroPivot();
  std::cout << "ldlt tests passed\n";
  return EXIT_SUCCESS;
}
