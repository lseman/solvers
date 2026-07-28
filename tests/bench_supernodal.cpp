#include "linear_system/eigen_interop/supernodal_eigen_interop.h"
#include "linear_system/ldlt/supernodal_ldlt.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Solver = supernodal::SupernodalLDLT<double, int>;
using SpMat = Eigen::SparseMatrix<double>;

struct Sample {
  double analyzeFactorMs;
  double refactorMs;
  double solveMs;
};

SpMat bandedSpd(int n, int halfBandwidth) {
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  for (int col = 0; col < n; ++col) {
    for (int offset = 1; offset <= halfBandwidth && col + offset < n;
         ++offset) {
      const double value = -0.05 / static_cast<double>(offset);
      dense(col, col + offset) = dense(col + offset, col) = value;
      dense(col, col) += std::abs(value);
      dense(col + offset, col + offset) += std::abs(value);
    }
    dense(col, col) += 1.0;
  }
  return dense.sparseView(0.0, 0.0);
}

SpMat blockArrowSpd(int n, int blockSize) {
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  dense.diagonal().array() = 3.0;
  for (int begin = 0; begin < n; begin += blockSize) {
    const int end = std::min(n, begin + blockSize);
    for (int col = begin; col < end; ++col) {
      for (int row = col + 1; row < end; ++row)
        dense(row, col) = dense(col, row) = 0.01;
      if (col + blockSize < n)
        dense(col, n - 1) = dense(n - 1, col) = 0.005;
    }
  }
  return dense.sparseView(0.0, 0.0);
}

template <typename Fn> double milliseconds(Fn &&fn) {
  const auto begin = Clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

Sample run(const SpMat &matrix, Solver::BackendPolicy policy) {
  constexpr int repetitions = 9;
  const auto csc = supernodal::eigen_to_csc<double, int>(matrix);
  const Eigen::VectorXd rhs =
      Eigen::VectorXd::LinSpaced(matrix.rows(), 0.25, 1.25);

  std::vector<double> analyzeFactor;
  std::vector<double> refactor;
  std::vector<double> solve;
  analyzeFactor.reserve(repetitions);
  refactor.reserve(repetitions);
  solve.reserve(repetitions);

  Solver warmup;
  warmup.setBackendPolicy(policy);
  warmup.compute(csc);
  (void)supernodal::solveEigen(warmup, rhs);

  for (int repetition = 0; repetition < repetitions; ++repetition) {
    Solver solver;
    solver.setBackendPolicy(policy);
    analyzeFactor.push_back(milliseconds([&] { solver.compute(csc); }));
    refactor.push_back(
        milliseconds([&] { solver.refactorizeSamePattern(csc); }));
    Eigen::VectorXd solution;
    solve.push_back(milliseconds(
        [&] { solution = supernodal::solveEigen(solver, rhs); }));
    const double residual = (matrix * solution - rhs).norm() /
                            std::max(1.0, rhs.norm());
    if (!std::isfinite(residual) || residual > 1e-9)
      throw std::runtime_error("benchmark solve residual is too large");
  }
  return {median(analyzeFactor), median(refactor), median(solve)};
}

void print(std::string_view matrixName, std::string_view backend,
           const SpMat &matrix, const Sample &sample) {
  std::cout << std::left << std::setw(14) << matrixName << std::setw(14)
            << backend << std::right << std::setw(8) << matrix.rows()
            << std::setw(11) << matrix.nonZeros() << std::fixed
            << std::setprecision(3) << std::setw(14)
            << sample.analyzeFactorMs << std::setw(14) << sample.refactorMs
            << std::setw(12) << sample.solveMs << '\n';
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, SpMat>> matrices{
      {"banded", bandedSpd(1000, 4)},
      {"block-arrow", blockArrowSpd(1000, 32)},
      {"dense-panel", bandedSpd(600, 80)},
  };
  const std::vector<std::pair<std::string_view, Solver::BackendPolicy>>
      backends{
          {"automatic", Solver::BackendPolicy::Automatic},
          {"simplicial", Solver::BackendPolicy::ForceSimplicial},
          {"supernodal", Solver::BackendPolicy::ForceSupernodal},
      };

  std::cout << std::left << std::setw(14) << "matrix" << std::setw(14)
            << "backend" << std::right << std::setw(8) << "n"
            << std::setw(11) << "nnz" << std::setw(14) << "analyze+fact"
            << std::setw(14) << "refactor" << std::setw(12) << "solve"
            << '\n';
  for (const auto &[matrixName, matrix] : matrices)
    for (const auto &[backendName, backend] : backends)
      print(matrixName, backendName, matrix, run(matrix, backend));
}
