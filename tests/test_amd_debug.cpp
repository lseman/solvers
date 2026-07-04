#include "linear_system/ldlt.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using SpMat = Eigen::SparseMatrix<double>;
using DenseMat = Eigen::MatrixXd;
using Solver = ldlt::SimplicialLDLT<double, int>;

SpMat denseSpd(int n) {
  DenseMat M = Eigen::MatrixXd::Zero(n, n);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      M(r, c) = 0.1 * static_cast<double>((r + 2) * (c + 3) % 7);
  return (M.transpose() * M + static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n)).sparseView(0.0, 0.0);
}

int main() {
  for (int n = 2; n <= 15; ++n) {
    SpMat A = denseSpd(n);
    Eigen::MatrixXd Adense(n, n);
    for (int j = 0; j < A.outerSize(); ++j)
      for (SpMat::InnerIterator it(A, j); it; ++it)
        Adense(static_cast<int>(it.row()), j) = it.value();
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1.0, static_cast<double>(n));
    
    ldlt::SparseCSC<double, int> csc = ldlt::SparseCSC<double, int>::fromEigenSparse(A);
    Solver solver(csc);
    
    std::vector<double> bv(n);
    for (int i = 0; i < n; ++i) bv[i] = b[i];
    auto xv = solver.solve(bv);
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; ++i) x[i] = xv[i];
    double residual = (Adense * x - b).norm();
    std::cout << "n=" << n << " residual=" << residual << std::endl;
  }
  return 0;
}
