#include "linear_system/ldlt.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using SpMat = Eigen::SparseMatrix<double>;
using DenseMat = Eigen::MatrixXd;

int main() {
  int n = 13;
  DenseMat M = Eigen::MatrixXd::Zero(n, n);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c)
      M(r, c) = 0.1 * static_cast<double>((r + 2) * (c + 3) % 7);
  Eigen::MatrixXd A = M.transpose() * M + static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n);
  
  Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1.0, static_cast<double>(n));
  
  // Dense solver
  Eigen::VectorXd x_dense = A.ldlt().solve(b);
  std::cout << "Dense solve residual = " << (A * x_dense - b).norm() << std::endl;
  
  // Sparse solver (full matrix)
  SpMat S_full = A.sparseView(0.0, 0.0);
  Eigen::SimplicialLDLT<SpMat> solver_full;
  solver_full.compute(S_full);
  Eigen::VectorXd x_full = solver_full.solve(b);
  std::cout << "Sparse full residual = " << (A * x_full - b).norm() << std::endl;
  
  // Sparse solver (upper triangle only)
  SpMat S_upper(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j)
      if (A(i,j) != 0) S_upper.insert(i, j) = A(i,j);
  S_upper.makeCompressed();
  
  Eigen::SimplicialLDLT<SpMat> solver_upper;
  solver_upper.compute(S_upper);
  Eigen::VectorXd x_upper = solver_upper.solve(b);
  std::cout << "Sparse upper-only residual = " << (A * x_upper - b).norm() << std::endl;
  
  return 0;
}
