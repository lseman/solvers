#include "linear_system/ldlt.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <iomanip>

using SpMat = Eigen::SparseMatrix<double>;
using DenseMat = Eigen::MatrixXd;
using Solver = ldlt::SimplicialLDLT<double, int>;

int main() {
    // tridiagSpd(2) = [[4,1],[1,4]]
    DenseMat M(2,2);
    M << 4, 1, 1, 4;
    SpMat S = M.sparseView(0.0, 0.0);
    
    // Check Eigen's SparseMatrix storage
    std::cout << "Eigen SparseMatrix:" << std::endl;
    for (int j = 0; j < S.outerSize(); ++j) {
        for (SpMat::InnerIterator it(S, j); it; ++it) {
            std::cout << "  (" << it.row() << "," << it.col() << ")=" << it.value() << std::endl;
        }
    }
    
    ldlt::SparseCSC<double, int> csc = ldlt::SparseCSC<double, int>::fromEigenSparse(S);
    std::cout << "CSC: n=" << csc.n << " Ap=" << csc.Ap[0] << "," << csc.Ap[1] << "," << csc.Ap[2]
              << " Ai=";
    for (auto v : csc.Ai) std::cout << v << " ";
    std::cout << " Ax=";
    for (auto v : csc.Ax) std::cout << v << " ";
    std::cout << std::endl;
    
    Solver solver(csc);
    std::cout << "D = " << solver.factors().D[0] << ", " << solver.factors().D[1] << std::endl;
    std::cout << "Lp =";
    for (int i = 0; i <= 2; ++i) std::cout << " " << solver.factors().Lp[i];
    std::cout << " Li=";
    for (auto v : solver.factors().Li) std::cout << " " << v;
    std::cout << " Lx=";
    for (auto v : solver.factors().Lx) std::cout << " " << v;
    std::cout << std::endl;
    
    Eigen::VectorXd b(2);
    b << 1.0, 2.0;
    std::vector<double> bv = {1.0, 2.0};
    auto xv = solver.solve(bv);
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(xv.data(), xv.size());
    std::cout << "x = " << x.transpose() << std::endl;
    std::cout << "S*x = " << (S * x).transpose() << std::endl;
    std::cout << "b = " << b.transpose() << std::endl;
    std::cout << "residual = " << (S * x - b).norm() << std::endl;
    
    return 0;
}
