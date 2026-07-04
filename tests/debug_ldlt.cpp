#include "linear_system/ldlt.h"
#include <iostream>
#include <iomanip>

int main() {
    // 1x1 SPD matrix
    ldlt::SparseCSC<double, int> csc;
    csc.n = 1;
    csc.Ap = {0, 1};
    csc.Ai = {0};
    csc.Ax = {5.0};
    
    ldlt::SimplicialLDLT<double, int> solver;
    solver.compute(csc);
    
    std::cout << "factorized: " << solver.factors().factorized << std::endl;
    std::cout << "info_val: " << solver.factors().info_val << std::endl;
    std::cout << "size: " << solver.size() << std::endl;
    std::cout << "D[0]: " << solver.factors().D[0] << std::endl;
    return 0;
}
