#include "ipm/ip_solver.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL ip_solver: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void testShiftedFiniteBounds() {
    Eigen::MatrixXd denseA(1, 2);
    denseA << 1.0, 1.0;
    Eigen::SparseMatrix<double> A = denseA.sparseView();

    Eigen::VectorXd b(1), c(2), lb(2), ub(2), sense(1);
    b << 0.0;
    c << -1.0, 0.0;
    lb << -2.0, -10.0;
    ub << 2.0, 10.0;
    sense << 1.0;

    ip_solver solver;
    solver.ls.setSolverType(SparseSolver::HIPO_LDLT);
    solver.solve(A, b, c, lb, ub, sense, 1e-8);

    const auto x = solver.getPrimals();
    require(x.size() == 2, "unexpected primal vector size");
    require(std::abs(x[0] - 2.0) < 1e-6, "finite upper bound was not shifted by the lower bound");
    require(std::abs(x[0] + x[1]) < 1e-6, "shifted-bound solution is infeasible");
    require(std::abs(solver.getObjective() + 2.0) < 1e-6, "shifted-bound objective is incorrect");
}

void testInequalitySense(double row_sign, double cost, double expected_x) {
    Eigen::MatrixXd denseA(1, 1);
    denseA << 1.0;
    Eigen::SparseMatrix<double> A = denseA.sparseView();

    Eigen::VectorXd b(1), c(1), lb(1), ub(1), sense(1);
    b << 1.0;
    c << cost;
    lb << 0.0;
    ub << 2.0;
    sense << row_sign;

    ip_solver solver;
    solver.ls.setSolverType(SparseSolver::HIPO_LDLT);
    solver.solve(A, b, c, lb, ub, sense, 1e-8);

    const auto x = solver.getPrimals();
    require(x.size() == 1, "unexpected inequality primal vector size");
    require(std::abs(x[0] - expected_x) < 1e-6, "inequality sense was converted incorrectly");
}

} // namespace

int main() {
    testShiftedFiniteBounds();
    testInequalitySense(0.0, -1.0, 1.0); // min -x subject to x <= 1
    testInequalitySense(-1.0, 1.0, 1.0); // min  x subject to x >= 1
    std::cout << "PASS ip_solver\n";
    return EXIT_SUCCESS;
}
