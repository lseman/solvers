#include "ipm/hipo_ldlt.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL hipo_ldlt: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Eigen::MatrixXd denseFactorD(const ipm::HiPOFactor& factor) {
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(factor.n, factor.n);
    ipm::HIndex offset = 0;
    for (ipm::HIndex k = 0; k < factor.n;) {
        if (factor.block_info[static_cast<size_t>(k)] == 1) {
            D(k, k) = factor.D[static_cast<size_t>(offset++)];
            ++k;
        } else {
            D(k, k) = factor.D[static_cast<size_t>(offset)];
            D(k + 1, k) = D(k, k + 1) = factor.D[static_cast<size_t>(offset + 1)];
            D(k + 1, k + 1) = factor.D[static_cast<size_t>(offset + 2)];
            offset += 3;
            k += 2;
        }
    }
    return D;
}

void testBkSwapPermutesPreviouslyAssembledLRows() {
    Eigen::MatrixXd denseA(3, 5);
    denseA << 0.0, 1.2, 2.2, 0.0, 4.2,
              1.2, 2.2, 0.0, 4.2, 5.2,
              2.2, 0.0, 4.2, 5.2, 0.0;
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A = denseA.sparseView();
    const std::vector<double> theta{0.01, 0.1, 1.0, 10.0, 100.0};
    const std::vector<double> regP(5, 1e-8);
    const std::vector<double> regD(3, 1e-8);

    ipm::HiPOLDLT solver;
    solver.analyzePattern(A);
    solver.factorize(theta, regP, regD);

    const auto& iperm = solver.debugFinalIperm();
    require(iperm[6] == 7 && iperm[7] == 6, "regression matrix must trigger the expected BK swap");

    Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(8, 8);
    for (int i = 0; i < 5; ++i)
        augmented(i, i) = -(theta[static_cast<size_t>(i)] + regP[static_cast<size_t>(i)]);
    augmented.topRightCorner(5, 3) = denseA.transpose();
    augmented.bottomLeftCorner(3, 5) = denseA;
    for (int i = 0; i < 3; ++i)
        augmented(5 + i, 5 + i) = regD[static_cast<size_t>(i)];

    Eigen::MatrixXd permuted(8, 8);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            permuted(i, j) = augmented(iperm[static_cast<size_t>(i)], iperm[static_cast<size_t>(j)]);

    const auto& factor = solver.debugFactor();
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(8, 8);
    for (int col = 0; col < 8; ++col)
        for (int p = factor.Lp[static_cast<size_t>(col)]; p < factor.Lp[static_cast<size_t>(col + 1)]; ++p)
            L(factor.Li[static_cast<size_t>(p)], col) = factor.Lx[static_cast<size_t>(p)];
    const Eigen::MatrixXd reconstructed = L * denseFactorD(factor) * L.transpose();
    const double reconstructionResidual = (permuted - reconstructed).norm() / augmented.norm();
    require(reconstructionResidual < 1e-12, "P A P^T != L D L^T after a later-supernode BK swap");

    const std::vector<double> rhs{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    const std::vector<double> solution = solver.solve(rhs);
    const Eigen::Map<const Eigen::VectorXd> x(solution.data(), 8);
    const Eigen::Map<const Eigen::VectorXd> b(rhs.data(), 8);
    const double solveResidual = (augmented * x - b).norm() /
                                 (augmented.norm() * x.norm() + b.norm());
    require(solveResidual < 1e-12, "augmented-system solve residual is too large");
}

void testDeterministicSparseSystems() {
    std::mt19937 generator(20260720);
    std::uniform_real_distribution<double> value(-2.0, 2.0);
    std::uniform_real_distribution<double> keep(0.0, 1.0);

    for (int trial = 0; trial < 20; ++trial) {
        const int n = 2 + trial % 7;
        const int m = 1 + (trial * 3) % 6;
        Eigen::MatrixXd denseA = Eigen::MatrixXd::Zero(m, n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                if (keep(generator) < 0.45)
                    denseA(i, j) = value(generator);
            }
            denseA(j % m, j) += value(generator) >= 0.0 ? 0.5 : -0.5;
        }
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> A = denseA.sparseView();
        std::vector<double> theta(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j)
            theta[static_cast<size_t>(j)] = std::pow(10.0, -2.0 + 4.0 * j / std::max(1, n - 1));
        const std::vector<double> regP(static_cast<size_t>(n), 1e-8);
        const std::vector<double> regD(static_cast<size_t>(m), 1e-8);
        std::vector<double> rhs(static_cast<size_t>(n + m));
        for (double& entry : rhs)
            entry = value(generator);

        ipm::HiPOLDLT solver;
        solver.analyzePattern(A);
        solver.factorize(theta, regP, regD);
        const std::vector<double> solution = solver.solve(rhs);

        Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(n + m, n + m);
        for (int j = 0; j < n; ++j)
            augmented(j, j) = -(theta[static_cast<size_t>(j)] + regP[static_cast<size_t>(j)]);
        augmented.topRightCorner(n, m) = denseA.transpose();
        augmented.bottomLeftCorner(m, n) = denseA;
        for (int i = 0; i < m; ++i)
            augmented(n + i, n + i) = regD[static_cast<size_t>(i)];
        const Eigen::Map<const Eigen::VectorXd> x(solution.data(), n + m);
        const Eigen::Map<const Eigen::VectorXd> b(rhs.data(), n + m);
        const double residual = (augmented * x - b).norm() /
                                (augmented.norm() * x.norm() + b.norm());
        require(std::isfinite(residual) && residual < 1e-10,
                "deterministic sparse-system solve residual is too large");
    }
}

} // namespace

int main() {
    testBkSwapPermutesPreviouslyAssembledLRows();
    testDeterministicSparseSystems();
    std::cout << "HiPO LDLT tests passed\n";
    return EXIT_SUCCESS;
}
