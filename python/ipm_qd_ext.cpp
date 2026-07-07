// ipm_qd_ext.cpp — Quasi-definite augmented-system solver for IPM
// Wraps BK pivoting + iterative refinement per Zanetti & Gondzio (2025)

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../ipm/qd_ldlt.h"

namespace nb = nanobind;

namespace {

/**
 * Python binding for the QDLDLT augmented-system solver.
 *
 * Usage:
 *   solver = IPMQDLDLT()
 *   solver.analyze(A)           # A is m×n constraint matrix (once)
 *   solver.factorize(theta, regP, regD)  # each IPM iteration
 *   x = solver.solve(rhs)       # solve without refinement
 *   (x, niters) = solver.solve_refine(rhs, theta, regP, regD)
 */
struct IPMQDLDLT {
    ipm::QDLDLT solver;

    void analyze(const Eigen::SparseMatrix< double, Eigen::ColMajor >& A) {
        solver.analyzePattern(A);
    }

    void factorize(const Eigen::VectorXd& theta, const Eigen::VectorXd& regP,
                   const Eigen::VectorXd& regD) {
        solver.factorize(std::vector< double >(theta.data(), theta.data() + theta.size()),
                         std::vector< double >(regP.data(), regP.data() + regP.size()),
                         std::vector< double >(regD.data(), regD.data() + regD.size()));
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
        auto x = solver.solve(std::vector< double >(rhs.data(), rhs.data() + rhs.size()));
        return Eigen::Map< Eigen::VectorXd >(x.data(), static_cast< Eigen::Index >(x.size()));
    }

    std::tuple< Eigen::VectorXd, int > solve_refine(const Eigen::VectorXd& rhs,
                                                    const Eigen::VectorXd& theta,
                                                    const Eigen::VectorXd& regP,
                                                    const Eigen::VectorXd& regD) const {
        auto [x, niters] = solver.solveWithRefinement(
            std::vector< double >(rhs.data(), rhs.data() + rhs.size()),
            std::vector< double >(theta.data(), theta.data() + theta.size()),
            std::vector< double >(regP.data(), regP.data() + regP.size()),
            std::vector< double >(regD.data(), regD.data() + regD.size()));
        return std::make_tuple(
            Eigen::Map< Eigen::VectorXd >(x.data(), static_cast< Eigen::Index >(x.size())), niters);
    }

    int n() const {
        return solver.n();
    }
    int m() const {
        return solver.m();
    }
    int numPos() const {
        return solver.numPos();
    }
    int numNeg() const {
        return solver.numNeg();
    }
    int numZero() const {
        return solver.numZero();
    }
};

/**
 * Convenience: build augmented system from A, theta, regP, regD and solve.
 * Returns {dx, dy, inertia_pos, inertia_neg, refinement_iters}.
 */
nb::dict solve_augmented_system(const Eigen::SparseMatrix< double, Eigen::ColMajor >& A,
                                const Eigen::VectorXd& r7, const Eigen::VectorXd& r1,
                                const Eigen::VectorXd& theta, const Eigen::VectorXd& regP,
                                const Eigen::VectorXd& regD) {
    int n = r7.size();
    int m = r1.size();
    if (A.cols() != n || A.rows() != m) {
        throw std::invalid_argument("A.cols != n or A.rows != m");
    }

    ipm::QDLDLT solver;
    solver.analyzePattern(A);
    solver.factorize(std::vector< double >(theta.data(), theta.data() + theta.size()),
                     std::vector< double >(regP.data(), regP.data() + regP.size()),
                     std::vector< double >(regD.data(), regD.data() + regD.size()));

    // RHS: [r7; r1]
    Eigen::VectorXd rhs(n + m);
    rhs << r7, r1;

    auto [x, niters] =
        solver.solveWithRefinement(std::vector< double >(rhs.data(), rhs.data() + rhs.size()),
                                   std::vector< double >(theta.data(), theta.data() + theta.size()),
                                   std::vector< double >(regP.data(), regP.data() + regP.size()),
                                   std::vector< double >(regD.data(), regD.data() + regD.size()));

    Eigen::VectorXd dx(n);
    Eigen::VectorXd dy(m);
    for (int i = 0; i < n; i++)
        dx[i] = x[static_cast< size_t >(i)];
    for (int i = 0; i < m; i++)
        dy[i] = x[static_cast< size_t >(n + i)];

    nb::dict out;
    out["dx"] = dx;
    out["dy"] = dy;
    out["inertia_pos"] = solver.numPos();
    out["inertia_neg"] = solver.numNeg();
    out["refinement_iters"] = niters;
    return out;
}

} // namespace

NB_MODULE(ipm_qd, m) {
    m.doc() = "quasi-definite IPM augmented-system solver (BK pivoting + iterative refinement)";

    nb::class_< IPMQDLDLT >(m, "IPMQDLDLT")
        .def(nb::init<>())
        .def("analyze", &IPMQDLDLT::analyze, nb::arg("A"),
             "Analyze sparsity of constraint matrix A (once).")
        .def("factorize", &IPMQDLDLT::factorize, nb::arg("theta"), nb::arg("regP"), nb::arg("regD"),
             "Factorize augmented system with given theta/regP/regD.")
        .def("solve", &IPMQDLDLT::solve, nb::arg("rhs"), "Solve without iterative refinement.")
        .def("solve_refine", &IPMQDLDLT::solve_refine, nb::arg("rhs"), nb::arg("theta"),
             nb::arg("regP"), nb::arg("regD"), "Solve with iterative refinement per paper.")
        .def("n", &IPMQDLDLT::n, "Number of primal variables.")
        .def("m", &IPMQDLDLT::m, "Number of constraints.")
        .def("num_pos", &IPMQDLDLT::numPos, "Number of positive eigenvalues.")
        .def("num_neg", &IPMQDLDLT::numNeg, "Number of negative eigenvalues.")
        .def("num_zero", &IPMQDLDLT::numZero, "Number of zero eigenvalues.");

    m.def("solve_augmented_system", &solve_augmented_system, nb::arg("A"), nb::arg("r7"),
          nb::arg("r1"), nb::arg("theta"), nb::arg("regP"), nb::arg("regD"),
          "Build augmented system [-(θ⁻¹+Rp) Aᵀ; A Rd] and solve.");
}
