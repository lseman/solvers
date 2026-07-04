// ipm_module.cpp — Standalone IPM solver Python bindings (solvers submodule)
// Exposes IPMSolver, IPMSolution, and convenience wrappers
// Can be built independently for use in other projects

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ipm/IPSolver.h"

namespace nb = nanobind;

namespace {

struct IPMSolution {
    std::vector<double> primals;
    std::vector<double> duals;
    double objective = std::numeric_limits<double>::quiet_NaN();
    std::string status;
};

Eigen::VectorXd default_sense(int rows) { return Eigen::VectorXd::Ones(rows); }

Eigen::VectorXd numeric_sense(const Eigen::VectorXd& raw, int rows) {
    if (raw.size() != rows) {
        throw std::invalid_argument("ipm: sense length must match the number of rows in A");
    }
    return raw;
}

Eigen::VectorXd parse_sense(nb::object sense, int rows) {
    if (sense.is_none()) {
        return default_sense(rows);
    }
    if (nb::isinstance<nb::list>(sense) || nb::isinstance<nb::tuple>(sense)) {
        std::vector<std::string> tokens = nb::cast<std::vector<std::string>>(sense);
        if (static_cast<int>(tokens.size()) != rows) {
            throw std::invalid_argument("ipm: sense length must match the number of rows in A");
        }
        Eigen::VectorXd sense_vec = Eigen::VectorXd::Zero(rows);
        for (int i = 0; i < rows; ++i) {
            if (tokens[i] == "=" || tokens[i] == "==") {
                sense_vec(i) = 1.0;
            } else if (tokens[i] == ">=" || tokens[i] == ">") {
                sense_vec(i) = -1.0;
            }
        }
        return sense_vec;
    }
    return numeric_sense(nb::cast<Eigen::VectorXd>(sense), rows);
}

IPMSolution solve_ipm(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                      const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub, nb::object sense, double tol) {
    if (A.cols() != c.size() || A.rows() != b.size() || c.size() != lb.size() ||
        c.size() != ub.size()) {
        throw std::invalid_argument(
            "ipm: dimension mismatch — A.cols != c.size or A.rows != b.size or c.size != "
            "lb.size or c.size != ub.size");
    }

    Eigen::VectorXd parsed_sense = parse_sense(std::move(sense), A.rows());

    IPSolver solver;
    solver.solve(A, b, c, lb, ub, parsed_sense, tol);

    IPMSolution out{solver.getPrimals(), solver.getDuals(), solver.getObjective(), "ipm"};
    if (std::isfinite(out.objective)) {
        return out;
    }

    out.status = "ipm_nonfinite";
    return out;
}

IPMSolution solve_ipm_dense(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                            const Eigen::VectorXd& ub, nb::object sense, double tol) {
    return solve_ipm(A.sparseView(), b, c, lb, ub, std::move(sense), tol);
}

}  // namespace

NB_MODULE(ipm_solver, m) {
    m.doc() = "simplinho IPM solver bindings (standalone, reusable across projects)";

    nb::class_<IPMSolution>(m, "IPMSolution")
        .def_prop_ro("x", [](const IPMSolution& self) { return self.primals; })
        .def_prop_ro("primals",
                               [](const IPMSolution& self) { return self.primals; })
        .def_prop_ro("duals", [](const IPMSolution& self) { return self.duals; })
        .def_prop_ro("status", [](const IPMSolution& self) { return self.status; })
        .def_prop_ro("obj", [](const IPMSolution& self) { return self.objective; })
        .def_prop_ro("objective",
                               [](const IPMSolution& self) { return self.objective; });

    nb::class_<IPSolver>(m, "IPSolver")
        .def(nb::init<>())
        .def(
            "solve",
            [](IPSolver&, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               nb::object sense,
               double tol) { return solve_ipm_dense(A, b, c, lb, ub, std::move(sense), tol); },
            nb::arg("A"), nb::arg("b"), nb::arg("c"), nb::arg("lb"), nb::arg("ub"),
            nb::arg("sense") = nb::none(), nb::arg("tol") = 1e-8,
            "Solve min c^T x subject to A x =/<=/>= b and lb <= x <= ub (dense).")
        .def(
            "solve",
            [](IPSolver&, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               nb::object sense,
               double tol) { return solve_ipm(A, b, c, lb, ub, std::move(sense), tol); },
            nb::arg("A"), nb::arg("b"), nb::arg("c"), nb::arg("lb"), nb::arg("ub"),
            nb::arg("sense") = nb::none(), nb::arg("tol") = 1e-8,
            "Solve min c^T x subject to A x =/<=/>= b and lb <= x <= ub (sparse).");

    m.def("solve_ipm", &solve_ipm_dense, nb::arg("A"), nb::arg("b"), nb::arg("c"), nb::arg("lb"),
          nb::arg("ub"), nb::arg("sense") = nb::none(), nb::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a dense matrix.");
    m.def("solve_ipm", &solve_ipm, nb::arg("A"), nb::arg("b"), nb::arg("c"), nb::arg("lb"),
          nb::arg("ub"), nb::arg("sense") = nb::none(), nb::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a sparse matrix.");
}
