// ipm_module.cpp — Standalone IPM solver Python bindings (solvers submodule)
// Exposes IPMSolver, IPMSolution, and convenience wrappers
// Can be built independently for use in other projects

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../ipm/IPSolver.h"

namespace py = pybind11;

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

Eigen::VectorXd parse_sense(py::object sense, int rows) {
    if (sense.is_none()) {
        return default_sense(rows);
    }
    if (py::isinstance<py::array>(sense)) {
        return numeric_sense(sense.cast<Eigen::VectorXd>(), rows);
    }
    std::vector<std::string> tokens = sense.cast<std::vector<std::string>>();
    if (static_cast<int>(tokens.size()) != rows) {
        throw std::invalid_argument("ipm: sense length must match the number of rows in A");
    }
    Eigen::VectorXd sense_vec = Eigen::VectorXd::Zero(rows);
    for (int i = 0; i < rows; ++i) {
        if (tokens[i] == "=" || tokens[i] == "==") {
            sense_vec(i) = 1.0;
        }
    }
    return sense_vec;
}

IPMSolution solve_ipm(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                      const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub, py::object sense, double tol) {
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
                            const Eigen::VectorXd& ub, py::object sense, double tol) {
    return solve_ipm(A.sparseView(), b, c, lb, ub, std::move(sense), tol);
}

}  // namespace

PYBIND11_MODULE(ipm_solver, m) {
    m.doc() = "simplinho IPM solver bindings (standalone, reusable across projects)";

    py::class_<IPMSolution>(m, "IPMSolution")
        .def_property_readonly("x", [](const IPMSolution& self) { return self.primals; })
        .def_property_readonly("primals",
                               [](const IPMSolution& self) { return self.primals; })
        .def_property_readonly("duals", [](const IPMSolution& self) { return self.duals; })
        .def_property_readonly("status", [](const IPMSolution& self) { return self.status; })
        .def_property_readonly("obj", [](const IPMSolution& self) { return self.objective; })
        .def_property_readonly("objective",
                               [](const IPMSolution& self) { return self.objective; });

    py::class_<IPSolver>(m, "IPSolver")
        .def(py::init<>())
        .def(
            "solve",
            [](IPSolver&, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               py::object sense,
               double tol) { return solve_ipm_dense(A, b, c, lb, ub, std::move(sense), tol); },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"), py::arg("ub"),
            py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
            "Solve min c^T x subject to A x =/<=/>= b and lb <= x <= ub (dense).")
        .def(
            "solve",
            [](IPSolver&, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               py::object sense,
               double tol) { return solve_ipm(A, b, c, lb, ub, std::move(sense), tol); },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"), py::arg("ub"),
            py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
            "Solve min c^T x subject to A x =/<=/>= b and lb <= x <= ub (sparse).");

    m.def("solve_ipm", &solve_ipm_dense, py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"),
          py::arg("ub"), py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a dense matrix.");
    m.def("solve_ipm", &solve_ipm, py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"),
          py::arg("ub"), py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a sparse matrix.");
}
