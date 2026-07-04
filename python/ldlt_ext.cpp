#include "linear_system/eigen_addon/ldlt_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace ldlt;

namespace {

using SpMat = Eigen::SparseMatrix<double>;

SpMat denseToSparse(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
  return matrix.sparseView(0.0, 0.0);
}

nb::dict diagnostics(const SimplicialLDLT<> &solver) {
  const auto &f = solver.factors();
  nb::dict out;
  out["info"] = static_cast<int>(f.info_val);
  out["factorized"] = f.factorized;
  out["size"] = solver.size();
  out["nonzeros_l"] = f.Li.size();
  out["perturbed_pivots"] = f.perturbed_pivots;
  out["min_abs_pivot"] = f.min_abs_pivot;
  return out;
}

class LDLTSolver {
public:
  LDLTSolver() = default;

  LDLTSolver(const SpMat &matrix) { computeSparse(matrix); }

  LDLTSolver(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
    computeDense(matrix);
  }

  LDLTSolver &computeSparse(const SpMat &matrix) {
    SparseCSC<> csc = ldlt::fromEigenSparse<double, int>(matrix);
    solver_.compute(csc);
    return *this;
  }

  LDLTSolver &computeDense(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
    SparseCSC<> csc = ldlt::fromDense<double, int>(matrix);
    solver_.compute(csc);
    return *this;
  }

  // Alias: compute() auto-detects dense and factorizes
  LDLTSolver &compute(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
    return computeDense(matrix);
  }

  Eigen::VectorXd solve(const Eigen::Ref<const Eigen::VectorXd> &rhs) const {
    return ldlt::solveEigen<double, int>(solver_, rhs);
  }

  nb::dict info() const { return diagnostics(solver_); }

  int rows() const { return static_cast<int>(solver_.size()); }

  void setRegularization(double eps) { solver_.setRegularization(eps); }

private:
  SimplicialLDLT<> solver_;
};

nb::dict solveSparse(const SpMat &matrix,
                     const Eigen::Ref<const Eigen::VectorXd> &rhs) {
  LDLTSolver solver(matrix);
  Eigen::VectorXd x = solver.solve(rhs);

  nb::dict out = solver.info();
  out["x"] = x;
  return out;
}

nb::dict solveDense(const Eigen::Ref<const Eigen::MatrixXd> &matrix,
                    const Eigen::Ref<const Eigen::VectorXd> &rhs) {
  LDLTSolver solver(matrix);
  Eigen::VectorXd x = solver.solve(rhs);

  nb::dict out = solver.info();
  out["x"] = x;
  return out;
}

} // namespace

nb::dict solve_generic(nb::object A_obj, const Eigen::Ref<const Eigen::VectorXd> &rhs) {
  // Try scipy.sparse: has .toarray() method
  try {
    auto toarray = A_obj.attr("toarray");
    Eigen::MatrixXd A_dense = nb::cast<Eigen::MatrixXd>(toarray());
    return solveDense(A_dense, rhs);
  } catch (...) {
    // Fall back: assume it's already castable (Eigen matrix)
    try {
      Eigen::MatrixXd A_dense = nb::cast<Eigen::MatrixXd>(A_obj);
      return solveDense(A_dense, rhs);
    } catch (...) {
      nb::dict err;
      err["x"] = nb::list();
      err["error"] = "Could not convert matrix to dense or sparse format";
      return err;
    }
  }
}

NB_MODULE(ldlt, m) {
  m.doc() = "Eigen-free LDLT solver (standalone, no Eigen dependency)";

  nb::class_<LDLTSolver>(m, "LDLTSolver")
      .def(nb::init<>())
      .def(nb::init<const SpMat &>(), nb::arg("A"))
      .def(nb::init<const Eigen::Ref<const Eigen::MatrixXd> &>(), nb::arg("A"))
      .def("compute_sparse", &LDLTSolver::computeSparse, nb::arg("A"),
           nb::rv_policy::reference_internal)
      .def("compute_dense", &LDLTSolver::computeDense, nb::arg("A"),
           nb::rv_policy::reference_internal)
      .def("compute", &LDLTSolver::compute, nb::arg("A"),
           nb::rv_policy::reference_internal)
      .def("solve", &LDLTSolver::solve, nb::arg("b"))
      .def("info", &LDLTSolver::info)
      .def_prop_ro("rows", &LDLTSolver::rows)
      .def("set_regularization", &LDLTSolver::setRegularization, nb::arg("eps"));

  m.def("solve", &solveSparse, nb::arg("A"), nb::arg("b"),
        "Solve Ax=b from a sparse symmetric matrix.");
  m.def("solve", &solveDense, nb::arg("A"), nb::arg("b"),
        "Solve Ax=b from a dense symmetric matrix.");
  m.def("solve", &solve_generic, nb::arg("A"), nb::arg("b"),
        "Solve Ax=b (accepts dense, sparse Eigen, or scipy.sparse matrices).");
}
