#include "linear_system/ldlt/supernodal_ldlt.h"
#include "linear_system/eigen_interop/supernodal_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace supernodal;

namespace {

using SpMat = Eigen::SparseMatrix<double>;

SpMat denseToSparse(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
    return matrix.sparseView(0.0, 0.0);
}

nb::dict diagnostics(const SupernodalLDLT<> &solver) {
    const auto &f = solver.factors();
    nb::dict out;
    out["info"] = static_cast<int>(f.info_val);
    out["factorized"] = f.factorized;
    out["size"] = solver.size();
    out["nonzeros_l"] = static_cast<int>(f.Li.size());
    out["perturbed_pivots"] = f.perturbed_pivots;
    out["min_abs_pivot"] = f.min_abs_pivot;
    return out;
}

nb::list supernode_ranges_list(const std::vector< std::pair< Int, Int > > &ranges) {
    nb::list result;
    for (const auto &r : ranges) {
        result.append(nb::make_tuple(r.first, r.second));
    }
    return result;
}

class SupernodalLDLTWrapper {
public:
    SupernodalLDLTWrapper() = default;

    SupernodalLDLTWrapper(const SpMat &matrix) { computeSparse(matrix); }

    SupernodalLDLTWrapper(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
        computeDense(matrix);
    }

    SupernodalLDLTWrapper &computeSparse(const SpMat &matrix) {
        auto csc = supernodal::eigen_to_csc<double, int>(matrix);
        solver_.compute(csc);
        return *this;
    }

    SupernodalLDLTWrapper &computeDense(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
        SpMat sparse = matrix.sparseView(0.0, 0.0);
        auto csc = supernodal::eigen_to_csc<double, int>(sparse);
        solver_.compute(csc);
        return *this;
    }

    // Alias: compute() auto-detects dense and factorizes
    SupernodalLDLTWrapper &compute(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
        return computeDense(matrix);
    }

    Eigen::VectorXd solve(const Eigen::Ref<const Eigen::VectorXd> &rhs) const {
        return supernodal::solveEigen<double, int>(solver_, rhs);
    }

    nb::dict info() const { return diagnostics(solver_); }

    int rows() const { return static_cast<int>(solver_.size()); }

    void setRegularization(double eps) { solver_.setRegularization(eps); }

    void setSupernodalFactorization(bool on) { solver_.setSupernodalFactorization(on); }

    bool isSupernodalFactorizationEnabled() const { return solver_.supernodalFactorization(); }

    nb::list supernodeRanges() const {
        return supernode_ranges_list(solver_.supernodeRanges());
    }

    bool isSupernodal() const { return solver_.isSupernodal(); }

private:
    SupernodalLDLT<> solver_;
};

nb::dict solveSparse(const SpMat &matrix,
                     const Eigen::Ref<const Eigen::VectorXd> &rhs) {
    SupernodalLDLTWrapper solver(matrix);
    Eigen::VectorXd x = solver.solve(rhs);

    nb::dict out = solver.info();
    out["x"] = x;
    return out;
}

nb::dict solveDense(const Eigen::Ref<const Eigen::MatrixXd> &matrix,
                    const Eigen::Ref<const Eigen::VectorXd> &rhs) {
    SupernodalLDLTWrapper solver(matrix);
    Eigen::VectorXd x = solver.solve(rhs);

    nb::dict out = solver.info();
    out["x"] = x;
    return out;
}

} // namespace

NB_MODULE(supernodal, m) {
    m.doc() = "Supernodal LDLT solver (dense-BLAS on supernodes + simplicial fallback)";

    nb::class_<SupernodalLDLTWrapper>(m, "SupernodalLDLT")
        .def(nb::init<>())
        .def(nb::init<const SpMat &>(), nb::arg("A"))
        .def(nb::init<const Eigen::Ref<const Eigen::MatrixXd> &>(), nb::arg("A"))
        .def("compute_sparse", &SupernodalLDLTWrapper::computeSparse, nb::arg("A"),
             nb::rv_policy::reference_internal)
        .def("compute_dense", &SupernodalLDLTWrapper::computeDense, nb::arg("A"),
             nb::rv_policy::reference_internal)
        .def("compute", &SupernodalLDLTWrapper::compute, nb::arg("A"),
             nb::rv_policy::reference_internal)
        .def("solve", &SupernodalLDLTWrapper::solve, nb::arg("b"))
        .def("info", &SupernodalLDLTWrapper::info)
        .def_prop_ro("rows", &SupernodalLDLTWrapper::rows)
        .def("set_regularization", &SupernodalLDLTWrapper::setRegularization, nb::arg("eps"))
        .def("set_supernodal_factorization", &SupernodalLDLTWrapper::setSupernodalFactorization,
             nb::arg("on"), "Enable/disable supernodal dense-BLAS factorization (default: off).")
        .def("supernodal_factorization", &SupernodalLDLTWrapper::isSupernodalFactorizationEnabled)
        .def("supernode_ranges", &SupernodalLDLTWrapper::supernodeRanges,
             "Return list of (lo, hi) supernode column ranges.")
        .def("is_supernodal", &SupernodalLDLTWrapper::isSupernodal,
             "True if merged supernodes were detected in the matrix.");

    m.def("solve", &solveSparse, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from a sparse symmetric matrix.");
    m.def("solve", &solveDense, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from a dense symmetric matrix.");
}
