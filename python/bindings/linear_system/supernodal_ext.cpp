// Python bindings for the supernodal LDLT solver.
#include "linear_system/ldlt/supernodal_ldlt.h"
#include "linear_system/eigen_interop/supernodal_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using namespace supernodal;

namespace {

using SpMat = Eigen::SparseMatrix<double>;

SpMat denseToSparse(const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
    return matrix.sparseView(0.0, 0.0);
}

nb::dict diagnostics(const SupernodalLDLT<> &solver) {
    const auto &f = solver.panelFactors();
    const auto symbolic = solver.symbolic();
    nb::dict out;
    out["info"] = static_cast<int>(f.info_val);
    out["factorized"] = f.factorized;
    out["size"] = solver.size();
    out["nonzeros_l"] = solver.nonZerosL();
    out["scalar_csc_materialized"] = solver.scalarCscMaterialized();
    out["perturbed_pivots"] = f.perturbed_pivots;
    out["min_abs_pivot"] = f.min_abs_pivot;
    out["backend"] =
        !symbolic ? "uninitialized"
                  : (symbolic->backend() == SymbolicLDLT::Backend::Supernodal
                         ? "supernodal"
                         : "simplicial");
    out["panel_storage"] = f.panel_values.size();
    out["pattern_hash"] = symbolic ? symbolic->patternHash() : 0;
    out["pivoted"] = f.pivoted;
    out["intranodal_pivoted"] = f.intranodal_pivoted;
    out["contributor_count"] =
        symbolic ? symbolic->contributors().size() : 0;
    out["positive_inertia"] = f.positive_inertia;
    out["negative_inertia"] = f.negative_inertia;
    out["zero_inertia"] = f.zero_inertia;
    out["symbolic_intensity"] =
        symbolic ? symbolic->symbolicIntensity() : 0.0;
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

    Eigen::MatrixXd
    solveMultiple(const Eigen::Ref<const Eigen::MatrixXd> &rhs) const {
        if (rhs.rows() != solver_.size())
            throw std::invalid_argument("supernodal: RHS row count mismatch");
        std::vector<double> packed(static_cast<size_t>(rhs.size()));
        for (Eigen::Index col = 0; col < rhs.cols(); ++col)
            for (Eigen::Index row = 0; row < rhs.rows(); ++row)
                packed[static_cast<size_t>(col * rhs.rows() + row)] =
                    rhs(row, col);
        const auto solution =
            solver_.solveMultiple(packed, static_cast<int>(rhs.cols()));
        Eigen::MatrixXd result(rhs.rows(), rhs.cols());
        for (Eigen::Index col = 0; col < rhs.cols(); ++col)
            for (Eigen::Index row = 0; row < rhs.rows(); ++row)
                result(row, col) =
                    solution[static_cast<size_t>(col * rhs.rows() + row)];
        return result;
    }

    nb::dict info() const { return diagnostics(solver_); }

    int rows() const { return static_cast<int>(solver_.size()); }

    void setRegularization(double eps) { solver_.setRegularization(eps); }
    void setRelativeRegularization(double eps) {
        solver_.setRelativeRegularization(eps);
    }
    void setStrictPivots(bool strict) { solver_.setStrictPivots(strict); }
    void setExpectedPivotSigns(const std::vector<int8_t> &signs) {
        solver_.setExpectedPivotSigns(signs);
    }
    void setBackend(const std::string &backend) {
        if (backend == "automatic" || backend == "auto")
            solver_.setBackendPolicy(SupernodalLDLT<>::BackendPolicy::Automatic);
        else if (backend == "simplicial")
            solver_.setBackendPolicy(
                SupernodalLDLT<>::BackendPolicy::ForceSimplicial);
        else if (backend == "supernodal")
            solver_.setBackendPolicy(
                SupernodalLDLT<>::BackendPolicy::ForceSupernodal);
        else
            throw std::invalid_argument(
                "backend must be 'automatic', 'simplicial', or 'supernodal'");
    }
    void setSupernodalThreshold(double threshold) {
        solver_.setSupernodalThreshold(threshold);
    }
    void setPivotPolicy(const std::string &policy) {
        if (policy == "regularized" || policy == "1x1")
            solver_.setPivotPolicy(
                SupernodalLDLT<>::PivotPolicy::Regularized1x1);
        else if (policy == "intranodal" || policy == "intranodal-bk")
            solver_.setPivotPolicy(
                SupernodalLDLT<>::PivotPolicy::IntranodalBunchKaufman);
        else if (policy == "bunch-kaufman" || policy == "bk")
            solver_.setPivotPolicy(
                SupernodalLDLT<>::PivotPolicy::BunchKaufman);
        else
            throw std::invalid_argument(
                "pivot policy must be 'regularized', 'intranodal', or "
                "'bunch-kaufman'");
    }
    void setSymmetricStorage(const std::string &storage) {
        using Storage = SupernodalLDLT<>::SymmetricStorage;
        if (storage == "auto")
            solver_.setSymmetricStorage(Storage::AutoDetect);
        else if (storage == "upper")
            solver_.setSymmetricStorage(Storage::Upper);
        else if (storage == "lower")
            solver_.setSymmetricStorage(Storage::Lower);
        else if (storage == "full")
            solver_.setSymmetricStorage(Storage::FullSymmetric);
        else
            throw std::invalid_argument(
                "storage must be 'auto', 'upper', 'lower', or 'full'");
    }

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
    m.doc() = "Supernodal LDLT solver with dense BLAS-3 frontal updates";

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
        .def("solve_multiple", &SupernodalLDLTWrapper::solveMultiple,
             nb::arg("B"))
        .def("info", &SupernodalLDLTWrapper::info)
        .def_prop_ro("rows", &SupernodalLDLTWrapper::rows)
        .def("set_regularization", &SupernodalLDLTWrapper::setRegularization, nb::arg("eps"))
        .def("set_relative_regularization",
             &SupernodalLDLTWrapper::setRelativeRegularization, nb::arg("eps"))
        .def("set_strict_pivots", &SupernodalLDLTWrapper::setStrictPivots,
             nb::arg("strict"))
        .def("set_expected_pivot_signs",
             &SupernodalLDLTWrapper::setExpectedPivotSigns, nb::arg("signs"))
        .def("set_backend", &SupernodalLDLTWrapper::setBackend,
             nb::arg("backend"))
        .def("set_supernodal_threshold",
             &SupernodalLDLTWrapper::setSupernodalThreshold,
             nb::arg("threshold"))
        .def("set_pivot_policy", &SupernodalLDLTWrapper::setPivotPolicy,
             nb::arg("policy"))
        .def("set_symmetric_storage",
             &SupernodalLDLTWrapper::setSymmetricStorage,
             nb::arg("storage"))
        .def("supernode_ranges", &SupernodalLDLTWrapper::supernodeRanges,
             "Return list of (lo, hi) supernode column ranges.")
        .def("is_supernodal", &SupernodalLDLTWrapper::isSupernodal,
             "True if merged supernodes were detected in the matrix.");

    m.def("solve", &solveSparse, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from a sparse symmetric matrix.");
    m.def("solve", &solveDense, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from a dense symmetric matrix.");
}
