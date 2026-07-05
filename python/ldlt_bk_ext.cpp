#include "linear_system/ldlt/ldlt_bk_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ldlt;

namespace {

using SpMat = Eigen::SparseMatrix<double>;

nb::dict solve_dense(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    BunchKaufmanLDLT solver;
    SparseCSC<double, int32_t> csc = ldlt::fromDense<double, int>(A);
    solver.compute(csc);

    std::vector<double> bv(b.data(), b.data() + b.size());
    auto x = solver.solve(bv);

    nb::dict result;
    Eigen::VectorXd xv = Eigen::Map<Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(x.size()));
    result["x"] = xv;
    const auto& factors = solver.factors();
    result["factorized"] = factors.factorized;
    result["inertia"] = nb::make_tuple(factors.num_pos, factors.num_neg, factors.num_zero);
    return result;
}

nb::dict solve_generic(nb::object A_obj, const Eigen::Ref<const Eigen::VectorXd>& rhs) {
    // Check for scipy.sparse: has .toarray() method
    try {
        [[maybe_unused]] auto toarray = A_obj.attr("toarray");
        Eigen::MatrixXd A_dense = nb::cast<Eigen::MatrixXd>(toarray());
        return solve_dense(A_dense, rhs);
    } catch (const std::exception& e) {
        // Not scipy.sparse, fall back: numpy ndarray -> Eigen::MatrixXd
        try {
            Eigen::MatrixXd A_dense = nb::cast<Eigen::MatrixXd>(A_obj);
            return solve_dense(A_dense, rhs);
        } catch (const std::exception& e2) {
            nb::dict err;
            err["x"] = nb::list();
            err["factorized"] = false;
            err["inertia"] = nb::make_tuple(0, 0, 0);
            err["error"] = std::string("Cast failed: ") + e2.what();
            return err;
        }
    }
}

} // namespace

NB_MODULE(ldlt_bk, m) {
    m.doc() = "Bunch-Kaufman LDL^T factorization for symmetric indefinite matrices";

    // BunchKaufmanFactors info
    nb::class_< BunchKaufmanFactors >(m, "BunchKaufmanFactors")
        .def_ro("n", &BunchKaufmanFactors::n, "Matrix size")
        .def_ro("num_pos", &BunchKaufmanFactors::num_pos, "# positive eigenvalues")
        .def_ro("num_neg", &BunchKaufmanFactors::num_neg, "# negative eigenvalues")
        .def_ro("num_zero", &BunchKaufmanFactors::num_zero, "# zero eigenvalues")
        .def_ro("factorized", &BunchKaufmanFactors::factorized, "Factorization complete");

    // Main solver class
    nb::class_< BunchKaufmanLDLT >(m, "BunchKaufmanLDLT")
        .def(nb::init<>(), "Create empty solver")
        .def(
            "compute",
            [](BunchKaufmanLDLT& self, const Eigen::SparseMatrix< double >& A) {
                SparseCSC< double, int32_t > csc = ldlt::eigen_to_csc(A);
                self.compute(csc);
            },
            nb::arg("A"), "Factorize sparse matrix")
        .def(
            "solve",
            [](const BunchKaufmanLDLT& self, const Eigen::VectorXd& b) -> Eigen::VectorXd {
                std::vector< double > bv(b.data(), b.data() + b.size());
                auto x = self.solve(bv);
                return Eigen::VectorXd(Eigen::Map< Eigen::VectorXd >(
                    x.data(), static_cast< Eigen::Index >(x.size())));
            },
            nb::arg("b"), "Solve Ax=b")
        .def(
            "info",
            [](const BunchKaufmanLDLT& self) {
                nb::dict info;
                const auto& factors = self.factors();
                info["n"] = factors.n;
                info["num_pos"] = factors.num_pos;
                info["num_neg"] = factors.num_neg;
                info["num_zero"] = factors.num_zero;
                info["factorized"] = factors.factorized;
                info["inertia"] =
                    nb::make_tuple(factors.num_pos, factors.num_neg, factors.num_zero);
                return info;
            },
            "Return factorization info dict")
        .def("set_pivot_tolerance", &BunchKaufmanLDLT::setPivotTolerance, nb::arg("tol"),
             "Set pivot tolerance");

    // Convenience function: direct solve (dense numpy/scipy first, then sparse Eigen)
    m.def(
        "solve",
        solve_generic,
        nb::arg("A"), nb::arg("b"),
        "Factorize A and solve Ax=b. Accepts dense ndarray, scipy.sparse, or Eigen::SparseMatrix. "
        "Return {x, factorized, inertia}");
    m.def(
        "solve",
        [](const Eigen::SparseMatrix< double >& A, const Eigen::VectorXd& b) -> nb::dict {
            BunchKaufmanLDLT solver;
            SparseCSC< double, int32_t > csc = ldlt::eigen_to_csc(A);
            solver.compute(csc);

            std::vector< double > bv(b.data(), b.data() + b.size());
            auto x = solver.solve(bv);

            nb::dict result;
            Eigen::VectorXd xv = Eigen::Map< Eigen::VectorXd >(
                x.data(), static_cast< Eigen::Index >(x.size()));
            result["x"] = xv;
            const auto& factors = solver.factors();
            result["factorized"] = factors.factorized;
            result["inertia"] = nb::make_tuple(factors.num_pos, factors.num_neg, factors.num_zero);
            return result;
        },
        nb::arg("A"), nb::arg("b"), "Factorize Eigen::SparseMatrix and solve Ax=b. Return {x, factorized, inertia}");
}
