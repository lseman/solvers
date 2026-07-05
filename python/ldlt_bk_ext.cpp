#include "linear_system/ldlt/ldlt_bk_eigen_interop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ldlt;

NB_MODULE(ldlt_bk_ext, m) {
  m.doc() = "Bunch-Kaufman LDL^T factorization for symmetric indefinite matrices";

  // BunchKaufmanFactors info
  nb::class_<BunchKaufmanFactors>(m, "BunchKaufmanFactors")
      .def_ro("n", &BunchKaufmanFactors::n, "Matrix size")
      .def_ro("num_pos", &BunchKaufmanFactors::num_pos, "# positive eigenvalues")
      .def_ro("num_neg", &BunchKaufmanFactors::num_neg, "# negative eigenvalues")
      .def_ro("num_zero", &BunchKaufmanFactors::num_zero, "# zero eigenvalues")
      .def_ro("factorized", &BunchKaufmanFactors::factorized, "Factorization complete");

  // Main solver class
  nb::class_<BunchKaufmanLDLT>(m, "BunchKaufmanLDLT")
      .def(nb::init<>(), "Create empty solver")
      .def("compute",
           [](BunchKaufmanLDLT &self, const Eigen::SparseMatrix<double> &A) {
             SparseCSC<double, int32_t> csc = ldlt::eigen_to_csc(A);
             self.compute(csc);
           },
           nb::arg("A"), "Factorize sparse matrix")
      .def("solve",
           [](const BunchKaufmanLDLT &self, const Eigen::VectorXd &b) -> Eigen::VectorXd {
             std::vector<double> bv(b.data(), b.data() + b.size());
             auto x = self.solve(bv);
             return Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
           },
           nb::arg("b"), "Solve Ax=b")
      .def("info",
           [](const BunchKaufmanLDLT &self) {
             nb::dict info;
             const auto &factors = self.factors();
             info["n"] = factors.n;
             info["num_pos"] = factors.num_pos;
             info["num_neg"] = factors.num_neg;
             info["num_zero"] = factors.num_zero;
             info["factorized"] = factors.factorized;
             info["inertia"] = nb::make_tuple(factors.num_pos, factors.num_neg,
                                              factors.num_zero);
             return info;
           },
           "Return factorization info dict")
      .def("set_pivot_tolerance",
           &BunchKaufmanLDLT::setPivotTolerance, nb::arg("tol"),
           "Set pivot tolerance");

  // Convenience function: direct solve
  m.def(
      "solve",
      [](const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b)
          -> nb::dict {
        BunchKaufmanLDLT solver;
        SparseCSC<double, int32_t> csc = ldlt::eigen_to_csc(A);
        solver.compute(csc);

        std::vector<double> bv(b.data(), b.data() + b.size());
        auto x = solver.solve(bv);

        nb::dict result;
        result["x"] = Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
        const auto &factors = solver.factors();
        result["factorized"] = factors.factorized;
        result["inertia"] = nb::make_tuple(factors.num_pos, factors.num_neg,
                                           factors.num_zero);
        return result;
      },
      nb::arg("A"), nb::arg("b"),
      "Factorize A and solve Ax=b. Return {x, factorized, inertia}");
}
