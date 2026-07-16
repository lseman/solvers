#include "linear_system/klu/klu.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

using DenseMatrix = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >;

// Convert dense matrix to CSC format (column-major)
std::pair<std::vector<int>, std::pair<std::vector<int>, std::vector<double>>>
dense_to_csc(const Eigen::Ref< const DenseMatrix >& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("A must be square");
    }

    const int n = static_cast< int >(A.rows());
    std::vector< int > Ap(static_cast< size_t >(n) + 1, 0);
    std::vector< int > Ai;
    std::vector< double > Ax;

    // Convert to CSC format (column-major)
    for (int j = 0; j < n; ++j) {
        Ap[static_cast< size_t >(j)] = static_cast< int >(Ai.size());
        for (int i = 0; i < n; ++i) {
            const double value = A(i, j);
            if (value != 0.0) {
                Ai.push_back(i);
                Ax.push_back(value);
            }
        }
    }
    Ap[static_cast< size_t >(n)] = static_cast< int >(Ai.size());

    return {Ap, {Ai, Ax}};
}

nb::dict solve_dense(const Eigen::Ref< const DenseMatrix >& A,
                     const Eigen::Ref< const Eigen::VectorXd >& b) {
    auto [Ap, AiAx] = dense_to_csc(A);
    const auto& [Ai, Ax] = AiAx;

    if (static_cast< int >(b.size()) != static_cast< int >(Ap.size()) - 1) {
        throw std::invalid_argument("b dimension mismatch");
    }

    klu::SparseCSC<int> csc;
    csc.n = static_cast< int >(Ap.size()) - 1;
    csc.Ap = std::move(Ap);
    csc.Ai = std::move(Ai);
    csc.Ax = std::move(Ax);

    klu::Solver solver;
    auto sym = solver.analyze(csc);
    auto num = solver.factorize(csc, sym);

    std::vector< double > bv(b.data(), b.data() + b.size());
    auto x = solver.solve(num, bv);

    nb::dict out;
    out["x"] = Eigen::Map< Eigen::VectorXd >(x.data(), static_cast< Eigen::Index >(x.size()));
    out["n"] = num.n;
    out["nblocks"] = num.nblocks;
    out["nnz_L"] = num.lnz;
    out["nnz_U"] = num.unz;
    out["factorized"] = true;
    return out;
}

nb::dict solve_generic(nb::object A_obj, const Eigen::Ref< const Eigen::VectorXd >& rhs) {
    // Try scipy.sparse: has .toarray() method
    try {
        auto toarray = A_obj.attr("toarray");
        Eigen::MatrixXd A_dense = nb::cast< Eigen::MatrixXd >(toarray());
        return solve_dense(A_dense, rhs);
    } catch (...) {
        // Fall back: assume it's already castable (Eigen matrix)
        try {
            Eigen::MatrixXd A_dense = nb::cast< Eigen::MatrixXd >(A_obj);
            return solve_dense(A_dense, rhs);
        } catch (...) {
            nb::dict err;
            err["x"] = nb::list();
            err["error"] = "Could not convert matrix to dense or sparse format";
            return err;
        }
    }
}

} // namespace

NB_MODULE(klu, m) {
    m.doc() = "KLU-style sparse LU solver from scratch (for circuit simulation matrices)";

    nb::class_< klu::Solver >(m, "Solver")
        .def(nb::init<>())
        .def("analyze", &klu::Solver::analyze, nb::arg("A"),
             nb::rv_policy::reference_internal)
        .def("factorize", &klu::Solver::factorize, nb::arg("A"), nb::arg("sym"),
             nb::rv_policy::reference_internal)
        .def("solve", &klu::Solver::solve, nb::arg("num"), nb::arg("b"))
        .def("tsolve", &klu::Solver::tsolve, nb::arg("num"), nb::arg("b"))
        .def("set_pivot_tolerance", &klu::Solver::setPivotTolerance, nb::arg("tol"))
        .def("set_row_scaling", &klu::Solver::setRowScaling, nb::arg("scale"));

    m.def("solve", &solve_dense, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from a dense matrix using KLU-style sparse LU factorization.");
    m.def("solve", &solve_generic, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b (accepts dense, sparse Eigen, or scipy.sparse matrices).");
}
