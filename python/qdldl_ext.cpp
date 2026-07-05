#include "linear_system/qdldl/qdldl.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

using DenseMatrix = Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >;

qdldl23::SparseD32 dense_upper_to_csc(const Eigen::Ref< const DenseMatrix >& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("A must be square");
    }

    const int n = static_cast< int >(A.rows());
    std::vector< int32_t > Ap(static_cast< size_t >(n) + 1, 0);
    std::vector< int32_t > Ai;
    std::vector< double > Ax;

    for (int j = 0; j < n; ++j) {
        Ap[static_cast< size_t >(j)] = static_cast< int32_t >(Ai.size());
        for (int i = 0; i <= j; ++i) {
            const double value = A(i, j);
            if (value != 0.0 || i == j) {
                Ai.push_back(i);
                Ax.push_back(value);
            }
        }
    }
    Ap[static_cast< size_t >(n)] = static_cast< int32_t >(Ai.size());

    qdldl23::SparseD32 mat;
    mat.n = n;
    mat.Ap = std::move(Ap);
    mat.Ai = std::move(Ai);
    mat.Ax = std::move(Ax);
    linsys::finalize_upper_inplace(mat);
    return mat;
}

nb::dict solve_dense(const Eigen::Ref< const DenseMatrix >& A,
                     const Eigen::Ref< const Eigen::VectorXd >& b) {
    auto csc = dense_upper_to_csc(A);
    if (b.size() != csc.n) {
        throw std::invalid_argument("b dimension mismatch");
    }

    auto symbolic = qdldl23::analyze_fast(csc);
    auto factors = qdldl23::refactorize(csc, symbolic);

    Eigen::VectorXd x = b;
    qdldl23::solve(factors, x.data());

    nb::dict out;
    out["x"] = x;
    out["n"] = csc.n;
    out["nnz"] = csc.nnz();
    return out;
}

} // namespace

NB_MODULE(qdldl, m) {
    m.doc() = "nanobind wrappers for QDLDL-style sparse LDLT factorization";

    m.def("solve", &solve_dense, nb::arg("A"), nb::arg("b"),
          "Solve Ax=b from the upper triangle of a dense symmetric matrix.");
}
