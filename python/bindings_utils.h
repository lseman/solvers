#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

namespace solvers_py {

using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Vector = Eigen::VectorXd;

inline SparseMatrix dense_to_sparse(const Eigen::Ref<const DenseMatrix>& dense) {
    return dense.sparseView();
}

inline std::optional<SparseMatrix> optional_dense_to_sparse(
    const std::optional<Eigen::Ref<const DenseMatrix>>& dense) {
    if (!dense) {
        return std::nullopt;
    }
    return dense_to_sparse(*dense);
}

inline nanobind::dict vector_result(const Vector& v) {
    nanobind::dict out;
    out["values"] = v;
    return out;
}

}  // namespace solvers_py
