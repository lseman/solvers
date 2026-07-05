#ifndef LDLT_BK_EIGEN_INTEROP_H
#define LDLT_BK_EIGEN_INTEROP_H

// Eigen interop for Bunch-Kaufman LDL^T
// Include this only if Eigen headers are available

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../ldlt/ldlt_bk.h"

namespace ldlt {

// Convert Eigen::SparseMatrix to standalone CSC
template < typename Scalar, typename StorageIndex >
inline SparseCSC< Scalar, int32_t >
eigen_to_csc(const Eigen::SparseMatrix< Scalar, Eigen::ColMajor, StorageIndex >& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("Matrix must be square");

    int32_t n = static_cast< int32_t >(A.rows());

    std::vector< int32_t > Ap(static_cast< size_t >(n) + 1);
    for (int32_t j = 0; j <= n; ++j)
        Ap[static_cast< size_t >(j)] =
            static_cast< int32_t >(A.outerIndexPtr()[static_cast< size_t >(j)]);

    std::vector< int32_t > Ai(A.nonZeros());
    for (int32_t i = 0; i < static_cast< int32_t >(A.nonZeros()); ++i)
        Ai[static_cast< size_t >(i)] =
            static_cast< int32_t >(A.innerIndexPtr()[static_cast< size_t >(i)]);

    std::vector< Scalar > Ax(A.valuePtr(), A.valuePtr() + A.nonZeros());

    return SparseCSC< Scalar, int32_t >(n, std::move(Ap), std::move(Ai), std::move(Ax));
}

// Convert dense Eigen matrix to CSC (sparseView then eigen_to_csc)
template < typename Scalar, typename Index >
inline SparseCSC< Scalar, int32_t > fromDense(const Eigen::Matrix< Scalar, -1, -1 >& mat) {
    Eigen::SparseMatrix< Scalar > sparse = mat.sparseView(Scalar(0.0), Scalar(0.0));
    return eigen_to_csc(sparse);
}

} // namespace ldlt

#endif // LDLT_BK_EIGEN_INTEROP_H
