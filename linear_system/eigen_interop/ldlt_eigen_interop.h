#ifndef LDLT_EIGEN_INTEROP_H
#define LDLT_EIGEN_INTEROP_H

// Optional Eigen interoperability for ldlt.h.
// Include this header ONLY if Eigen is available. ldlt.h core itself is Eigen-free.
// Usage: #include <ldlt_eigen_interop.h>  (requires Eigen headers already included)

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "../ldlt/ldlt.h"

namespace ldlt {

template < typename Scalar, typename Index >
Eigen::VectorXd solveEigen(const SimplicialLDLT< Scalar, Index >& ldlt_solver,
                           const Eigen::VectorXd& b) {
    std::vector< Scalar > bv(static_cast< size_t >(b.size()));
    for (int i = 0; i < b.size(); ++i)
        bv[static_cast< size_t >(i)] = b[i];
    auto x = ldlt_solver.solve(bv);
    Eigen::VectorXd result(b.size());
    for (int i = 0; i < b.size(); ++i)
        result[i] = x[static_cast< size_t >(i)];
    return result;
}

template < typename Scalar, typename Index >
SparseCSC< Scalar, Index > fromEigenSparse(const Eigen::SparseMatrix< Scalar >& mat) {
    SparseCSC< Scalar, Index > result;
    result.n = static_cast< Index >(mat.rows());

    std::vector< std::tuple< Index, Index, Scalar > > trips;
    trips.reserve(static_cast< size_t >(mat.nonZeros()));
    for (typename Eigen::SparseMatrix< Scalar >::Index j = 0; j < mat.cols(); ++j) {
        for (typename Eigen::SparseMatrix< Scalar >::InnerIterator it(mat, j); it; ++it) {
            trips.emplace_back(static_cast< Index >(it.row()), static_cast< Index >(j),
                               static_cast< Scalar >(it.value()));
        }
    }

    std::sort(trips.begin(), trips.end(), [](const auto& a, const auto& b) {
        if (std::get< 1 >(a) != std::get< 1 >(b))
            return std::get< 1 >(a) < std::get< 1 >(b);
        return std::get< 0 >(a) < std::get< 0 >(b);
    });

    std::vector< Index > Ai;
    std::vector< Scalar > Ax;
    Ai.reserve(trips.size());
    Ax.reserve(trips.size());

    for (size_t i = 0; i < trips.size(); i++) {
        const auto& t = trips[i];
        if (i > 0 && std::get< 0 >(t) == std::get< 0 >(trips[i - 1]) &&
            std::get< 1 >(t) == std::get< 1 >(trips[i - 1])) {
            Ax.back() += std::get< 2 >(t);
        } else {
            Ai.push_back(std::get< 0 >(t));
            Ax.push_back(std::get< 2 >(t));
        }
    }

    result.Ap.assign(static_cast< size_t >(result.n) + 1, 0);
    for (const auto& t : trips) {
        result.Ap[static_cast< size_t >(std::get< 1 >(t)) + 1]++;
    }
    for (Index j = 0; j < result.n; ++j) {
        result.Ap[static_cast< size_t >(j) + 1] += result.Ap[static_cast< size_t >(j)];
    }

    result.Ai.resize(static_cast< size_t >(result.nnz()));
    result.Ax.resize(static_cast< size_t >(result.nnz()));
    std::vector< Index > pos(static_cast< size_t >(result.n), 0);
    for (const auto& t : trips) {
        Index j = std::get< 1 >(t);
        Index p = result.Ap[static_cast< size_t >(j)] + pos[static_cast< size_t >(j)]++;
        result.Ai[static_cast< size_t >(p)] = std::get< 0 >(t);
        result.Ax[static_cast< size_t >(p)] = std::get< 2 >(t);
    }

    return result;
}

template < typename Scalar, typename Index >
SparseCSC< Scalar, Index > fromDense(const Eigen::Matrix< Scalar, -1, -1 >& mat) {
    Eigen::SparseMatrix< Scalar > sparse = mat.sparseView(0.0, 0.0);
    return fromEigenSparse< Scalar, Index >(sparse);
}

} // namespace ldlt

#endif // LDLT_EIGEN_INTEROP_H
