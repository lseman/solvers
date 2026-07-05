/*
 * sparse_csc.h — shared compressed-sparse-column storage for the custom
 * (Eigen-free) linear-system solvers (ldlt, ldlt_bk, qdldl, ...).
 *
 * Plain CSC container with structural validation. Solver-specific invariants
 * (e.g. upper-triangular-only, explicit diagonal) are enforced by the solvers
 * themselves, not here.
 */

#ifndef LINSYS_SPARSE_CSC_H
#define LINSYS_SPARSE_CSC_H

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace linsys {

template < typename Scalar = double, typename Index = int32_t > struct SparseCSC {
    std::vector< Index > Ap;  // column pointers, size n+1
    std::vector< Index > Ai;  // row indices, size nnz
    std::vector< Scalar > Ax; // values, size nnz
    Index n = 0;

    SparseCSC() = default;
    SparseCSC(Index n_, std::vector< Index > Ap_, std::vector< Index > Ai_,
              std::vector< Scalar > Ax_)
        : Ap(std::move(Ap_)), Ai(std::move(Ai_)), Ax(std::move(Ax_)), n(n_) {
        if (n < 0)
            throw std::invalid_argument("linsys: n < 0");
        if (Ap.size() != static_cast< size_t >(n) + 1)
            throw std::invalid_argument("linsys: Ap size != n+1");
        auto nnz_ = Ap.back();
        if (static_cast< size_t >(nnz_) != Ai.size() || static_cast< size_t >(nnz_) != Ax.size())
            throw std::invalid_argument("linsys: Ai/Ax size mismatch");
    }

    [[nodiscard]] size_t nnz() const {
        return static_cast< size_t >(Ap.back());
    }
};

} // namespace linsys

#endif // LINSYS_SPARSE_CSC_H
