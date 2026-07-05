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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
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
        return Ap.empty() ? 0 : static_cast< size_t >(Ap.back());
    }
};

// ---------------------------------------------------------------------------
// finalize_upper_inplace — enforce upper-triangular CSC invariant
// ---------------------------------------------------------------------------
// Coalesces duplicate entries in each column, checks that row indices
// satisfy i >= j (upper triangular), and (by default) requires an explicit
// diagonal element in every column.
//
// Throws std::runtime_error on invariant violations.

template < typename Scalar, typename Index >
void finalize_upper_inplace(SparseCSC< Scalar, Index >& A, bool require_diag = true) {
    using std::size_t;
    const Index n = A.n;
    if (n == 0)
        return;

    struct Pair {
        Index row;
        Scalar val;
    };
    auto col_size = static_cast< size_t >(A.Ap[1] - A.Ap[0]);
    std::vector< Pair > col_workspace;
    col_workspace.reserve(std::max(size_t{64}, A.nnz() / static_cast< size_t >(n)));

    const auto oldAp = A.Ap;
    std::vector< Index > newAp(static_cast< size_t >(n) + 1, Index{0});
    std::vector< Index > newAi;
    std::vector< Scalar > newAx;
    newAi.reserve(A.Ai.size());
    newAx.reserve(A.Ax.size());

    for (Index j = 0; j < n; ++j) {
        const Index p0 = oldAp[static_cast< size_t >(j)];
        const Index p1 = oldAp[static_cast< size_t >(j) + 1];

        if (p0 > p1)
            throw std::runtime_error("linsys::finalize_upper_inplace: Ap must be nondecreasing");
        if (p0 == p1)
            throw std::runtime_error("linsys::finalize_upper_inplace: empty column");

        const size_t sz = static_cast< size_t >(p1 - p0);
        col_workspace.clear();
        if (col_workspace.capacity() < sz)
            col_workspace.reserve(sz * 2);

        for (Index p = p0; p < p1; ++p) {
            const Index i = A.Ai[static_cast< size_t >(p)];
            if (i < 0 || i >= n)
                throw std::runtime_error("linsys::finalize_upper_inplace: row index OOB");
            if (i > j)
                throw std::runtime_error(
                    "linsys::finalize_upper_inplace: lower-triangular entry at col " +
                    std::to_string(static_cast< int >(j)));
            col_workspace.emplace_back(i, A.Ax[static_cast< size_t >(p)]);
        }

        std::sort(col_workspace.begin(), col_workspace.end(),
                  [](const Pair& a, const Pair& b) { return a.row < b.row; });

        bool has_diag = false;
        for (size_t k = 0; k < col_workspace.size();) {
            const Index r = col_workspace[k].row;
            Scalar sum = col_workspace[k].val;
            size_t k2 = k + 1;
            while (k2 < col_workspace.size() && col_workspace[k2].row == r) {
                sum += col_workspace[k2].val;
                ++k2;
            }
            newAi.push_back(r);
            newAx.push_back(sum);
            if (r == j)
                has_diag = true;
            k = k2;
        }
        newAp[static_cast< size_t >(j) + 1] = static_cast< Index >(newAi.size());

        if (require_diag && !has_diag)
            throw std::runtime_error(
                "linsys::finalize_upper_inplace: missing explicit diagonal at col " +
                std::to_string(static_cast< int >(j)));
    }

    A.Ap.swap(newAp);
    A.Ai.swap(newAi);
    A.Ax.swap(newAx);
    A.Ai.shrink_to_fit();
    A.Ax.shrink_to_fit();
}

} // namespace linsys

#endif // LINSYS_SPARSE_CSC_H
