/*
 * trisolve.h — shared triangular-solve and rhs-permutation kernels for
 * LDL^T-family solvers.
 *
 * All kernels operate on a unit lower-triangular L stored as CSC with only
 * the strict lower part (Lp/Li/Lx); the implicit unit diagonal is not stored.
 * Diagonal (D) solves stay in each solver — that is where they differ
 * (scalar D, precomputed 1/D, Bunch-Kaufman 1x1/2x2 blocks).
 */

#ifndef LINSYS_TRISOLVE_H
#define LINSYS_TRISOLVE_H

#include <cstddef>

namespace linsys {

/// Forward substitution, in place: x <- L^{-1} x.
template < typename Scalar, typename Index >
inline void lsolve_unit(Index n, const Index* Lp, const Index* Li, const Scalar* Lx, Scalar* x) {
    for (Index k = 0; k < n; ++k) {
        const Scalar xk = x[static_cast< size_t >(k)];
        if (xk == Scalar(0))
            continue;
        for (Index p = Lp[static_cast< size_t >(k)]; p < Lp[static_cast< size_t >(k) + 1]; ++p) {
            x[static_cast< size_t >(Li[static_cast< size_t >(p)])] -=
                Lx[static_cast< size_t >(p)] * xk;
        }
    }
}

/// Backward substitution, in place: x <- L^{-T} x.
template < typename Scalar, typename Index >
inline void ltsolve_unit(Index n, const Index* Lp, const Index* Li, const Scalar* Lx, Scalar* x) {
    for (Index k = n - 1; k >= 0; --k) {
        Scalar acc = x[static_cast< size_t >(k)];
        for (Index p = Lp[static_cast< size_t >(k)]; p < Lp[static_cast< size_t >(k) + 1]; ++p) {
            acc -= Lx[static_cast< size_t >(p)] *
                   x[static_cast< size_t >(Li[static_cast< size_t >(p)])];
        }
        x[static_cast< size_t >(k)] = acc;
    }
}

/// Gather: out[i] = in[map[i]].
template < typename Scalar, typename Index >
inline void permute_gather(Index n, const Index* map, const Scalar* in, Scalar* out) {
    for (Index i = 0; i < n; ++i)
        out[static_cast< size_t >(i)] = in[static_cast< size_t >(map[static_cast< size_t >(i)])];
}

/// Scatter: out[map[i]] = in[i].
template < typename Scalar, typename Index >
inline void permute_scatter(Index n, const Index* map, const Scalar* in, Scalar* out) {
    for (Index i = 0; i < n; ++i)
        out[static_cast< size_t >(map[static_cast< size_t >(i)])] = in[static_cast< size_t >(i)];
}

} // namespace linsys

#endif // LINSYS_TRISOLVE_H
