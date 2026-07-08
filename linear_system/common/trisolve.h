/*
 * trisolve.h — shared triangular-solve and rhs-permutation kernels for
 * LDL^T-family solvers.
 *
 * All kernels operate on a unit lower-triangular L stored as CSC with only
 * the strict lower part (Lp/Li/Lx); the implicit unit diagonal is not stored.
 * Diagonal (D) solves stay in each solver — that is where they differ
 * (scalar D, precomputed 1/D, Bunch-Kaufman 1x1/2x2 blocks).
 *
 * SIMD acceleration:
 *   - lsolve_unit: dispatches to ldlt::simd_sparse_macc (AVX-512/AVX2/SSE2)
 *                   for double/int32_t; scalar fallback otherwise.
 *   - ltsolve_unit: AVX-512 gather-accelerate for the scattered-read dot;
 *                   scalar for AVX2/SSE2/fallback.
 */

#ifndef LINSYS_TRISOLVE_H
#define LINSYS_TRISOLVE_H

#include <cstddef>
#include <limits>

#include "../ldlt/ldlt_simd.h"

namespace linsys {

// ── Forward substitution ────────────────────────────────────────────────────
// x <- L^{-1} x  (unit lower triangular)
// Pattern: for each column k, scatter x[Li[p]] -= Lx[p] * x[k]
//
// Dispatch to ldlt::simd_sparse_macc when Scalar==double, Index==int32_t:
//   - AVX-512: vectorized load/mul of 8 Lx values per iteration
//   - AVX2:    vectorized load/mul of 4 Lx values per iteration
//   - SSE2:    vectorized load/mul of 2 Lx values per iteration
//   - scalar:  pure C fallback

/// Forward substitution, in place: x <- L^{-1} x.
template < typename Scalar, typename Index >
inline void lsolve_unit(Index n, const Index* Lp, const Index* Li, const Scalar* Lx, Scalar* x) {
    for (Index k = 0; k < n; ++k) {
        const Scalar xk = x[static_cast< size_t >(k)];
        if (xk == Scalar(0))
            continue;

        const Index p0 = Lp[static_cast< size_t >(k)];
        const Index p1 = Lp[static_cast< size_t >(k + 1)];
        const Index len = p1 - p0;
        if (len <= 0)
            continue;

        // ── SIMD path: double + int32_t ──────────────────────────────
        if constexpr (std::is_same_v< Scalar, double > && std::is_same_v< Index, int32_t >) {
            ldlt::simd_sparse_macc(
                x, reinterpret_cast< const int32_t* >(Li) + static_cast< size_t >(p0),
                reinterpret_cast< const double* >(Lx) + static_cast< size_t >(p0),
                static_cast< size_t >(len), static_cast< double >(xk));
        } else {
            // ── Scalar path ──────────────────────────────────────────
            for (Index p = p0; p < p1; ++p) {
                x[static_cast< size_t >(Li[static_cast< size_t >(p)])] -=
                    Lx[static_cast< size_t >(p)] * xk;
            }
        }
    }
}

// ── Backward substitution ───────────────────────────────────────────────────
// x <- L^{-T} x  (unit lower triangular, transposed)
// Pattern: for each column k (reverse), gather x[Li[p]] dot Lx, accumulate
//
// AVX-512: vectorized gather for 8 x-values per iteration (gather_pd)
// AVX2:    no gather — scalar fallback

/// Backward substitution, in place: x <- L^{-T} x.
template < typename Scalar, typename Index >
inline void ltsolve_unit(Index n, const Index* Lp, const Index* Li, const Scalar* Lx, Scalar* x) {
    // ── Scalar path (all types) ──────────────────────────────────────
    if constexpr (!std::is_same_v< Scalar, double > || !std::is_same_v< Index, int32_t > ||
                  LDLT_HAS_AVX512 == 0) {
        for (Index k = n - 1; k >= 0; --k) {
            Scalar acc = x[static_cast< size_t >(k)];
            for (Index p = Lp[static_cast< size_t >(k)]; p < Lp[static_cast< size_t >(k) + 1];
                 ++p) {
                acc -= Lx[static_cast< size_t >(p)] *
                       x[static_cast< size_t >(Li[static_cast< size_t >(p)])];
            }
            x[static_cast< size_t >(k)] = acc;
        }
    } else {
        // ── AVX-512 gather-accelerated path ──────────────────────────
#if LDLT_HAS_AVX512
        const size_t SN = 8; // AVX-512 double width

        for (Index k = n - 1; k >= 0; --k) {
            const Index p0 = Lp[static_cast< size_t >(k)];
            const Index p1 = Lp[static_cast< size_t >(k) + 1];
            const size_t len = static_cast< size_t >(p1 - p0);

            double acc = x[static_cast< size_t >(k)];

            // Process 8 elements at a time via gather
            for (size_t i = 0; i + SN <= len; i += SN) {
                // Gather 8 x-values from scattered positions
                const __m512i vidx = _mm512_loadu_si512(
                    reinterpret_cast< const __m512i* >(Li + static_cast< size_t >(p0 + i)));
                __m512d vgather = _mm512_i32gather_pd(vidx, x, 8);

                // Load 8 Lx values (contiguous)
                __m512d vLx = _mm512_loadu_pd(Lx + static_cast< size_t >(p0 + i));

                // Dot product: sum(vLx * vgather)
                __m512d vprod = _mm512_mul_pd(vLx, vgather);
                // Horizontal reduce: add pairs -> quad -> scalar
                __m512d vhi = _mm512_extractf64x4_pd(vprod, 1);
                __m512d vlo = _mm512_extractf64x4_pd(vprod, 0);
                __m512d vsum = _mm512_add_pd(vhi, vlo);
                vhi = _mm512_shuffle_pd(vsum, vsum, 1); // [0,0,3,3] -> [3,3,3,3]
                vsum = _mm512_add_pd(vsum, vhi);
                vhi = _mm512_shuffle_pd(vsum, vsum, 2); // [0,0,0,0]
                acc -= static_cast< double >(_mm512_cvtss_f32(vhi));
            }

            // Scalar tail
            for (size_t i = (len / SN) * SN; i < len; ++i) {
                acc -= Lx[static_cast< size_t >(p0 + i)] *
                       x[static_cast< size_t >(Li[static_cast< size_t >(p0 + i)])];
            }

            x[static_cast< size_t >(k)] = static_cast< Scalar >(acc);
        }
#else
        // Fallback: same scalar path as the outer branch
        for (Index k = n - 1; k >= 0; --k) {
            Scalar acc = x[static_cast< size_t >(k)];
            for (Index p = Lp[static_cast< size_t >(k)]; p < Lp[static_cast< size_t >(k) + 1];
                 ++p) {
                acc -= Lx[static_cast< size_t >(p)] *
                       x[static_cast< size_t >(Li[static_cast< size_t >(p)])];
            }
            x[static_cast< size_t >(k)] = acc;
        }
#endif
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
