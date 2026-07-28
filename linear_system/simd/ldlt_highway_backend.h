// ldlt_highway_backend.h — Dense SIMD kernels via Google Highway with runtime ISA dispatch.
// Used when LDLT_WITH_HIGHWAY is defined. Sparse kernels keep scalar fallback.
//
// Usage: CMakeLists.txt must define -DLDTLT_WITH_HIGHWAY and link libhighway.
// At runtime, Highway auto-selects best ISA (AVX512/AVX2/SSE2/NEON/Scalar).

#ifndef LDLT_HIGHWAY_BACKEND_H
#define LDLT_HIGHWAY_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <hwy/highway.h>

// Sparse/transposed ops: keep intrinsics (Highway has no sparse scatter)
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define HWY_HAS_AVX512 1
#define HWY_HAS_AVX2 1
#define HWY_HAS_SSE2 1
#include <immintrin.h>
#elif defined(__AVX2__) && defined(__FMA__)
#define HWY_HAS_AVX512 0
#define HWY_HAS_AVX2 1
#define HWY_HAS_SSE2 1
#include <immintrin.h>
#elif defined(__SSE2__)
#define HWY_HAS_AVX512 0
#define HWY_HAS_AVX2 0
#define HWY_HAS_SSE2 1
#include <emmintrin.h>
#else
#define HWY_HAS_AVX512 0
#define HWY_HAS_AVX2 0
#define HWY_HAS_SSE2 0
#endif

namespace ldlt {
namespace hw {

// ── Helper: get lane count for ScalableTag<double> ─────────────────────────
inline size_t lanes_d() {
    hwy::ScalableTag< double > d;
    return hwy::Lanes(d);
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense kernels — Highway with runtime dispatch
// ═══════════════════════════════════════════════════════════════════════════

// Forward declarations for HWY_DYNAMIC_DISPATCH
void simd_scale_array(double* __restrict__ x, const double* __restrict__ scale, size_t n);
void simd_sparse_macc(double* __restrict__ x, const int32_t* __restrict__ rows,
                      const double* __restrict__ vals, size_t n, double scale);
void simd_dense_axpy(double* __restrict__ x, double alpha, const double* __restrict__ y, size_t n);
void simd_vector_add(double* __restrict__ result, const double* __restrict__ a,
                     const double* __restrict__ b, size_t n);
void simd_vector_sub(double* __restrict__ result, const double* __restrict__ a,
                     const double* __restrict__ b, size_t n);
void simd_swap_arrays(double* __restrict__ x, double* __restrict__ y, size_t n);
void simd_rank2_update(double* __restrict__ x, const double* __restrict__ a,
                       const double* __restrict__ b, double c0, double c1, size_t n);
void simd_transpose_square(double* __restrict__ m, size_t n);
void simd_transpose_square_copy(const double* __restrict__ src, double* __restrict__ dst, size_t n);
double simd_strided_dot(const double* __restrict__ a, const double* __restrict__ b, size_t n,
                        size_t stride);
void simd_strided_col_mirror(const double* __restrict__ row_src, double* __restrict__ col_dst,
                             size_t start, size_t len, size_t stride);
void simd_scatter_masked(double* __restrict__ out, const double* __restrict__ src,
                         const int32_t* __restrict__ indices, size_t n);
void simd_gather_permute(const double* __restrict__ src, const int32_t* __restrict__ indices,
                         double* __restrict__ out, size_t n);

// ── scale_array: x[i] *= scale[i] ──────────────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_scale_array) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto vx = hwy::Load(d, x + i);
        const auto vs = hwy::Load(d, scale + i);
        hwy::Store(hwy::Mul(vx, vs), d, x + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── sparse_macc: x[rows[i]] -= scale * vals[i] ─────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_sparse_macc) {
    const hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    const auto vscale = hwy::Set(d, -scale);
    for (size_t i = 0; i + l <= n; i += l) {
        auto vals = hwy::Load(d, &vals_arr[i]);
        auto prods = hwy::Mul(vals, vscale);
        for (size_t j = 0; j < l; ++j) {
            const int32_t row = rows[i + j];
            if (row >= 0) {
                out[static_cast< size_t >(row)] +=
                    hwy::GetLane(hwy::ExtractLane< size_t >(j, prods));
            }
        }
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── dense_axpy: x[i] -= alpha * y[i] ───────────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_dense_axpy) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    const auto valpha = hwy::Set(d, -alpha);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto vy = hwy::Load(d, y + i);
        auto vx = hwy::Load(d, x + i);
        hwy::Store(hwy::FMSub(valpha, vy, vx), d, x + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── vector_add: result[i] = a[i] + b[i] ─────────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_vector_add) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto va = hwy::Load(d, a + i);
        const auto vb = hwy::Load(d, b + i);
        hwy::Store(hwy::Add(va, vb), d, result + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── vector_sub: result[i] = a[i] - b[i] ─────────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_vector_sub) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto va = hwy::Load(d, a + i);
        const auto vb = hwy::Load(d, b + i);
        hwy::Store(hwy::Sub(va, vb), d, result + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── swap_arrays: swap x[i] <-> y[i] ─────────────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_swap_arrays) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto vx = hwy::Load(d, x + i);
        const auto vy = hwy::Load(d, y + i);
        hwy::Store(vy, d, x + i);
        hwy::Store(vx, d, y + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── rank2_update: x[i] -= a[i]*c0 + b[i]*c1 ─────────────────────────────────
HWY_DYNAMIC_DISPATCH(simd_rank2_update) {
    hwy::ScalableTag< double > d;
    const size_t l = hwy::Lanes(d);
    const auto vc0 = hwy::Set(d, c0);
    const auto vc1 = hwy::Set(d, c1);
    for (size_t i = 0; i + l <= n; i += l) {
        const auto va = hwy::Load(d, a + i);
        const auto vb = hwy::Load(d, b + i);
        auto vx = hwy::Load(d, x + i);
        hwy::Store(hwy::Sub(vx, hwy::FMSub(va, vc0, hwy::Mul(vb, vc1))), d, x + i);
    }
}
HWY_DYNAMIC_DISPATCH_END

// ── transpose_square: in-place transpose (8×8 blocked) ──────────────────────
void simd_transpose_square(double* __restrict__ m, size_t n) {
    for (size_t ii = 0; ii < n; ii += 8) {
        const size_t ir = std::min(static_cast< size_t >(8), n - ii);
        for (size_t jj = 0; jj < n; jj += 8) {
            const size_t jc = std::min(static_cast< size_t >(8), n - jj);
            for (size_t r = 0; r < ir; ++r)
                for (size_t c = 0; c < jc; ++c) {
                    const double tmp = m[(ii + r) * n + (jj + c)];
                    m[(ii + r) * n + (jj + c)] = m[(jj + c) * n + (ii + r)];
                    m[(jj + c) * n + (ii + r)] = tmp;
                }
        }
    }
}

// ── transpose_square_copy: copy column-major to row-major ───────────────────
void simd_transpose_square_copy(const double* __restrict__ src, double* __restrict__ dst,
                                size_t n) {
    for (size_t ii = 0; ii < n; ii += 8) {
        const size_t ir = std::min(static_cast< size_t >(8), n - ii);
        for (size_t jj = 0; jj < n; jj += 8) {
            const size_t jc = std::min(static_cast< size_t >(8), n - jj);
            for (size_t r = 0; r < ir; ++r)
                for (size_t c = 0; c < jc; ++c)
                    dst[r * jc + c] = src[(ii + r) * n + (jj + c)];
        }
    }
}

// ── strided dot product: Σ a[i*stride] * b[i*stride] ───────────────────────
double simd_strided_dot(const double* __restrict__ a, const double* __restrict__ b, size_t n,
                        size_t stride) {
    double sum = 0.0;
#if HWY_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    const __m512i idx512 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_i32gather_pd(&a[i * stride], idx512, 8);
        __m512d vb = _mm512_i32gather_pd(&b[i * stride], idx512, 8);
        __m512d vmul = _mm512_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(_mm512_extractf64x2_pd(vmul, 0));
        sum += _mm_cvtsd_f64(_mm512_extractf64x2_pd(vmul, 1));
    }
    for (size_t i = simd_end; i < n; ++i) {
        sum += a[i * stride] * b[i * stride];
    }
#elif HWY_HAS_AVX2
    const __m128i idx256 = _mm_set_epi32(3, 2, 1, 0);
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m256d va = _mm256_i32gather_pd(&a[i * stride], idx256, 8);
        __m256d vb = _mm256_i32gather_pd(&b[i * stride], idx256, 8);
        __m256d vmul = _mm256_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(_mm256_extractf128_pd(vmul, 0));
        sum += _mm_cvtsd_f64(_mm256_extractf128_pd(vmul, 1));
    }
    for (size_t i = (n / 4) * 4; i < n; ++i) {
        sum += a[i * stride] * b[i * stride];
    }
#elif HWY_HAS_SSE2
    for (size_t i = 0; i + 2 <= n; i += 2) {
        double va0 = a[i * stride], va1 = a[(i + 1) * stride];
        double vb0 = b[i * stride], vb1 = b[(i + 1) * stride];
        __m128d va = _mm_set_pd(va1, va0);
        __m128d vb = _mm_set_pd(vb1, vb0);
        __m128d vmul = _mm_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(vmul);
        __m128d vshuf = _mm_shuffle_pd(vmul, vmul, 1);
        sum += _mm_cvtsd_f64(vshuf);
    }
    for (size_t i = (n / 2) * 2; i < n; ++i) {
        sum += a[i * stride] * b[i * stride];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        sum += a[i * stride] * b[i * stride];
    }
#endif
    return sum;
}

// ── strided column mirror (row → column) ────────────────────────────────────
void simd_strided_col_mirror(const double* __restrict__ row_src, double* __restrict__ col_dst,
                             size_t start, size_t len, size_t stride) {
    const double* __restrict__ src = &row_src[start * stride];
    for (size_t i = 0; i < len; ++i) {
        col_dst[(start + i) * stride] = src[i];
    }
}

// ── masked scatter (AVX-512) / extract+scatter (AVX2/SSE2) ─────────────────
void simd_scatter_masked(double* __restrict__ out, const double* __restrict__ src,
                         const int32_t* __restrict__ indices, size_t n) {
    if (n == 0)
        return;

#if HWY_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512d vsrc = _mm512_loadu_pd(&src[i]);
        __m512i vidx = _mm512_loadu_epi32(&indices[i]);
        __m512i cmp = _mm512_cmpgt_epi32(vidx, _mm512_setzero_si512());
        unsigned mask = _mm512_movemask_ps(_mm512_castsi512_ps(cmp));
        _mm512_mask_i32scatter_pd(out, static_cast< unsigned char >(mask & 0xFF), &indices[i], vsrc,
                                  8);
    }
#elif HWY_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d vsrc = _mm256_loadu_pd(&src[i]);
        double vals[4];
        _mm256_storeu_pd(vals, vsrc);
        for (int k = 0; k < 4; ++k) {
            const int32_t idx = indices[i + k];
            if (idx >= 0) {
                out[static_cast< size_t >(idx)] = vals[k];
            }
        }
    }
#elif HWY_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (size_t i = 0; i < simd_end; i += 2) {
        const int32_t i0 = indices[i];
        const int32_t i1 = indices[i + 1];
        if (i0 >= 0)
            out[static_cast< size_t >(i0)] = src[i];
        if (i1 >= 0)
            out[static_cast< size_t >(i1)] = src[i + 1];
    }
#endif

    for (size_t i = simd_end; i < n; ++i) {
        const int32_t idx = indices[i];
        if (idx >= 0) {
            out[static_cast< size_t >(idx)] = src[i];
        }
    }
}

// ── gather permutation ─────────────────────────────────────────────────────
void simd_gather_permute(const double* __restrict__ src, const int32_t* __restrict__ indices,
                         double* __restrict__ out, size_t n) {
    if (n == 0)
        return;

#if HWY_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    const __m512i idx = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512i vperm = _mm512_loadu_epi32(&indices[i]);
        __m512d vgather = _mm512_i32gather_pd(src, vperm, 8);
        _mm512_storeu_pd(&out[i], vgather);
    }
#elif HWY_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    const __m128i idx4 = _mm_set_epi32(3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d vgather = _mm256_i32gather_pd(src, idx4, 8);
        _mm256_storeu_pd(&out[i], vgather);
    }
#elif HWY_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (size_t i = 0; i < simd_end; i += 2) {
        out[i] = src[static_cast< size_t >(indices[i])];
        out[i + 1] = src[static_cast< size_t >(indices[i + 1])];
    }
#endif

    for (size_t i = simd_end; i < n; ++i) {
        out[i] = src[static_cast< size_t >(indices[i])];
    }
}

} // namespace hw
} // namespace ldlt

#endif // LDLT_HIGHWAY_BACKEND_H
