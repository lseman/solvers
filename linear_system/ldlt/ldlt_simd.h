#ifndef LDLT_SIMD_H
#define LDLT_SIMD_H

// SIMD-accelerated kernels for ldlt.h: AVX-512, AVX2, SSE2 with scalar fallback.
// Mirrored from qdldl.h patterns for consistency.

#include <cstring>

// SIMD capability detection
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define LDLT_HAS_AVX512 1
#define LDLT_HAS_AVX2 1
#define LDLT_SIMD_WIDTH 8
#include <immintrin.h>
#elif defined(__AVX2__) && defined(__FMA__)
#define LDLT_HAS_AVX512 0
#define LDLT_HAS_AVX2 1
#define LDLT_SIMD_WIDTH 4
#include <immintrin.h>
#elif defined(__SSE2__)
#define LDLT_HAS_AVX512 0
#define LDLT_HAS_AVX2 0
#define LDLT_HAS_SSE2 1
#define LDLT_SIMD_WIDTH 2
#include <xmmintrin.h>
#else
#define LDLT_HAS_AVX512 0
#define LDLT_HAS_AVX2 0
#define LDLT_HAS_SSE2 0
#define LDLT_SIMD_WIDTH 1
#endif

#include <thread>
#include <vector>

namespace ldlt {

// Cache prefetching
inline void prefetch_read(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(static_cast< const char* >(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void* addr) {
#if defined(__SSE2__)
    _mm_prefetch(static_cast< const char* >(addr), _MM_HINT_T0);
#endif
}

// SIMD: scale array by vector (x[i] *= scale[i])
inline void simd_scale_array(double* __restrict__ x, const double* __restrict__ scale, size_t n) {
    size_t i = 0;

#if LDLT_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    for (; i < simd_end; i += 8) {
        if ((i & 63) == 0) {
            prefetch_read(&scale[i + 64]);
            prefetch_write(&x[i + 64]);
        }
        __m512d vx = _mm512_loadu_pd(&x[i]);
        __m512d vs = _mm512_loadu_pd(&scale[i]);
        vx = _mm512_mul_pd(vx, vs);
        _mm512_storeu_pd(&x[i], vx);
    }
#elif LDLT_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (; i < simd_end; i += 4) {
        if ((i & 31) == 0) {
            prefetch_read(&scale[i + 32]);
            prefetch_write(&x[i + 32]);
        }
        __m256d vx = _mm256_loadu_pd(&x[i]);
        __m256d vs = _mm256_loadu_pd(&scale[i]);
        vx = _mm256_mul_pd(vx, vs);
        _mm256_storeu_pd(&x[i], vx);
    }
#elif LDLT_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (; i < simd_end; i += 2) {
        __m128d vx = _mm_loadu_pd(&x[i]);
        __m128d vs = _mm_loadu_pd(&scale[i]);
        vx = _mm_mul_pd(vx, vs);
        _mm_storeu_pd(&x[i], vx);
    }
#endif

    for (; i < n; ++i)
        x[i] *= scale[i];
}

// SIMD: sparse multiply-accumulate for factorization kernels
// Computes: x[rows[i]] -= scale * vals[i] for contiguous i
// Used in forward/backward solves for small dense segments
inline void simd_sparse_macc(double* __restrict__ x, const int32_t* __restrict__ rows,
                             const double* __restrict__ vals, size_t n, double scale) {
    if (n == 0)
        return;

#if LDLT_HAS_AVX512
    // AVX-512: load vals in SIMD, broadcast scale, FMAD-sub to scattered x
    const size_t simd_end = n - (n % 8);
    const size_t prefetch_dist = 8;
    __m512d vscale = _mm512_set1_pd(-scale);

    for (size_t i = 0; i < simd_end; i += 8) {
        // Prefetch future x targets
        if (i + prefetch_dist < simd_end) {
            for (int k = 0; k < 4; ++k) {
                const int32_t r = rows[i + prefetch_dist + k];
                if (r >= 0)
                    prefetch_write(&x[static_cast< size_t >(r)]);
            }
        }

        // Load 8 values (SIMD width for double)
        __m512d vvals = _mm512_loadu_pd(&vals[i]);
        __m512d vprod = _mm512_mul_pd(vvals, vscale); // -scale * vals[i..i+7]

        // Store all 8 for scatter
        double tmp8[8];
        _mm512_storeu_pd(tmp8, vprod);
        for (int lane = 0; lane < 8; ++lane) {
            const int32_t row = rows[i + lane];
            if (row >= 0) {
                x[static_cast< size_t >(row)] += tmp8[lane];
            }
        }
    }

    // Scalar tail
    for (size_t i = simd_end; i < n; ++i) {
        const int32_t j = rows[i];
        if (j >= 0) {
            x[static_cast< size_t >(j)] -= scale * vals[i];
        }
    }
#elif LDLT_HAS_AVX2
    // AVX2: load vals in SIMD, broadcast scale, FMAD-sub to scattered x
    const size_t simd_end = n - (n % 4);
    const size_t prefetch_dist = 4;
    __m256d vscale = _mm256_set1_pd(-scale);

    for (size_t i = 0; i < simd_end; i += 4) {
        if (i + prefetch_dist < n) {
            for (int k = 0; k < 2; ++k) {
                const int32_t r = rows[i + prefetch_dist + k];
                if (r >= 0)
                    prefetch_write(&x[static_cast< size_t >(r)]);
            }
        }

        __m256d vvals = _mm256_loadu_pd(&vals[i]);
        __m256d vprod = _mm256_mul_pd(vvals, vscale);

        // Unpack and scatter — AVX2 doesn't have masked scatter,
        // so we extract and store individually (still wins via vectorized load+mul)
        double tmp[4];
        _mm256_storeu_pd(tmp, vprod);
        for (int lane = 0; lane < 4; ++lane) {
            const int32_t row = rows[i + lane];
            if (row >= 0) {
                x[static_cast< size_t >(row)] += tmp[lane];
            }
        }
    }

    // Scalar tail
    for (size_t i = simd_end; i < n; ++i) {
        const int32_t j = rows[i];
        if (j >= 0) {
            x[static_cast< size_t >(j)] -= scale * vals[i];
        }
    }
#elif LDLT_HAS_SSE2
    // SSE2: load 2 values at a time
    const size_t simd_end = n - (n % 2);
    __m128d vscale = _mm_set1_pd(-scale);

    for (size_t i = 0; i < simd_end; i += 2) {
        __m128d vvals = _mm_loadu_pd(&vals[i]);
        __m128d vprod = _mm_mul_pd(vvals, vscale);

        double tmp[2];
        _mm_storeu_pd(tmp, vprod);
        for (int lane = 0; lane < 2; ++lane) {
            const int32_t row = rows[i + lane];
            if (row >= 0) {
                x[static_cast< size_t >(row)] += tmp[lane];
            }
        }
    }

    // Scalar tail
    for (size_t i = simd_end; i < n; ++i) {
        const int32_t j = rows[i];
        if (j >= 0) {
            x[static_cast< size_t >(j)] -= scale * vals[i];
        }
    }
#else
    // Pure scalar fallback
    for (size_t i = 0; i < n; ++i) {
        const int32_t j = rows[i];
        if (j >= 0) {
            x[static_cast< size_t >(j)] -= scale * vals[i];
        }
    }
#endif
}

// SIMD: dense segment accumulation (contiguous scatter for column ops)
// Computes: x[k:k+n] -= alpha * y[p0:p1] where access is sequential
// Useful for dense frontal blocks in supernodal factorization
inline void simd_dense_axpy(double* __restrict__ x, double alpha, const double* __restrict__ y,
                            size_t n) {
    size_t i = 0;

#if LDLT_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    __m512d valpha = _mm512_set1_pd(-alpha);
    for (; i < simd_end; i += 8) {
        __m512d vy = _mm512_loadu_pd(&y[i]);
        __m512d vx = _mm512_loadu_pd(&x[i]);
        vx = _mm512_fmadd_pd(valpha, vy, vx);
        _mm512_storeu_pd(&x[i], vx);
    }
#elif LDLT_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    __m256d valpha = _mm256_set1_pd(-alpha);
    for (; i < simd_end; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y[i]);
        __m256d vx = _mm256_loadu_pd(&x[i]);
        vx = _mm256_fmadd_pd(valpha, vy, vx);
        _mm256_storeu_pd(&x[i], vx);
    }
#elif LDLT_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (; i < simd_end; i += 2) {
        __m128d vy = _mm_loadu_pd(&y[i]);
        __m128d vx = _mm_loadu_pd(&x[i]);
        __m128d vprod = _mm_mul_pd(vy, _mm_set1_pd(-alpha));
        vx = _mm_add_pd(vx, vprod);
        _mm_storeu_pd(&x[i], vx);
    }
#endif

    for (; i < n; ++i)
        x[i] -= alpha * y[i];
}

// Parallel for loop: divides n iterations across hardware threads
// Automatically serializes if n is small or only 1 thread available
// ── SIMD: strided dot product ──────────────────────────────────────────────
// Computes dot(a[0], a[stride], a[2*stride], ...) · dot(b[0], b[stride], ...)
inline double simd_strided_dot(const double* __restrict__ a, const double* __restrict__ b, size_t n,
                               size_t stride) {
    double sum = 0.0;
#if LDLT_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    const __m512i idx512 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_i32gather_pd(&a[i * stride], idx512, 8);
        __m512d vb = _mm512_i32gather_pd(&b[i * stride], idx512, 8);
        __m512d vmul = _mm512_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(_mm512_extractf64x2_pd(vmul, 0));
        sum += _mm_cvtsd_f64(_mm512_extractf64x2_pd(vmul, 1));
    }
#elif LDLT_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    const __m128i idx256 = _mm_set_epi32(3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d va = _mm256_i32gather_pd(&a[i * stride], idx256, 8);
        __m256d vb = _mm256_i32gather_pd(&b[i * stride], idx256, 8);
        __m256d vmul = _mm256_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(_mm256_extractf128_pd(vmul, 0));
        sum += _mm_cvtsd_f64(_mm256_extractf128_pd(vmul, 1));
    }
#elif LDLT_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (size_t i = 0; i < simd_end; i += 2) {
        double va0 = a[i * stride], va1 = a[(i + 1) * stride];
        double vb0 = b[i * stride], vb1 = b[(i + 1) * stride];
        __m128d va = _mm_set_pd(va1, va0);
        __m128d vb = _mm_set_pd(vb1, vb0);
        __m128d vmul = _mm_mul_pd(va, vb);
        sum += _mm_cvtsd_f64(vmul);
        __m128d vshuf = _mm_shuffle_pd(vmul, vmul, 1);
        sum += _mm_cvtsd_f64(vshuf);
    }
#endif

    for (size_t i = simd_end; i < n; ++i)
        sum += a[i * stride] * b[i * stride];
    return sum;
}

// ── SIMD: strided column mirror (row → column) ─────────────────────────────
inline void simd_strided_col_mirror(const double* __restrict__ row_src,
                                    double* __restrict__ col_dst, size_t start, size_t len,
                                    size_t stride) {
    const double* __restrict__ src = &row_src[start * stride];
    for (size_t i = 0; i < len; ++i)
        col_dst[(start + i) * stride] = src[i];
}

// ── SIMD: masked scatter ───────────────────────────────────────────────────
inline void simd_scatter_masked(double* __restrict__ out, const double* __restrict__ src,
                                const int32_t* __restrict__ indices, size_t n) {
    if (n == 0)
        return;
#if LDLT_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512d vsrc = _mm512_loadu_pd(&src[i]);
        __m512i vidx = _mm512_loadu_epi32(&indices[i]);
        __m512i cmp = _mm512_cmpgt_epi32(vidx, _mm512_setzero_si512());
        unsigned mask = _mm512_movemask_ps(_mm512_castsi512_ps(cmp));
        _mm512_mask_i32scatter_pd(out, static_cast< unsigned char >(mask & 0xFF), &indices[i], vsrc,
                                  8);
    }
#elif LDLT_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d vsrc = _mm256_loadu_pd(&src[i]);
        double vals[4];
        _mm256_storeu_pd(vals, vsrc);
        for (int k = 0; k < 4; ++k) {
            const int32_t idx = indices[i + k];
            if (idx >= 0)
                out[static_cast< size_t >(idx)] = vals[k];
        }
    }
#elif LDLT_HAS_SSE2
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
        if (idx >= 0)
            out[static_cast< size_t >(idx)] = src[i];
    }
}

// ── SIMD: gather permutation ───────────────────────────────────────────────
inline void simd_gather_permute(const double* __restrict__ src, const int32_t* __restrict__ indices,
                                double* __restrict__ out, size_t n) {
    if (n == 0)
        return;
#if LDLT_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    const __m512i idx = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 8) {
        __m512i vperm = _mm512_loadu_epi32(&indices[i]);
        __m512d vgather = _mm512_i32gather_pd(src, vperm, 8);
        _mm512_storeu_pd(&out[i], vgather);
    }
#elif LDLT_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    const __m128i idx4 = _mm_set_epi32(3, 2, 1, 0);
    for (size_t i = 0; i < simd_end; i += 4) {
        __m256d vgather = _mm256_i32gather_pd(src, idx4, 8);
        _mm256_storeu_pd(&out[i], vgather);
    }
#elif LDLT_HAS_SSE2
    const size_t simd_end = n - (n % 2);
    for (size_t i = 0; i < simd_end; i += 2) {
        out[i] = src[static_cast< size_t >(indices[i])];
        out[i + 1] = src[static_cast< size_t >(indices[i + 1])];
    }
#endif

    for (size_t i = simd_end; i < n; ++i)
        out[i] = src[static_cast< size_t >(indices[i])];
}

template < class F > inline void par_for_n(std::size_t n, F&& f) {
    constexpr std::size_t min_parallel_size = 1000;

    if (n < min_parallel_size) {
        for (std::size_t i = 0; i < n; ++i)
            f(i);
        return;
    }

    const auto num_threads = std::max(1u, std::thread::hardware_concurrency());
    const auto chunk_size = (n + num_threads - 1) / num_threads;

    if (num_threads > 1 && n > 2 * min_parallel_size) {
        std::vector< std::thread > threads;
        threads.reserve(num_threads);

        for (unsigned t = 0; t < num_threads; ++t) {
            threads.emplace_back([=]() {
                const auto start = static_cast< std::size_t >(t) * chunk_size;
                const auto end = std::min(start + chunk_size, n);
                for (std::size_t i = start; i < end; ++i) {
                    f(i);
                }
            });
        }

        for (auto& thread : threads)
            thread.join();
    } else {
        for (std::size_t i = 0; i < n; ++i)
            f(i);
    }
}

} // namespace ldlt

#endif // LDLT_SIMD_H
