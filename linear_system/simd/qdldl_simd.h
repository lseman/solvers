#ifndef QDLDL_SIMD_H
#define QDLDL_SIMD_H

// SIMD acceleration for qdldl23: AVX-512, AVX2, SSE2 with scalar fallback.
// Extracted from qdldl.h for modular organization.

#include <algorithm>
#include <thread>
#include <vector>

// ===== qdldl23 execution config (optional stdexec) ==========================
#if defined(QDLDL23_USE_STDEXEC)
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#define QDLDL23_HAS_STDEXEC 1
#else
#define QDLDL23_HAS_STDEXEC 0
#endif

// Enhanced SIMD support detection with AVX-512
#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#define QDLDL23_HAS_AVX512 1
#define QDLDL23_HAS_AVX2 1
#define QDLDL23_SIMD_WIDTH 8
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define QDLDL23_HAS_AVX512 0
#define QDLDL23_HAS_AVX2 1
#define QDLDL23_SIMD_WIDTH 4
#elif defined(__SSE2__)
#include <emmintrin.h>
#define QDLDL23_HAS_AVX512 0
#define QDLDL23_HAS_AVX2 0
#define QDLDL23_HAS_SSE2 1
#define QDLDL23_SIMD_WIDTH 2
#else
#define QDLDL23_HAS_AVX512 0
#define QDLDL23_HAS_AVX2 0
#define QDLDL23_HAS_SSE2 0
#define QDLDL23_SIMD_WIDTH 1
#endif

// Cache line and memory prefetching
#define QDLDL23_CACHE_LINE_SIZE 64
#define QDLDL23_LIKELY [[likely]]
#define QDLDL23_UNLIKELY [[unlikely]]

namespace qdldl23 {

namespace detail {

// Memory prefetching hints
inline void prefetch_read(const void* addr) noexcept {
#if defined(__builtin_prefetch)
    __builtin_prefetch(addr, 0, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast< const char* >(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void* addr) noexcept {
#if defined(__builtin_prefetch)
    __builtin_prefetch(addr, 1, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast< const char* >(addr), _MM_HINT_T0);
#endif
}

// Parallel for loop with stdexec support (stub — full impl in qdldl.h)
template < class F > inline void qd_par_for_n(std::size_t n, F&& f) {
    constexpr std::size_t min_par_size = 1000;
    if (n < min_par_size) {
        for (std::size_t i = 0; i < n; ++i)
            f(i);
        return;
    }
    const auto num_threads = std::max(1u, std::thread::hardware_concurrency());
    const auto chunk_size = (n + num_threads - 1) / num_threads;
    if (num_threads > 1 && n > 2 * min_par_size) {
        std::vector< std::thread > threads;
        threads.reserve(num_threads);
        for (std::size_t t = 0; t < num_threads; ++t) {
            const auto start = t * chunk_size;
            const auto end = std::min(start + chunk_size, n);
            if (start < end) {
                threads.emplace_back([start, end, &f]() {
                    for (std::size_t i = start; i < end; ++i)
                        f(i);
                });
            }
        }
        for (auto& thread : threads)
            thread.join();
    } else {
        for (std::size_t i = 0; i < n; ++i)
            f(i);
    }
}

// SIMD: scale array by vector
inline void simd_scale_array(double* __restrict__ x, const double* __restrict__ scale, size_t n) {
    size_t i = 0;
#if QDLDL23_HAS_AVX512
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
#elif QDLDL23_HAS_AVX2
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
#elif QDLDL23_HAS_SSE2
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

// SIMD: vector subtraction
inline void simd_vector_sub(double* __restrict__ result, const double* __restrict__ a,
                            const double* __restrict__ b, size_t n) {
    size_t i = 0;
#if QDLDL23_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    for (; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vr = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(&result[i], vr);
    }
#elif QDLDL23_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (; i < simd_end; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vr);
    }
#endif
    for (; i < n; ++i)
        result[i] = a[i] - b[i];
}

// SIMD: vector addition
inline void simd_vector_add(double* __restrict__ result, const double* __restrict__ a,
                            const double* __restrict__ b, size_t n) {
    size_t i = 0;
#if QDLDL23_HAS_AVX512
    const size_t simd_end = n - (n % 8);
    for (; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vr = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&result[i], vr);
    }
#elif QDLDL23_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (; i < simd_end; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vr);
    }
#endif
    for (; i < n; ++i)
        result[i] = a[i] + b[i];
}

// Cache-friendly sorting for small arrays
template < typename T > inline void optimized_sort(T* begin, T* end) {
    const auto size = end - begin;
    if (size <= 1)
        return;
    if (size <= 16) {
        for (auto it = begin + 1; it != end; ++it) {
            auto key = *it;
            auto pos = it;
            while (pos > begin && *(pos - 1) > key) {
                *pos = *(pos - 1);
                --pos;
            }
            *pos = key;
        }
    } else {
        std::sort(begin, end);
    }
}

// Optimized scattered memory access
template < typename FloatT, typename IntT >
inline void scatter_update(FloatT* __restrict__ x, const IntT* __restrict__ indices,
                           const FloatT* __restrict__ values, IntT n, FloatT scale) {
    constexpr IntT prefetch_distance = 8;
    for (IntT i = 0; i < n; ++i) {
        if (i + prefetch_distance < n) {
            const IntT future_idx = indices[i + prefetch_distance];
            if (future_idx >= 0) {
                prefetch_write(&x[static_cast< size_t >(future_idx)]);
            }
        }
        const IntT idx = indices[i];
        if (idx >= 0) {
            x[static_cast< size_t >(idx)] -= values[i] * scale;
        }
    }
}

} // namespace detail
} // namespace qdldl23

#endif // QDLDL_SIMD_H
