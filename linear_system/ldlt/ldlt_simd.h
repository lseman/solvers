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

namespace ldlt {

// Cache prefetching
inline void prefetch_read(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
#endif
}

// SIMD: scale array by vector (x[i] *= scale[i])
inline void simd_scale_array(double *__restrict__ x,
                             const double *__restrict__ scale, size_t n) {
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

  for (; i < n; ++i) x[i] *= scale[i];
}

// SIMD: sparse multiply-accumulate for factorization kernels
// Computes: x[rows[i]] -= scale * vals[i] for contiguous i
// Used in forward/backward solves for small dense segments
inline void simd_sparse_macc(double *__restrict__ x,
                             const int32_t *__restrict__ rows,
                             const double *__restrict__ vals, size_t n,
                             double scale) {
#if LDLT_HAS_AVX2
  const size_t prefetch_dist = 4;
  for (size_t i = 0; i < n; ++i) {
    if (i + prefetch_dist < n) {
      const int32_t future_row = rows[i + prefetch_dist];
      if (future_row >= 0) {
        prefetch_write(&x[static_cast<size_t>(future_row)]);
      }
    }
    const int32_t j = rows[i];
    if (j >= 0) {
      x[static_cast<size_t>(j)] -= scale * vals[i];
    }
  }
#else
  for (size_t i = 0; i < n; ++i) {
    const int32_t j = rows[i];
    if (j >= 0) {
      x[static_cast<size_t>(j)] -= scale * vals[i];
    }
  }
#endif
}

// SIMD: dense segment accumulation (contiguous scatter for column ops)
// Computes: x[k:k+n] -= alpha * y[p0:p1] where access is sequential
// Useful for dense frontal blocks in supernodal factorization
inline void simd_dense_axpy(double *__restrict__ x, double alpha,
                            const double *__restrict__ y, size_t n) {
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

  for (; i < n; ++i) x[i] -= alpha * y[i];
}

}  // namespace ldlt

#endif  // LDLT_SIMD_H
