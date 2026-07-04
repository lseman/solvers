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

// SIMD: vector subtraction (result = a - b)
inline void simd_vector_sub(double *__restrict__ result,
                            const double *__restrict__ a,
                            const double *__restrict__ b, size_t n) {
  size_t i = 0;

#if LDLT_HAS_AVX512
  const size_t simd_end = n - (n % 8);
  for (; i < simd_end; i += 8) {
    __m512d va = _mm512_loadu_pd(&a[i]);
    __m512d vb = _mm512_loadu_pd(&b[i]);
    __m512d vr = _mm512_sub_pd(va, vb);
    _mm512_storeu_pd(&result[i], vr);
  }
#elif LDLT_HAS_AVX2
  const size_t simd_end = n - (n % 4);
  for (; i < simd_end; i += 4) {
    __m256d va = _mm256_loadu_pd(&a[i]);
    __m256d vb = _mm256_loadu_pd(&b[i]);
    __m256d vr = _mm256_sub_pd(va, vb);
    _mm256_storeu_pd(&result[i], vr);
  }
#endif

  for (; i < n; ++i) result[i] = a[i] - b[i];
}

// SIMD: vector addition (result = a + b)
inline void simd_vector_add(double *__restrict__ result,
                            const double *__restrict__ a,
                            const double *__restrict__ b, size_t n) {
  size_t i = 0;

#if LDLT_HAS_AVX512
  const size_t simd_end = n - (n % 8);
  for (; i < simd_end; i += 8) {
    __m512d va = _mm512_loadu_pd(&a[i]);
    __m512d vb = _mm512_loadu_pd(&b[i]);
    __m512d vr = _mm512_add_pd(va, vb);
    _mm512_storeu_pd(&result[i], vr);
  }
#elif LDLT_HAS_AVX2
  const size_t simd_end = n - (n % 4);
  for (; i < simd_end; i += 4) {
    __m256d va = _mm256_loadu_pd(&a[i]);
    __m256d vb = _mm256_loadu_pd(&b[i]);
    __m256d vr = _mm256_add_pd(va, vb);
    _mm256_storeu_pd(&result[i], vr);
  }
#endif

  for (; i < n; ++i) result[i] = a[i] + b[i];
}

// SIMD: dot product (sum of a[i]*b[i])
inline double simd_dot_product(const double *__restrict__ a,
                               const double *__restrict__ b, size_t n) {
  double sum = 0.0;
  size_t i = 0;

#if LDLT_HAS_AVX512
  __m512d vsum = _mm512_setzero_pd();
  const size_t simd_end = n - (n % 8);
  for (; i < simd_end; i += 8) {
    __m512d va = _mm512_loadu_pd(&a[i]);
    __m512d vb = _mm512_loadu_pd(&b[i]);
    vsum = _mm512_fmadd_pd(va, vb, vsum);
  }
  sum = _mm512_reduce_add_pd(vsum);
#elif LDLT_HAS_AVX2
  __m256d vsum = _mm256_setzero_pd();
  const size_t simd_end = n - (n % 4);
  for (; i < simd_end; i += 4) {
    __m256d va = _mm256_loadu_pd(&a[i]);
    __m256d vb = _mm256_loadu_pd(&b[i]);
    vsum = _mm256_fmadd_pd(va, vb, vsum);
  }
  __m128d sum_low = _mm256_castpd256_pd128(vsum);
  __m128d sum_high = _mm256_extractf128_pd(vsum, 1);
  sum_low = _mm_add_pd(sum_low, sum_high);
  sum_low = _mm_hadd_pd(sum_low, sum_low);
  sum = _mm_cvtsd_f64(sum_low);
#elif LDLT_HAS_SSE2
  __m128d vsum = _mm_setzero_pd();
  const size_t simd_end = n - (n % 2);
  for (; i < simd_end; i += 2) {
    __m128d va = _mm_loadu_pd(&a[i]);
    __m128d vb = _mm_loadu_pd(&b[i]);
    __m128d vp = _mm_mul_pd(va, vb);
    vsum = _mm_add_pd(vsum, vp);
  }
  __m128d temp = _mm_shuffle_pd(vsum, vsum, 1);
  vsum = _mm_add_sd(vsum, temp);
  sum = _mm_cvtsd_f64(vsum);
#endif

  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace ldlt

#endif  // LDLT_SIMD_H
