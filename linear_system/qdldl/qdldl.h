#pragma once

/// qdldl.h — Header-only LDLᵀ factorization (QDLDL-style) for sparse upper CSC
///
/// Modern C++20/23 implementation of the QDLDL algorithm for symmetric
/// positive-definite systems. Emphasizes cache efficiency, SIMD acceleration
/// (AVX-512, AVX2, SSE2), and numerical stability.
///
/// Core features:
///   - Exact-zero pivot detection (no fill-in tricks)
///   - Pattern analysis and numeric factorization phases
///   - Permutation support via Ordering struct
///   - Iterative refinement for improved accuracy
///   - SIMD-accelerated solves and memory operations
///   - Optional stdexec parallel execution
///
/// © 2025 — MIT / Apache-2.0 compatible with original QDLDL spirit.

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// ===== qdldl23 execution config (optional stdexec) ==========================
#if defined(QDLDL23_USE_STDEXEC)
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#define QDLDL23_HAS_STDEXEC 1
#else
#define QDLDL23_HAS_STDEXEC 0
#endif

#include "../common/ordering.h"
#include "../common/sparse_csc.h"
#include "qdldl_simd.h"

namespace qdldl23 {

// --------- Errors ----------
struct InvalidMatrixError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct FactorizationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// SparseUpperCSC is now an alias to linsys::SparseCSC.
// All upper-triangular validation is in linsys::finalize_upper_inplace.

// --------- Ordering (perm/iperm) ----------
// Shared with the other custom solvers (linear_system/common/ordering.h).
// Note: from_perm throws std::invalid_argument on a bad permutation.
using linsys::Ordering;

// Enhanced symmetric permutation with better memory access patterns
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline linsys::SparseCSC< FloatT, IntT >
permute_symmetric_upper(const linsys::SparseCSC< FloatT, IntT >& A, const Ordering< IntT >& ord) {
    if (ord.n != A.n)
        QDLDL23_UNLIKELY
    throw InvalidMatrixError("Ordering size mismatch");
    const IntT n = A.n;

    std::vector< IntT > cnt(static_cast< size_t >(n), IntT{0});

    // Count entries per permuted column
    for (IntT j = 0; j < n; ++j) {
        const IntT pj = ord.perm[static_cast< size_t >(j)];
        for (IntT p = A.Ap[static_cast< size_t >(j)]; p < A.Ap[static_cast< size_t >(j) + 1]; ++p) {
            const IntT i = A.Ai[static_cast< size_t >(p)];
            const IntT pi = ord.perm[static_cast< size_t >(i)];
            const IntT col = std::max(pi, pj);
            ++cnt[static_cast< size_t >(col)];
        }
    }

    // Build pointers
    std::vector< IntT > Bp(static_cast< size_t >(n) + 1, IntT{0});
    for (IntT j = 0; j < n; ++j) {
        Bp[static_cast< size_t >(j) + 1] =
            Bp[static_cast< size_t >(j)] + cnt[static_cast< size_t >(j)];
    }
    const size_t nnz = static_cast< size_t >(Bp[static_cast< size_t >(n)]);

    std::vector< IntT > Bi(nnz);
    std::vector< FloatT > Bx(nnz);
    std::vector< IntT > head = Bp;

    // Scatter entries with prefetching
    for (IntT j = 0; j < n; ++j) {
        const IntT pj = ord.perm[static_cast< size_t >(j)];
        const IntT p0 = A.Ap[static_cast< size_t >(j)];
        const IntT p1 = A.Ap[static_cast< size_t >(j) + 1];

        if (j + 1 < n) {
            const IntT next_p0 = A.Ap[static_cast< size_t >(j + 1)];
            if (next_p0 < static_cast< IntT >(A.Ai.size())) {
                detail::prefetch_read(&A.Ai[static_cast< size_t >(next_p0)]);
                detail::prefetch_read(&A.Ax[static_cast< size_t >(next_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            const IntT i = A.Ai[static_cast< size_t >(p)];
            const FloatT v = A.Ax[static_cast< size_t >(p)];
            IntT pi = ord.perm[static_cast< size_t >(i)];
            IntT col = pj, row = pi;
            if (row > col) {
                std::swap(row, col);
            }
            const IntT dst = head[static_cast< size_t >(col)]++;
            Bi[static_cast< size_t >(dst)] = row;
            Bx[static_cast< size_t >(dst)] = v;
        }
    }

    // Sort and coalesce each column with compatible approach
    for (IntT j = 0; j < n; ++j) {
        const IntT p0 = Bp[static_cast< size_t >(j)];
        const IntT p1 = Bp[static_cast< size_t >(j) + 1];
        const IntT len = p1 - p0;

        if (len <= 1)
            QDLDL23_LIKELY continue;

        // Create index array for indirect sorting (more compatible)
        std::vector< IntT > indices(static_cast< size_t >(len));
        std::iota(indices.begin(), indices.end(), IntT{0});

        // Sort indices by corresponding row values
        std::stable_sort(indices.begin(), indices.end(), [&](IntT a, IntT b) {
            return Bi[static_cast< size_t >(p0 + a)] < Bi[static_cast< size_t >(p0 + b)];
        });

        // Apply permutation in-place using a more efficient approach
        std::vector< IntT > tmp_i(static_cast< size_t >(len));
        std::vector< FloatT > tmp_x(static_cast< size_t >(len));

        for (IntT k = 0; k < len; ++k) {
            tmp_i[static_cast< size_t >(k)] =
                Bi[static_cast< size_t >(p0 + indices[static_cast< size_t >(k)])];
            tmp_x[static_cast< size_t >(k)] =
                Bx[static_cast< size_t >(p0 + indices[static_cast< size_t >(k)])];
        }

        // In-place coalescing
        IntT w = 0;
        bool has_diag = false;
        for (IntT k = 0; k < len;)
            QDLDL23_LIKELY {
                const IntT r = tmp_i[static_cast< size_t >(k)];
                FloatT s = tmp_x[static_cast< size_t >(k)];
                IntT k2 = k + 1;

                while (k2 < len && tmp_i[static_cast< size_t >(k2)] == r)
                    QDLDL23_UNLIKELY {
                        s += tmp_x[static_cast< size_t >(k2)];
                        ++k2;
                    }

                Bi[static_cast< size_t >(p0 + w)] = r;
                Bx[static_cast< size_t >(p0 + w)] = s;
                if (r == j)
                    QDLDL23_LIKELY
                has_diag = true;
                ++w;
                k = k2;
            }

        const IntT drop = len - w;
        if (drop > 0)
            QDLDL23_UNLIKELY {
                for (IntT jj = j + 1; jj <= n; ++jj) {
                    Bp[static_cast< size_t >(jj)] -= drop;
                }
            }

        if (!has_diag)
            QDLDL23_UNLIKELY
        throw InvalidMatrixError("Missing diagonal after permutation at column " +
                                 std::to_string(j));
    }

    linsys::SparseCSC< FloatT, IntT > B;
    B.n = n;
    B.Ap = std::move(Bp);
    B.Ai = std::move(Bi);
    B.Ax = std::move(Bx);
    return B;
}

// --------- Symbolic structure ----------
template < std::signed_integral IntT = int32_t > struct Symbolic {
    std::vector< IntT > etree; // parent[j] or -1
    std::vector< IntT > Lnz;   // strictly-lower count per column of L
    std::vector< IntT > Lp;    // size n+1, cumulated Lnz
    IntT n{0};
};

// Enhanced Liu elimination tree with path compression
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void etree_upper_liu(const linsys::SparseCSC< FloatT, IntT >& A,
                            std::vector< IntT >& parent) {
    const IntT n = A.n;
    parent.assign(static_cast< size_t >(n), IntT{-1});
    std::vector< IntT > ancestor(static_cast< size_t >(n), IntT{-1});

    auto find_root = [&](IntT j) -> IntT {
        if (ancestor[static_cast< size_t >(j)] == IntT{-1})
            QDLDL23_LIKELY {
                return j;
            }

        IntT r = j;
        while (ancestor[static_cast< size_t >(r)] != IntT{-1}) {
            r = ancestor[static_cast< size_t >(r)];
        }

        // Path compression
        IntT cur = j;
        while (ancestor[static_cast< size_t >(cur)] != IntT{-1}) {
            const IntT nxt = ancestor[static_cast< size_t >(cur)];
            ancestor[static_cast< size_t >(cur)] = r;
            cur = nxt;
        }
        return r;
    };

    for (IntT j = 0; j < n; ++j) {
        const IntT p0 = A.Ap[static_cast< size_t >(j)];
        const IntT p1 = A.Ap[static_cast< size_t >(j) + 1];

        if (j + 1 < n) {
            const IntT next_p0 = A.Ap[static_cast< size_t >(j + 1)];
            if (next_p0 < static_cast< IntT >(A.Ai.size())) {
                detail::prefetch_read(&A.Ai[static_cast< size_t >(next_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            IntT i = A.Ai[static_cast< size_t >(p)];
            if (i == j)
                QDLDL23_LIKELY continue;

            for (IntT iter = 0; iter < n; ++iter) {
                const IntT anc_i = ancestor[static_cast< size_t >(i)];
                if (anc_i == IntT{-1})
                    QDLDL23_LIKELY {
                        ancestor[static_cast< size_t >(i)] = j;
                        parent[static_cast< size_t >(i)] = j;
                        break;
                    }

                const IntT r = find_root(i);
                if (r == j)
                    QDLDL23_LIKELY {
                        break;
                    }

                ancestor[static_cast< size_t >(r)] = j;
                parent[static_cast< size_t >(r)] = j;
                i = r;
            }
        }
    }
}

// Enhanced Gilbert–Ng–Peyton column counts
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void colcounts_upper_gnp(const linsys::SparseCSC< FloatT, IntT >& A,
                                const std::vector< IntT >& parent, std::vector< IntT >& Lnz) {
    const IntT n = A.n;
    Lnz.assign(static_cast< size_t >(n), IntT{0});
    std::vector< IntT > prevnz(static_cast< size_t >(n), IntT{-1});

    for (IntT j = 0; j < n; ++j) {
        const IntT p0 = A.Ap[static_cast< size_t >(j)];
        const IntT p1 = A.Ap[static_cast< size_t >(j) + 1];

        if (j + 1 < n) {
            const IntT next_p0 = A.Ap[static_cast< size_t >(j + 1)];
            if (next_p0 < static_cast< IntT >(A.Ai.size())) {
                detail::prefetch_read(&A.Ai[static_cast< size_t >(next_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            const IntT i = A.Ai[static_cast< size_t >(p)];
            if (i == j)
                QDLDL23_LIKELY continue;

            IntT k = i;
            // Unroll first iterations for common case
            if (k != -1 && k < j && prevnz[static_cast< size_t >(k)] != j)
                QDLDL23_LIKELY {
                    ++Lnz[static_cast< size_t >(k)];
                    prevnz[static_cast< size_t >(k)] = j;
                    k = parent[static_cast< size_t >(k)];

                    if (k != -1 && k < j && prevnz[static_cast< size_t >(k)] != j) {
                        ++Lnz[static_cast< size_t >(k)];
                        prevnz[static_cast< size_t >(k)] = j;
                        k = parent[static_cast< size_t >(k)];
                    }
                }

            while (k != -1 && k < j && prevnz[static_cast< size_t >(k)] != j)
                QDLDL23_UNLIKELY {
                    ++Lnz[static_cast< size_t >(k)];
                    prevnz[static_cast< size_t >(k)] = j;
                    k = parent[static_cast< size_t >(k)];
                }
        }
    }
}

// Enhanced symbolic analysis
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline Symbolic< IntT > analyze_fast(const linsys::SparseCSC< FloatT, IntT >& A) {
    Symbolic< IntT > S;
    S.n = A.n;
    S.etree.reserve(static_cast< size_t >(A.n));
    S.Lnz.reserve(static_cast< size_t >(A.n));
    S.Lp.reserve(static_cast< size_t >(A.n) + 1);

    etree_upper_liu(A, S.etree);
    colcounts_upper_gnp(A, S.etree, S.Lnz);

    S.Lp.resize(static_cast< size_t >(S.n) + 1);
    S.Lp[0] = 0;
    for (IntT j = 0; j < S.n; ++j) {
        S.Lp[static_cast< size_t >(j) + 1] =
            S.Lp[static_cast< size_t >(j)] + S.Lnz[static_cast< size_t >(j)];
    }
    return S;
}

// --------- Enhanced Numeric factorization ----------
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
struct LDLFactors {
    std::vector< IntT > Lp;     // size n+1 (strict lower)
    std::vector< IntT > Li;     // size nnz(L)
    std::vector< FloatT > Lx;   // size nnz(L)
    std::vector< FloatT > D;    // size n
    std::vector< FloatT > Dinv; // size n
    IntT n{0};
    IntT num_pos{0};
    bool is_cache_aligned{false};
};

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline LDLFactors< FloatT, IntT > refactorize(const linsys::SparseCSC< FloatT, IntT >& A,
                                              const Symbolic< IntT >& S) {
    const IntT n = S.n;
    if (A.n != n)
        QDLDL23_UNLIKELY
    throw InvalidMatrixError("Symbolic/numeric size mismatch");

    LDLFactors< FloatT, IntT > R;
    R.n = n;
    R.num_pos = 0;
    R.Lp = S.Lp;
    const IntT nnzL = R.Lp[static_cast< size_t >(n)];

    R.Li.assign(static_cast< size_t >(nnzL), IntT{0});
    R.Lx.assign(static_cast< size_t >(nnzL), FloatT{0});
    R.D.assign(static_cast< size_t >(n), FloatT{0});
    R.Dinv.assign(static_cast< size_t >(n), FloatT{0});

    // Enhanced work arrays with cache alignment
    std::vector< IntT > Lnext(static_cast< size_t >(n));
    for (IntT i = 0; i < n; ++i) {
        Lnext[static_cast< size_t >(i)] = R.Lp[static_cast< size_t >(i)];
    }

    constexpr size_t alignment = QDLDL23_CACHE_LINE_SIZE / sizeof(FloatT);
    const size_t aligned_n = ((static_cast< size_t >(n) + alignment - 1) / alignment) * alignment;

    std::vector< FloatT > y(aligned_n, FloatT{0});
    std::vector< IntT > yIdx(static_cast< size_t >(n));
    std::vector< IntT > Ebuf(static_cast< size_t >(n));
    std::vector< IntT > mark(static_cast< size_t >(n), IntT{0});

    IntT curmark = 1;

    auto is_marked = [&](IntT i) -> bool {
        return i >= 0 && i < n && mark[static_cast< size_t >(i)] == curmark;
    };
    auto set_mark = [&](IntT i) {
        if (i >= 0 && i < n)
            mark[static_cast< size_t >(i)] = curmark;
    };

    // Handle k=0 case
    if (n > 0)
        QDLDL23_LIKELY {
            if (A.Ap[0] >= A.Ap[1] || A.Ai[static_cast< size_t >(A.Ap[0])] != 0)
                QDLDL23_UNLIKELY
            throw FactorizationError("Missing diagonal at column 0");
            R.D[0] = A.Ax[static_cast< size_t >(A.Ap[0])];
            if (R.D[0] == FloatT{0})
                QDLDL23_UNLIKELY
            throw FactorizationError("Zero pivot at column 0");
            if (R.D[0] > FloatT{0})
                QDLDL23_LIKELY
            ++R.num_pos;
            R.Dinv[0] = FloatT{1} / R.D[0];
        }

    // Main factorization loop with enhanced optimizations
    for (IntT k = 1; k < n; ++k) {
        if (++curmark == std::numeric_limits< IntT >::max())
            QDLDL23_UNLIKELY {
                std::fill(mark.begin(), mark.end(), IntT{0});
                curmark = 1;
            }

        IntT nnzY = 0;
        bool diag_seen = false;

        const IntT p0 = A.Ap[static_cast< size_t >(k)];
        const IntT p1 = A.Ap[static_cast< size_t >(k) + 1];

        // Prefetch next column
        if (k + 1 < n) {
            const IntT next_p0 = A.Ap[static_cast< size_t >(k + 1)];
            if (next_p0 < static_cast< IntT >(A.Ai.size())) {
                detail::prefetch_read(&A.Ai[static_cast< size_t >(next_p0)]);
                detail::prefetch_read(&A.Ax[static_cast< size_t >(next_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            const IntT i = A.Ai[static_cast< size_t >(p)];
            const FloatT v = A.Ax[static_cast< size_t >(p)];

            if (i == k)
                QDLDL23_LIKELY {
                    R.D[static_cast< size_t >(k)] = v;
                    diag_seen = true;
                    continue;
                }

            if (i < 0 || i >= n)
                QDLDL23_UNLIKELY continue;

            y[static_cast< size_t >(i)] = v;
            if (!is_marked(i))
                QDLDL23_LIKELY {
                    set_mark(i);

                    IntT next = i;
                    IntT nnzE = 0;
                    constexpr IntT max_path = 1000;

                    while (next != -1 && next < k && nnzE < max_path) {
                        if (nnzE < static_cast< IntT >(Ebuf.size())) {
                            Ebuf[static_cast< size_t >(nnzE++)] = next;
                        }

                        const IntT parent_next = S.etree[static_cast< size_t >(next)];
                        if (parent_next >= 0 && parent_next < k && is_marked(parent_next))
                            break;
                        if (parent_next >= 0 && parent_next < k)
                            set_mark(parent_next);
                        next = parent_next;
                    }

                    for (IntT e = nnzE - 1; e >= 0 && nnzY < n; --e) {
                        if (e < static_cast< IntT >(Ebuf.size()) &&
                            nnzY < static_cast< IntT >(yIdx.size())) {
                            yIdx[static_cast< size_t >(nnzY++)] = Ebuf[static_cast< size_t >(e)];
                        }
                    }
                }
        }

        if (!diag_seen)
            QDLDL23_UNLIKELY
        throw FactorizationError("Missing diagonal at column " + std::to_string(k));

        // Enhanced elimination with optimized scatter updates
        for (IntT t = nnzY - 1; t >= 0; --t) {
            if (t >= static_cast< IntT >(yIdx.size()))
                continue;
            const IntT c = yIdx[static_cast< size_t >(t)];
            if (c < 0 || c >= n)
                QDLDL23_UNLIKELY continue;

            const IntT j0 = R.Lp[static_cast< size_t >(c)];
            IntT& j1 = Lnext[static_cast< size_t >(c)];
            const FloatT yc = y[static_cast< size_t >(c)];

            const IntT update_len = j1 - j0;
            if (update_len > 0) {
                if (j0 < static_cast< IntT >(R.Li.size())) {
                    detail::prefetch_read(&R.Li[static_cast< size_t >(j0)]);
                    detail::prefetch_read(&R.Lx[static_cast< size_t >(j0)]);
                }

                // Use optimized scatter update
                if (update_len > 8 && std::is_same_v< FloatT, double >) {
                    detail::scatter_update(y.data(), &R.Li[static_cast< size_t >(j0)],
                                           &R.Lx[static_cast< size_t >(j0)], update_len, yc);
                } else {
                    for (IntT j = j0; j < j1; ++j) {
                        if (j >= static_cast< IntT >(R.Li.size()))
                            break;
                        const IntT idx = R.Li[static_cast< size_t >(j)];
                        if (idx >= 0 && idx < n) {
                            y[static_cast< size_t >(idx)] -= R.Lx[static_cast< size_t >(j)] * yc;
                        }
                    }
                }
            }

            if (j1 < static_cast< IntT >(R.Li.size())) {
                R.Li[static_cast< size_t >(j1)] = k;
                const FloatT lk = yc * R.Dinv[static_cast< size_t >(c)];
                R.Lx[static_cast< size_t >(j1)] = lk;
                R.D[static_cast< size_t >(k)] -= yc * lk;
                ++j1;
            }

            y[static_cast< size_t >(c)] = FloatT{0};
        }

        if (R.D[static_cast< size_t >(k)] == FloatT{0})
            QDLDL23_UNLIKELY
        throw FactorizationError("Zero pivot at column " + std::to_string(k));
        if (R.D[static_cast< size_t >(k)] > FloatT{0})
            QDLDL23_LIKELY
        ++R.num_pos;
        R.Dinv[static_cast< size_t >(k)] = FloatT{1} / R.D[static_cast< size_t >(k)];
    }

    R.is_cache_aligned = true;
    return R;
}

// Maintained API compatibility functions
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline LDLFactors< FloatT, IntT >
refactorize_with_ordering(const linsys::SparseCSC< FloatT, IntT >& A, const Symbolic< IntT >& Sperm,
                          const Ordering< IntT >& ord) {
    const auto B = permute_symmetric_upper(A, ord);
    return refactorize(B, Sperm);
}

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline LDLFactors< FloatT, IntT > factorize(const linsys::SparseCSC< FloatT, IntT >& A) {
    auto S = analyze_fast(A);
    return refactorize(A, S);
}

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline LDLFactors< FloatT, IntT >
factorize_with_ordering(const linsys::SparseCSC< FloatT, IntT >& A, const Ordering< IntT >& ord) {
    auto B = permute_symmetric_upper(A, ord);
    auto S = analyze_fast(B);
    return refactorize(B, S);
}

// --------- Enhanced Solves with SIMD and prefetching ----------
template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void L_solve(const LDLFactors< FloatT, IntT >& F, FloatT* x) {
    const IntT n = F.n;
    for (IntT i = 0; i < n; ++i) {
        const FloatT xi = x[static_cast< size_t >(i)];
        const IntT p0 = F.Lp[static_cast< size_t >(i)];
        const IntT p1 = F.Lp[static_cast< size_t >(i) + 1];

        // Prefetch next iteration
        if (i + 1 < n) {
            const IntT next_p0 = F.Lp[static_cast< size_t >(i + 1)];
            if (next_p0 < static_cast< IntT >(F.Li.size())) {
                detail::prefetch_read(&F.Li[static_cast< size_t >(next_p0)]);
                detail::prefetch_read(&F.Lx[static_cast< size_t >(next_p0)]);
            }
        }

        const IntT col_len = p1 - p0;
        if (col_len > 8 && std::is_same_v< FloatT, double >) {
            detail::scatter_update(x, &F.Li[static_cast< size_t >(p0)],
                                   &F.Lx[static_cast< size_t >(p0)], col_len, xi);
        } else {
            for (IntT p = p0; p < p1; ++p) {
                const IntT idx = F.Li[static_cast< size_t >(p)];
                if (idx >= 0 && idx < n) {
                    x[static_cast< size_t >(idx)] -= F.Lx[static_cast< size_t >(p)] * xi;
                }
            }
        }
    }
}

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void Lt_solve(const LDLFactors< FloatT, IntT >& F, FloatT* x) {
    const IntT n = F.n;
    for (IntT i = n - 1; i >= 0; --i) {
        FloatT xi = x[static_cast< size_t >(i)];
        const IntT p0 = F.Lp[static_cast< size_t >(i)];
        const IntT p1 = F.Lp[static_cast< size_t >(i) + 1];

        // Prefetch previous iteration
        if (i > 0) {
            const IntT prev_p0 = F.Lp[static_cast< size_t >(i - 1)];
            if (prev_p0 < static_cast< IntT >(F.Li.size())) {
                detail::prefetch_read(&F.Li[static_cast< size_t >(prev_p0)]);
                detail::prefetch_read(&F.Lx[static_cast< size_t >(prev_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            const IntT idx = F.Li[static_cast< size_t >(p)];
            if (idx >= 0 && idx < n) {
                xi -= F.Lx[static_cast< size_t >(p)] * x[static_cast< size_t >(idx)];
            }
        }
        x[static_cast< size_t >(i)] = xi;
        if (i == 0)
            break;
    }
}

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void solve(const LDLFactors< FloatT, IntT >& F, FloatT* x) {
    L_solve(F, x);

    const IntT n = F.n;
    if constexpr (std::is_same_v< FloatT, double >) {
        detail::simd_scale_array(x, F.Dinv.data(), static_cast< size_t >(n));
    } else {
        detail::qd_par_for_n(static_cast< std::size_t >(n),
                             [&](std::size_t i) { x[i] *= F.Dinv[i]; });
    }

    Lt_solve(F, x);
}

template < std::floating_point FloatT = double, std::signed_integral IntT = int32_t >
inline void solve_with_ordering(const LDLFactors< FloatT, IntT >& F, const Ordering< IntT >& ord,
                                FloatT* x) {
    const IntT n = F.n;
    if (ord.n != n)
        QDLDL23_UNLIKELY
    throw InvalidMatrixError("ordering mismatch");

    std::vector< FloatT > xp(static_cast< size_t >(n));

    // Permute input: xp = P x
    detail::qd_par_for_n(static_cast< std::size_t >(n), [&](std::size_t i) {
        const IntT pi = ord.perm[i];
        if (pi >= 0 && pi < n) {
            xp[static_cast< size_t >(pi)] = x[i];
        }
    });

    L_solve(F, xp.data());

    // Diagonal scaling
    if constexpr (std::is_same_v< FloatT, double >) {
        detail::simd_scale_array(xp.data(), F.Dinv.data(), static_cast< size_t >(n));
    } else {
        detail::qd_par_for_n(static_cast< std::size_t >(n),
                             [&](std::size_t i) { xp[i] *= F.Dinv[i]; });
    }

    Lt_solve(F, xp.data());

    // Permute output: x = P xp
    detail::qd_par_for_n(static_cast< std::size_t >(n), [&](std::size_t i) {
        const IntT pi = ord.perm[i];
        if (pi >= 0 && pi < n) {
            x[i] = xp[static_cast< size_t >(pi)];
        }
    });
}

// Enhanced symmetric SpMV with cache optimizations
template < std::floating_point FloatT, std::signed_integral IntT >
inline void sym_spmv_upper(const linsys::SparseCSC< FloatT, IntT >& A, const FloatT* __restrict__ x,
                           FloatT* __restrict__ y) {
    const IntT n = A.n;
    std::fill(y, y + static_cast< size_t >(n), FloatT{0});

    for (IntT j = 0; j < n; ++j) {
        const FloatT xj = x[static_cast< size_t >(j)];
        const IntT p0 = A.Ap[static_cast< size_t >(j)];
        const IntT p1 = A.Ap[static_cast< size_t >(j) + 1];

        // Prefetch next column
        if (j + 1 < n) {
            const IntT next_p0 = A.Ap[static_cast< size_t >(j + 1)];
            if (next_p0 < static_cast< IntT >(A.Ai.size())) {
                detail::prefetch_read(&A.Ai[static_cast< size_t >(next_p0)]);
                detail::prefetch_read(&A.Ax[static_cast< size_t >(next_p0)]);
            }
        }

        for (IntT p = p0; p < p1; ++p) {
            const IntT i = A.Ai[static_cast< size_t >(p)];
            if (i >= 0 && i < n) {
                const FloatT v = A.Ax[static_cast< size_t >(p)];
                y[static_cast< size_t >(i)] += v * xj;
                if (i != j) {
                    y[static_cast< size_t >(j)] += v * x[static_cast< size_t >(i)];
                }
            }
        }
    }
}

// Enhanced iterative refinement
template < std::floating_point FloatT, std::signed_integral IntT >
inline void refine(const linsys::SparseCSC< FloatT, IntT >& A, const LDLFactors< FloatT, IntT >& F,
                   FloatT* x, const FloatT* b, int iters = 2,
                   const Ordering< IntT >* ord = nullptr) {
    const IntT n = A.n;
    if (n <= 0)
        return;

    std::vector< FloatT > r(static_cast< size_t >(n));
    std::vector< FloatT > t(static_cast< size_t >(n));

    for (int it = 0; it < iters; ++it) {
        // Compute residual: r = b - A*x
        sym_spmv_upper(A, x, r.data());

        if constexpr (std::is_same_v< FloatT, double >) {
            detail::simd_vector_sub(r.data(), b, r.data(), static_cast< size_t >(n));
        } else {
            for (IntT i = 0; i < n; ++i) {
                r[static_cast< size_t >(i)] =
                    b[static_cast< size_t >(i)] - r[static_cast< size_t >(i)];
            }
        }

        // Solve for correction
        std::copy(r.begin(), r.end(), t.begin());
        if (ord) {
            solve_with_ordering(F, *ord, t.data());
        } else {
            solve(F, t.data());
        }

        // Apply correction
        if constexpr (std::is_same_v< FloatT, double >) {
            detail::simd_vector_add(x, x, t.data(), static_cast< size_t >(n));
        } else {
            detail::qd_par_for_n(static_cast< size_t >(n), [&](size_t i) { x[i] += t[i]; });
        }
    }
}

// --------- Type Aliases ----------
using SparseD32 = linsys::SparseCSC< double, int32_t >;
using SparseD64 = linsys::SparseCSC< double, int64_t >;
using SparseF32 = linsys::SparseCSC< float, int32_t >;
using Symb32 = Symbolic< int32_t >;
using Symb64 = Symbolic< int64_t >;
using LDL32 = LDLFactors< double, int32_t >;
using LDL64 = LDLFactors< double, int64_t >;

} // namespace qdldl23
