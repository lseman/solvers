#ifndef AMD_H
#define AMD_H

/// amd.h — Approximate Minimum Degree matrix reordering
///
/// Graph-based implementation of AMD for fill reduction in sparse
/// factorization. Groups variables by identical connectivity patterns and
/// eliminates the quotient graph using weighted minimum-degree heuristics.
///
/// Features:
///   - Hash-based variable grouping (coalescence)
///   - Dense variable postponement with adaptive thresholding
///   - Statistics reporting (coalesced, postponed dense, bandwidth reduction)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------
static inline uint64_t mix64_(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

template < class T > static inline void dedup_sorted_inplace(std::vector< T >& a) {
    if (a.empty())
        return;
    auto it = std::unique(a.begin(), a.end());
    a.erase(it, a.end());
}

//------------------------------------------------------------------------------
// CSR (pattern-only)
//------------------------------------------------------------------------------
struct CSR {
    i32 n{0};
    std::vector< i32 > indptr;  // size n+1
    std::vector< i32 > indices; // size nnz

    CSR() = default;
    explicit CSR(i32 n_) : n(n_), indptr(n_ + 1, 0) {
    }
    i32 nnz() const {
        return (i32)indices.size();
    }

    // Fast strictly upper of A ∪ Aᵀ (i<j), allocation-light, optionally
    // parallel
    CSR strict_upper_union_transpose() const {
        const i32 N = n;
        const auto& AI = indptr;
        const auto& AJ = indices;
        if (N == 0)
            return CSR(0);

        std::vector< std::vector< i32 > > rows(static_cast< size_t >(N));
        for (i32 i = 0; i < N; ++i) {
            for (i32 p = AI[i]; p < AI[i + 1]; ++p) {
                const i32 j = AJ[p];
                if (i == j)
                    continue;
                const i32 a = (i < j) ? i : j;
                const i32 b = (i < j) ? j : i;
                rows[static_cast< size_t >(a)].push_back(b);
            }
        }

        CSR U(N);
        U.indptr.assign(N + 1, 0);
        for (i32 i = 0; i < N; ++i) {
            auto& row = rows[static_cast< size_t >(i)];
            std::sort(row.begin(), row.end());
            dedup_sorted_inplace(row);
            U.indptr[static_cast< size_t >(i) + 1] =
                U.indptr[static_cast< size_t >(i)] + static_cast< i32 >(row.size());
        }
        U.indices.resize(static_cast< size_t >(U.indptr.back()));
        for (i32 i = 0; i < N; ++i) {
            std::copy(rows[static_cast< size_t >(i)].begin(), rows[static_cast< size_t >(i)].end(),
                      U.indices.begin() + U.indptr[static_cast< size_t >(i)]);
        }

        return U;
    }
};

// inverse permutation (p[new]=old) → ip[old]=new
static std::vector< i32 > inverse_permutation(const std::vector< i32 >& p) {
    if (p.size() > static_cast< size_t >(std::numeric_limits< i32 >::max()))
        throw std::invalid_argument("permutation is too large for i32 indices");

    std::vector< i32 > ip(p.size(), i32{-1});
    for (size_t i = 0; i < p.size(); ++i) {
        const i32 value = p[i];
        if (value < 0 || static_cast< size_t >(value) >= p.size() ||
            ip[static_cast< size_t >(value)] != i32{-1}) {
            throw std::invalid_argument("invalid permutation");
        }
        ip[static_cast< size_t >(value)] = static_cast< i32 >(i);
    }
    return ip;
}

//------------------------------------------------------------------------------
// Stats
//------------------------------------------------------------------------------
struct AMDStats {
    i32 original_nnz{0};
    i32 original_bandwidth{0};
    i32 reordered_bandwidth{0};
    double bandwidth_reduction{0.0};
    i32 matrix_size{0};
    std::vector< i32 > inverse_permutation;
    // Retained for source compatibility with the former element-based engine.
    i32 absorbed_elements{0};
    i32 coalesced_variables{0};
    i32 iw_capacity_peak{0};
    i32 postponed_dense_variables{0};
};

//------------------------------------------------------------------------------
// AMD (array-based) with SoTA-leaning micro-architecture
//------------------------------------------------------------------------------
class AMDReorderingArray {
  public:
    explicit AMDReorderingArray(bool /*aggressive_absorption*/ = true, int dense_cutoff = -1)
        : dense_cutoff_(dense_cutoff) {
    }

    // Main API: permutation p[new] = old
    std::vector< i32 > amd_order(const CSR& A, bool symmetrize = true) {
        validate_csr_(A);
        CSR Awork = symmetrize ? A.strict_upper_union_transpose() : A;
        stats_dense_ = 0;
        stats_coalesced_ = 0;
        perm_ = approximate_minimum_degree_order_(Awork, dense_cutoff_, &stats_coalesced_,
                                                  &stats_dense_);
        return perm_;
    }

    std::pair< std::vector< i32 >, AMDStats >
    compute_fill_reducing_permutation(const CSR& A, bool symmetrize = true) {
        AMDStats st;
        st.original_nnz = A.nnz();
        st.original_bandwidth = bandwidth_(A);
        auto p = amd_order(A, symmetrize);
        CSR Apr = permute_(A, p);
        st.reordered_bandwidth = bandwidth_(Apr);
        st.bandwidth_reduction = (st.original_bandwidth == 0)
                                     ? 0.0
                                     : double(st.original_bandwidth - st.reordered_bandwidth) /
                                           double(st.original_bandwidth);
        st.matrix_size = A.n;
        st.inverse_permutation = inverse_permutation(p);
        st.coalesced_variables = stats_coalesced_;
        st.postponed_dense_variables = stats_dense_;
        return {p, st};
    }

  private:
    static void validate_csr_(const CSR& A) {
        if (A.n < 0)
            throw std::invalid_argument("CSR.n must be nonnegative");
        if (A.indptr.size() != static_cast< size_t >(A.n + 1))
            throw std::invalid_argument("CSR.indptr length must be n+1");
        if (A.indptr.empty())
            return;
        if (A.indptr.front() != 0)
            throw std::invalid_argument("CSR.indptr[0] must be zero");
        for (i32 i = 0; i < A.n; ++i) {
            const i32 p0 = A.indptr[static_cast< size_t >(i)];
            const i32 p1 = A.indptr[static_cast< size_t >(i) + 1];
            if (p0 > p1)
                throw std::invalid_argument("CSR.indptr must be nondecreasing");
            if (p0 < 0 || p1 < 0 || p1 > static_cast< i32 >(A.indices.size()))
                throw std::invalid_argument("CSR.indptr contains out-of-range offsets");
            for (i32 p = p0; p < p1; ++p) {
                const i32 j = A.indices[static_cast< size_t >(p)];
                if (j < 0 || j >= A.n)
                    throw std::invalid_argument("CSR.indices contains an out-of-range column");
            }
        }
        if (A.indptr.back() != static_cast< i32 >(A.indices.size()))
            throw std::invalid_argument("CSR.indptr.back() must equal indices length");
    }

    static uint64_t hash_pattern_(const std::vector< i32 >& pattern) {
        uint64_t h = 0x9e3779b97f4a7c15ULL ^ static_cast< uint64_t >(pattern.size());
        for (i32 v : pattern) {
            h = mix64_(h ^ (static_cast< uint64_t >(static_cast< u32 >(v)) + 0x9e3779b97f4a7c15ULL +
                            (h << 6) + (h >> 2)));
        }
        return h;
    }

    static std::vector< i32 > approximate_minimum_degree_order_(const CSR& A, int dense_cutoff,
                                                                i32* coalesced_out,
                                                                i32* dense_out) {
        const i32 n = A.n;
        if (n == 0)
            return {};

        std::vector< i32 > input_degree(static_cast< size_t >(n), 0);
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = A.indptr[static_cast< size_t >(i)];
                 p < A.indptr[static_cast< size_t >(i) + 1]; ++p) {
                if (A.indices[static_cast< size_t >(p)] != i)
                    ++input_degree[static_cast< size_t >(i)];
            }
        }

        // Sorted-vector adjacency instead of per-vertex unordered_set: avoids
        // hashmap allocation/rehash overhead on the O(n) build below. Each
        // row is appended to (both directions) then sorted+deduped once,
        // same as the CSR degree pass above.
        std::vector< std::vector< i32 > > adj(static_cast< size_t >(n));
        for (i32 i = 0; i < n; ++i)
            adj[static_cast< size_t >(i)].reserve(
                static_cast< size_t >(input_degree[static_cast< size_t >(i)]));
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = A.indptr[static_cast< size_t >(i)];
                 p < A.indptr[static_cast< size_t >(i) + 1]; ++p) {
                const i32 j = A.indices[static_cast< size_t >(p)];
                if (i == j)
                    continue;
                adj[static_cast< size_t >(i)].push_back(j);
                adj[static_cast< size_t >(j)].push_back(i);
            }
        }
        for (i32 i = 0; i < n; ++i) {
            auto& row = adj[static_cast< size_t >(i)];
            std::sort(row.begin(), row.end());
            dedup_sorted_inplace(row);
        }

        std::unordered_map< uint64_t, std::vector< i32 > > groups_by_hash;
        groups_by_hash.reserve(static_cast< size_t >(n) * 2);
        std::vector< std::vector< i32 > > group_members;
        std::vector< std::vector< i32 > > group_patterns;
        std::vector< i32 > vertex_group(static_cast< size_t >(n), -1);
        std::vector< i32 > pattern;
        for (i32 v = 0; v < n; ++v) {
            pattern.assign(adj[static_cast< size_t >(v)].begin(),
                           adj[static_cast< size_t >(v)].end());
            std::sort(pattern.begin(), pattern.end());

            const uint64_t h = hash_pattern_(pattern);
            i32 g = -1;
            auto& bucket = groups_by_hash[h];
            for (i32 candidate : bucket) {
                if (group_patterns[static_cast< size_t >(candidate)] == pattern) {
                    g = candidate;
                    break;
                }
            }
            if (g < 0) {
                g = static_cast< i32 >(group_members.size());
                bucket.push_back(g);
                group_members.push_back({});
                group_patterns.push_back(pattern);
            }
            vertex_group[static_cast< size_t >(v)] = g;
            group_members[static_cast< size_t >(g)].push_back(v);
        }

        const i32 ng = static_cast< i32 >(group_members.size());
        std::vector< i32 > weight(static_cast< size_t >(ng), 0);
        for (i32 g = 0; g < ng; ++g) {
            auto& members = group_members[static_cast< size_t >(g)];
            std::sort(members.begin(), members.end());
            weight[static_cast< size_t >(g)] = static_cast< i32 >(members.size());
        }
        if (coalesced_out)
            *coalesced_out = n - ng;

        std::vector< std::unordered_set< i32 > > gadj(static_cast< size_t >(ng));
        for (i32 g = 0; g < ng; ++g) {
            gadj[static_cast< size_t >(g)].reserve(group_patterns[static_cast< size_t >(g)].size());
        }
        for (i32 v = 0; v < n; ++v) {
            const i32 gv = vertex_group[static_cast< size_t >(v)];
            for (i32 u : adj[static_cast< size_t >(v)]) {
                const i32 gu = vertex_group[static_cast< size_t >(u)];
                if (gv != gu)
                    gadj[static_cast< size_t >(gv)].insert(gu);
            }
        }

        std::vector< i32 > initial_degree(static_cast< size_t >(ng), 0);
        std::vector< i32 > degree(static_cast< size_t >(ng), 0);
        i64 degree_sum = 0;
        for (i32 g = 0; g < ng; ++g) {
            i32 d = 0;
            for (i32 h : gadj[static_cast< size_t >(g)])
                d += weight[static_cast< size_t >(h)];
            initial_degree[static_cast< size_t >(g)] = d;
            degree[static_cast< size_t >(g)] = d;
            degree_sum += d;
        }

        struct HeapNode {
            i32 degree;
            i32 initial_degree;
            i32 vertex;
        };
        struct HeapGreater {
            bool operator()(const HeapNode& a, const HeapNode& b) const {
                if (a.degree != b.degree)
                    return a.degree > b.degree;
                if (a.initial_degree != b.initial_degree)
                    return a.initial_degree > b.initial_degree;
                return a.vertex > b.vertex;
            }
        };

        const double avg_degree = (ng > 0) ? double(degree_sum) / double(ng) : 0.0;
        const i32 dense_threshold =
            (dense_cutoff == 0)
                ? std::numeric_limits< i32 >::max()
                : ((dense_cutoff > 0)
                       ? dense_cutoff
                       : std::max< i32 >(
                             16, std::min< i32 >(
                                     n - 1, static_cast< i32 >(std::floor(
                                                0.35 * avg_degree +
                                                12.0 * std::sqrt(std::max(1.0, avg_degree)))))));

        std::vector< char > active(static_cast< size_t >(ng), 1);
        std::vector< char > dense(static_cast< size_t >(ng), 0);
        std::priority_queue< HeapNode, std::vector< HeapNode >, HeapGreater > heap;
        i32 dense_count = 0;
        i32 active_groups = ng;
        for (i32 g = 0; g < ng; ++g) {
            if (degree[static_cast< size_t >(g)] >= dense_threshold) {
                dense[static_cast< size_t >(g)] = 1;
                ++dense_count;
                continue;
            }
            heap.push(
                {degree[static_cast< size_t >(g)], initial_degree[static_cast< size_t >(g)], g});
        }
        if (dense_out)
            *dense_out = dense_count;

        std::vector< i32 > perm;
        perm.reserve(static_cast< size_t >(n));
        std::vector< i32 > nbrs;

        while (active_groups > 0) {
            i32 best = -1;
            while (!heap.empty()) {
                const HeapNode top = heap.top();
                heap.pop();
                const size_t g = static_cast< size_t >(top.vertex);
                if (!active[g] || dense[g])
                    continue;
                if (top.degree != degree[g])
                    continue;
                if (top.initial_degree != initial_degree[g])
                    continue;
                best = top.vertex;
                break;
            }

            if (best < 0) {
                i32 best_degree = std::numeric_limits< i32 >::max();
                for (i32 g = 0; g < ng; ++g) {
                    if (!active[static_cast< size_t >(g)])
                        continue;
                    const i32 d = degree[static_cast< size_t >(g)];
                    if (d < best_degree ||
                        (d == best_degree &&
                         (best < 0 ||
                          initial_degree[static_cast< size_t >(g)] <
                              initial_degree[static_cast< size_t >(best)] ||
                          (initial_degree[static_cast< size_t >(g)] ==
                               initial_degree[static_cast< size_t >(best)] &&
                           group_members[static_cast< size_t >(g)].front() <
                               group_members[static_cast< size_t >(best)].front())))) {
                        best = g;
                        best_degree = d;
                    }
                }
            }

            if (best < 0)
                throw std::logic_error("internal AMD error: no active pivot found");

            nbrs.clear();
            for (i32 u : gadj[static_cast< size_t >(best)])
                if (active[static_cast< size_t >(u)])
                    nbrs.push_back(u);
            std::sort(nbrs.begin(), nbrs.end());

            for (i32 member : group_members[static_cast< size_t >(best)])
                perm.push_back(member);
            active[static_cast< size_t >(best)] = 0;
            --active_groups;

            for (i32 u : nbrs) {
                auto& au = gadj[static_cast< size_t >(u)];
                if (au.erase(best) != 0) {
                    degree[static_cast< size_t >(u)] -= weight[static_cast< size_t >(best)];
                }
            }
            for (size_t a = 0; a < nbrs.size(); ++a) {
                const i32 u = nbrs[a];
                for (size_t b = a + 1; b < nbrs.size(); ++b) {
                    const i32 v = nbrs[b];
                    auto inserted_uv = gadj[static_cast< size_t >(u)].insert(v);
                    auto inserted_vu = gadj[static_cast< size_t >(v)].insert(u);
                    if (inserted_uv.second)
                        degree[static_cast< size_t >(u)] += weight[static_cast< size_t >(v)];
                    if (inserted_vu.second)
                        degree[static_cast< size_t >(v)] += weight[static_cast< size_t >(u)];
                }
            }
            for (i32 u : nbrs) {
                if (!dense[static_cast< size_t >(u)])
                    heap.push({degree[static_cast< size_t >(u)],
                               initial_degree[static_cast< size_t >(u)], u});
            }
            gadj[static_cast< size_t >(best)].clear();
        }
        return perm;
    }

    std::vector< i32 > perm_;
    int dense_cutoff_{-1};
    i32 stats_dense_{0};
    i32 stats_coalesced_{0};

  public:
    // A is n×n CSR (pattern-only). Return B = A[p, :][:, p].
    // EXPECTS: p[new] = old
    static CSR permute_(const CSR& A, const std::vector< i32 >& p, bool sort_cols = true,
                        bool dedup = false) {
        const i32 n = A.n;
        validate_csr_(A);
        if (p.size() != static_cast< size_t >(n))
            throw std::invalid_argument("permutation size must equal CSR.n");
        if (n == 0)
            return CSR(0);

        const auto& AI = A.indptr;
        const auto& AJ = A.indices;
        std::vector< i32 > ip = inverse_permutation(p);

        CSR B(n);
        B.indptr.assign(n + 1, 0);

        for (i32 i = 0; i < n; ++i) {
            const i32 oi = p[i];
            B.indptr[i + 1] = B.indptr[i] + (AI[oi + 1] - AI[oi]);
        }
        B.indices.resize(B.indptr.back());

        for (i32 i = 0; i < n; ++i) {
            const i32 oi = p[i];
            const i32 begA = AI[oi], endA = AI[oi + 1];
            i32 out = B.indptr[i];
            for (i32 k = begA; k < endA; ++k) {
                const i32 j_old = AJ[k];
                B.indices[out++] = ip[j_old];
            }
            if (sort_cols) {
                auto beg = B.indices.begin() + B.indptr[i];
                auto end = B.indices.begin() + out;
                std::sort(beg, end);
                if (dedup) {
                    auto new_end = std::unique(beg, end);
                    (void)new_end; // second pass compacts
                }
            }
        }

        if (dedup) {
            std::vector< i32 > nip(n + 1, 0);
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i], re = B.indptr[i + 1];
                if (re <= rb) {
                    nip[i + 1] = nip[i];
                    continue;
                }
                i32 len = 1;
                for (i32 k = rb + 1; k < re; ++k)
                    if (B.indices[k] != B.indices[k - 1])
                        ++len;
                nip[i + 1] = nip[i] + len;
            }
            std::vector< i32 > nidx(nip.back());
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i], re = B.indptr[i + 1];
                i32 out = nip[i];
                if (re > rb) {
                    nidx[out++] = B.indices[rb];
                    for (i32 k = rb + 1; k < re; ++k)
                        if (B.indices[k] != B.indices[k - 1])
                            nidx[out++] = B.indices[k];
                }
            }
            B.indptr.swap(nip);
            B.indices.swap(nidx);
        }

        return B;
    }

    static i32 bandwidth_(const CSR& A) {
        if (A.nnz() == 0)
            return 0;
        i32 bw = 0;
        for (i32 i = 0; i < A.n; ++i) {
            for (i32 p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
                const i32 j = A.indices[p];
                bw = std::max(bw, std::abs(j - i));
            }
        }
        return bw;
    }
};

#endif // AMD_H
