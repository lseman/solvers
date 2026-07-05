/*
 * ordering.h — shared permutation (perm/iperm) container and AMD
 * fill-reducing ordering wrapper for the custom linear-system solvers.
 */

#ifndef LINSYS_ORDERING_H
#define LINSYS_ORDERING_H

// GCC 16 <cstring> bug fix: <string.h> must be the very first include so
// that ::memchr, ::memcpy etc. are in the global namespace before <cstring>
// does 'using ::memchr;'.
#include <string.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

// amd.h must be included at global scope: it includes <iostream> which
// references C symbols (::memchr, ::__libc_single_threaded etc.) that would
// otherwise resolve into the linsys namespace.
#include "amd.h"

namespace linsys {

template < typename Index = int32_t > struct Ordering {
    std::vector< Index > perm;  // size n: perm[old] = new
    std::vector< Index > iperm; // size n: iperm[new] = old
    Index n = 0;

    static Ordering identity(Index n) {
        Ordering o;
        o.n = n;
        o.perm.resize(static_cast< size_t >(n));
        o.iperm.resize(static_cast< size_t >(n));
        std::iota(o.perm.begin(), o.perm.end(), Index{0});
        std::iota(o.iperm.begin(), o.iperm.end(), Index{0});
        return o;
    }

    /// Build from a permutation array (old->new). Validates and computes iperm.
    static Ordering from_perm(std::vector< Index > p) {
        Ordering o;
        o.n = static_cast< Index >(p.size());
        o.perm = std::move(p);
        o.iperm.assign(static_cast< size_t >(o.n), Index{-1});

        for (Index i = 0; i < o.n; ++i) {
            const Index pi = o.perm[static_cast< size_t >(i)];
            if (pi < 0 || pi >= o.n || o.iperm[static_cast< size_t >(pi)] != Index{-1})
                throw std::invalid_argument("linsys: invalid permutation");
            o.iperm[static_cast< size_t >(pi)] = i;
        }
        return o;
    }
};

/// AMD ordering via amd.h's AMDReorderingArray.
/// Builds CSR from (row,col) edge list, calls AMD, returns perm[old] = new.
inline std::vector< int32_t >
amd_ordering(int32_t n, const std::vector< std::pair< int32_t, int32_t > >& edges) {
    if (n <= 0)
        return {};

    // Build CSR from upper-triangular edge list.
    std::vector< int32_t > row_counts(static_cast< size_t >(n), 0);
    for (const auto& e : edges) {
        if (e.first >= 0 && e.first < n && e.second >= 0 && e.second < n) {
            row_counts[static_cast< size_t >(e.first)]++;
        }
    }

    std::vector< int32_t > row_indptr(static_cast< size_t >(n) + 1, 0);
    for (int32_t i = 0; i < n; ++i) {
        row_indptr[static_cast< size_t >(i) + 1] =
            row_indptr[static_cast< size_t >(i)] + row_counts[static_cast< size_t >(i)];
    }
    int32_t nnz = row_indptr[static_cast< size_t >(n)];

    std::vector< int32_t > row_indices(static_cast< size_t >(nnz));
    std::vector< int32_t > row_cur(static_cast< size_t >(n), 0);
    for (const auto& e : edges) {
        if (e.first >= 0 && e.first < n && e.second >= 0 && e.second < n) {
            row_indices[static_cast< size_t >(row_indptr[static_cast< size_t >(e.first)] +
                                              row_cur[static_cast< size_t >(e.first)]++)] =
                e.second;
        }
    }

    CSR csr;
    csr.n = n;
    csr.indptr = std::move(row_indptr);
    csr.indices = std::move(row_indices);

    AMDReorderingArray amd_orderer;
    auto perm = amd_orderer.amd_order(csr, true); // symmetrize
    return perm;
}

} // namespace linsys

#endif // LINSYS_ORDERING_H
