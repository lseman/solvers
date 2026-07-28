#pragma once

#ifndef LINSYS_SYMBOLIC_LDLT_H
#define LINSYS_SYMBOLIC_LDLT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse_csc.h"
#include "ordering.h"

namespace ldlt {
template < typename Scalar, typename Index > class SimplicialLDLT;
}
namespace supernodal {
template < typename Scalar, typename Index > class SupernodalLDLT;
}

namespace linsys {

class SymbolicLDLTBuilder;
template < typename Scalar, typename Index >
std::uint64_t symmetric_pattern_hash(const SparseCSC< Scalar, Index >& a);

/// Immutable, shareable symbolic analysis for sparse LDLT implementations.
///
/// The ordering and symmetric pattern identity are backend-neutral. Supernodal
/// analyzers additionally populate the elimination tree, panel layout, and
/// contributor graph. Published instances are exposed only through
/// std::shared_ptr<const SymbolicLDLT>.
class SymbolicLDLT {
  public:
    enum class Backend { Simplicial, Supernodal };

    int32_t size() const { return m_n; }
    Backend backend() const { return m_backend; }
    std::uint64_t patternHash() const { return m_pattern_hash; }
    double symbolicIntensity() const { return m_symbolic_intensity; }
    size_t structuralNonZerosL() const { return m_structural_nnz_l; }
    size_t panelStorage() const {
        return m_panel_value_ptr.empty()
                   ? 0
                   : static_cast< size_t >(m_panel_value_ptr.back());
    }
    int32_t maxFrontSize() const { return m_max_front_size; }
    int32_t maxUpdateRows() const { return m_max_update_rows; }
    size_t frontalWorkspaceSize() const {
        return static_cast< size_t >(m_max_front_size) *
               static_cast< size_t >(m_max_front_size);
    }
    const std::vector< int32_t >& permutation() const { return m_perm; }
    const std::vector< int32_t >& inversePermutation() const { return m_iperm; }
    const std::vector< std::pair< int32_t, int32_t > >& supernodeRanges() const {
        return m_supernode_ranges;
    }
    const std::vector< int32_t >& eliminationTree() const { return m_etree; }
    const std::vector< int32_t >& supernodeParents() const {
        return m_supernode_parent;
    }
    const std::vector< int32_t >& supernodePostorder() const {
        return m_supernode_post;
    }
    const std::vector< int32_t >& supernodeRowPointers() const {
        return m_supernode_row_ptr;
    }
    const std::vector< int32_t >& supernodeRows() const {
        return m_supernode_rows;
    }
    const std::vector< int32_t >& contributorPointers() const {
        return m_contributor_ptr;
    }
    const std::vector< int32_t >& contributors() const {
        return m_contributors;
    }
    const std::vector< int32_t >& panelValuePointers() const {
        return m_panel_value_ptr;
    }

  private:
    friend class SymbolicLDLTBuilder;
    template < typename, typename > friend class ::ldlt::SimplicialLDLT;
    template < typename, typename > friend class ::supernodal::SupernodalLDLT;
    SymbolicLDLT() = default;

    int32_t m_n = 0;
    std::vector< std::pair< int32_t, int32_t > > m_supernode_ranges;
    std::vector< int32_t > m_etree;
    std::vector< int32_t > m_supernode_parent;
    std::vector< int32_t > m_supernode_post;
    std::vector< int32_t > m_supernode_row_ptr;
    std::vector< int32_t > m_supernode_rows;
    std::vector< int32_t > m_contributor_ptr;
    std::vector< int32_t > m_contributors;
    std::vector< int32_t > m_panel_value_ptr;
    std::vector< int32_t > m_perm;
    std::vector< int32_t > m_iperm;
    std::uint64_t m_pattern_hash = 0;
    Backend m_backend = Backend::Simplicial;
    double m_symbolic_intensity = 0.0;
    size_t m_structural_nnz_l = 0;
    int32_t m_max_front_size = 0;
    int32_t m_max_update_rows = 0;
    int m_resolved_storage = 0;
};

class SymbolicLDLTBuilder {
  public:
    template < typename Scalar, typename Index >
    static std::shared_ptr< const SymbolicLDLT >
    analyzeSimplicial(const SparseCSC< Scalar, Index >& a) {
        if (a.n < 0)
            throw std::invalid_argument("ldlt: negative matrix dimension");
        if (a.Ap.size() != static_cast< size_t >(a.n) + 1 ||
            a.Ap.empty() || a.Ap.front() != Index{0} ||
            a.Ai.size() != a.Ax.size() ||
            static_cast< size_t >(a.Ap.back()) != a.Ai.size())
            throw std::invalid_argument("ldlt: invalid CSC structure");

        std::vector< std::pair< int32_t, int32_t > > edges;
        edges.reserve(a.Ai.size());
        for (Index col = 0; col < a.n; ++col) {
            const Index begin = a.Ap[static_cast< size_t >(col)];
            const Index end = a.Ap[static_cast< size_t >(col) + 1];
            if (begin < 0 || begin > end ||
                static_cast< size_t >(end) > a.Ai.size())
                throw std::invalid_argument("ldlt: invalid CSC column pointers");
            for (Index p = begin; p < end; ++p) {
                const Index row = a.Ai[static_cast< size_t >(p)];
                if (row < 0 || row >= a.n)
                    throw std::invalid_argument("ldlt: row index out of range");
                if (row != col)
                    edges.emplace_back(
                        static_cast< int32_t >(std::min(row, col)),
                        static_cast< int32_t >(std::max(row, col)));
            }
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        auto symbolic =
            std::shared_ptr< SymbolicLDLT >(new SymbolicLDLT);
        symbolic->m_n = static_cast< int32_t >(a.n);
        symbolic->m_backend = SymbolicLDLT::Backend::Simplicial;
        const auto ordering =
            a.n > 20
                ? Ordering< int32_t >::from_perm(
                      amd_ordering(static_cast< int32_t >(a.n), edges))
                : Ordering< int32_t >::identity(static_cast< int32_t >(a.n));
        symbolic->m_perm = ordering.perm;
        symbolic->m_iperm = ordering.iperm;
        symbolic->m_pattern_hash = symmetric_pattern_hash(a);
        return symbolic;
    }
};

template < typename Scalar, typename Index >
std::uint64_t symmetric_pattern_hash(const SparseCSC< Scalar, Index >& a) {
    std::vector< std::pair< Index, Index > > entries;
    entries.reserve(a.Ai.size());
    for (Index col = 0; col < a.n; ++col) {
        for (Index p = a.Ap[static_cast< size_t >(col)];
             p < a.Ap[static_cast< size_t >(col) + 1]; ++p) {
            const Index row = a.Ai[static_cast< size_t >(p)];
            entries.emplace_back(std::min(row, col), std::max(row, col));
        }
    }
    std::sort(entries.begin(), entries.end());
    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());

    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    mix(static_cast< std::uint64_t >(a.n));
    for (const auto& [row, col] : entries) {
        mix(static_cast< std::uint64_t >(row));
        mix(static_cast< std::uint64_t >(col));
    }
    return hash;
}

} // namespace linsys

#endif
