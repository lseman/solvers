// schur_frontal_eigen_interop.h
// Interop layer between Eigen and Schur-frontal standalone LDLT.
//
// Provides conversions and templated wrapper for using Eigen sparse matrices
// with the standalone Schur-frontal algorithm.

#pragma once

#ifndef SCHUR_FRONTAL_EIGEN_INTEROP_H
#define SCHUR_FRONTAL_EIGEN_INTEROP_H

// GCC 16 <cstring> bug fix: <string.h> must precede all other includes.
#include <string.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../common/ordering.h"
#include "../ldlt/schur_frontal_ldlt.h"

namespace schur_frontal {

// Convert Eigen::SparseMatrix to standalone CSC
template < typename Scalar, typename StorageIndex >
inline SparseCSC< Scalar, int32_t >
eigen_to_csc(const Eigen::SparseMatrix< Scalar, Eigen::ColMajor, StorageIndex >& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    Int n = static_cast< Int >(A.rows());
    SparseCSC< Scalar, int32_t > csc;
    csc.n = n;

    csc.Ap.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    for (auto& p : csc.Ap)
        p = static_cast< int32_t >(p);

    csc.Ai.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    for (auto& i : csc.Ai)
        i = static_cast< int32_t >(i);

    csc.Ax.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());

    return csc;
}

// Convert standalone CSC to Eigen::SparseMatrix
template < typename Scalar >
inline Eigen::SparseMatrix< Scalar, Eigen::ColMajor, int32_t >
csc_to_eigen(const SparseCSC< Scalar, int32_t >& csc) {
    Eigen::SparseMatrix< Scalar, Eigen::ColMajor, int32_t > A(csc.n, csc.n);

    std::vector< Eigen::Triplet< Scalar, int32_t > > trips;
    trips.reserve(csc.nnz());

    for (int32_t j = 0; j < csc.n; ++j) {
        for (int32_t p = csc.Ap[static_cast< size_t >(j)]; p < csc.Ap[static_cast< size_t >(j + 1)];
             ++p) {
            int32_t i = csc.Ai[static_cast< size_t >(p)];
            Scalar v = csc.Ax[static_cast< size_t >(p)];
            trips.emplace_back(i, j, v);
        }
    }

    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}

// Full frontal factorization: left-looking sparse LDL^T with AMD ordering.
// Fills result.L (CSC, strict lower), result.diag (D), result.perm/iperm.
inline FrontalLDLT factor_frontal(const Eigen::SparseMatrix< double, Eigen::ColMajor, int32_t >& A,
                                  Real pivot_tolerance = 1e-14) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("Matrix must be square");

    FrontalLDLT result;
    Int n = static_cast< Int >(A.rows());
    result.n = n;
    result.pivot_tolerance = pivot_tolerance;

    if (n == 0) {
        result.factorized = true;
        result.L.n = 0;
        result.L.Ap.assign(1, 0);
        return result;
    }

    // ── AMD ordering ─────────────────────────────────────────────────────
    std::vector< std::pair< Int, Int > > edges;
    edges.reserve(static_cast< size_t >(A.nonZeros()));
    for (Int j = 0; j < n; ++j) {
        for (Eigen::SparseMatrix< double, Eigen::ColMajor, int32_t >::InnerIterator it(A, j); it;
             ++it) {
            Int i = static_cast< Int >(it.row());
            if (i != j) {
                edges.emplace_back(std::min(i, j), std::max(i, j));
            }
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector< Int > perm_vec;
    if (n > 20) {
        perm_vec = linsys::amd_ordering(n, edges);
    } else {
        perm_vec.resize(static_cast< size_t >(n));
        std::iota(perm_vec.begin(), perm_vec.end(), Int{0});
    }

    result.perm = perm_vec; // perm[old] = new
    result.iperm.resize(static_cast< size_t >(n), Int{-1});
    for (Int i = 0; i < n; ++i)
        result.iperm[static_cast< size_t >(perm_vec[static_cast< size_t >(i)])] = i;

    // ── Build permuted lower-triangle columns ─────────────────────────────
    // The permuted system: row_new = perm[row_old], col_new = perm[col_old].
    // We only need the lower triangle (row_new >= col_new).
    const auto& iperm = result.iperm;

    using ColEntry = std::pair< Int, Real >; // (row_new, value)
    std::vector< std::vector< ColEntry > > Acols(static_cast< size_t >(n));

    for (Int j = 0; j < n; ++j) {
        Int nc = iperm[static_cast< size_t >(j)];
        for (Eigen::SparseMatrix< double, Eigen::ColMajor, int32_t >::InnerIterator it(A, j); it;
             ++it) {
            Int nr = iperm[static_cast< size_t >(it.row())];
            if (nr < nc)
                continue; // skip strict upper triangle
            Acols[static_cast< size_t >(nc)].emplace_back(nr, it.value());
        }
    }

    // Sort and accumulate duplicate entries per column.
    for (Int k = 0; k < n; ++k) {
        auto& col = Acols[static_cast< size_t >(k)];
        std::sort(col.begin(), col.end(),
                  [](const ColEntry& a, const ColEntry& b) { return a.first < b.first; });
        size_t w = 1;
        for (size_t i = 1; i < col.size(); ++i) {
            if (col[i].first == col[w - 1].first)
                col[w - 1].second += col[i].second;
            else
                col[w++] = col[i];
        }
        if (!col.empty())
            col.resize(w);
    }

    // ── Left-looking LDL^T ────────────────────────────────────────────────
    result.diag.assign(static_cast< size_t >(n), Real(0));

    // Lcols[j]: strict lower column j of L (sorted by row).
    std::vector< std::vector< ColEntry > > Lcols(static_cast< size_t >(n));
    // rowToColumns[i]: list of column indices j < i where L[i,j] != 0.
    std::vector< std::vector< Int > > rowToColumns(static_cast< size_t >(n));

    std::vector< ColEntry > w_work;
    std::vector< ColEntry > merge_buf;
    w_work.reserve(static_cast< size_t >(n));
    merge_buf.reserve(static_cast< size_t >(n));

    auto lookupRow = [](const std::vector< ColEntry >& col, Int row) -> Real {
        auto it = std::lower_bound(
            col.begin(), col.end(), ColEntry{row, Real(0)},
            [](const ColEntry& a, const ColEntry& b) { return a.first < b.first; });
        return (it != col.end() && it->first == row) ? it->second : Real(0);
    };

    for (Int k = 0; k < n; ++k) {
        // Start with column k of permuted A (lower triangle).
        w_work = Acols[static_cast< size_t >(k)];

        // Left-looking updates from all columns j < k that have an entry in row k.
        for (Int j : rowToColumns[static_cast< size_t >(k)]) {
            Real L_kj = lookupRow(Lcols[static_cast< size_t >(j)], k);
            if (L_kj == Real(0))
                continue;
            Real alpha = result.diag[static_cast< size_t >(j)] * L_kj;

            const auto& Lj = Lcols[static_cast< size_t >(j)];
            // Only entries with row >= k contribute to column k and below.
            auto start = std::lower_bound(
                Lj.begin(), Lj.end(), ColEntry{k, Real(0)},
                [](const ColEntry& a, const ColEntry& b) { return a.first < b.first; });

            merge_buf.clear();
            auto wi = w_work.begin();
            auto li = start;
            while (wi != w_work.end() && li != Lj.end()) {
                if (wi->first < li->first) {
                    merge_buf.push_back(*wi++);
                } else if (wi->first > li->first) {
                    merge_buf.emplace_back(li->first, -li->second * alpha);
                    ++li;
                } else {
                    merge_buf.emplace_back(wi->first, wi->second - li->second * alpha);
                    ++wi;
                    ++li;
                }
            }
            while (wi != w_work.end())
                merge_buf.push_back(*wi++);
            while (li != Lj.end()) {
                merge_buf.emplace_back(li->first, -li->second * alpha);
                ++li;
            }
            w_work = std::move(merge_buf);
        }

        // Extract the diagonal entry (row == k).
        Real d = Real(0);
        {
            auto dit = std::lower_bound(
                w_work.begin(), w_work.end(), ColEntry{k, Real(0)},
                [](const ColEntry& a, const ColEntry& b) { return a.first < b.first; });
            if (dit != w_work.end() && dit->first == k) {
                d = dit->second;
                w_work.erase(dit);
            }
        }

        // Regularize near-zero pivot.
        if (std::abs(d) < pivot_tolerance)
            d = (d < Real(0)) ? -pivot_tolerance : pivot_tolerance;
        result.diag[static_cast< size_t >(k)] = d;

        // Compute L[i,k] = w[i] / D[k] for i > k.
        auto& Lk = Lcols[static_cast< size_t >(k)];
        Lk.reserve(w_work.size());
        for (const auto& e : w_work) {
            if (e.first <= k)
                continue;
            Real lij = e.second / d;
            if (lij == Real(0))
                continue;
            Lk.emplace_back(e.first, lij);
            rowToColumns[static_cast< size_t >(e.first)].push_back(k);
        }
    }

    // ── Pack L into CSC ───────────────────────────────────────────────────
    result.L.n = n;
    result.L.Ap.resize(static_cast< size_t >(n) + 1, Int{0});
    for (Int j = 0; j < n; ++j)
        result.L.Ap[static_cast< size_t >(j) + 1] =
            result.L.Ap[static_cast< size_t >(j)] +
            static_cast< Int >(Lcols[static_cast< size_t >(j)].size());

    Int nnzL = result.L.Ap[static_cast< size_t >(n)];
    result.L.Ai.resize(static_cast< size_t >(nnzL));
    result.L.Ax.resize(static_cast< size_t >(nnzL));
    for (Int j = 0; j < n; ++j) {
        Int p = result.L.Ap[static_cast< size_t >(j)];
        for (const auto& e : Lcols[static_cast< size_t >(j)]) {
            result.L.Ai[static_cast< size_t >(p)] = e.first;
            result.L.Ax[static_cast< size_t >(p)] = e.second;
            ++p;
        }
    }

    result.elimination_order.resize(static_cast< size_t >(n));
    std::iota(result.elimination_order.begin(), result.elimination_order.end(), Int{0});

    result.factorized = true;
    return result;
}

} // namespace schur_frontal

#endif // SCHUR_FRONTAL_EIGEN_INTEROP_H
