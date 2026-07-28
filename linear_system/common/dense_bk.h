/*
 * dense_bk.h — shared dense/frontal Bunch-Kaufman factorization kernel
 *
 * Factors the leading pivot_count columns of a symmetric column-major front:
 *
 *   P F P^T = L D L^T
 *
 * Rows below pivot_count participate in the multipliers but are not pivot
 * candidates. This makes the same kernel usable for a complete dense
 * factorization and for intranodal factorization of a supernodal front.
 */

#ifndef LDLT_DENSE_BK_H
#define LDLT_DENSE_BK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dense_matrix.h"

namespace ldlt::detail {

template < typename Scalar, typename Index >
struct DenseBunchKaufmanOptions {
    Scalar absolute_regularization = Scalar(0);
    Scalar relative_regularization = Scalar(0);
    Scalar inertia_tolerance = Scalar(0);
    bool regularize_singular_pivots = false;
    bool strict_pivots = false;
    Index global_column_begin = Index(0);
    const std::vector< int8_t >* expected_pivot_signs = nullptr;

    // Optional extra regularization floor, folded in via max() with the
    // abs/rel threshold before a pivot is judged singular (e.g. a
    // sign-preservation bound computed from the trailing front, as in
    // Zanetti & Gondzio's Proposition 3). local_col is the pivot-block-local
    // column index k; default_threshold is what would be used without this
    // hook. Left unset, behavior is identical to today's.
    std::function< Scalar(Index local_col, Scalar default_threshold) > custom_regularization_floor;

    // When a 2x2 pivot is judged singular: false (default) shifts d11/d22 by
    // +-threshold and recomputes the determinant (today's behavior); true
    // instead replaces the determinant directly with a signed floor
    // (matches Proposition 3's guarantee, which bounds the Schur complement
    // via the determinant rather than the individual diagonal entries).
    bool direct_2x2_replacement = false;
};

template < typename Scalar, typename Index >
struct DenseBunchKaufmanResult {
    std::vector< Scalar > diagonal;
    std::vector< Scalar > subdiagonal;
    std::vector< int8_t > blocks;
    std::vector< Index > permutation;
    Index positive_inertia = 0;
    Index negative_inertia = 0;
    Index zero_inertia = 0;
    Index perturbed_pivots = 0;
    Scalar min_abs_pivot = Scalar(0);
};

template < typename Scalar, typename Index >
DenseBunchKaufmanResult< Scalar, Index >
denseBunchKaufman(linsys::DenseMatrix< Scalar >& front, Index pivot_count,
                  const DenseBunchKaufmanOptions< Scalar, Index >& options = {}) {
    const Index front_size = static_cast< Index >(front.rows);
    if (front_size < 0 || pivot_count < 0 || pivot_count > front_size ||
        front.cols < front.rows) {
        throw std::invalid_argument("dense BK: invalid front dimensions");
    }

    DenseBunchKaufmanResult< Scalar, Index > result;
    const size_t pivot_size = static_cast< size_t >(pivot_count);
    result.diagonal.assign(pivot_size, Scalar(0));
    result.subdiagonal.assign(pivot_size, Scalar(0));
    result.blocks.assign(pivot_size, int8_t{1});
    result.permutation.resize(pivot_size);
    std::iota(result.permutation.begin(), result.permutation.end(), Index(0));

    // The lower triangle is authoritative on entry. Keeping the leading rows
    // symmetric makes full row/column swaps straightforward and also permutes
    // multipliers from already factored columns.
    for (Index j = 0; j < pivot_count; ++j)
        for (Index i = j + 1; i < front_size; ++i)
            front(j, i) = front(i, j);

    auto symmetric_swap = [&](Index lhs, Index rhs) {
        if (lhs == rhs)
            return;
        for (Index j = 0; j < front_size; ++j)
            std::swap(front(lhs, j), front(rhs, j));
        for (Index i = 0; i < front_size; ++i)
            std::swap(front(i, lhs), front(i, rhs));
        std::swap(result.permutation[static_cast< size_t >(lhs)],
                  result.permutation[static_cast< size_t >(rhs)]);
    };

    auto update_minimum = [&](Scalar magnitude) {
        if (result.min_abs_pivot == Scalar(0) ||
            magnitude < result.min_abs_pivot)
            result.min_abs_pivot = magnitude;
    };

    auto classify = [&](Scalar eigenvalue, Scalar tolerance) {
        if (eigenvalue > tolerance)
            ++result.positive_inertia;
        else if (eigenvalue < -tolerance)
            ++result.negative_inertia;
        else
            ++result.zero_inertia;
    };

    auto regularize_1x1 = [&](Scalar value, Index local, Scalar scale) {
        const Scalar threshold =
            std::max(options.absolute_regularization,
                     options.relative_regularization * scale);
        if (!options.regularize_singular_pivots ||
            (std::isfinite(value) && std::abs(value) > threshold))
            return value;
        if (options.strict_pivots)
            throw std::domain_error("dense BK: zero or non-finite 1x1 pivot");
        int sign = value < Scalar(0) ? -1 : 1;
        if (options.expected_pivot_signs != nullptr) {
            const size_t original = static_cast< size_t >(
                options.global_column_begin +
                result.permutation[static_cast< size_t >(local)]);
            if (original < options.expected_pivot_signs->size() &&
                (*options.expected_pivot_signs)[original] != 0)
                sign = (*options.expected_pivot_signs)[original];
        }
        ++result.perturbed_pivots;
        Scalar floor_value =
            threshold > Scalar(0) ? threshold
                                  : std::numeric_limits< Scalar >::epsilon();
        if (options.custom_regularization_floor)
            floor_value = std::max(floor_value, options.custom_regularization_floor(local, floor_value));
        return static_cast< Scalar >(sign) * floor_value;
    };

    constexpr Scalar alpha =
        static_cast< Scalar >((1.0 + 4.1231056256176605498) / 8.0);
    std::vector< Scalar > column0(static_cast< size_t >(front_size));
    std::vector< Scalar > column1(static_cast< size_t >(front_size));

    Index k = 0;
    while (k < pivot_count) {
        Scalar lambda = Scalar(0);
        Index pivot_row = k;
        for (Index i = k + 1; i < pivot_count; ++i) {
            const Scalar candidate = std::abs(front(i, k));
            if (candidate > lambda) {
                lambda = candidate;
                pivot_row = i;
            }
        }

        const Scalar abs_diagonal = std::abs(front(k, k));
        bool use_2x2 = false;
        Index swap_row = k;
        if (lambda > Scalar(0) && abs_diagonal < alpha * lambda) {
            Scalar sigma = Scalar(0);
            for (Index i = k; i < pivot_count; ++i)
                if (i != pivot_row)
                    sigma = std::max(sigma, std::abs(front(i, pivot_row)));
            if (abs_diagonal * sigma >= alpha * lambda * lambda) {
                // Keep the current 1x1 pivot.
            } else if (std::abs(front(pivot_row, pivot_row)) >= alpha * sigma) {
                swap_row = pivot_row;
            } else if (k + 1 < pivot_count) {
                use_2x2 = true;
            }
        }

        if (!use_2x2) {
            symmetric_swap(k, swap_row);
            Scalar column_scale = std::abs(front(k, k));
            for (Index i = k + 1; i < front_size; ++i)
                column_scale =
                    std::max(column_scale, std::abs(front(i, k)));
            const Scalar d = regularize_1x1(front(k, k), k, column_scale);
            front(k, k) = d;
            result.diagonal[static_cast< size_t >(k)] = d;
            result.blocks[static_cast< size_t >(k)] = int8_t{1};
            classify(d, options.inertia_tolerance);
            update_minimum(std::abs(d));

            const Scalar inverse =
                d != Scalar(0) ? Scalar(1) / d : Scalar(0);
            for (Index i = k + 1; i < front_size; ++i)
                front(i, k) *= inverse;
            for (Index j = k + 1; j < pivot_count; ++j) {
                const Scalar scaled = front(j, k) * d;
                for (Index i = j; i < front_size; ++i) {
                    front(i, j) -= front(i, k) * scaled;
                    front(j, i) = front(i, j);
                }
            }
            ++k;
            continue;
        }

        symmetric_swap(k + 1, pivot_row);
        Scalar d11 = front(k, k);
        const Scalar d21 = front(k + 1, k);
        Scalar d22 = front(k + 1, k + 1);
        Scalar determinant = d11 * d22 - d21 * d21;
        const Scalar scale = std::max(
            {std::abs(d11), std::abs(d21), std::abs(d22), Scalar(1)});
        const Scalar determinant_threshold =
            std::max(options.absolute_regularization,
                     options.relative_regularization * scale);
        if (options.regularize_singular_pivots &&
            (!std::isfinite(determinant) ||
             std::abs(determinant) <=
                 determinant_threshold * determinant_threshold)) {
            if (options.strict_pivots)
                throw std::domain_error("dense BK: singular 2x2 pivot");
            Scalar floor =
                determinant_threshold > Scalar(0)
                    ? determinant_threshold
                    : std::numeric_limits< Scalar >::epsilon();
            if (options.custom_regularization_floor)
                floor = std::max(floor, options.custom_regularization_floor(k, floor));
            if (options.direct_2x2_replacement) {
                determinant = determinant < Scalar(0) ? -floor : floor;
            } else {
                d11 += d11 < Scalar(0) ? -floor : floor;
                d22 += d22 < Scalar(0) ? -floor : floor;
                determinant = d11 * d22 - d21 * d21;
            }
            ++result.perturbed_pivots;
        }

        result.diagonal[static_cast< size_t >(k)] = d11;
        result.diagonal[static_cast< size_t >(k + 1)] = d22;
        result.subdiagonal[static_cast< size_t >(k)] = d21;
        result.blocks[static_cast< size_t >(k)] = int8_t{2};
        result.blocks[static_cast< size_t >(k + 1)] = int8_t{0};
        front(k + 1, k) = Scalar(0);
        front(k, k + 1) = Scalar(0);

        const Scalar trace_half = Scalar(0.5) * (d11 + d22);
        const Scalar radius =
            std::hypot(Scalar(0.5) * (d11 - d22), d21);
        const Scalar eigenvalue0 = trace_half + radius;
        const Scalar eigenvalue1 = trace_half - radius;
        const Scalar inertia_tolerance =
            options.regularize_singular_pivots
                ? determinant_threshold
                : options.inertia_tolerance;
        classify(eigenvalue0, inertia_tolerance);
        classify(eigenvalue1, inertia_tolerance);
        update_minimum(
            std::min(std::abs(eigenvalue0), std::abs(eigenvalue1)));

        const Scalar determinant_inverse =
            determinant != Scalar(0) ? Scalar(1) / determinant : Scalar(0);
        for (Index i = k + 2; i < front_size; ++i) {
            const Scalar c0 = front(i, k);
            const Scalar c1 = front(i, k + 1);
            column0[static_cast< size_t >(i)] = c0;
            column1[static_cast< size_t >(i)] = c1;
            front(i, k) =
                (c0 * d22 - c1 * d21) * determinant_inverse;
            front(i, k + 1) =
                (-c0 * d21 + c1 * d11) * determinant_inverse;
        }
        for (Index j = k + 2; j < pivot_count; ++j) {
            const Scalar l0 = front(j, k);
            const Scalar l1 = front(j, k + 1);
            for (Index i = j; i < front_size; ++i) {
                front(i, j) -=
                    column0[static_cast< size_t >(i)] * l0 +
                    column1[static_cast< size_t >(i)] * l1;
                front(j, i) = front(i, j);
            }
        }
        k += 2;
    }

    return result;
}

} // namespace ldlt::detail

#endif // LDLT_DENSE_BK_H
