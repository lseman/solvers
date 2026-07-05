/*
 * dense_matrix.h — shared column-major dense matrix for frontal/supernode
 * LDLᵀ factorization. Wraps std::vector<Scalar> with (i,j) accessors.
 */

#ifndef LINSYS_DENSE_MATRIX_H
#define LINSYS_DENSE_MATRIX_H

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace linsys {

template < typename Scalar = double >
struct DenseMatrix {
    std::vector< Scalar > data;
    int32_t rows = 0;
    int32_t cols = 0;

    DenseMatrix() = default;
    explicit DenseMatrix(int32_t m, int32_t n)
        : data(static_cast< size_t >(m) * static_cast< size_t >(n)), rows(m), cols(n) {}

    inline Scalar& operator()(int32_t i, int32_t j) {
        return data[static_cast< size_t >(j) * static_cast< size_t >(rows) + static_cast< size_t >(i)];
    }
    inline const Scalar& operator()(int32_t i, int32_t j) const {
        return data[static_cast< size_t >(j) * static_cast< size_t >(rows) + static_cast< size_t >(i)];
    }

    void resize(int32_t m, int32_t n) {
        rows = m;
        cols = n;
        data.assign(static_cast< size_t >(m) * static_cast< size_t >(n), Scalar(0));
    }
    void setZero() {
        std::fill(data.begin(), data.end(), Scalar(0));
    }
};

} // namespace linsys

#endif // LINSYS_DENSE_MATRIX_H
