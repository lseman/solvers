/*
 * normal_eq_ldlt.h — Normal-equations IPM solver.
 *
 * Eliminates Delta_x from the augmented system
 *
 *   [ -(Theta + R_p)   A^T ] [dx]   [xi_d]
 *   [       A           R_d] [dy] = [xi_p]
 *
 * via dx = D^-1 (A^T dy - xi_d), D = Theta + R_p, giving the SPD normal
 * equations
 *
 *   (A D^-1 A^T + R_d) dy = xi_p + A D^-1 xi_d
 *
 * M = A D^-1 A^T + R_d is m x m (vs. the (n+m) x (n+m) augmented system) and
 * symmetric positive definite, so it factorizes with plain AMD (no block
 * splitting needed — unlike the quasi-definite augmented system, there is
 * only one sign to preserve). Cheaper than the augmented system whenever
 * m << n or A is sparse enough that A D^-1 A^T doesn't fill in badly; the
 * caller (sparse_solver.h) picks between the two via a fill-in estimate.
 */

#pragma once

#ifndef IPM_NORMAL_EQ_LDLT_H
#define IPM_NORMAL_EQ_LDLT_H

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <utility>
#include <vector>
#include "../linear_system/common/sparse_csc.h"
#include "../linear_system/ldlt/supernodal_ldlt.h"

namespace ipm {

using NEReal = double;
using NEIndex = int32_t;

/// Estimate nnz(A D^-1 A^T) upper triangle without forming it, and nnz of the
/// augmented system's upper triangle, so callers can pick the cheaper path
/// from the (IPM-invariant) sparsity pattern alone.
struct FillEstimate {
    long long normal_eq_nnz = 0;
    long long augmented_nnz = 0;
};

inline FillEstimate
estimate_fill(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& A) {
    const int m = A.rows();
    const int n = A.cols();

    // Row -> cols incidence (structural pattern of A).
    std::vector< int > row_degree(static_cast< size_t >(m), 0);
    for (int j = 0; j < n; ++j) {
        for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(A, j); it; ++it) {
            row_degree[static_cast< size_t >(it.row())]++;
        }
    }

    // Upper bound on nnz(upper(A A^T)): each row of A with degree d
    // contributes at most d*(d+1)/2 upper-triangle entries (row-col pairs it
    // induces via the outer product of its nonzero columns). This is a loose
    // bound (entries from different rows may coincide, reducing true fill)
    // but cheap to compute from the pattern alone and enough to pick a path.
    long long fanout_pairs = 0;
    for (int i = 0; i < m; ++i) {
        const long long deg = row_degree[static_cast< size_t >(i)];
        fanout_pairs += deg * (deg + 1) / 2;
    }
    const long long normal_eq_nnz = std::min< long long >(fanout_pairs, static_cast< long long >(m) * m);

    const long long aug_upper = static_cast< long long >(n) /*diag*/ +
                                static_cast< long long >(m) /*diag*/ +
                                static_cast< long long >(A.nonZeros());

    FillEstimate est;
    est.normal_eq_nnz = normal_eq_nnz;
    est.augmented_nnz = aug_upper;
    return est;
}

/**
 * SPD normal-equations solver for the IPM augmented system.
 */
class NormalEqLDLT {
  public:
    NormalEqLDLT() : m_n(0), m_m(0) {
    }

    void analyzePattern(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& A) {
        m_A = A;
        m_n = A.cols();
        m_m = A.rows();
        m_At = Eigen::SparseMatrix< double, Eigen::ColMajor, int >(m_A.transpose());
        m_ldlt.reset();
        m_ldlt.setRegularization(1e-11);
        // Full-matrix AMD is safe here: M is SPD, every pivot has the same
        // sign, so there is no block-boundary sign-change risk (unlike the
        // augmented system).
    }

    void factorize(const std::vector< NEReal >& theta, const std::vector< NEReal >& regP,
                  const std::vector< NEReal >& regD) {
        if (m_n == 0 || m_m == 0)
            return;

        m_Dinv.assign(static_cast< size_t >(m_n), 0.0);
        for (int j = 0; j < m_n; ++j) {
            const NEReal d = theta[static_cast< size_t >(j)] + regP[static_cast< size_t >(j)];
            m_Dinv[static_cast< size_t >(j)] = (d != 0.0) ? 1.0 / d : 0.0;
        }

        // M = A * Dinv * A^T + diag(regD), full symmetric matrix — both
        // triangles are built explicitly since SupernodalLDLT::factorizeMatrix
        // expects a full symmetric CSC (it extracts the lower triangle
        // itself; see buildPermutedLowerColumns in supernodal_ldlt.h).
        std::vector< Eigen::Triplet< double > > triplets;
        triplets.reserve(static_cast< size_t >(m_A.nonZeros()) * 4);

        // For each primal column j (contraction index), the rows it touches
        // in A contribute a rank-1 update scaled by Dinv[j] to M.
        for (int j = 0; j < m_n; ++j) {
            const NEReal scale = m_Dinv[static_cast< size_t >(j)];
            if (scale == 0.0)
                continue;
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator ii(m_A, j); ii;
                 ++ii) {
                const int row_i = ii.row();
                const double val_i = ii.value();
                for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator jj(m_A, j);
                     jj; ++jj) {
                    const int row_j = jj.row();
                    triplets.emplace_back(row_i, row_j, val_i * jj.value() * scale);
                }
            }
        }
        for (int i = 0; i < m_m; ++i) {
            triplets.emplace_back(i, i, regD[static_cast< size_t >(i)]);
        }

        Eigen::SparseMatrix< double, Eigen::ColMajor, int > M(m_m, m_m);
        M.setFromTriplets(triplets.begin(), triplets.end(),
                          [](const double& a, const double& b) { return a + b; });
        M.makeCompressed();

        linsys::SparseCSC< NEReal, NEIndex > csc;
        csc.n = m_m;
        csc.Ap.assign(static_cast< size_t >(m_m) + 1, 0);
        for (int j = 0; j < m_m; ++j) {
            csc.Ap[static_cast< size_t >(j) + 1] =
                csc.Ap[static_cast< size_t >(j)] + static_cast< NEIndex >(M.col(j).nonZeros());
        }
        csc.Ai.reserve(static_cast< size_t >(M.nonZeros()));
        csc.Ax.reserve(static_cast< size_t >(M.nonZeros()));
        for (int j = 0; j < m_m; ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(M, j); it;
                 ++it) {
                csc.Ai.push_back(it.row());
                csc.Ax.push_back(it.value());
            }
        }

        m_ldlt.factorizeMatrix(csc);
        m_theta = theta;
        m_regP = regP;
    }

    /**
     * Solve the augmented system given RHS halves (xi_d for dx's equation,
     * xi_p for dy's equation). Returns (dx, dy).
     */
    std::pair< std::vector< NEReal >, std::vector< NEReal > >
    solve(const std::vector< NEReal >& xi_d, const std::vector< NEReal >& xi_p) const {
        if (m_n == 0 || m_m == 0 || !m_ldlt.isFactorized())
            return {};

        // rhs_y = xi_p + A * Dinv * xi_d
        std::vector< NEReal > rhs_y(xi_p);
        for (int j = 0; j < m_n; ++j) {
            const NEReal scaled = m_Dinv[static_cast< size_t >(j)] * xi_d[static_cast< size_t >(j)];
            if (scaled == 0.0)
                continue;
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                rhs_y[static_cast< size_t >(it.row())] += it.value() * scaled;
            }
        }

        std::vector< NEReal > dy = m_ldlt.solve(rhs_y);

        // dx = Dinv * (A^T dy - xi_d)
        std::vector< NEReal > dx(static_cast< size_t >(m_n));
        for (int j = 0; j < m_n; ++j) {
            double atdy = 0.0;
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(m_A, j); it;
                 ++it) {
                atdy += it.value() * dy[static_cast< size_t >(it.row())];
            }
            dx[static_cast< size_t >(j)] =
                m_Dinv[static_cast< size_t >(j)] * (atdy - xi_d[static_cast< size_t >(j)]);
        }

        return {std::move(dx), std::move(dy)};
    }

    int n() const {
        return m_n;
    }
    int m() const {
        return m_m;
    }

  private:
    int m_n;
    int m_m;
    Eigen::SparseMatrix< double, Eigen::ColMajor, int > m_A;
    Eigen::SparseMatrix< double, Eigen::ColMajor, int > m_At;
    std::vector< NEReal > m_Dinv;
    std::vector< NEReal > m_theta;
    std::vector< NEReal > m_regP;
    supernodal::SupernodalLDLT< NEReal, NEIndex > m_ldlt;
};

} // namespace ipm

#endif // IPM_NORMAL_EQ_LDLT_H
