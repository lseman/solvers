/*
 * supernodal_ldlt_standalone.h — Eigen-free supernodal sparse LDLᵀ factorization
 *
 * Hybrid sparse LDL^T using a symbolic work-per-nonzero estimate to select
 * simplicial or supernodal factorization. The supernodal path retains dense
 * panels for BLAS updates and solves.
 * Uses snode::identify_supernodes (supernodes.h) for supernode detection.
 *
 * API mirrors the Eigen version for drop-in replacement:
 *   SparseCSC<Scalar,Index>      — compressed sparse column (upper+diag)
 *   Ordering<int>                — perm/iperm
 *   SupernodalFactor             — L (CSC) + D + metadata
 *   SupernodalLDLT<Scalar,Index> — solver class
 *
 * Algorithm:
 *   - Pattern analysis: AMD ordering (or externally supplied, see
 *     setExternalOrdering) + supernode detection via supernodes.h
 *   - Numeric factorization: automatic simplicial/supernodal selection
 *   - Solve: forward (L) + diagonal (D) + backward (Lᵀ) + permutation
 *
 * Usage:
 *   SupernodalLDLT<double,int> solver;
 *   solver.compute(csc);
 *   auto x = solver.solve(b);
 */

#ifndef SUPERSONAL_LDLT_STANDALONE_H
#define SUPERSONAL_LDLT_STANDALONE_H

// GCC 16 <cstring> bug fix: <string.h> must be the very first include so
// that ::memchr, ::memcpy etc. are in the global namespace before <cstring>
// does 'using ::memchr;'.
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(LINSYS_HAS_BLAS)
#include <cblas.h>
#endif

// ===== AMD ordering ========================================================
// amd.h must be in global scope: it includes <iostream> which references C
// symbols (::memchr, ::__libc_single_threaded etc.) that would otherwise
// resolve to supernodal:: inside the namespace block.
#include "../common/amd.h"

// ===== Supernode detection =================================================
#include "../supernodes.h"

#include "../common/dense_matrix.h"
#include "../common/dense_bk.h"
#include "../common/ordering.h"
#include "../common/sparse_csc.h"
#include "../common/symbolic_ldlt.h"
#include "../common/trisolve.h"
#include "ldlt.h"

// ===== Shared types (see linear_system/common/) ============================

namespace supernodal {

using Int = int32_t;
using Real = double;

using linsys::amd_ordering;
using linsys::lsolve_unit;
using linsys::ltsolve_unit;
using linsys::Ordering;
using linsys::permute_gather;
using linsys::SparseCSC;
using linsys::SymbolicLDLT;

// Dense matrix from shared linsys::
using linsys::DenseMatrix;

// ===== Symbolic analysis and factorization result ===========================

struct SupernodalFactor {
    // L in CSC format (unit lower triangular)
    Int n = 0;
    std::vector< Int > Lp;    // size n+1
    std::vector< Int > Li;    // size nnz(L)
    std::vector< double > Lx; // size nnz(L)

    // D diagonal
    std::vector< double > D; // size n
    std::vector< double > D_subdiag; // nonzero at starts of 2x2 blocks
    std::vector< int8_t > pivot_blocks; // 1, 2 (start), or 0 (second)
    std::vector< Int > intranodal_perm; // factor position -> unpivoted local position
    std::vector< Int > intranodal_iperm; // unpivoted position -> factor position

    std::vector< double > panel_values;

    // Metadata
    Int perturbed_pivots = 0;
    double min_abs_pivot = 0.0;
    bool pivoted = false;
    bool intranodal_pivoted = false;
    Int positive_inertia = 0;
    Int negative_inertia = 0;
    Int zero_inertia = 0;
    bool scalar_csc_materialized = false;
    bool factorized = false;

    enum Info { Success = 0, NumericalIssue = 1, NotInitialized = 2 };
    Info info_val = NotInitialized;

    size_t nnzL() const {
        return Li.size();
    }
};

// ===== Supernodal LDLᵀ solver ==============================================

template < typename Scalar = Real, typename Index = Int > class SupernodalLDLT {
  public:
    using MatrixType = SparseCSC< Scalar, Index >;
    using RealScalar = double;

    struct RelaxationThreshold {
        Index max_columns;
        double max_zero_fraction;
    };

    enum class BackendPolicy { Automatic, ForceSimplicial, ForceSupernodal };
    enum class PivotPolicy {
        Regularized1x1,
        IntranodalBunchKaufman,
        BunchKaufman
    };
    enum class SymmetricStorage { AutoDetect, Upper, Lower, FullSymmetric };

    SupernodalLDLT() = default;

    explicit SupernodalLDLT(const MatrixType& a) {
        compute(a);
    }

    void reset() {
        m_size = 0;
        m_factors = SupernodalFactor{};
        m_symbolic.reset();
        m_symbolic_builder = nullptr;
        m_ordering = Ordering< Index >{};
        m_externalOrdering = Ordering< Index >{};
        m_regularization = 1e-12;
        m_relativeRegularization = 0.0;
        m_strictPivots = false;
        m_backendPolicy = BackendPolicy::Automatic;
        m_pivotPolicy = PivotPolicy::Regularized1x1;
        m_supernodalThreshold = 2.0;
        m_relaxation = defaultRelaxation();
        m_expectedPivotSigns.clear();
        m_inputStorage = SymmetricStorage::AutoDetect;
        m_patternAnalyzed = false;
        m_useExternalOrdering = false;
    }

    void compute(const MatrixType& a) {
        analyzePattern(a);
        factorize(a);
    }

    std::shared_ptr< const SymbolicLDLT >
    analyzeSymbolic(const MatrixType& a) {
        analyzePattern(a);
        return m_symbolic;
    }

    void factorizeMatrix(const MatrixType& a) {
        validateMatrix(a);
        const std::uint64_t hash = structuralHash(a);
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n) ||
            !m_symbolic || hash != m_symbolic->m_pattern_hash) {
            analyzePattern(a);
        }
        factorize(a);
    }

    /// Refactorize values while requiring the sparsity pattern to be unchanged.
    void refactorizeSamePattern(const MatrixType& a) {
        validateMatrix(a);
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n) ||
            !m_symbolic || structuralHash(a) != m_symbolic->m_pattern_hash) {
            throw std::invalid_argument("supernodal: refactorization pattern mismatch");
        }
        factorize(a);
    }

    /// Solve Ax = b given already computed factorization.
    std::vector< Scalar > solve(const std::vector< Scalar >& b) const {
        if (!m_factors.factorized) {
            throw std::runtime_error("supernodal: solver is not factorized");
        }
        if (static_cast< Index >(b.size()) != m_size) {
            throw std::invalid_argument("supernodal: rhs size mismatch");
        }
        return solveImpl(b);
    }

    /// Solve AX=B for column-major B with `nrhs` right-hand sides.
    std::vector< Scalar > solveMultiple(const std::vector< Scalar >& rhs,
                                        Index nrhs) const {
        if (!m_factors.factorized)
            throw std::runtime_error("supernodal: solver is not factorized");
        if (nrhs < 0 ||
            rhs.size() != static_cast< size_t >(m_size) *
                              static_cast< size_t >(nrhs))
            throw std::invalid_argument("supernodal: multiple-RHS size mismatch");
        if (nrhs == 0)
            return {};
        if (m_symbolic->m_backend != SymbolicLDLT::Backend::Supernodal ||
            m_factors.pivoted) {
            std::vector< Scalar > result(rhs.size());
            std::vector< Scalar > column(static_cast< size_t >(m_size));
            for (Index j = 0; j < nrhs; ++j) {
                std::copy_n(rhs.data() + static_cast< size_t >(j) * m_size,
                            m_size, column.data());
                const auto solution = solveImpl(column);
                std::copy(solution.begin(), solution.end(),
                          result.begin() + static_cast< size_t >(j) * m_size);
            }
            return result;
        }
        return solveMultipleSupernodal(rhs, nrhs);
    }

    const SupernodalFactor& factors() const {
        ensureScalarCsc();
        return m_factors;
    }
    const SupernodalFactor& panelFactors() const {
        return m_factors;
    }
    bool scalarCscMaterialized() const {
        return m_factors.scalar_csc_materialized;
    }
    Index size() const {
        return m_size;
    }
    Index info() const {
        return static_cast< Index >(m_factors.info_val);
    }
    bool isFactorized() const {
        return m_factors.factorized;
    }
    Index nonZerosL() const {
        return static_cast< Index >(m_symbolic ? m_symbolic->m_structural_nnz_l : 0);
    }
    Index perturbedPivots() const {
        return m_factors.perturbed_pivots;
    }
    Scalar minAbsPivot() const {
        return static_cast< Scalar >(m_factors.min_abs_pivot);
    }

    void setRegularization(RealScalar eps) {
        m_regularization = std::max(eps, RealScalar(0));
    }

    void setRelativeRegularization(RealScalar eps) {
        m_relativeRegularization = std::max(eps, RealScalar(0));
        m_patternAnalyzed = false;
    }

    void setStrictPivots(bool strict) {
        m_strictPivots = strict;
        m_patternAnalyzed = false;
    }

    void setExpectedPivotSigns(std::vector< int8_t > signs) {
        for (int8_t sign : signs) {
            if (sign < -1 || sign > 1)
                throw std::invalid_argument("supernodal: pivot signs must be -1, 0, or 1");
        }
        m_expectedPivotSigns = std::move(signs);
        m_patternAnalyzed = false;
    }

    void setBackendPolicy(BackendPolicy policy) {
        if (m_backendPolicy != policy) {
            m_backendPolicy = policy;
            m_patternAnalyzed = false;
        }
    }

    void setSupernodalThreshold(double threshold) {
        if (!(threshold >= 0.0))
            throw std::invalid_argument("supernodal: threshold must be nonnegative");
        m_supernodalThreshold = threshold;
        m_patternAnalyzed = false;
    }

    void setPivotPolicy(PivotPolicy policy) {
        m_pivotPolicy = policy;
        m_patternAnalyzed = false;
    }

    void setSymmetricStorage(SymmetricStorage storage) {
        m_inputStorage = storage;
    }

    void setRelaxationThresholds(std::vector< RelaxationThreshold > thresholds) {
        for (const auto& threshold : thresholds) {
            if (threshold.max_columns <= 0 || threshold.max_zero_fraction < 0.0 ||
                threshold.max_zero_fraction > 1.0)
                throw std::invalid_argument("supernodal: invalid relaxation threshold");
        }
        m_relaxation = std::move(thresholds);
        m_patternAnalyzed = false;
    }

    SymbolicLDLT::Backend backend() const {
        return m_symbolic ? m_symbolic->m_backend
                          : SymbolicLDLT::Backend::Supernodal;
    }

    /// Supply a precomputed permutation instead of letting analyzePattern
    /// run full-matrix AMD. Useful when the caller has structural knowledge
    /// AMD can't see (e.g. block-quasi-definite systems where pivots must
    /// not cross block boundaries). Must be set before compute()/factorizeMatrix().
    void setExternalOrdering(Ordering< Index > ordering) {
        m_externalOrdering = std::move(ordering);
        m_useExternalOrdering = true;
        m_patternAnalyzed = false;
    }

    const Ordering< Index >& permutation() const {
        return m_ordering;
    }

    const std::vector< std::pair< Int, Int > >& supernodeRanges() const {
        static const std::vector< std::pair< Int, Int > > empty;
        return m_symbolic ? m_symbolic->m_supernode_ranges : empty;
    }

    const std::vector< Int >& etree() const {
        static const std::vector< Int > empty;
        return m_symbolic ? m_symbolic->m_etree : empty;
    }

    bool isSupernodal() const {
        return !m_factors.pivoted &&
               m_symbolic &&
               m_symbolic->m_backend == SymbolicLDLT::Backend::Supernodal;
    }

    std::shared_ptr< const SymbolicLDLT > symbolic() const {
        return m_symbolic;
    }

    void setSymbolic(std::shared_ptr< const SymbolicLDLT > symbolic) {
        if (!symbolic || symbolic->m_n <= 0)
            throw std::invalid_argument("supernodal: invalid symbolic analysis");
        m_symbolic = std::move(symbolic);
        m_size = static_cast< Index >(m_symbolic->m_n);
        m_ordering = Ordering< Index >::from_perm(
            std::vector< Index >(m_symbolic->m_perm.begin(),
                                 m_symbolic->m_perm.end()));
        m_resolvedStorage =
            static_cast< SymmetricStorage >(m_symbolic->m_resolved_storage);
        m_patternAnalyzed = true;
        m_factors = SupernodalFactor{};
        m_factors.info_val = SupernodalFactor::Success;
    }

    void factorizeWithSymbolic(
        const MatrixType& a, std::shared_ptr< const SymbolicLDLT > symbolic) {
        validateMatrix(a);
        if (!symbolic || a.n != symbolic->m_n ||
            structuralHash(a) != symbolic->m_pattern_hash)
            throw std::invalid_argument("supernodal: symbolic pattern mismatch");
        setSymbolic(std::move(symbolic));
        factorize(a);
    }

  private:
    struct NumericEntry {
        Int col;
        Int row;
        double value;
        int8_t source_triangle;
    };

    static std::vector< RelaxationThreshold > defaultRelaxation() {
        return {{Index{4}, 1.0},
                {Index{16}, 0.8},
                {Index{48}, 0.1},
                {std::numeric_limits< Index >::max(), 0.05}};
    }

    std::uint64_t structuralHash(const MatrixType& a) {
        return linsys::symmetric_pattern_hash(a);
    }

    SymmetricStorage resolveStorage(const MatrixType& a) const {
        if (m_inputStorage != SymmetricStorage::AutoDetect)
            return m_inputStorage;
        bool has_upper = false;
        bool has_lower = false;
        for (Index col = 0; col < a.n; ++col) {
            for (Index p = a.Ap[static_cast< size_t >(col)];
                 p < a.Ap[static_cast< size_t >(col) + 1]; ++p) {
                const Index row = a.Ai[static_cast< size_t >(p)];
                has_upper = has_upper || row < col;
                has_lower = has_lower || row > col;
            }
        }
        if (has_upper && has_lower)
            return SymmetricStorage::FullSymmetric;
        if (!has_upper && !has_lower)
            return SymmetricStorage::FullSymmetric;
        return has_lower ? SymmetricStorage::Lower : SymmetricStorage::Upper;
    }

    // ===== Dense LDLᵀ on a frontal matrix (used within supernodes) ==========
    // Factorizes F(0:npiv, 0:fsize) in-place. D_local gets the diagonal.

    static void updateTrailingBlock(DenseMatrix< Real >& F, Int first, Int panel_begin,
                                    Int panel_end, const std::vector< double >& D_local,
                                    std::vector< double >& scaled_panel,
                                    std::vector< double >& product) {
        const Int trailing_size = F.rows - first;
        const Int panel_size = panel_end - panel_begin;
        if (trailing_size <= 0 || panel_size <= 0)
            return;

#if defined(LINSYS_HAS_BLAS)
        scaled_panel.resize(static_cast< size_t >(trailing_size) *
                            static_cast< size_t >(panel_size));
        product.resize(static_cast< size_t >(trailing_size) *
                       static_cast< size_t >(trailing_size));

        for (Int k = 0; k < panel_size; ++k) {
            const double d = D_local[static_cast< size_t >(panel_begin + k)];
            for (Int i = 0; i < trailing_size; ++i) {
                scaled_panel[static_cast< size_t >(k) * static_cast< size_t >(trailing_size) +
                             static_cast< size_t >(i)] = F(first + i, panel_begin + k) * d;
            }
        }

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, trailing_size, trailing_size,
                    panel_size, 1.0, scaled_panel.data(), trailing_size,
                    &F(first, panel_begin), F.rows, 0.0, product.data(), trailing_size);

        for (Int j = 0; j < trailing_size; ++j) {
            for (Int i = j; i < trailing_size; ++i) {
                F(first + i, first + j) -=
                    product[static_cast< size_t >(j) * static_cast< size_t >(trailing_size) +
                            static_cast< size_t >(i)];
            }
        }
#else
        (void)scaled_panel;
        (void)product;
        for (Int j = first; j < F.rows; ++j) {
            for (Int i = j; i < F.rows; ++i) {
                double update = 0.0;
                for (Int k = panel_begin; k < panel_end; ++k) {
                    update += F(i, k) * D_local[static_cast< size_t >(k)] * F(j, k);
                }
                F(i, j) -= update;
            }
        }
#endif
    }

    static void denseLDLT(DenseMatrix< Real >& F, Int fsize, Int npiv,
                          std::vector< double >& D_local, Int& perturbed_pivots,
                          double& min_abs_pivot, double regularization,
                          double relative_regularization, bool strict_pivots,
                          const std::vector< int8_t >& expected_signs,
                          Int global_column_begin) {
        constexpr Int block_size = 32;
        std::vector< double > scaled_panel;
        std::vector< double > product;

        for (Int panel_begin = 0; panel_begin < npiv; panel_begin += block_size) {
            const Int panel_end = std::min(npiv, panel_begin + block_size);

            // Factor the diagonal panel and compute the rectangular L panel.
            for (Int k = panel_begin; k < panel_end; ++k) {
                double d = F(k, k);
                double column_scale = std::abs(d);
                for (Int i = k + 1; i < fsize; ++i)
                    column_scale = std::max(column_scale, std::abs(F(i, k)));
                const double threshold =
                    std::max(regularization, relative_regularization * column_scale);

                if (!std::isfinite(d) || std::abs(d) <= threshold) {
                    if (strict_pivots) {
                        throw std::domain_error("supernodal: zero or non-finite pivot");
                    }
                    int sign = d < 0.0 ? -1 : 1;
                    const size_t global_column =
                        static_cast< size_t >(global_column_begin + k);
                    if (global_column < expected_signs.size() &&
                        expected_signs[global_column] != 0)
                        sign = expected_signs[global_column];
                    const double replacement =
                        threshold > 0.0 ? threshold : std::numeric_limits< double >::epsilon();
                    d = static_cast< double >(sign) * replacement;
                    ++perturbed_pivots;
                }
                D_local[static_cast< size_t >(k)] = d;

                const double absd = std::abs(d);
                if (min_abs_pivot == 0.0 || absd < min_abs_pivot) {
                    min_abs_pivot = absd;
                }

                const double dinv = 1.0 / d;
                for (Int i = k + 1; i < fsize; ++i) {
                    F(i, k) *= dinv;
                }

                // Only update columns still inside the panel. The entire
                // trailing block is updated by DGEMM after the panel closes.
                for (Int j = k + 1; j < panel_end; ++j) {
                    const double dljk = F(j, k) * d;
                    for (Int i = j; i < fsize; ++i) {
                        F(i, j) -= F(i, k) * dljk;
                    }
                }
            }

            updateTrailingBlock(F, panel_end, panel_begin, panel_end, D_local,
                                scaled_panel, product);
        }
    }

    void denseIntranodalBunchKaufman(
        DenseMatrix< Real >& F, Int fsize, Int npiv,
        std::vector< double >& D_local, std::vector< double >& D_subdiag_local,
        std::vector< int8_t >& block_local, std::vector< Int >& perm_local,
        Int global_column_begin) {
        if (F.rows != fsize)
            throw std::logic_error("supernodal: BK front size mismatch");
        ldlt::detail::DenseBunchKaufmanOptions< Real, Int > options;
        options.absolute_regularization = m_regularization;
        options.relative_regularization = m_relativeRegularization;
        options.regularize_singular_pivots = true;
        options.strict_pivots = m_strictPivots;
        options.global_column_begin = global_column_begin;
        options.expected_pivot_signs = &m_factorPivotSigns;
        auto result =
            ldlt::detail::denseBunchKaufman(F, npiv, options);

        D_local = std::move(result.diagonal);
        D_subdiag_local = std::move(result.subdiagonal);
        block_local = std::move(result.blocks);
        perm_local = std::move(result.permutation);
        m_factors.positive_inertia += result.positive_inertia;
        m_factors.negative_inertia += result.negative_inertia;
        m_factors.zero_inertia += result.zero_inertia;
        m_factors.perturbed_pivots += result.perturbed_pivots;
        if (m_factors.min_abs_pivot == 0.0 ||
            (result.min_abs_pivot != 0.0 &&
             result.min_abs_pivot < m_factors.min_abs_pivot))
            m_factors.min_abs_pivot = result.min_abs_pivot;
    }

    // ===== Pattern analysis: compute ordering + supernodes ==================

    static void validateMatrix(const MatrixType& a) {
        if (a.n < 0)
            throw std::invalid_argument("supernodal: matrix dimension must be nonnegative");
        if (a.Ap.size() != static_cast< size_t >(a.n) + 1)
            throw std::invalid_argument("supernodal: Ap size mismatch");
        if (a.Ap.empty() || a.Ap.front() != Index{0})
            throw std::invalid_argument("supernodal: Ap must start at zero");
        if (a.Ai.size() != a.Ax.size())
            throw std::invalid_argument("supernodal: Ai/Ax size mismatch");

        for (Index col = 0; col < a.n; ++col) {
            const Index begin = a.Ap[static_cast< size_t >(col)];
            const Index end = a.Ap[static_cast< size_t >(col) + 1];
            if (begin < 0 || begin > end || static_cast< size_t >(end) > a.Ai.size())
                throw std::invalid_argument("supernodal: invalid CSC column pointers");
            for (Index p = begin; p < end; ++p) {
                const Index row = a.Ai[static_cast< size_t >(p)];
                if (row < 0 || row >= a.n)
                    throw std::invalid_argument("supernodal: row index out of range");
            }
        }
        if (static_cast< size_t >(a.Ap.back()) != a.Ai.size())
            throw std::invalid_argument("supernodal: Ap.back() must equal Ai/Ax size");
    }

    void analyzePattern(const MatrixType& a) {
        validateMatrix(a);
        if (a.n <= 0) {
            m_factors.info_val = SupernodalFactor::NotInitialized;
            m_factors.factorized = false;
            m_patternAnalyzed = false;
            return;
        }
        m_size = static_cast< Index >(a.n);
        m_resolvedStorage = resolveStorage(a);
        auto symbolic = std::shared_ptr< SymbolicLDLT >(new SymbolicLDLT);
        m_symbolic_builder = symbolic.get();
        m_symbolic = std::move(symbolic);
        m_symbolic_builder->m_n = static_cast< Int >(m_size);
        m_symbolic_builder->m_resolved_storage =
            static_cast< int >(m_resolvedStorage);

        // Compute permutation: AMD if n > threshold, else natural.
        computeOrdering(a);

        // Detect supernodes using snode::identify_supernodes.
        computeSupernodes(a);

        m_symbolic_builder->m_pattern_hash = structuralHash(a);
        selectBackend();
        m_symbolic_builder = nullptr;

        m_patternAnalyzed = true;
        m_factors = SupernodalFactor{};
        m_factors.info_val = SupernodalFactor::Success;
    }

    // ===== Numeric factorization: hybrid simplicial/supernodal/pivoted =====

    void factorize(const MatrixType& a) {
        if (!m_patternAnalyzed || m_size != static_cast< Index >(a.n)) {
            analyzePattern(a);
        }
        if (!m_patternAnalyzed)
            return;

        m_factors.pivoted = false;
        m_factors.intranodal_pivoted =
            m_pivotPolicy == PivotPolicy::IntranodalBunchKaufman;
        m_factors.D_subdiag.assign(static_cast< size_t >(m_size), 0.0);
        m_factors.pivot_blocks.assign(static_cast< size_t >(m_size), int8_t{1});
        m_factors.intranodal_perm.resize(static_cast< size_t >(m_size));
        m_factors.intranodal_iperm.resize(static_cast< size_t >(m_size));
        std::iota(m_factors.intranodal_perm.begin(),
                  m_factors.intranodal_perm.end(), Int{0});
        std::iota(m_factors.intranodal_iperm.begin(),
                  m_factors.intranodal_iperm.end(), Int{0});
        m_factors.positive_inertia = 0;
        m_factors.negative_inertia = 0;
        m_factors.zero_inertia = 0;
        if (m_pivotPolicy == PivotPolicy::BunchKaufman) {
            factorizeFullBunchKaufman(a);
        } else if (m_symbolic->m_backend == SymbolicLDLT::Backend::Simplicial) {
            m_simplicial.setRegularization(m_regularization);
            m_simplicial.factorizeWithSymbolic(a, m_symbolic);
            const auto& source = m_simplicial.factors();
            m_factors.n = source.n;
            m_factors.Lp.assign(source.Lp.begin(), source.Lp.end());
            m_factors.Li.assign(source.Li.begin(), source.Li.end());
            m_factors.Lx.assign(source.Lx.begin(), source.Lx.end());
            m_factors.D.assign(source.D.begin(), source.D.end());
            m_ordering = m_simplicial.permutation();
            m_factors.perturbed_pivots = source.perturbed_pivots;
            m_factors.min_abs_pivot = source.min_abs_pivot;
            m_factors.info_val = source.factorized ? SupernodalFactor::Success
                                                   : SupernodalFactor::NumericalIssue;
            m_factors.scalar_csc_materialized = true;
        } else {
            factorizeSupernodal(a);
        }
        m_factors.factorized = (m_factors.info_val == SupernodalFactor::Success);
    }

    void factorizeFullBunchKaufman(const MatrixType& a) {
        DenseMatrix< Real > front(static_cast< Int >(m_size),
                                  static_cast< Int >(m_size));
        for (Index j = 0; j < m_size; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)];
                 p < a.Ap[static_cast< size_t >(j + 1)]; ++p) {
                const Index i = a.Ai[static_cast< size_t >(p)];
                const Real value =
                    static_cast< Real >(a.Ax[static_cast< size_t >(p)]);
                front(static_cast< Int >(i), static_cast< Int >(j)) = value;
                front(static_cast< Int >(j), static_cast< Int >(i)) = value;
            }
        }

        ldlt::detail::DenseBunchKaufmanOptions< Real, Int > options;
        options.inertia_tolerance = m_regularization;
        auto result = ldlt::detail::denseBunchKaufman(
            front, static_cast< Int >(m_size), options);

        m_factors.n = static_cast< Int >(m_size);
        m_factors.D = std::move(result.diagonal);
        m_factors.D_subdiag = std::move(result.subdiagonal);
        m_factors.pivot_blocks = std::move(result.blocks);
        m_factors.intranodal_perm = std::move(result.permutation);
        m_factors.intranodal_iperm.assign(static_cast< size_t >(m_size), Int{0});
        for (Int factor_position = 0;
             factor_position < static_cast< Int >(m_size); ++factor_position) {
            m_factors.intranodal_iperm[static_cast< size_t >(
                m_factors.intranodal_perm[
                    static_cast< size_t >(factor_position)])] = factor_position;
        }

        m_factors.Lp.assign(static_cast< size_t >(m_size) + 1, Int{0});
        m_factors.Li.clear();
        m_factors.Lx.clear();
        for (Int j = 0; j < static_cast< Int >(m_size); ++j) {
            const Int first =
                m_factors.pivot_blocks[static_cast< size_t >(j)] == int8_t{2}
                    ? j + 2
                    : j + 1;
            for (Int i = first; i < static_cast< Int >(m_size); ++i) {
                const Real value = front(i, j);
                if (value != Real(0)) {
                    m_factors.Li.push_back(i);
                    m_factors.Lx.push_back(value);
                }
            }
            m_factors.Lp[static_cast< size_t >(j + 1)] =
                static_cast< Int >(m_factors.Li.size());
        }

        m_factors.pivoted = true;
        m_factors.positive_inertia = result.positive_inertia;
        m_factors.negative_inertia = result.negative_inertia;
        m_factors.zero_inertia = result.zero_inertia;
        m_factors.min_abs_pivot = result.min_abs_pivot;
        m_factors.scalar_csc_materialized = true;
        m_factors.info_val = SupernodalFactor::Success;
    }

    void selectBackend() {
        long double work = 0.0;
        std::uint64_t nnz = 0;
        std::uint64_t strict_nnz = 0;
        for (size_t si = 0; si < m_symbolic->m_supernode_ranges.size(); ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] -
                             m_symbolic->m_supernode_row_ptr[si];
            for (Int col = 0; col < width; ++col) {
                const long double count = static_cast< long double >(rows - col);
                nnz += static_cast< std::uint64_t >(rows - col);
                strict_nnz += static_cast< std::uint64_t >(rows - col - 1);
                work += count * (count + 1.0L) * 0.5L;
            }
        }
        const double intensity =
            nnz == 0 ? 0.0 : static_cast< double >(work / static_cast< long double >(nnz));
        m_symbolic_builder->m_symbolic_intensity = intensity;
        m_symbolic_builder->m_structural_nnz_l =
            static_cast< size_t >(strict_nnz);

        if (m_backendPolicy == BackendPolicy::ForceSimplicial) {
            if (m_strictPivots || m_relativeRegularization > 0.0 ||
                !m_expectedPivotSigns.empty())
                throw std::invalid_argument(
                    "supernodal: forced simplicial backend does not support "
                    "strict, relative, or signed regularization");
            m_symbolic_builder->m_backend = SymbolicLDLT::Backend::Simplicial;
            return;
        }
        if (m_backendPolicy == BackendPolicy::ForceSupernodal || m_useExternalOrdering ||
            m_strictPivots || m_relativeRegularization > 0.0 ||
            !m_expectedPivotSigns.empty() ||
            m_pivotPolicy == PivotPolicy::IntranodalBunchKaufman ||
            m_inputStorage != SymmetricStorage::AutoDetect ||
            m_resolvedStorage != SymmetricStorage::FullSymmetric) {
            m_symbolic_builder->m_backend = SymbolicLDLT::Backend::Supernodal;
            return;
        }

        m_symbolic_builder->m_backend =
            intensity > m_supernodalThreshold
                ? SymbolicLDLT::Backend::Supernodal
                : SymbolicLDLT::Backend::Simplicial;
    }

    // ===== Supernodal factorization =========================================
    //
    // Fundamental supernode structure from supernodes.h:
    //   A supernode S spans columns [lo, hi] where:
    //     - etree[k] = k+1 for all k in [lo, hi-1]  (chain)
    //     - all columns share the same L-pattern below the diagonal
    //
    //   Children of supernode S are supernodes C where
    //     etree[hi(C)] ∈ columns of S.
    //
    //   Processing order: increasing columns (left-looking).
    //
    //   For each supernode:
    //     1. Gather A(p,q) for p,q in pivot set + update rows.
    //     2. Apply all intersecting earlier panels directly to the frontal matrix.
    //     3. Dense LDLᵀ on the frontal matrix (BLAS-level 3).
    //     4. Retain the dense L panel for later updates and solves.

    void factorizeSupernodal(const MatrixType& a) {
        m_factors.n = m_size;
        m_factors.D.assign(static_cast< size_t >(m_size), 0.0);
        m_factors.perturbed_pivots = 0;
        m_factors.min_abs_pivot = 0.0;
        m_factors.factorized = false;
        m_factors.info_val = SupernodalFactor::Success;
        m_factorPivotSigns.assign(static_cast< size_t >(m_size), int8_t{0});
        if (!m_expectedPivotSigns.empty()) {
            if (m_expectedPivotSigns.size() != static_cast< size_t >(m_size))
                throw std::invalid_argument("supernodal: pivot-sign vector size mismatch");
            for (Int new_index = 0; new_index < m_size; ++new_index) {
                const Int old_index =
                    m_symbolic->m_iperm[static_cast< size_t >(new_index)];
                m_factorPivotSigns[static_cast< size_t >(new_index)] =
                    m_expectedPivotSigns[static_cast< size_t >(old_index)];
            }
        }

        // perm[old] = new, so it maps input coordinates directly
        // into A_perm.
        const auto& perm_idx = m_symbolic->m_perm;
        const SymmetricStorage storage = resolveStorage(a);

        // Build a canonical lower-triangular A_perm = P A P^T. Averaging
        // mirrored entries accepts upper-only, lower-only, or full symmetric
        // input without double counting a full representation.
        auto& entries = m_numericEntries;
        entries.clear();
        entries.reserve(a.nnz());
        for (Index j = 0; j < a.n; ++j) {
            const Int pj = perm_idx[static_cast< size_t >(j)];
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Int pi = perm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const Index original_row = a.Ai[static_cast< size_t >(p)];
                const int8_t source_triangle =
                    original_row < j ? int8_t{-1}
                                     : (original_row > j ? int8_t{1} : int8_t{0});
                if (storage == SymmetricStorage::Upper && source_triangle > 0)
                    throw std::invalid_argument(
                        "supernodal: lower entry supplied for upper storage");
                if (storage == SymmetricStorage::Lower && source_triangle < 0)
                    throw std::invalid_argument(
                        "supernodal: upper entry supplied for lower storage");
                entries.push_back(
                    {std::min(pi, pj), std::max(pi, pj),
                     static_cast< double >(a.Ax[static_cast< size_t >(p)]),
                     source_triangle});
            }
        }
        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
            return std::pair{lhs.col, lhs.row} < std::pair{rhs.col, rhs.row};
        });
        auto& canonical = m_canonicalEntries;
        canonical.clear();
        canonical.reserve(entries.size());
        for (size_t pos = 0; pos < entries.size();) {
            size_t end = pos + 1;
            double diagonal_sum = 0.0;
            double upper_sum = 0.0;
            double lower_sum = 0.0;
            size_t upper_count = 0;
            size_t lower_count = 0;
            auto accumulate = [&](const NumericEntry& entry) {
                if (entry.source_triangle < 0) {
                    upper_sum += entry.value;
                    ++upper_count;
                } else if (entry.source_triangle > 0) {
                    lower_sum += entry.value;
                    ++lower_count;
                } else {
                    diagonal_sum += entry.value;
                }
            };
            accumulate(entries[pos]);
            while (end < entries.size() && entries[end].col == entries[pos].col &&
                   entries[end].row == entries[pos].row) {
                accumulate(entries[end]);
                ++end;
            }
            double value = diagonal_sum;
            if (entries[pos].row != entries[pos].col) {
                if (storage == SymmetricStorage::FullSymmetric) {
                    if (upper_count == 0 || lower_count == 0)
                        throw std::invalid_argument(
                            "supernodal: full symmetric storage is missing a mirror");
                    const double scale =
                        std::max({1.0, std::abs(upper_sum), std::abs(lower_sum)});
                    if (std::abs(upper_sum - lower_sum) >
                        32.0 * std::numeric_limits< double >::epsilon() * scale)
                        throw std::invalid_argument(
                            "supernodal: mirrored entries have inconsistent sums");
                    value = 0.5 * (upper_sum + lower_sum);
                } else {
                    value = storage == SymmetricStorage::Upper ? upper_sum
                                                               : lower_sum;
                }
            }
            canonical.push_back(
                {entries[pos].col, entries[pos].row, value, int8_t{0}});
            pos = end;
        }

        auto& permAp = m_permutedAp;
        permAp.assign(static_cast< size_t >(m_size) + 1, Int{0});
        for (const auto& entry : canonical)
            ++permAp[static_cast< size_t >(entry.col) + 1];
        for (Int j = 0; j < m_size; ++j) {
            permAp[static_cast< size_t >(j) + 1] += permAp[static_cast< size_t >(j)];
        }

        auto& permAi = m_permutedAi;
        auto& permAx = m_permutedAx;
        permAi.resize(canonical.size());
        permAx.resize(canonical.size());
        for (size_t pos = 0; pos < canonical.size(); ++pos) {
            permAi[pos] = canonical[pos].row;
            permAx[pos] = canonical[pos].value;
        }

        const size_t ns = m_symbolic->m_supernode_ranges.size();
        const size_t panel_storage =
            static_cast< size_t >(m_symbolic->m_panel_value_ptr.back());
        m_factors.panel_values.assign(panel_storage, 0.0);

        m_globalToLocal.assign(static_cast< size_t >(m_size), Int{-1});
        m_touchedRows.clear();
        m_touchedRows.reserve(static_cast< size_t >(m_size));

        // Left-looking numerical factorization. Each earlier panel contributes
        // -L_d D_d L_d^T directly to the current frontal panel; explicit child
        // Schur matrices are never materialized.
        for (size_t si = 0; si < ns; ++si) {
            const auto [col_lo, col_hi] = m_symbolic->m_supernode_ranges[si];
            const Int npiv = col_hi - col_lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[si];
            const Int row_end = m_symbolic->m_supernode_row_ptr[si + 1];
            const Int fsize = row_end - row_begin;

            m_front.resize(fsize, fsize);
            for (Int local = 0; local < fsize; ++local) {
                const Int global =
                    m_symbolic->m_supernode_rows[static_cast< size_t >(row_begin + local)];
                m_globalToLocal[static_cast< size_t >(global)] = local;
                m_touchedRows.push_back(global);
            }
            auto& F = m_front;

            // Gather A_perm entries into the pivot/pivot and update/pivot
            // blocks of the frontal matrix. Only lower triangle is needed
            // since Dense LDLT only touches lower triangle; rows are either
            // within the pivot block [col_lo, col_hi] or one of this
            // supernode's update_rows (rows > col_hi with an entry in a
            // pivot column) — both map via globalToLocal.
            for (Int pj = col_lo; pj <= col_hi; ++pj) {
                for (Int p = permAp[static_cast< size_t >(pj)];
                     p < permAp[static_cast< size_t >(pj) + 1]; ++p) {
                    Int pi = permAi[static_cast< size_t >(p)];
                    if (pi < pj)
                        continue; // strictly upper entries: covered by the symmetric (pj,pi) pass
                    Int li = m_globalToLocal[static_cast< size_t >(pi)];
                    Int lj = m_globalToLocal[static_cast< size_t >(pj)];
                    if (li < 0 || lj < 0)
                        continue;
                    double v = permAx[static_cast< size_t >(p)];
                    F(li, lj) += v;
                }
            }

            const Int contributor_begin = m_symbolic->m_contributor_ptr[si];
            const Int contributor_end = m_symbolic->m_contributor_ptr[si + 1];
            for (Int contributor_pos = contributor_begin;
                 contributor_pos < contributor_end; ++contributor_pos) {
                const size_t descendant = static_cast< size_t >(
                    m_symbolic->m_contributors[static_cast< size_t >(contributor_pos)]);
                const auto [dlo, dhi] = m_symbolic->m_supernode_ranges[descendant];
                const Int dwidth = dhi - dlo + 1;
                const Int drow_begin = m_symbolic->m_supernode_row_ptr[descendant];
                const Int drow_end = m_symbolic->m_supernode_row_ptr[descendant + 1];
                const Int drows = drow_end - drow_begin;
                const double* panel =
                    m_factors.panel_values.data() +
                    m_symbolic->m_panel_value_ptr[descendant];

                m_mappedPanelRows.clear();
                for (Int dlocal = dwidth; dlocal < drows; ++dlocal) {
                    const Int global =
                        m_symbolic->m_supernode_rows[
                            static_cast< size_t >(drow_begin + dlocal)];
                    const Int local = m_globalToLocal[static_cast< size_t >(global)];
                    if (local >= 0)
                        m_mappedPanelRows.emplace_back(dlocal, local);
                }
                if (m_mappedPanelRows.empty())
                    continue;

#if defined(LINSYS_HAS_BLAS)
                const Int mapped = static_cast< Int >(m_mappedPanelRows.size());
                m_updatePanel.resize(static_cast< size_t >(mapped) *
                                     static_cast< size_t >(dwidth));
                m_scaledUpdatePanel.resize(static_cast< size_t >(mapped) *
                                           static_cast< size_t >(dwidth));
                m_updateProduct.resize(static_cast< size_t >(mapped) *
                                       static_cast< size_t >(mapped));
                for (Int k = 0; k < dwidth; ++k) {
                    const bool two_by_two =
                        m_factors.intranodal_pivoted &&
                        m_factors.pivot_blocks[static_cast< size_t >(dlo + k)] ==
                            int8_t{2};
                    for (Int i = 0; i < mapped; ++i) {
                        const Int drow = m_mappedPanelRows[static_cast< size_t >(i)].first;
                        const double value =
                            panel[static_cast< size_t >(k) * drows + drow];
                        m_updatePanel[static_cast< size_t >(k) * mapped + i] = value;
                        if (two_by_two) {
                            const double next =
                                panel[static_cast< size_t >(k + 1) * drows +
                                      drow];
                            m_updatePanel[static_cast< size_t >(k + 1) * mapped +
                                          i] = next;
                            const double offdiag = m_factors.D_subdiag[
                                static_cast< size_t >(dlo + k)];
                            m_scaledUpdatePanel[
                                static_cast< size_t >(k) * mapped + i] =
                                value * m_factors.D[static_cast< size_t >(dlo + k)] +
                                next * offdiag;
                            m_scaledUpdatePanel[
                                static_cast< size_t >(k + 1) * mapped + i] =
                                value * offdiag +
                                next * m_factors.D[
                                           static_cast< size_t >(dlo + k + 1)];
                        } else if (!m_factors.intranodal_pivoted ||
                                   m_factors.pivot_blocks[
                                       static_cast< size_t >(dlo + k)] !=
                                       int8_t{0}) {
                            m_scaledUpdatePanel[
                                static_cast< size_t >(k) * mapped + i] =
                                value *
                                m_factors.D[static_cast< size_t >(dlo + k)];
                        }
                    }
                    if (two_by_two)
                        ++k;
                }
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, mapped,
                            mapped, dwidth, 1.0, m_scaledUpdatePanel.data(),
                            mapped, m_updatePanel.data(), mapped, 0.0,
                            m_updateProduct.data(), mapped);
                for (Int j = 0; j < mapped; ++j) {
                    const Int local_j =
                        m_mappedPanelRows[static_cast< size_t >(j)].second;
                    for (Int i = 0; i < mapped; ++i) {
                        const Int local_i =
                            m_mappedPanelRows[static_cast< size_t >(i)].second;
                        if (local_i >= local_j)
                            F(local_i, local_j) -=
                                m_updateProduct[static_cast< size_t >(j) * mapped + i];
                    }
                }
#else
                for (size_t jb = 0; jb < m_mappedPanelRows.size(); ++jb) {
                    const auto [drow_j, local_j] = m_mappedPanelRows[jb];
                    for (size_t ib = jb; ib < m_mappedPanelRows.size(); ++ib) {
                        const auto [drow_i, local_i] = m_mappedPanelRows[ib];
                        if (local_i < local_j)
                            continue;
                        double update = 0.0;
                        for (Int k = 0; k < dwidth; ++k) {
                            const double li =
                                panel[static_cast< size_t >(k) * drows + drow_i];
                            const double lj =
                                panel[static_cast< size_t >(k) * drows + drow_j];
                            if (m_factors.intranodal_pivoted &&
                                m_factors.pivot_blocks[
                                    static_cast< size_t >(dlo + k)] ==
                                    int8_t{2}) {
                                const double li1 = panel[
                                    static_cast< size_t >(k + 1) * drows +
                                    drow_i];
                                const double lj1 = panel[
                                    static_cast< size_t >(k + 1) * drows +
                                    drow_j];
                                const double offdiag = m_factors.D_subdiag[
                                    static_cast< size_t >(dlo + k)];
                                update +=
                                    li * (m_factors.D[
                                              static_cast< size_t >(dlo + k)] *
                                              lj +
                                          offdiag * lj1) +
                                    li1 *
                                        (offdiag * lj +
                                         m_factors.D[static_cast< size_t >(
                                             dlo + k + 1)] *
                                             lj1);
                                ++k;
                            } else {
                                update +=
                                    li *
                                    m_factors.D[
                                        static_cast< size_t >(dlo + k)] *
                                    lj;
                            }
                        }
                        F(local_i, local_j) -= update;
                    }
                }
#endif
            }

            m_localDiagonal.resize(static_cast< size_t >(npiv));
            if (m_factors.intranodal_pivoted) {
                denseIntranodalBunchKaufman(
                    F, fsize, npiv, m_localDiagonal, m_localSubdiagonal,
                    m_localPivotBlocks, m_localPivotPermutation, col_lo);
            } else {
                denseLDLT(F, fsize, npiv, m_localDiagonal,
                          m_factors.perturbed_pivots,
                          m_factors.min_abs_pivot, m_regularization,
                          m_relativeRegularization, m_strictPivots,
                          m_factorPivotSigns, col_lo);
            }

            for (Int k = 0; k < npiv; ++k) {
                m_factors.D[static_cast< size_t >(col_lo + k)] =
                    m_localDiagonal[static_cast< size_t >(k)];
                if (m_factors.intranodal_pivoted) {
                    m_factors.D_subdiag[static_cast< size_t >(col_lo + k)] =
                        m_localSubdiagonal[static_cast< size_t >(k)];
                    m_factors.pivot_blocks[static_cast< size_t >(col_lo + k)] =
                        m_localPivotBlocks[static_cast< size_t >(k)];
                    m_factors.intranodal_perm[static_cast< size_t >(col_lo + k)] =
                        m_localPivotPermutation[static_cast< size_t >(k)];
                    m_factors.intranodal_iperm[static_cast< size_t >(
                        col_lo + m_localPivotPermutation[static_cast< size_t >(k)])] =
                        col_lo + k;
                }
            }

            double* panel = m_factors.panel_values.data() +
                            m_symbolic->m_panel_value_ptr[si];
            for (Int k = 0; k < npiv; ++k) {
                for (Int i = 0; i < fsize; ++i)
                    panel[static_cast< size_t >(k) * fsize + i] =
                        i < k ? 0.0 : (i == k ? 1.0 : F(i, k));
            }

            for (Int global : m_touchedRows)
                m_globalToLocal[static_cast< size_t >(global)] = Int{-1};
            m_touchedRows.clear();
        }

        m_factors.Lp.clear();
        m_factors.Li.clear();
        m_factors.Lx.clear();
        m_factors.scalar_csc_materialized = false;
    }

    void ensureScalarCsc() const {
        if (m_factors.scalar_csc_materialized ||
            !m_symbolic ||
            m_symbolic->m_backend != SymbolicLDLT::Backend::Supernodal ||
            m_factors.pivoted)
            return;
        buildScalarCscFromPanels();
        m_factors.scalar_csc_materialized = true;
    }

    void buildScalarCscFromPanels() const {
        m_factors.Lp.assign(static_cast< size_t >(m_size) + 1, Int{0});
        const size_t ns = m_symbolic->m_supernode_ranges.size();
        for (size_t si = 0; si < ns; ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] -
                             m_symbolic->m_supernode_row_ptr[si];
            for (Int k = 0; k < width; ++k)
                m_factors.Lp[static_cast< size_t >(lo + k + 1)] = rows - k - 1;
        }
        std::partial_sum(m_factors.Lp.begin(), m_factors.Lp.end(),
                         m_factors.Lp.begin());
        const Int nnz = m_factors.Lp.back();
        if (static_cast< size_t >(nnz) != m_symbolic->m_structural_nnz_l)
            throw std::logic_error("supernodal: scalar CSC count mismatch");
        m_factors.Li.resize(static_cast< size_t >(nnz));
        m_factors.Lx.resize(static_cast< size_t >(nnz));
        for (size_t si = 0; si < ns; ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[si];
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] - row_begin;
            const double* panel = m_factors.panel_values.data() +
                                  m_symbolic->m_panel_value_ptr[si];
            for (Int k = 0; k < width; ++k) {
                Int out = m_factors.Lp[static_cast< size_t >(lo + k)];
                for (Int i = k + 1; i < rows; ++i, ++out) {
                    m_factors.Li[static_cast< size_t >(out)] =
                        m_factors.intranodal_pivoted
                            ? m_factors.intranodal_iperm[static_cast< size_t >(
                                  m_symbolic->m_supernode_rows[
                                      static_cast< size_t >(row_begin + i)])]
                            : m_symbolic->m_supernode_rows[
                                  static_cast< size_t >(row_begin + i)];
                    m_factors.Lx[static_cast< size_t >(out)] =
                        panel[static_cast< size_t >(k) * rows + i];
                }
            }
        }
    }

    // ===== Solve: forward + diagonal + backward + un-permute ==============

    std::vector< Scalar > solveImpl(const std::vector< Scalar >& b) const {
        if (m_factors.pivoted)
            return solveFullBunchKaufman(b);
        if (m_symbolic->m_backend == SymbolicLDLT::Backend::Simplicial)
            return m_simplicial.solve(b);

        std::vector< Scalar > x(b.size());

        // Permute: x_perm[new] = b[iperm[new]].
        if (!m_symbolic->m_iperm.empty()) {
            std::vector< Scalar > y(static_cast< size_t >(m_size));
            permute_gather(m_size, m_symbolic->m_iperm.data(), b.data(), y.data());
            x = std::move(y);
        } else {
            x = b;
        }

        // Forward solve using the retained dense supernodal panels.
        const size_t ns = m_symbolic->m_supernode_ranges.size();
        for (size_t si = 0; si < ns; ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[si];
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] - row_begin;
            const double* panel = m_factors.panel_values.data() +
                                  m_symbolic->m_panel_value_ptr[si];
            if (m_factors.intranodal_pivoted)
                permuteIntranodalRhs(x, lo, width, Int{1});
#if defined(LINSYS_HAS_BLAS)
            if constexpr (std::is_same_v< Scalar, double >) {
                cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                            width, panel, rows, x.data() + lo, 1);
                const Int update_count = rows - width;
                m_solveWorkspace.resize(static_cast< size_t >(update_count));
                for (Int i = 0; i < update_count; ++i) {
                    const Int global = m_symbolic->m_supernode_rows[
                        static_cast< size_t >(row_begin + width + i)];
                    m_solveWorkspace[static_cast< size_t >(i)] =
                        x[static_cast< size_t >(global)];
                }
                if (update_count > 0)
                    cblas_dgemv(CblasColMajor, CblasNoTrans, update_count, width,
                                -1.0, panel + width, rows, x.data() + lo, 1,
                                1.0, m_solveWorkspace.data(), 1);
                for (Int i = 0; i < update_count; ++i) {
                    const Int global = m_symbolic->m_supernode_rows[
                        static_cast< size_t >(row_begin + width + i)];
                    x[static_cast< size_t >(global)] =
                        m_solveWorkspace[static_cast< size_t >(i)];
                }
                continue;
            }
#endif
            for (Int k = 0; k < width; ++k) {
                const Scalar pivot = x[static_cast< size_t >(lo + k)];
                for (Int i = k + 1; i < width; ++i)
                    x[static_cast< size_t >(lo + i)] -=
                        static_cast< Scalar >(panel[static_cast< size_t >(k) * rows + i]) *
                        pivot;
                for (Int i = width; i < rows; ++i) {
                    const Int global =
                        m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + i)];
                    x[static_cast< size_t >(global)] -=
                        static_cast< Scalar >(panel[static_cast< size_t >(k) * rows + i]) *
                        pivot;
                }
            }
        }

        solveBlockDiagonal(x, Int{1});

        // Backward solve using dense panel transposes.
        for (size_t rev = ns; rev-- > 0;) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[rev];
            const Int width = hi - lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[rev];
            const Int rows = m_symbolic->m_supernode_row_ptr[rev + 1] - row_begin;
            const double* panel = m_factors.panel_values.data() +
                                  m_symbolic->m_panel_value_ptr[rev];
#if defined(LINSYS_HAS_BLAS)
            if constexpr (std::is_same_v< Scalar, double >) {
                const Int update_count = rows - width;
                m_solveWorkspace.resize(static_cast< size_t >(update_count));
                for (Int i = 0; i < update_count; ++i) {
                    const Int global = m_symbolic->m_supernode_rows[
                        static_cast< size_t >(row_begin + width + i)];
                    m_solveWorkspace[static_cast< size_t >(i)] =
                        x[static_cast< size_t >(global)];
                }
                if (update_count > 0)
                    cblas_dgemv(CblasColMajor, CblasTrans, update_count, width,
                                -1.0, panel + width, rows,
                                m_solveWorkspace.data(), 1, 1.0, x.data() + lo, 1);
                cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                            width, panel, rows, x.data() + lo, 1);
                if (m_factors.intranodal_pivoted)
                    unpermuteIntranodalRhs(x, lo, width, Int{1});
                continue;
            }
#endif
            for (Int k = width; k-- > 0;) {
                Scalar value = x[static_cast< size_t >(lo + k)];
                for (Int i = k + 1; i < width; ++i)
                    value -=
                        static_cast< Scalar >(panel[static_cast< size_t >(k) * rows + i]) *
                        x[static_cast< size_t >(lo + i)];
                for (Int i = width; i < rows; ++i) {
                    const Int global =
                        m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + i)];
                    value -=
                        static_cast< Scalar >(panel[static_cast< size_t >(k) * rows + i]) *
                        x[static_cast< size_t >(global)];
                }
                x[static_cast< size_t >(lo + k)] = value;
            }
            if (m_factors.intranodal_pivoted)
                unpermuteIntranodalRhs(x, lo, width, Int{1});
        }

        // Un-permute: x_old[old] = x_perm[perm[old]].
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        if (!m_symbolic->m_perm.empty()) {
            permute_gather(m_size, m_symbolic->m_perm.data(), x.data(), result.data());
        } else {
            result = std::move(x);
        }

        return result;
    }

    std::vector< Scalar >
    solveFullBunchKaufman(const std::vector< Scalar >& b) const {
        std::vector< Scalar > x(static_cast< size_t >(m_size));
        permute_gather(m_size, m_factors.intranodal_perm.data(), b.data(),
                       x.data());
        lsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(),
                    m_factors.Lx.data(), x.data());

        for (Index k = 0; k < m_size;) {
            if (m_factors.pivot_blocks[static_cast< size_t >(k)] ==
                int8_t{2}) {
                const Scalar d11 =
                    static_cast< Scalar >(m_factors.D[static_cast< size_t >(k)]);
                const Scalar d21 = static_cast< Scalar >(
                    m_factors.D_subdiag[static_cast< size_t >(k)]);
                const Scalar d22 = static_cast< Scalar >(
                    m_factors.D[static_cast< size_t >(k + 1)]);
                const Scalar determinant = d11 * d22 - d21 * d21;
                const Scalar b0 = x[static_cast< size_t >(k)];
                const Scalar b1 = x[static_cast< size_t >(k + 1)];
                if (determinant != Scalar(0)) {
                    x[static_cast< size_t >(k)] =
                        (d22 * b0 - d21 * b1) / determinant;
                    x[static_cast< size_t >(k + 1)] =
                        (d11 * b1 - d21 * b0) / determinant;
                } else {
                    x[static_cast< size_t >(k)] =
                        d11 != Scalar(0) ? b0 / d11 : Scalar(0);
                    x[static_cast< size_t >(k + 1)] =
                        d22 != Scalar(0) ? b1 / d22 : Scalar(0);
                }
                k += 2;
            } else {
                const Scalar d =
                    static_cast< Scalar >(m_factors.D[static_cast< size_t >(k)]);
                x[static_cast< size_t >(k)] =
                    d != Scalar(0) ? x[static_cast< size_t >(k)] / d
                                   : Scalar(0);
                ++k;
            }
        }

        ltsolve_unit(m_size, m_factors.Lp.data(), m_factors.Li.data(),
                     m_factors.Lx.data(), x.data());
        std::vector< Scalar > result(static_cast< size_t >(m_size));
        linsys::permute_scatter(m_size, m_factors.intranodal_perm.data(),
                                x.data(), result.data());
        return result;
    }

    void permuteIntranodalRhs(std::vector< Scalar >& x, Int lo, Int width,
                              Int nrhs) const {
        m_rhsPermutationWorkspace.resize(static_cast< size_t >(width) *
                                         static_cast< size_t >(nrhs));
        for (Int rhs_col = 0; rhs_col < nrhs; ++rhs_col) {
            Scalar* column = x.data() + static_cast< size_t >(rhs_col) * m_size;
            Scalar* workspace =
                m_rhsPermutationWorkspace.data() +
                static_cast< size_t >(rhs_col) * width;
            for (Int k = 0; k < width; ++k) {
                const Int original_local = m_factors.intranodal_perm[
                    static_cast< size_t >(lo + k)];
                workspace[k] = column[lo + original_local];
            }
            std::copy_n(workspace, width, column + lo);
        }
    }

    void unpermuteIntranodalRhs(std::vector< Scalar >& x, Int lo, Int width,
                                Int nrhs) const {
        m_rhsPermutationWorkspace.resize(static_cast< size_t >(width) *
                                         static_cast< size_t >(nrhs));
        for (Int rhs_col = 0; rhs_col < nrhs; ++rhs_col) {
            Scalar* column = x.data() + static_cast< size_t >(rhs_col) * m_size;
            Scalar* workspace =
                m_rhsPermutationWorkspace.data() +
                static_cast< size_t >(rhs_col) * width;
            std::copy_n(column + lo, width, workspace);
            for (Int k = 0; k < width; ++k) {
                const Int original_local = m_factors.intranodal_perm[
                    static_cast< size_t >(lo + k)];
                column[lo + original_local] = workspace[k];
            }
        }
    }

    void solveBlockDiagonal(std::vector< Scalar >& x, Int nrhs) const {
        for (Int rhs_col = 0; rhs_col < nrhs; ++rhs_col) {
            Scalar* column = x.data() + static_cast< size_t >(rhs_col) * m_size;
            for (Int k = 0; k < m_size; ++k) {
                if (m_factors.intranodal_pivoted &&
                    m_factors.pivot_blocks[static_cast< size_t >(k)] ==
                        int8_t{2}) {
                    const Scalar d11 =
                        static_cast< Scalar >(m_factors.D[static_cast< size_t >(k)]);
                    const Scalar d21 = static_cast< Scalar >(
                        m_factors.D_subdiag[static_cast< size_t >(k)]);
                    const Scalar d22 = static_cast< Scalar >(
                        m_factors.D[static_cast< size_t >(k + 1)]);
                    const Scalar determinant = d11 * d22 - d21 * d21;
                    const Scalar first = column[k];
                    const Scalar second = column[k + 1];
                    column[k] = (d22 * first - d21 * second) / determinant;
                    column[k + 1] =
                        (-d21 * first + d11 * second) / determinant;
                    ++k;
                } else {
                    column[k] /= static_cast< Scalar >(
                        m_factors.D[static_cast< size_t >(k)]);
                }
            }
        }
    }

    std::vector< Scalar >
    solveMultipleSupernodal(const std::vector< Scalar >& rhs, Index nrhs) const {
        std::vector< Scalar > x(rhs.size());
        for (Index j = 0; j < nrhs; ++j) {
            const Scalar* source = rhs.data() + static_cast< size_t >(j) * m_size;
            Scalar* target = x.data() + static_cast< size_t >(j) * m_size;
            if (!m_symbolic->m_iperm.empty())
                permute_gather(m_size, m_symbolic->m_iperm.data(), source, target);
            else
                std::copy_n(source, m_size, target);
        }

        const size_t ns = m_symbolic->m_supernode_ranges.size();
        for (size_t si = 0; si < ns; ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[si];
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] - row_begin;
            const Int update_count = rows - width;
            const double* panel = m_factors.panel_values.data() +
                                  m_symbolic->m_panel_value_ptr[si];
            if (m_factors.intranodal_pivoted)
                permuteIntranodalRhs(x, lo, width, nrhs);
#if defined(LINSYS_HAS_BLAS)
            if constexpr (std::is_same_v< Scalar, double >) {
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasUnit, width, nrhs, 1.0, panel, rows,
                            x.data() + lo, m_size);
                m_solveWorkspace.resize(static_cast< size_t >(update_count) *
                                        static_cast< size_t >(nrhs));
                for (Index rhs_col = 0; rhs_col < nrhs; ++rhs_col)
                    for (Int i = 0; i < update_count; ++i) {
                        const Int global = m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + width + i)];
                        m_solveWorkspace[static_cast< size_t >(rhs_col) *
                                             update_count +
                                         i] =
                            x[static_cast< size_t >(rhs_col) * m_size + global];
                    }
                if (update_count > 0)
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                update_count, nrhs, width, -1.0, panel + width,
                                rows, x.data() + lo, m_size, 1.0,
                                m_solveWorkspace.data(), update_count);
                for (Index rhs_col = 0; rhs_col < nrhs; ++rhs_col)
                    for (Int i = 0; i < update_count; ++i) {
                        const Int global = m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + width + i)];
                        x[static_cast< size_t >(rhs_col) * m_size + global] =
                            m_solveWorkspace[static_cast< size_t >(rhs_col) *
                                                 update_count +
                                             i];
                    }
                continue;
            }
#endif
            for (Index rhs_col = 0; rhs_col < nrhs; ++rhs_col) {
                Scalar* column = x.data() + static_cast< size_t >(rhs_col) * m_size;
                for (Int k = 0; k < width; ++k) {
                    const Scalar pivot = column[lo + k];
                    for (Int i = k + 1; i < width; ++i)
                        column[lo + i] -= static_cast< Scalar >(
                                                  panel[static_cast< size_t >(k) *
                                                            rows +
                                                        i]) *
                                              pivot;
                    for (Int i = width; i < rows; ++i) {
                        const Int global = m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + i)];
                        column[global] -= static_cast< Scalar >(
                                              panel[static_cast< size_t >(k) *
                                                        rows +
                                                    i]) *
                                          pivot;
                    }
                }
            }
        }

        solveBlockDiagonal(x, nrhs);

        for (size_t rev = ns; rev-- > 0;) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[rev];
            const Int width = hi - lo + 1;
            const Int row_begin = m_symbolic->m_supernode_row_ptr[rev];
            const Int rows = m_symbolic->m_supernode_row_ptr[rev + 1] - row_begin;
            const Int update_count = rows - width;
            const double* panel = m_factors.panel_values.data() +
                                  m_symbolic->m_panel_value_ptr[rev];
#if defined(LINSYS_HAS_BLAS)
            if constexpr (std::is_same_v< Scalar, double >) {
                m_solveWorkspace.resize(static_cast< size_t >(update_count) *
                                        static_cast< size_t >(nrhs));
                for (Index rhs_col = 0; rhs_col < nrhs; ++rhs_col)
                    for (Int i = 0; i < update_count; ++i) {
                        const Int global = m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + width + i)];
                        m_solveWorkspace[static_cast< size_t >(rhs_col) *
                                             update_count +
                                         i] =
                            x[static_cast< size_t >(rhs_col) * m_size + global];
                    }
                if (update_count > 0)
                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, width,
                                nrhs, update_count, -1.0, panel + width, rows,
                                m_solveWorkspace.data(), update_count, 1.0,
                                x.data() + lo, m_size);
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasUnit, width, nrhs, 1.0, panel, rows,
                            x.data() + lo, m_size);
                if (m_factors.intranodal_pivoted)
                    unpermuteIntranodalRhs(x, lo, width, nrhs);
                continue;
            }
#endif
            for (Index rhs_col = 0; rhs_col < nrhs; ++rhs_col) {
                Scalar* column = x.data() + static_cast< size_t >(rhs_col) * m_size;
                for (Int k = width; k-- > 0;) {
                    Scalar value = column[lo + k];
                    for (Int i = k + 1; i < width; ++i)
                        value -= static_cast< Scalar >(
                                     panel[static_cast< size_t >(k) * rows + i]) *
                                 column[lo + i];
                    for (Int i = width; i < rows; ++i) {
                        const Int global = m_symbolic->m_supernode_rows[
                            static_cast< size_t >(row_begin + i)];
                        value -= static_cast< Scalar >(
                                     panel[static_cast< size_t >(k) * rows + i]) *
                                 column[global];
                    }
                    column[lo + k] = value;
                }
            }
            if (m_factors.intranodal_pivoted)
                unpermuteIntranodalRhs(x, lo, width, nrhs);
        }

        std::vector< Scalar > result(rhs.size());
        for (Index j = 0; j < nrhs; ++j) {
            const Scalar* source = x.data() + static_cast< size_t >(j) * m_size;
            Scalar* target = result.data() + static_cast< size_t >(j) * m_size;
            if (!m_symbolic->m_perm.empty())
                permute_gather(m_size, m_symbolic->m_perm.data(), source, target);
            else
                std::copy_n(source, m_size, target);
        }
        return result;
    }

    // ===== Pattern analysis helpers =========================================

    void computeOrdering(const MatrixType& a) {
        if (m_useExternalOrdering) {
            if (m_externalOrdering.n != m_size ||
                m_externalOrdering.perm.size() != static_cast< size_t >(m_size))
                throw std::invalid_argument("supernodal: external ordering size mismatch");
            m_ordering = Ordering< Index >::from_perm(m_externalOrdering.perm);
            if (!m_externalOrdering.iperm.empty() &&
                m_externalOrdering.iperm != m_ordering.iperm)
                throw std::invalid_argument("supernodal: inconsistent external inverse permutation");
            m_symbolic_builder->m_perm = m_ordering.perm;
            m_symbolic_builder->m_iperm = m_ordering.iperm;
            return;
        }

        m_ordering.n = m_size;

        // Build symmetric adjacency edges (structural pattern).
        std::vector< std::pair< Index, Index > > edges;
        edges.reserve(static_cast< size_t >(a.nnz()) * 2u + 1u);
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                Index i = a.Ai[static_cast< size_t >(p)];
                Index r = std::min(i, j);
                Index c = std::max(i, j);
                if (r == c)
                    continue;
                edges.emplace_back(r, c);
            }
        }

        // Compute permutation: AMD if n > threshold, else natural.
        if (m_size > 20) {
            m_ordering = Ordering< Index >::from_perm(amd_ordering(m_size, edges));
        } else {
            m_ordering = Ordering< Index >::identity(m_size);
        }

        // Store in factor output.
        m_symbolic_builder->m_perm = m_ordering.perm;
        m_symbolic_builder->m_iperm = m_ordering.iperm;
    }

    void computeSupernodes(const MatrixType& a) {
        // Build upper-triangular CSC in permuted space for supernode detection.
        const auto& perm_idx = m_symbolic->m_perm;
        std::vector< Index > ap(static_cast< size_t >(m_size) + 1, 0);
        std::vector< Index > ai;
        ai.reserve(static_cast< size_t >(a.nnz()));

        // Collect structural upper-triangular entries in the permuted space.
        std::vector< std::pair< Index, Index > > entries;
        entries.reserve(static_cast< size_t >(a.nnz()));
        for (Index j = 0; j < a.n; ++j) {
            for (Index p = a.Ap[static_cast< size_t >(j)]; p < a.Ap[static_cast< size_t >(j) + 1];
                 ++p) {
                const Index i = perm_idx[static_cast< size_t >(a.Ai[static_cast< size_t >(p)])];
                const Index j2 = perm_idx[static_cast< size_t >(j)];
                if (i < 0 || i >= m_size || j2 < 0 || j2 >= m_size)
                    continue;

                const Index row = std::min(i, j2);
                const Index col = std::max(i, j2);
                entries.emplace_back(col, row);
            }
        }
        std::sort(entries.begin(), entries.end());
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());

        // Build CSC.
        std::vector< Index > colCounts(static_cast< size_t >(m_size), 0);
        for (const auto& entry : entries)
            ++colCounts[static_cast< size_t >(entry.first)];
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += colCounts[static_cast< size_t >(j)];
        for (Index j = 0; j < m_size; ++j)
            ap[static_cast< size_t >(j) + 1] += ap[static_cast< size_t >(j)];
        ai.resize(static_cast< size_t >(ap.back()));
        std::vector< Index > curPos(static_cast< size_t >(m_size), 0);
        for (const auto& entry : entries) {
            const Index pj = entry.first;
            const Index pos = ap[static_cast< size_t >(pj)] + curPos[static_cast< size_t >(pj)]++;
            ai[static_cast< size_t >(pos)] = entry.second;
        }

        // Compute elimination tree using ancestor path-compression.
        std::vector< Index > parent(static_cast< size_t >(m_size), Index{-1});
        std::vector< Index > ancestor(static_cast< size_t >(m_size), Index{-1});

        for (Index j = 0; j < m_size; ++j) {
            for (Index p = ap[static_cast< size_t >(j)]; p < ap[static_cast< size_t >(j) + 1];
                 ++p) {
                Index i = ai[static_cast< size_t >(p)];
                while (i != Index{-1} && i < j) {
                    const Index next = ancestor[static_cast< size_t >(i)];
                    ancestor[static_cast< size_t >(i)] = j;
                    if (next == Index{-1}) {
                        parent[static_cast< size_t >(i)] = j;
                    }
                    i = next;
                }
            }
        }

        // Detect supernodes using snode::identify_supernodes.
        snode::SparseUpperCSC< Index > B;
        B.n = m_size;
        B.Ap = &ap;
        B.Ai = &ai;

        snode::Symbolic< Index > Sn;
        Sn.n = m_size;
        Sn.etree = &parent;

        std::vector< std::pair< Index, double > > relaxation;
        relaxation.reserve(m_relaxation.size());
        for (const auto& threshold : m_relaxation)
            relaxation.emplace_back(threshold.max_columns,
                                    threshold.max_zero_fraction);
        auto sn = snode::identify_supernodes< Index >(
            B, Sn, Index{-1}, -1.0, 0.0,
            std::numeric_limits< Index >::max(), &relaxation);

        m_symbolic_builder->m_supernode_ranges.clear();
        m_symbolic_builder->m_supernode_ranges.reserve(sn.ranges.size());
        for (auto rit = sn.ranges.begin(); rit != sn.ranges.end(); ++rit) {
            m_symbolic_builder->m_supernode_ranges.emplace_back(
                static_cast< Int >(rit->first), static_cast< Int >(rit->second));
        }
        m_symbolic_builder->m_etree.assign(parent.begin(), parent.end());
        m_symbolic_builder->m_supernode_parent.assign(sn.parent.begin(),
                                                     sn.parent.end());
        m_symbolic_builder->m_supernode_post.assign(sn.supernode_post.begin(),
                                                   sn.supernode_post.end());
        m_symbolic_builder->m_supernode_row_ptr.assign(sn.row_ptr.begin(),
                                                      sn.row_ptr.end());
        m_symbolic_builder->m_supernode_rows.assign(sn.rows.begin(),
                                                   sn.rows.end());

        const size_t nsuper = sn.ranges.size();
        std::vector< Int > contributor_counts(nsuper, Int{0});
        for (size_t descendant = 0; descendant < nsuper; ++descendant) {
            const Int width = sn.ranges[descendant].second -
                              sn.ranges[descendant].first + 1;
            const Int begin = sn.row_ptr[descendant] + width;
            const Int end = sn.row_ptr[descendant + 1];
            Int previous_target = Int{-1};
            for (Int pos = begin; pos < end; ++pos) {
                const Int row = sn.rows[static_cast< size_t >(pos)];
                const Int target = sn.col2sn[static_cast< size_t >(row)];
                if (target != previous_target) {
                    ++contributor_counts[static_cast< size_t >(target)];
                    previous_target = target;
                }
            }
        }
        m_symbolic_builder->m_contributor_ptr.assign(nsuper + 1, Int{0});
        for (size_t target = 0; target < nsuper; ++target)
            m_symbolic_builder->m_contributor_ptr[target + 1] =
                m_symbolic_builder->m_contributor_ptr[target] +
                contributor_counts[target];
        m_symbolic_builder->m_contributors.resize(static_cast< size_t >(
            m_symbolic_builder->m_contributor_ptr.back()));
        std::vector< Int > contributor_position(
            m_symbolic_builder->m_contributor_ptr.begin(),
            m_symbolic_builder->m_contributor_ptr.end() - 1);
        for (size_t descendant = 0; descendant < nsuper; ++descendant) {
            const Int width = sn.ranges[descendant].second -
                              sn.ranges[descendant].first + 1;
            const Int begin = sn.row_ptr[descendant] + width;
            const Int end = sn.row_ptr[descendant + 1];
            Int previous_target = Int{-1};
            for (Int pos = begin; pos < end; ++pos) {
                const Int row = sn.rows[static_cast< size_t >(pos)];
                const Int target = sn.col2sn[static_cast< size_t >(row)];
                if (target != previous_target) {
                    const Int out =
                        contributor_position[static_cast< size_t >(target)]++;
                    m_symbolic_builder
                        ->m_contributors[static_cast< size_t >(out)] =
                        static_cast< Int >(descendant);
                    previous_target = target;
                }
            }
        }

        m_symbolic_builder->m_panel_value_ptr.assign(nsuper + 1, Int{0});
        size_t panel_storage = 0;
        Int max_front = 0;
        Int max_update_rows = 0;
        for (size_t si = 0; si < nsuper; ++si) {
            const auto [lo, hi] = m_symbolic->m_supernode_ranges[si];
            const Int width = hi - lo + 1;
            const Int rows = m_symbolic->m_supernode_row_ptr[si + 1] -
                             m_symbolic->m_supernode_row_ptr[si];
            max_front = std::max(max_front, rows);
            max_update_rows = std::max(max_update_rows, rows - width);
            panel_storage += static_cast< size_t >(width) *
                             static_cast< size_t >(rows);
            if (panel_storage >
                static_cast< size_t >(std::numeric_limits< Int >::max()))
                throw std::overflow_error(
                    "supernodal: panel storage index overflow");
            m_symbolic_builder->m_panel_value_ptr[si + 1] =
                static_cast< Int >(panel_storage);
        }
        m_symbolic_builder->m_max_front_size = max_front;
        m_symbolic_builder->m_max_update_rows = max_update_rows;
    }


    // ===== State ==========================================================

    Index m_size = 0;
    mutable SupernodalFactor m_factors;
    std::shared_ptr< const SymbolicLDLT > m_symbolic;
    SymbolicLDLT* m_symbolic_builder = nullptr;
    Ordering< Index > m_ordering;
    Ordering< Index > m_externalOrdering;
    bool m_useExternalOrdering = false;
    bool m_patternAnalyzed = false;

    double m_regularization = 1e-12;
    double m_relativeRegularization = 0.0;
    bool m_strictPivots = false;
    BackendPolicy m_backendPolicy = BackendPolicy::Automatic;
    PivotPolicy m_pivotPolicy = PivotPolicy::Regularized1x1;
    SymmetricStorage m_inputStorage = SymmetricStorage::AutoDetect;
    SymmetricStorage m_resolvedStorage = SymmetricStorage::Upper;
    double m_supernodalThreshold = 2.0;
    std::vector< RelaxationThreshold > m_relaxation = defaultRelaxation();
    std::vector< int8_t > m_expectedPivotSigns;
    std::vector< int8_t > m_factorPivotSigns;

    ldlt::SimplicialLDLT< Scalar, Index > m_simplicial;
    DenseMatrix< Real > m_front;
    std::vector< Int > m_globalToLocal;
    std::vector< Int > m_touchedRows;
    std::vector< std::pair< Int, Int > > m_mappedPanelRows;
    std::vector< double > m_localDiagonal;
    std::vector< double > m_localSubdiagonal;
    std::vector< int8_t > m_localPivotBlocks;
    std::vector< Int > m_localPivotPermutation;
    std::vector< NumericEntry > m_numericEntries;
    std::vector< NumericEntry > m_canonicalEntries;
    std::vector< Int > m_permutedAp;
    std::vector< Int > m_permutedAi;
    std::vector< double > m_permutedAx;
    std::vector< double > m_updatePanel;
    std::vector< double > m_scaledUpdatePanel;
    std::vector< double > m_updateProduct;
    mutable std::vector< double > m_solveWorkspace;
    mutable std::vector< Scalar > m_rhsPermutationWorkspace;

};

} // namespace supernodal

#endif // SUPERSONAL_LDLT_STANDALONE_H
