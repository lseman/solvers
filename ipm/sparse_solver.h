/*
 * @file sparse_solver.h
 * @brief Hybrid sparse linear solver for IPM KKT systems.
 *
 * Provides SparseSolver wrapper with three factorization methods:
 * - CustomSimplicialLDLT (LDLT): Standard sparse LDL^T (sorted-vector optimized)
 * - SupernodalLDLT (SUPERNOODAL): Dense BLAS kernels on supernodes + sparse fallback
 * - FrontalLDLT (FRONTAL): Schur-complement frontal method (SOTA for structured)
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <numeric>

#include "../ipm/hipo_ldlt.h"
#include "../ipm/normal_eq_ldlt.h"
#include "../linear_system/eigen_interop/ldlt_eigen_interop.h"
#include "../linear_system/eigen_interop/schur_frontal_eigen_interop.h"
#include "../linear_system/eigen_interop/supernodal_eigen_interop.h"

/**
 * @class SparseSolver
 * @brief Hybrid sparse linear solver supporting LDLT and frontal factorization.
 */
class SparseSolver {
  public:
    int n;
    int m;
    Eigen::VectorXd theta;
    Eigen::VectorXd regP;
    Eigen::VectorXd regD;
    Eigen::SparseMatrix< double > A;
    Eigen::SparseMatrix< double > S;
    Eigen::SparseMatrix< double > AD;
    Eigen::SparseMatrix< double > D;
    bool firstFactorization = true;
    bool useInternalAugmentedSolver = false;

    enum SolverType {
        LDLT,
        SUPERNOODAL,
        FRONTAL,
        HIPO_LDLT,
        NORMAL_EQ,
        AUTO,
    };

    SparseSolver(SolverType type = LDLT)
        : solverType(type), useInternalAugmentedSolver(type == HIPO_LDLT) {
        switch (type) {
        case LDLT:
            solver = new CustomLDLTWrapper();
            break;
        case SUPERNOODAL:
            solver = new SupernodalLDLTWrapper();
            break;
        case FRONTAL:
            solver = new FrontalLDLTWrapper();
            break;
        case HIPO_LDLT: {
            auto* w = new HiPOLDLTWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        case NORMAL_EQ: {
            auto* w = new NormalEqWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        case AUTO: {
            auto* w = new AutoWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        }
    }

    ~SparseSolver() {
        delete solver;
    }

    void factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) {
        solver->factorizeMatrix(matrix);
    }

    void reset() {
        solver->reset();
    }

    int info() {
        return solver->info();
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) {
        return solver->solve(rhs);
    }

    // Query active solver type
    SolverType activeSolverType() const {
        return solverType;
    }
    std::string solverName() const {
        switch (solverType) {
        case LDLT:
            return "CustomSimplicialLDLT";
        case SUPERNOODAL:
            return "SupernodalLDLT";
        case FRONTAL:
            return "FrontalLDLT";
        case HIPO_LDLT:
            return "HiPOLDLT (multifrontal BK+iterative refinement)";
        case NORMAL_EQ:
            return "NormalEqLDLT (A Theta^-1 A^T)";
        case AUTO:
            return "Auto (augmented vs normal-eq by fill estimate)";
        }
        return "unknown";
    }

    /** Switch active solver type. Call before factorize. */
    void setSolverType(SolverType type) {
        solverType = type;
        useInternalAugmentedSolver = (type == HIPO_LDLT || type == AUTO);
        reset();
        switch (type) {
        case LDLT:
            solver = new CustomLDLTWrapper();
            break;
        case SUPERNOODAL:
            solver = new SupernodalLDLTWrapper();
            break;
        case FRONTAL:
            solver = new FrontalLDLTWrapper();
            break;
        case HIPO_LDLT: {
            auto* w = new HiPOLDLTWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        case NORMAL_EQ: {
            auto* w = new NormalEqWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        case AUTO: {
            auto* w = new AutoWrapper();
            w->parent = this;
            solver = w;
            break;
        }
        }
    }

  private:
    SolverType solverType;

    struct SolverBase {
        virtual void
        factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& rhs) = 0;
        virtual ~SolverBase() = default;
        virtual void reset() = 0;
        virtual int info() = 0;
    };

    struct SupernodalLDLTWrapper : public SolverBase {
        supernodal::SupernodalLDLT< double, int > ldlt;
        int info_code = 0;

        void factorizeMatrix(
            const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) override {
            try {
                supernodal::factorizeEigen(ldlt, matrix);
                info_code = ldlt.info();
            } catch (...) {
                info_code = 1;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            if (info_code != 0 || rhs.size() != static_cast< int >(ldlt.size()))
                return rhs;
            try {
                return supernodal::solveEigen(ldlt, rhs);
            } catch (...) {
                return rhs;
            }
        }

        void reset() override {
            ldlt.reset();
        }

        int info() override {
            return info_code;
        }
    };

    struct CustomLDLTWrapper : public SolverBase {
        ldlt::SimplicialLDLT< double, int > ldlt;
        int info_code = 0;

        void factorizeMatrix(
            const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) override {
            try {
                auto csc = ldlt::fromEigenSparse< double, int >(matrix);
                ldlt.factorizeMatrix(csc);
                info_code = ldlt.info();
            } catch (...) {
                info_code = 1;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            if (info_code != 0 || rhs.size() != static_cast< int >(ldlt.size()))
                return rhs;
            try {
                return ldlt::solveEigen< double, int >(ldlt, rhs);
            } catch (...) {
                return rhs;
            }
        }

        void reset() override {
            ldlt = ldlt::SimplicialLDLT< double, int >();
            info_code = 0;
        }

        int info() override {
            return info_code;
        }
    };

    struct FrontalLDLTWrapper : public SolverBase {
        schur_frontal::FrontalLDLT ldlt;
        int info_code = 0;

        void factorizeMatrix(
            const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) override {
            try {
                ldlt = schur_frontal::factor_frontal(matrix);
                info_code = ldlt.factorized ? 0 : 1;
            } catch (...) {
                info_code = 1;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            if (!ldlt.factorized || rhs.size() != ldlt.n)
                return rhs;
            std::vector< double > rhs_std(rhs.data(), rhs.data() + rhs.size());
            auto sol = schur_frontal::solve(ldlt, rhs_std);
            return Eigen::Map< Eigen::VectorXd >(sol.data(), static_cast< int >(sol.size()));
        }

        void reset() override {
            ldlt.fronts.clear();
            ldlt.factorized = false;
        }

        int info() override {
            return info_code;
        }
    };

    template < typename Solver > struct SolverWrapper : public SolverBase {
        Solver solver;
        void factorizeMatrix(
            const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& matrix) override {
            solver.factorizeMatrix(matrix);
        }
        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            return solver.solve(rhs);
        }
        void reset() override {
            solver.reset();
        }
        int info() override {
            return solver.info();
        }
    };

    /** Extract A block from augmented system S. S = [-(Θ⁻¹+R_p) Aᵀ; A R_d].
     *  A block is rows n..n+m-1, cols 0..n-1. */
    Eigen::SparseMatrix< double, Eigen::ColMajor, int > extractA() const {
        Eigen::SparseMatrix< double, Eigen::ColMajor, int > A_out(m, n);
        std::vector< Eigen::Triplet< double > > triplets;
        triplets.reserve(static_cast< size_t >(m) * 5);
        if (S.rows() != n + m || S.cols() != n + m) {
            // S is not the augmented system — return empty
            std::cerr << "[HiPOLDLT] S size mismatch: " << S.rows() << "x" << S.cols() << " expected "
                      << (n + m) << "x" << n << std::endl;
            return A_out;
        }
        for (int j = 0; j < n; ++j) {
            for (Eigen::SparseMatrix< double, Eigen::ColMajor, int >::InnerIterator it(S, j); it;
                 ++it) {
                int row = it.row() - n;
                if (row >= 0 && row < m) {
                    triplets.emplace_back(row, j, it.value());
                }
            }
        }
        A_out.setFromTriplets(triplets.begin(), triplets.end());
        return A_out;
    }

    friend struct HiPOLDLTWrapper;

    /** HiPO wrapper: builds the augmented system from A + theta/regP/regD
     *  with multifrontal BK pivoting + iterative refinement. */
    struct HiPOLDLTWrapper : public SolverBase {
        SparseSolver* parent = nullptr;
        ipm::HiPOLDLT hipo;
        int info_code = 0;

        void
        factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& /*S*/) override {
            try {
                Eigen::SparseMatrix< double, Eigen::ColMajor, int > A_copy(parent->extractA());
                hipo.analyzePattern(A_copy);

                std::vector< ipm::HReal > theta_v(parent->theta.data(),
                                                  parent->theta.data() + parent->n);
                std::vector< ipm::HReal > regP_v(parent->regP.data(),
                                                 parent->regP.data() + parent->n);
                std::vector< ipm::HReal > regD_v(parent->regD.data(),
                                                 parent->regD.data() + parent->m);
                hipo.factorize(theta_v, regP_v, regD_v);
                info_code = 0;
            } catch (...) {
                info_code = 1;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            try {
                std::vector< ipm::HReal > rhs_v(rhs.data(), rhs.data() + rhs.size());
                std::vector< ipm::HReal > theta_v(parent->theta.data(),
                                                  parent->theta.data() + parent->n);
                std::vector< ipm::HReal > regP_v(parent->regP.data(),
                                                 parent->regP.data() + parent->n);
                std::vector< ipm::HReal > regD_v(parent->regD.data(),
                                                 parent->regD.data() + parent->m);
                auto [result, refinement_iters] =
                    hipo.solveWithRefinement(rhs_v, theta_v, regP_v, regD_v);
                (void) refinement_iters;
                Eigen::VectorXd out(static_cast< int >(result.size()));
                for (int i = 0; i < static_cast< int >(result.size()); ++i) {
                    out[i] = result[static_cast< size_t >(i)];
                }
                return out;
            } catch (...) {
                return rhs;
            }
        }

        void reset() override {
            hipo = ipm::HiPOLDLT();
            info_code = 0;
            // parent stays valid — points to SparseSolver::ls member
        }

        int info() override {
            return info_code;
        }
    };

    /** Normal-equations wrapper: solves (A Theta^-1 A^T + R_d) dy = ... and
     *  recovers dx, bypassing the augmented system entirely. */
    struct NormalEqWrapper : public SolverBase {
        SparseSolver* parent = nullptr;
        ipm::NormalEqLDLT ne;
        int info_code = 0;
        int last_n = -1;

        void
        factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& /*S*/) override {
            try {
                Eigen::SparseMatrix< double, Eigen::ColMajor, int > A_copy(parent->extractA());
                if (parent->n != last_n) {
                    ne.analyzePattern(A_copy);
                    last_n = parent->n;
                }

                std::vector< ipm::NEReal > theta_v(parent->theta.data(),
                                                   parent->theta.data() + parent->n);
                std::vector< ipm::NEReal > regP_v(parent->regP.data(),
                                                  parent->regP.data() + parent->n);
                std::vector< ipm::NEReal > regD_v(parent->regD.data(),
                                                  parent->regD.data() + parent->m);
                ne.factorize(theta_v, regP_v, regD_v);
                info_code = 0;
            } catch (...) {
                info_code = 1;
            }
        }

        /** rhs is the stacked augmented-system RHS: [xi_d (n); xi_p (m)]. */
        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            try {
                const int n = ne.n();
                const int m = ne.m();
                std::vector< ipm::NEReal > xi_d(rhs.data(), rhs.data() + n);
                std::vector< ipm::NEReal > xi_p(rhs.data() + n, rhs.data() + n + m);
                auto [dx, dy] = ne.solve(xi_d, xi_p);

                Eigen::VectorXd out(n + m);
                for (int i = 0; i < n; ++i)
                    out[i] = dx[static_cast< size_t >(i)];
                for (int i = 0; i < m; ++i)
                    out[n + i] = dy[static_cast< size_t >(i)];
                return out;
            } catch (...) {
                return rhs;
            }
        }

        void reset() override {
            ne = ipm::NormalEqLDLT();
            info_code = 0;
            last_n = -1;
        }

        int info() override {
            return info_code;
        }
    };

    /** Picks between the augmented-system (HiPOLDLT) and normal-equations
     *  paths once per analyzePattern, using a structural fill estimate —
     *  the sparsity pattern (and hence the right choice) is invariant
     *  across IPM iterations, so this only runs when n/m change. */
    struct AutoWrapper : public SolverBase {
        SparseSolver* parent = nullptr;
        HiPOLDLTWrapper aug;
        NormalEqWrapper ne;
        bool use_normal_eq = false;
        int last_n = -1;
        int info_code = 0;

        void
        factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& S) override {
            if (parent->n != last_n) {
                Eigen::SparseMatrix< double, Eigen::ColMajor, int > A_copy(parent->extractA());
                auto est = ipm::estimate_fill(A_copy);
                use_normal_eq = est.normal_eq_nnz < est.augmented_nnz;
                last_n = parent->n;
            }
            aug.parent = parent;
            ne.parent = parent;
            if (use_normal_eq) {
                ne.factorizeMatrix(S);
                info_code = ne.info_code;
            } else {
                aug.factorizeMatrix(S);
                info_code = aug.info_code;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            return use_normal_eq ? ne.solve(rhs) : aug.solve(rhs);
        }

        void reset() override {
            aug.reset();
            ne.reset();
            use_normal_eq = false;
            last_n = -1;
            info_code = 0;
        }

        int info() override {
            return info_code;
        }
    };

    SolverBase* solver;
};
