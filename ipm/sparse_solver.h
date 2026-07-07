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

#include "../ipm/qd_ldlt.h"
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
    bool useQDLDLT = false;

    enum SolverType {
        LDLT,
        SUPERNOODAL,
        FRONTAL,
        QD_LDLT,
    };

    SparseSolver(SolverType type = LDLT) : solverType(type), useQDLDLT(type == QD_LDLT) {
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
        case QD_LDLT: {
            auto* w = new QDLDLTWrapper();
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
        case QD_LDLT:
            return "QDLDLT (BK+iterative refinement)";
        }
        return "unknown";
    }

    /** Switch active solver type. Call before factorize. */
    void setSolverType(SolverType type) {
        solverType = type;
        useQDLDLT = (type == QD_LDLT);
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
        case QD_LDLT: {
            auto* w = new QDLDLTWrapper();
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
            std::cerr << "[QDLDLT] S size mismatch: " << S.rows() << "x" << S.cols() << " expected "
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

    friend struct QDLDLTWrapper;

    /** QDLDLT wrapper: builds augmented system from A + theta/regP/regD
     *  with BK pivoting + iterative refinement. */
    struct QDLDLTWrapper : public SolverBase {
        SparseSolver* parent = nullptr;
        ipm::QDLDLT qd;
        int info_code = 0;

        void
        factorizeMatrix(const Eigen::SparseMatrix< double, Eigen::ColMajor, int >& /*S*/) override {
            try {
                Eigen::SparseMatrix< double, Eigen::ColMajor, int > A_copy(parent->extractA());
                qd.analyzePattern(A_copy);

                std::vector< ipm::QDReal > theta_v(parent->theta.data(),
                                                   parent->theta.data() + parent->n);
                std::vector< ipm::QDReal > regP_v(parent->regP.data(),
                                                  parent->regP.data() + parent->n);
                std::vector< ipm::QDReal > regD_v(parent->regD.data(),
                                                  parent->regD.data() + parent->m);
                qd.factorize(theta_v, regP_v, regD_v);
                info_code = 0;
            } catch (...) {
                info_code = 1;
            }
        }

        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override {
            try {
                std::vector< ipm::QDReal > rhs_v(rhs.data(), rhs.data() + rhs.size());
                auto result = qd.solve(rhs_v);
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
            qd = ipm::QDLDLT();
            info_code = 0;
            // parent stays valid — points to SparseSolver::ls member
        }

        int info() override {
            return info_code;
        }
    };

    SolverBase* solver;
};
