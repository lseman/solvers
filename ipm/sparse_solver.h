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

#include "../linear_system/eigen_addon/ldlt_eigen_interop.h"
#include "../linear_system/schur_frontal_ldlt.h"
#include "../linear_system/supernodal_ldlt.h"

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
  Eigen::SparseMatrix<double> A;
  Eigen::SparseMatrix<double> S;
  Eigen::SparseMatrix<double> AD;
  Eigen::SparseMatrix<double> D;
  bool firstFactorization = true;

  enum SolverType {
    LDLT,
    SUPERNOODAL,
    FRONTAL,
  };

  SparseSolver(SolverType type = LDLT) : solverType(type) {
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
    }
  }

  ~SparseSolver() { delete solver; }

  void factorizeMatrix(
      const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &matrix) {
    solver->factorizeMatrix(matrix);
  }

  void reset() { solver->reset(); }

  int info() { return solver->info(); }

  Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
    return solver->solve(rhs);
  }

  // Query active solver type
  SolverType activeSolverType() const { return solverType; }
  std::string solverName() const {
    switch (solverType) {
    case LDLT:           return "CustomSimplicialLDLT";
    case SUPERNOODAL:    return "SupernodalLDLT";
    case FRONTAL:        return "FrontalLDLT";
    }
    return "unknown";
  }

  /** Switch active solver type. Call before factorize. */
  void setSolverType(SolverType type) {
    solverType = type;
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
    }
  }

private:
  SolverType solverType;

  struct SolverBase {
    virtual void factorizeMatrix(
        const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &matrix) = 0;
    virtual Eigen::VectorXd solve(const Eigen::VectorXd &rhs) = 0;
    virtual ~SolverBase() = default;
    virtual void reset() = 0;
    virtual int info() = 0;
  };

  struct SupernodalLDLTWrapper : public SolverBase {
    Eigen::SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower,
                           Eigen::AMDOrdering<int>> ldlt;
    int info_code = 0;

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>
                             &matrix) override {
      try {
        ldlt.factorizeMatrix(matrix);
        info_code = (ldlt.info() == Eigen::Success) ? 0 : 1;
      } catch (...) {
        info_code = 1;
      }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) override {
      if (info_code != 0 || rhs.size() != static_cast<int>(ldlt.size())) return rhs;
      try {
        return ldlt.solve(rhs);
      } catch (...) {
        return rhs;
      }
    }

    void reset() override { ldlt.reset(); }

    int info() override { return info_code; }
  };

  struct CustomLDLTWrapper : public SolverBase {
    ldlt::SimplicialLDLT<double, int> ldlt;
    int info_code = 0;

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>
                             &matrix) override {
      try {
        auto csc = ldlt::fromEigenSparse<double, int>(matrix);
        ldlt.factorizeMatrix(csc);
        info_code = ldlt.info();
      } catch (...) {
        info_code = 1;
      }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) override {
      if (info_code != 0 || rhs.size() != static_cast<int>(ldlt.size())) return rhs;
      try {
        return ldlt::solveEigen<double, int>(ldlt, rhs);
      } catch (...) {
        return rhs;
      }
    }

    void reset() override {
      ldlt = ldlt::SimplicialLDLT<double, int>();
      info_code = 0;
    }

    int info() override { return info_code; }
  };

  struct FrontalLDLTWrapper : public SolverBase {
    schur_frontal::FrontalLDLT ldlt;
    int info_code = 0;

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>
                             &matrix) override {
      try {
        // Build supernodes via pattern merging (group consecutive columns
        // with overlapping row patterns into dense frontal matrices)
        std::vector<std::pair<int, int>> supernode_ranges;
        std::vector<int> col2sn(matrix.cols());
        int sn_id = 0;

        int start_col = 0;
        for (int j = 1; j <= matrix.cols(); ++j) {
          bool should_merge = false;
          if (j < matrix.cols()) {
            // Heuristic: merge if column density is high and consecutive
            // (in production: use AMD graph + elimination tree)
            int nnz_j = matrix.innerVector(j).nonZeros();
            int nnz_prev = matrix.innerVector(j - 1).nonZeros();
            should_merge = (nnz_j > 3 && nnz_prev > 3);
          }

          if (!should_merge || j == matrix.cols()) {
            int end_col = (j == matrix.cols()) ? j - 1 : j - 1;
            supernode_ranges.push_back({start_col, end_col});
            for (int k = start_col; k <= end_col; ++k)
              col2sn[k] = sn_id;
            sn_id++;
            if (j < matrix.cols())
              start_col = j;
          }
        }

        // Build elimination tree (simple: parent is next supernode)
        std::vector<int> etree(matrix.cols(), -1);
        for (int j = 0; j < matrix.cols() - 1; ++j) {
          etree[j] = j + 1;
        }

        ldlt = schur_frontal::build_and_factor_frontal(matrix, supernode_ranges,
                                                       col2sn, etree);
        info_code = ldlt.factorized ? 0 : 1;
      } catch (...) {
        info_code = 1;
      }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) override {
      if (!ldlt.factorized || rhs.size() != ldlt.n)
        return rhs;
      try {
        std::vector<double> rhs_std(rhs.data(), rhs.data() + rhs.size());
        auto sol_std = schur_frontal::solve(ldlt, rhs_std);
        return Eigen::VectorXd::Map(sol_std.data(), sol_std.size());
      } catch (...) {
        return rhs;
      }
    }

    void reset() override {
      ldlt.fronts.clear();
      ldlt.factorized = false;
    }

    int info() override { return info_code; }
  };

  template <typename Solver> struct SolverWrapper : public SolverBase {
    Solver solver;
    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>
                             &matrix) override {
      solver.factorizeMatrix(matrix);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) override {
      return solver.solve(rhs);
    }
    void reset() override { solver.reset(); }
    int info() override { return solver.info(); }
  };

  SolverBase *solver;
};
