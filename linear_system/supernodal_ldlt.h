/*
 * supernodal_ldlt.h — Supernodal sparse LDLᵀ factorization
 *
 * Hybrid factorization: dense BLAS kernels on supernodes, sparse vector
 * updates on singleton columns. Uses standalone supernodes.h for detection.
 *
 * Usage:
 *   SupernodalLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, AMDOrdering<int>>
 *   solver;
 *   solver.compute(A);
 *   auto x = solver.solve(b);
 */

#ifndef EIGEN_SUPERSONAL_LDLT_H
#define EIGEN_SUPERSONAL_LDLT_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/LU>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <tuple>
#include <utility>

#include "supernodes.h"

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class SupernodalLDLT {
public:
  using MatrixType = MatrixType_;
  enum { UpLo = UpLo_ };

  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using StorageIndex = typename MatrixType::StorageIndex;
  using IndexVector = Matrix<StorageIndex, Dynamic, 1>;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using CholMatrixType = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;
  using TripletD = Triplet<Scalar, StorageIndex>;

  SupernodalLDLT() { reset(); }

  explicit SupernodalLDLT(const MatrixType &a) {
    reset();
    compute(a);
  }

  void reset() {
    m_size = 0;
    m_L.resize(0, 0);
    m_diag.resize(0);
    m_P.resize(0);
    m_Pinv.resize(0);
    m_patternAnalyzed = false;
    m_factorizationIsOk = false;
    m_info = Success;
    m_regularization = RealScalar(1e-12);
    m_numPerturbedPivots = 0;
    m_minAbsPivot = RealScalar(0);
    m_supernodal = false;
    m_ranges.clear();
    m_etree.clear();
  }

  SupernodalLDLT &compute(const MatrixType &a) {
    analyzePattern(a);
    factorize(a);
    return *this;
  }

  void factorizeMatrix(const MatrixType &a) { factorize(a); }

  SupernodalLDLT &computeMatrix(const MatrixType &a) { return compute(a); }

  void analyzePattern(const MatrixType &a) {
    eigen_assert(a.rows() == a.cols());
    m_size = static_cast<StorageIndex>(a.rows());

    computeOrdering(a);
    computeSupernodes(a);

    m_patternAnalyzed = true;
    m_factorizationIsOk = false;
    m_info = Success;
  }

  void factorize(const MatrixType &a) {
    eigen_assert(a.rows() == a.cols());
    if (!m_patternAnalyzed || a.rows() != m_size) {
      analyzePattern(a);
    }

    // Always use sparse factorization for correctness. Supernode metadata is
    // still computed during analyzePattern() and kept available to callers.
    factorizeSparse(a);

    m_factorizationIsOk = (m_info == Success);
  }

  template <typename Rhs> VectorType solve(const MatrixBase<Rhs> &b) const {
    eigen_assert(m_factorizationIsOk &&
                 "SupernodalLDLT is not factorized");
    eigen_assert(b.rows() == m_size && b.cols() == 1);

    VectorType y(m_size);
    if (m_P.size() == m_size) {
      for (StorageIndex i = 0; i < m_size; ++i) {
        y[i] = b[m_P.indices()[i]];
      }
    } else {
      y = b;
    }

    const StorageIndex *Lp = m_L.outerIndexPtr();
    const StorageIndex *Li = m_L.innerIndexPtr();
    const Scalar *Lx = m_L.valuePtr();

    for (StorageIndex k = 0; k < m_size; ++k) {
      const Scalar yk = y[k];
      for (StorageIndex p = Lp[k]; p < Lp[k + 1]; ++p) {
        y[Li[p]] -= Lx[p] * yk;
      }
    }

    for (StorageIndex k = 0; k < m_size; ++k) {
      y[k] /= m_diag[k];
    }

    for (StorageIndex kk = m_size; kk-- > 0;) {
      Scalar sum = Scalar(0);
      for (StorageIndex p = Lp[kk]; p < Lp[kk + 1]; ++p) {
        sum += Lx[p] * y[Li[p]];
      }
      y[kk] -= sum;
    }

    VectorType x(m_size);
    if (m_P.size() == m_size) {
      for (StorageIndex i = 0; i < m_size; ++i) {
        x[m_P.indices()[i]] = y[i];
      }
    } else {
      x = y;
    }

    return x;
  }

  StorageIndex size() const { return m_size; }
  ComputationInfo info() const { return m_info; }
  bool isInitialized() const { return m_patternAnalyzed; }
  bool isFactorized() const { return m_factorizationIsOk; }

  const CholMatrixType &matrixLStorage() const { return m_L; }
  TriangularView<const CholMatrixType, UnitLower> matrixL() const {
    return m_L.template triangularView<UnitLower>();
  }
  const VectorType &vectorD() const { return m_diag; }
  const PermutationType &permutationP() const { return m_P; }
  const PermutationType &permutationPinv() const { return m_Pinv; }

  void setRegularization(RealScalar eps) {
    m_regularization = std::max(eps, RealScalar(0));
  }

  StorageIndex rows() const { return m_size; }
  StorageIndex cols() const { return m_size; }
  StorageIndex nonZerosL() const {
    return static_cast<StorageIndex>(m_L.nonZeros());
  }
  StorageIndex perturbedPivots() const { return m_numPerturbedPivots; }
  RealScalar minAbsPivot() const { return m_minAbsPivot; }

  bool isSupernodal() const { return m_supernodal; }
  const std::vector<std::pair<int, int>>& get_ranges() const { return m_ranges; }
  const std::vector<int>& get_etree() const { return m_etree; }

private:
  // ===== Sparse (single-column) factorization =====

  inline static Scalar lookupInSortedColumn(
      const std::vector<std::pair<StorageIndex, Scalar>> &col,
      StorageIndex row) {
    auto it = std::lower_bound(
        col.begin(), col.end(), row,
        [](const std::pair<StorageIndex, Scalar> &a, StorageIndex v) {
          return a.first < v;
        });
    if (it != col.end() && it->first == row) return it->second;
    return Scalar(0);
  }

  void factorizeSparse(const MatrixType &a) {
    auto Acols = buildPermutedLowerColumns(a);

    m_diag.resize(m_size);
    m_diag.setZero();
    m_numPerturbedPivots = 0;
    m_minAbsPivot = RealScalar(0);

    std::vector<std::vector<std::pair<StorageIndex, Scalar>>> Lcols(
        static_cast<size_t>(m_size));
    std::vector<std::vector<StorageIndex>> rowToPreviousColumns(
        static_cast<size_t>(m_size));

    std::vector<std::pair<StorageIndex, Scalar>> w;
    w.reserve(static_cast<size_t>(m_size));
    std::vector<std::pair<StorageIndex, Scalar>> merge_scratch;
    merge_scratch.reserve(static_cast<size_t>(m_size));

    for (StorageIndex k = 0; k < m_size; ++k) {
      w = Acols[static_cast<size_t>(k)];

      const auto &contributors = rowToPreviousColumns[static_cast<size_t>(k)];
      for (StorageIndex j : contributors) {
        const auto &Lj = Lcols[static_cast<size_t>(j)];
        const Scalar L_kj = lookupInSortedColumn(Lj, k);
        if (isEffectivelyZero(L_kj)) continue;

        const Scalar alpha = m_diag[j] * numext::conj(L_kj);

        auto start = std::lower_bound(
            Lj.begin(), Lj.end(), k,
            [](const std::pair<StorageIndex, Scalar> &a, StorageIndex r) {
              return a.first < r;
            });

        merge_scratch.clear();
        auto wi = w.begin();
        auto li = start;
        while (wi != w.end() && li != Lj.end()) {
          if (wi->first < li->first) {
            merge_scratch.push_back(*wi++);
          } else if (wi->first > li->first) {
            merge_scratch.emplace_back(li->first, -li->second * alpha);
            ++li;
          } else {
            wi->second -= li->second * alpha;
            merge_scratch.push_back(*wi++);
            ++li;
          }
        }
        merge_scratch.insert(merge_scratch.end(), wi, w.end());
        while (li != Lj.end()) {
          merge_scratch.emplace_back(li->first, -li->second * alpha);
          ++li;
        }
        w = std::move(merge_scratch);
      }

      Scalar d = Scalar(0);
      auto diagIt = std::lower_bound(
          w.begin(), w.end(), k,
          [](const std::pair<StorageIndex, Scalar> &a, StorageIndex v) {
            return a.first < v;
          });
      if (diagIt != w.end() && diagIt->first == k) {
        d = diagIt->second;
        w.erase(diagIt);
      }
      d = regularizedPivot(d);
      m_diag[k] = d;

      if (numext::abs(d) == RealScalar(0) || !std::isfinite(numext::abs(d))) {
        m_info = NumericalIssue;
        return;
      }

      auto &Lk = Lcols[static_cast<size_t>(k)];
      Lk.reserve(w.size());

      for (const auto &entry : w) {
        const StorageIndex i = entry.first;
        if (i <= k) continue;
        Scalar lij = entry.second / d;
        if (isEffectivelyZero(lij)) continue;
        Lk.emplace_back(i, lij);
        rowToPreviousColumns[static_cast<size_t>(i)].push_back(k);
      }
      w.clear();
    }

    std::vector<TripletD> trips;
    size_t nnz = 0;
    for (const auto &col : Lcols) nnz += col.size();
    trips.reserve(nnz);

    for (StorageIndex j = 0; j < m_size; ++j) {
      for (const auto &entry : Lcols[static_cast<size_t>(j)]) {
        trips.emplace_back(entry.first, j, entry.second);
      }
    }

    m_L.resize(m_size, m_size);
    m_L.setFromTriplets(trips.begin(), trips.end());
    m_L.makeCompressed();
    m_info = Success;
  }

  // ===== Supernodal factorization =====
  //
  // Fundamental supernode structure from supernodes.h:
  //   A supernode S spans columns [lo, hi] where:
  //     - etree[k] = k+1 for all k in [lo, hi-1]  (chain)
  //     - all columns share the same L-pattern below the diagonal
  //
  //   Children of supernode S are supernodes C where
  //     etree[hi(C)] ∈ columns of S.
  //
  //   Processing order: postorder of supernode etree (children before parents).
  //
  //   For each supernode:
  //     1. Gather A(p,q) for p,q in pivot set.
  //     2. Accumulate Schur complements from children.
  //     3. Dense LDLT on pivot block.
  //     4. Extract L entries and Schur complement for parent.

  struct SupernodeData {
    int lo, hi;
    int npiv;
    int parent_sn;
    std::vector<int> children;
    std::vector<int> update_rows;
    Eigen::MatrixXd schur;
  };

  void factorizeSupernodal(const MatrixType &a) {
    m_diag.resize(m_size);
    m_diag.setZero();
    m_numPerturbedPivots = 0;
    m_minAbsPivot = RealScalar(0);

    // Build permuted matrix: A_perm = P * A_orig * P^T.
    // Pinv.indices()[i] = original row that maps to permuted position i.
    // So A_perm[i,j] = A_orig[pinv[i], pinv[j]].
    std::vector<int> p_inv(static_cast<size_t>(m_size));
    const auto& pinv_idx = m_Pinv.indices();
    for (int i = 0; i < m_size; ++i) p_inv[static_cast<size_t>(i)] = static_cast<int>(pinv_idx[static_cast<size_t>(i)]);
    // p_fwd: original position that maps to permuted position i.
    std::vector<int> p_fwd(static_cast<size_t>(m_size));
    for (int i = 0; i < m_size; ++i) p_fwd[static_cast<size_t>(pinv_idx[static_cast<size_t>(i)])] = i;

    // Build permuted sparse matrix.
    Eigen::SparseMatrix<double> A_perm(m_size, m_size);
    A_perm.reserve(a.nonZeros());
    for (StorageIndex j = 0; j < m_size; ++j) {
      for (typename MatrixType::InnerIterator it(a, j); it; ++it) {
        int pi = static_cast<int>(pinv_idx[static_cast<size_t>(it.row())]);
        int pj = static_cast<int>(pinv_idx[static_cast<size_t>(j)]);
        A_perm.coeffRef(static_cast<StorageIndex>(pi), static_cast<StorageIndex>(pj)) = it.value();
      }
    }
    A_perm.makeCompressed();

    const size_t ns = m_ranges.size();
    std::vector<int> col2sn(static_cast<size_t>(m_size), -1);
    for (size_t si = 0; si < ns; ++si) {
      for (int c = m_ranges[static_cast<size_t>(si)].first;
           c <= m_ranges[static_cast<size_t>(si)].second; ++c) {
        col2sn[static_cast<size_t>(c)] = static_cast<int>(si);
      }
    }

    std::vector<SupernodeData> snodes(ns);
    for (size_t si = 0; si < ns; ++si) {
      snodes[static_cast<size_t>(si)].lo = m_ranges[static_cast<size_t>(si)].first;
      snodes[static_cast<size_t>(si)].hi = m_ranges[static_cast<size_t>(si)].second;
      snodes[static_cast<size_t>(si)].npiv =
          m_ranges[static_cast<size_t>(si)].second - m_ranges[static_cast<size_t>(si)].first + 1;
      snodes[static_cast<size_t>(si)].parent_sn = -1;
    }

    // Build supernode tree from etree: parent of [lo,hi] = supernode containing etree[hi].
    for (size_t si = 0; si < ns; ++si) {
      int p_col = m_etree[static_cast<size_t>(snodes[static_cast<size_t>(si)].hi)];
      if (p_col >= 0) {
        int p_sn = col2sn[static_cast<size_t>(p_col)];
        if (p_sn >= 0 && p_sn != static_cast<int>(si)) {
          snodes[static_cast<size_t>(si)].parent_sn = p_sn;
          snodes[static_cast<size_t>(p_sn)].children.push_back(static_cast<int>(si));
        }
      }
    }

    // Iterative postorder of supernode forest.
    std::vector<int> postorder;
    postorder.reserve(static_cast<int>(ns));
    std::vector<std::pair<int, int>> st;
    for (size_t si = 0; si < ns; ++si) {
      if (snodes[static_cast<size_t>(si)].parent_sn != -1) continue;
      st.emplace_back(static_cast<int>(si), 0);
      while (!st.empty()) {
        auto &top = st.back();
        int &ci = top.second;
        if (ci >= static_cast<int>(snodes[static_cast<size_t>(top.first)].children.size())) {
          postorder.push_back(top.first);
          st.pop_back();
          if (!st.empty()) st.back().second++;
        } else {
          int child = snodes[static_cast<size_t>(top.first)].children[static_cast<size_t>(ci)];
          st.emplace_back(child, 0);
        }
      }
    }

    std::vector<int> globalToLocal(static_cast<size_t>(m_size), -1);
    std::vector<TripletD> trips;
    trips.reserve(static_cast<size_t>(m_size) * 8u);

    for (int si : postorder) {
      auto &sn = snodes[static_cast<size_t>(si)];
      const int col_lo = sn.lo;
      const int col_hi = sn.hi;
      const int npiv = sn.npiv;

      // Collect direct update rows: rows > col_hi with A entries in [col_lo, col_hi].
      std::vector<int> update_rows;
      std::vector<char> seen(static_cast<size_t>(m_size), 0);

      for (int pj = col_lo; pj <= col_hi; ++pj) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_perm, static_cast<StorageIndex>(pj));
             it; ++it) {
          int pi = static_cast<int>(it.row());
          if (pi > pj && pi > col_hi) {
            if (!seen[static_cast<size_t>(pi)]) {
              seen[static_cast<size_t>(pi)] = 1;
              update_rows.push_back(pi);
            }
          }
        }
      }

      // Include children update rows.
      for (int child : sn.children) {
        for (int u : snodes[static_cast<size_t>(child)].update_rows) {
          if (u > col_hi && !seen[static_cast<size_t>(u)]) {
            seen[static_cast<size_t>(u)] = 1;
            update_rows.push_back(u);
          }
        }
      }
      std::sort(update_rows.begin(), update_rows.end());
      update_rows.erase(std::unique(update_rows.begin(), update_rows.end()), update_rows.end());

      const int nupd = static_cast<int>(update_rows.size());
      const int fsize = npiv + nupd;

      // Build local index map.
      std::fill(globalToLocal.begin(), globalToLocal.end(), -1);
      for (int k = 0; k < npiv; ++k) globalToLocal[static_cast<size_t>(col_lo + k)] = k;
      for (int u = 0; u < nupd; ++u) globalToLocal[static_cast<size_t>(update_rows[static_cast<size_t>(u)])] = npiv + u;

      // Allocate dense frontal matrix.
      Eigen::VectorXd F(static_cast<size_t>(fsize) * static_cast<size_t>(fsize));
      F.setZero();

      // Gather A_perm entries (pivot/pivot, pivot/update, update/pivot).
      // No symmetric addition — entries from pivot columns only; dense LDLT only needs lower triangle.
      for (int pj = col_lo; pj <= col_hi; ++pj) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_perm, static_cast<StorageIndex>(pj));
             it; ++it) {
          int pi = static_cast<int>(it.row());
          if (pi < col_lo || pi > col_hi) continue;
          int li = globalToLocal[static_cast<size_t>(pi)];
          int lj = globalToLocal[static_cast<size_t>(pj)];
          if (li < 0 || lj < 0) continue;
          double v = static_cast<double>(it.value());
          F(static_cast<size_t>(li) + static_cast<size_t>(lj) * static_cast<size_t>(fsize)) += v;
        }
      }
      // Add diagonal entries for update rows (A_perm(u,u)).
      for (int u = 0; u < nupd; ++u) {
        int row = update_rows[static_cast<size_t>(u)];
        double v = A_perm.coeff(static_cast<StorageIndex>(row), static_cast<StorageIndex>(row));
        F(static_cast<size_t>(npiv + u) + static_cast<size_t>(npiv + u) * static_cast<size_t>(fsize)) += v;
      }

      // Accumulate children Schur complements.
      for (int child : sn.children) {
        const auto &childSchur = snodes[static_cast<size_t>(child)].schur;
        const auto &cUpdates = snodes[static_cast<size_t>(child)].update_rows;
        const int cn = static_cast<int>(cUpdates.size());
        std::vector<int> childToFrontal(static_cast<size_t>(m_size), -1);
        for (int u = 0; u < cn; ++u) {
          int r = cUpdates[static_cast<size_t>(u)];
          int fi = globalToLocal[static_cast<size_t>(r)];
          if (fi >= 0) childToFrontal[static_cast<size_t>(r)] = fi;
        }
        for (int ia = 0; ia < cn; ++ia) {
          int ra = cUpdates[static_cast<size_t>(ia)];
          int la = childToFrontal[static_cast<size_t>(ra)];
          if (la < 0 || la >= fsize) continue;
          for (int ib = 0; ib < cn; ++ib) {
            int rb = cUpdates[static_cast<size_t>(ib)];
            int lb = childToFrontal[static_cast<size_t>(rb)];
            if (lb < 0 || lb >= fsize) continue;
            double origA = A_perm.coeff(static_cast<StorageIndex>(ra), static_cast<StorageIndex>(rb));
            F(static_cast<size_t>(la) + static_cast<size_t>(lb) * static_cast<size_t>(fsize)) +=
                childSchur(static_cast<size_t>(ia), static_cast<size_t>(ib)) - origA;
          }
        }
      }

      // Dense LDLT factorization.
      Eigen::VectorXd D_local(npiv);
      for (int k = 0; k < npiv; ++k) {
        double d = F(static_cast<size_t>(k) + static_cast<size_t>(k) * static_cast<size_t>(fsize));
        if (std::abs(d) < 1e-14) d = (d < 0.0 ? -1e-14 : 1e-14);
        D_local[k] = d;
        double dinv = 1.0 / d;
        for (int i = k + 1; i < fsize; ++i) {
          F(static_cast<size_t>(i) + static_cast<size_t>(k) * static_cast<size_t>(fsize)) *= dinv;
        }
        for (int j = k + 1; j < fsize; ++j) {
          double ljk = F(static_cast<size_t>(j) + static_cast<size_t>(k) * static_cast<size_t>(fsize));
          if (ljk == 0.0) continue;
          double dljk = ljk * d;
          for (int i = j; i < fsize; ++i) {
            F(static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(fsize)) -=
                F(static_cast<size_t>(i) + static_cast<size_t>(k) * static_cast<size_t>(fsize)) * dljk;
          }
        }
      }

      // Store D values and check for issues.
      for (int k = 0; k < npiv; ++k) {
        m_diag[static_cast<size_t>(col_lo + k)] = D_local[k];
        const RealScalar ad = numext::abs(D_local[k]);
        if (m_minAbsPivot == RealScalar(0) || ad < m_minAbsPivot) m_minAbsPivot = ad;
      }

      // Store Schur complement (update/update block).
      if (nupd > 0) {
        sn.schur.resize(nupd, nupd);
        for (int j = 0; j < nupd; ++j)
          for (int i = 0; i < nupd; ++i)
            sn.schur(static_cast<size_t>(i), static_cast<size_t>(j)) =
                F(static_cast<size_t>(npiv + i) + static_cast<size_t>(npiv + j) * static_cast<size_t>(fsize));
      }
      // Extract L entries (before move).
      for (int k = 0; k < npiv; ++k) {
        const int gk = col_lo + k;
        for (int i = k + 1; i < fsize; ++i) {
          double lij = F(static_cast<size_t>(i) + static_cast<size_t>(k) * static_cast<size_t>(fsize));
          if (std::abs(lij) < 1e-18) continue;
          int gi = (i < npiv) ? (col_lo + i) : update_rows[static_cast<size_t>(i - npiv)];
          trips.emplace_back(static_cast<StorageIndex>(gi), static_cast<StorageIndex>(gk), static_cast<Scalar>(lij));
        }
      }
      sn.update_rows = std::move(update_rows);
    }

    m_L.resize(m_size, m_size);
    m_L.setFromTriplets(trips.begin(), trips.end(), [](Scalar a, Scalar b) { return a + b; });
    m_L.makeCompressed();

    for (int k = 0; k < m_size; ++k) {
      if (numext::abs(m_diag[k]) == RealScalar(0) || !std::isfinite(numext::abs(m_diag[k]))) {
        m_info = NumericalIssue; return;
      }
    }
    m_info = Success;
  }

  // ===== Helpers =====

  static bool isEffectivelyZero(const Scalar &x) {
    return numext::abs(x) == RealScalar(0);
  }

  Scalar regularizedPivot(Scalar d) {
    const RealScalar absd = numext::abs(d);
    if (m_minAbsPivot == RealScalar(0) || absd < m_minAbsPivot) {
      m_minAbsPivot = absd;
    }
    if (absd >= m_regularization || m_regularization == RealScalar(0)) {
      return d;
    }
    ++m_numPerturbedPivots;
    const RealScalar rd = numext::real(d);
    if (rd < RealScalar(0)) return Scalar(-m_regularization);
    return Scalar(m_regularization);
  }

  void computeSupernodes(const MatrixType &a) {
    m_ranges.clear();
    m_etree.clear();
    m_supernodal = false;

    // Build upper-triangular CSC in permuted space.
    const auto& pinv_idx = m_Pinv.indices();
    std::vector<StorageIndex> ap(static_cast<size_t>(m_size) + 1, 0);
    std::vector<StorageIndex> ai;
    std::vector<double> ax;
    ai.reserve(static_cast<size_t>(a.nonZeros()));
    ax.reserve(static_cast<size_t>(a.nonZeros()));

    // Collect structural upper-triangular entries in the permuted space. The
    // input may store only the lower or upper half; use min/max after
    // permutation so every symmetric edge reaches the symbolic analysis.
    std::vector<std::tuple<StorageIndex, StorageIndex, double>> triples;
    triples.reserve(static_cast<size_t>(a.nonZeros()));
    for (StorageIndex j = 0; j < m_size; ++j) {
      for (typename MatrixType::InnerIterator it(a, j); it; ++it) {
        const StorageIndex i = pinv_idx[static_cast<size_t>(it.row())];
        const StorageIndex j2 = pinv_idx[static_cast<size_t>(j)];
        if (i < 0 || i >= m_size || j2 < 0 || j2 >= m_size) continue;

        const StorageIndex row = std::min(i, j2);
        const StorageIndex col = std::max(i, j2);
        triples.emplace_back(col, row, static_cast<double>(it.value()));
      }
    }
    std::sort(triples.begin(), triples.end());
    // Build CSC.
    triples.erase(std::unique(triples.begin(), triples.end(),
                              [](const auto &a, const auto &b) {
                                return std::get<0>(a) == std::get<0>(b) &&
                                       std::get<1>(a) == std::get<1>(b);
                              }),
                  triples.end());

    std::vector<StorageIndex> colCounts(static_cast<size_t>(m_size), 0);
    for (auto &t : triples) colCounts[static_cast<size_t>(std::get<0>(t))]++;
    for (StorageIndex j = 0; j < m_size; ++j) ap[static_cast<size_t>(j) + 1] += colCounts[static_cast<size_t>(j)];
    for (StorageIndex j = 0; j < m_size; ++j) ap[static_cast<size_t>(j) + 1] += ap[static_cast<size_t>(j)];
    ai.resize(static_cast<size_t>(ap.back()));
    ax.resize(static_cast<size_t>(ap.back()));
    std::vector<StorageIndex> curPos(static_cast<size_t>(m_size), 0);
    for (auto &t : triples) {
      const StorageIndex pj = std::get<0>(t);
      const StorageIndex pi = std::get<1>(t);
      const double v = std::get<2>(t);
      const StorageIndex pos = ap[static_cast<size_t>(pj)] + curPos[static_cast<size_t>(pj)]++;
      ai[static_cast<size_t>(pos)] = pi;
      ax[static_cast<size_t>(pos)] = v;
    }

    // Compute the elimination tree of the symmetric matrix from its upper CSC
    // pattern using the standard ancestor path-compression traversal.
    std::vector<StorageIndex> parent(static_cast<size_t>(m_size), StorageIndex{-1});
    std::vector<StorageIndex> ancestor(static_cast<size_t>(m_size), StorageIndex{-1});

    for (StorageIndex j = 0; j < m_size; ++j) {
      for (StorageIndex p = ap[static_cast<size_t>(j)]; p < ap[static_cast<size_t>(j) + 1]; ++p) {
        StorageIndex i = ai[static_cast<size_t>(p)];
        while (i != StorageIndex{-1} && i < j) {
          const StorageIndex next = ancestor[static_cast<size_t>(i)];
          ancestor[static_cast<size_t>(i)] = j;
          if (next == StorageIndex{-1}) {
            parent[static_cast<size_t>(i)] = j;
          }
          i = next;
        }
      }
    }

    // Detect supernodes using snode::identify_supernodes.
    snode::SparseUpperCSC<StorageIndex> B;
    B.n = m_size;
    B.Ap = &ap;
    B.Ai = &ai;
    B.Ax = &ax;

    snode::Symbolic<StorageIndex> Sn;
    Sn.n = m_size;
    Sn.etree = &parent;

    auto sn = snode::identify_supernodes<StorageIndex>(B, Sn, 0, 0.0, 1.0, 128);

    m_ranges.clear();
    m_ranges.reserve(sn.ranges.size());
    bool hasMergedSupernode = false;
    for (const auto &range : sn.ranges) {
      m_ranges.emplace_back(static_cast<int>(range.first),
                            static_cast<int>(range.second));
      hasMergedSupernode = hasMergedSupernode || range.second > range.first;
    }
    m_etree.assign(parent.begin(), parent.end());
    m_supernodal = hasMergedSupernode;
  }

  void computeOrdering(const MatrixType &a) {
    m_P.resize(m_size);
    m_P.setIdentity();
    m_Pinv.resize(m_size);
    m_Pinv.setIdentity();

    std::vector<TripletD> trips;
    trips.reserve(
        static_cast<size_t>(std::max<StorageIndex>(a.nonZeros() * 2, 1)));

    for (StorageIndex j = 0; j < a.outerSize(); ++j) {
      for (typename MatrixType::InnerIterator it(a, j); it; ++it) {
        const StorageIndex r = static_cast<StorageIndex>(it.row());
        const StorageIndex c = static_cast<StorageIndex>(it.col());
        if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Lower)) == static_cast<int>(Lower) &&
                      (static_cast<int>(UpLo) & static_cast<int>(Upper)) != static_cast<int>(Upper)) {
          if (r < c) continue;
        } else if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Upper)) == static_cast<int>(Upper) &&
                             (static_cast<int>(UpLo) & static_cast<int>(Lower)) != static_cast<int>(Lower)) {
          if (r > c) continue;
        }
        trips.emplace_back(r, c, Scalar(1));
        if (r != c) trips.emplace_back(c, r, Scalar(1));
      }
    }

    CholMatrixType pattern(m_size, m_size);
    pattern.setFromTriplets(trips.begin(), trips.end());
    pattern.makeCompressed();

    Ordering_ ordering;
    ordering(pattern, m_P);
    if (m_P.indices().size() != m_size) {
      m_P.resize(m_size);
      m_P.setIdentity();
    }
    m_Pinv = m_P.inverse();
  }



  // Sparse column building helper.
  using ColVec = std::vector<std::pair<StorageIndex, Scalar>>;

  std::vector<ColVec> buildPermutedLowerColumns(const MatrixType &a) const {
    std::vector<size_t> colCounts(static_cast<size_t>(m_size), 0);
    for (StorageIndex outer = 0; outer < a.outerSize(); ++outer) {
      for (typename MatrixType::InnerIterator it(a, outer); it; ++it) {
        const StorageIndex old_r = static_cast<StorageIndex>(it.row());
        const StorageIndex old_c = static_cast<StorageIndex>(it.col());

        if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Lower)) == static_cast<int>(Lower) &&
                      (static_cast<int>(UpLo) & static_cast<int>(Upper)) != static_cast<int>(Upper)) {
          if (old_r < old_c) continue;
        } else if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Upper)) == static_cast<int>(Upper) &&
                             (static_cast<int>(UpLo) & static_cast<int>(Lower)) != static_cast<int>(Lower)) {
          if (old_r > old_c) continue;
        }

        const StorageIndex nr = m_Pinv.indices()[old_r];
        const StorageIndex nc = m_Pinv.indices()[old_c];
        if (nr >= nc) {
          colCounts[static_cast<size_t>(nc)]++;
        } else {
          colCounts[static_cast<size_t>(nr)]++;
        }
      }
    }

    std::vector<ColVec> Acols(static_cast<size_t>(m_size));
    for (StorageIndex k = 0; k < m_size; ++k) {
      Acols[static_cast<size_t>(k)].reserve(colCounts[static_cast<size_t>(k)]);
    }

    for (StorageIndex outer = 0; outer < a.outerSize(); ++outer) {
      for (typename MatrixType::InnerIterator it(a, outer); it; ++it) {
        const StorageIndex old_r = static_cast<StorageIndex>(it.row());
        const StorageIndex old_c = static_cast<StorageIndex>(it.col());

        if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Lower)) == static_cast<int>(Lower) &&
                      (static_cast<int>(UpLo) & static_cast<int>(Upper)) != static_cast<int>(Upper)) {
          if (old_r < old_c) continue;
        } else if constexpr ((static_cast<int>(UpLo) & static_cast<int>(Upper)) == static_cast<int>(Upper) &&
                             (static_cast<int>(UpLo) & static_cast<int>(Lower)) != static_cast<int>(Lower)) {
          if (old_r > old_c) continue;
        }

        const StorageIndex nr = m_Pinv.indices()[old_r];
        const StorageIndex nc = m_Pinv.indices()[old_c];
        Scalar v = it.value();

        if (nr >= nc) {
          Acols[static_cast<size_t>(nc)].emplace_back(nr, v);
        } else {
          Acols[static_cast<size_t>(nr)].emplace_back(nc, numext::conj(v));
        }
      }
    }

    for (StorageIndex k = 0; k < m_size; ++k) {
      auto &col = Acols[static_cast<size_t>(k)];
      if (col.empty()) continue;
      std::sort(col.begin(), col.end(),
                [](const ColVec::value_type &a, const ColVec::value_type &b) {
                  return a.first < b.first;
                });
      size_t w = 1;
      for (size_t i = 1; i < col.size(); ++i) {
        if (col[static_cast<size_t>(i)].first ==
            col[static_cast<size_t>(w - 1)].first) {
          col[static_cast<size_t>(w - 1)].second +=
              col[static_cast<size_t>(i)].second;
        } else {
          col[static_cast<size_t>(w++)] = col[static_cast<size_t>(i)];
        }
      }
      col.resize(static_cast<size_t>(w));
    }

    return Acols;
  }

private:
  StorageIndex m_size = 0;
  CholMatrixType m_L;
  VectorType m_diag;
  PermutationType m_P;
  PermutationType m_Pinv;

  bool m_patternAnalyzed = false;
  bool m_factorizationIsOk = false;
  ComputationInfo m_info = Success;

  RealScalar m_regularization = RealScalar(1e-12);

  StorageIndex m_numPerturbedPivots = 0;
  RealScalar m_minAbsPivot = RealScalar(0);

  // Supernodal data (currently disabled — always uses sparse path).
  bool m_supernodal = false;
  std::vector<std::pair<int, int>> m_ranges; // supernode column ranges
  std::vector<int> m_etree;                  // elimination tree from supernode detection
};

} // namespace Eigen

#endif // EIGEN_SUPERSONAL_LDLT_H
