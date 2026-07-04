#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>

namespace solvers::ruiz {

using Scalar = double;
using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

struct Residuals {
    Scalar pri_inf = 0.0;
    Scalar dua_inf = 0.0;
};

struct Termination {
    Scalar eps_pri = 0.0;
    Scalar eps_dua = 0.0;
};

struct Scaling {
    // xbar = D^{-1} x, zbar = E z, ybar = c E^{-1} y
    Vec Dx;
    Vec Ez;
    Scalar c = 1.0;
    bool enabled = false;

    void reset(int n, int m) {
        Dx = Vec::Ones(n);
        Ez = Vec::Ones(m);
        c = 1.0;
        enabled = false;
    }
};

inline Scalar inf_norm(const Vec& v) {
    return (v.size() ? v.lpNorm<Eigen::Infinity>() : 0.0);
}

inline void scale_cols_inplace(SpMat& M, const Vec& s) {
    const int n = M.cols();
    for (int j = 0; j < n; ++j) {
        Scalar sj = (j < s.size() ? s[j] : 1.0);
        if (sj == 1.0) continue;
        for (SpMat::InnerIterator it(M, j); it; ++it) {
            it.valueRef() *= sj;
        }
    }
}

inline void scale_rows_inplace(SpMat& M, const Vec& s) {
    const int n = M.cols();
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(M, j); it; ++it) {
            int i = it.row();
            Scalar si = (i < s.size() ? s[i] : 1.0);
            if (si != 1.0) it.valueRef() *= si;
        }
    }
}

inline Scalar col_inf_norm(const SpMat& M, int j) {
    Scalar v = 0.0;
    for (SpMat::InnerIterator it(M, j); it; ++it)
        v = std::max(v, std::abs(it.value()));
    return v;
}

inline Scalar row_inf_norm(const SpMat& M, int i) {
    Scalar v = 0.0;
    for (int k = 0; k < M.outerSize(); ++k)
        for (SpMat::InnerIterator it(M, k); it; ++it)
            if (it.row() == i) v = std::max(v, std::abs(it.value()));
    return v;
}

inline void ruiz_equilibrate_modified(const SpMat& P, const Vec& q,
                                      const SpMat& A, const Vec& l,
                                      const Vec& u, int max_iter, Scalar tol,
                                      SpMat& Pbar, Vec& qbar, SpMat& Abar,
                                      Vec& lbar, Vec& ubar, Scaling& S) {
    const int n = int(P.rows());
    const int m = int(A.rows());

    S.reset(n, m);
    Pbar = P;
    Abar = A;
    qbar = q;
    lbar = l;
    ubar = u;

    Vec delta_x = Vec::Ones(n);
    Vec delta_z = Vec::Ones(m);

    for (int it = 0; it < max_iter; ++it) {
        Scalar max_dev = 0.0;

        for (int j = 0; j < n; ++j) {
            Scalar nP = col_inf_norm(Pbar, j);
            Scalar nA = col_inf_norm(Abar, j);
            Scalar s = std::max(nP, nA);
            Scalar dj = (s > 0) ? 1.0 / std::sqrt(s) : 1.0;
            delta_x[j] = dj;
            max_dev = std::max(max_dev, std::abs(1.0 - dj));
        }

        for (int i = 0; i < m; ++i) {
            Scalar nAr = row_inf_norm(Abar, i);
            Scalar di = (nAr > 0) ? 1.0 / std::sqrt(nAr) : 1.0;
            delta_z[i] = di;
            max_dev = std::max(max_dev, std::abs(1.0 - di));
        }

        scale_cols_inplace(Pbar, delta_x);
        scale_rows_inplace(Pbar, delta_x);
        scale_rows_inplace(Abar, delta_z);
        scale_cols_inplace(Abar, delta_x);
        qbar = delta_x.cwiseProduct(qbar);
        if (lbar.size()) lbar = delta_z.cwiseProduct(lbar);
        if (ubar.size()) ubar = delta_z.cwiseProduct(ubar);

        S.Dx = S.Dx.cwiseProduct(delta_x);
        S.Ez = S.Ez.cwiseProduct(delta_z);

        Scalar meanPcol = 0.0;
        if (n) {
            Scalar sum = 0.0;
            for (int j = 0; j < n; ++j) sum += col_inf_norm(Pbar, j);
            meanPcol = sum / n;
        }
        Scalar qinf = (qbar.size() ? qbar.lpNorm<Eigen::Infinity>() : 1.0);
        Scalar gamma = 1.0 / std::max(meanPcol, qinf);
        if (!std::isfinite(gamma) || gamma <= 0) gamma = 1.0;

        Pbar *= gamma;
        qbar *= gamma;
        S.c *= gamma;

        if (max_dev <= tol) break;
    }

    S.enabled = true;
}

inline void map_initial_to_scaled(const Scaling& S, Vec& x0, Vec& z0, Vec& y0) {
    if (!S.enabled) return;
    if (x0.size()) x0 = x0.cwiseQuotient(S.Dx);
    if (z0.size()) z0 = S.Ez.cwiseProduct(z0);
    if (y0.size()) y0 = S.c * y0.cwiseQuotient(S.Ez);
}

inline void map_solution_from_scaled(const Scaling& S, Vec& x, Vec& z, Vec& y) {
    if (!S.enabled) return;
    if (x.size()) x = S.Dx.cwiseProduct(x);
    if (z.size()) z = z.cwiseQuotient(S.Ez);
    if (y.size()) y = (1.0 / S.c) * S.Ez.cwiseProduct(y);
}

inline Termination compute_thresholds_unscaled(
    const SpMat& Pbar, const SpMat& Abar, const Vec& xbar, const Vec& zbar,
    const Vec& qbar, const Vec& ybar, const Scaling& S, Scalar eps_abs,
    Scalar eps_rel) {
    Vec Ax_bar = (Abar.rows() ? Abar * xbar : Vec());
    Scalar term_pri = 0.0;
    if (zbar.size()) {
        Scalar tA =
            (Ax_bar.size()
                 ? (Ax_bar.cwiseQuotient(S.Ez)).lpNorm<Eigen::Infinity>()
                 : 0.0);
        Scalar tz = (zbar.cwiseQuotient(S.Ez)).lpNorm<Eigen::Infinity>();
        term_pri = std::max(tA, tz);
    }

    Vec Px_bar = (Pbar.rows() ? Pbar * xbar : Vec());
    Vec ATy_bar = (Abar.rows() ? Abar.transpose() * ybar : Vec());
    Scalar t1 =
        (Px_bar.size() ? (Px_bar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                       : 0.0);
    Scalar t2 = (ATy_bar.size()
                     ? (ATy_bar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                     : 0.0);
    Scalar t3 =
        (qbar.size() ? (qbar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                     : 0.0);
    Scalar term_dua = (1.0 / S.c) * std::max({t1, t2, t3});

    return {eps_abs + eps_rel * term_pri, eps_abs + eps_rel * term_dua};
}

inline Residuals compute_residuals_unscaled(const SpMat& Pbar,
                                            const SpMat& Abar, const Vec& xbar,
                                            const Vec& zbar, const Vec& qbar,
                                            const Vec& ybar, const Scaling& S) {
    Vec r_p, r_d;
    if (Abar.rows()) {
        Vec tmp = Abar * xbar - zbar;
        r_p = tmp.cwiseQuotient(S.Ez);
    } else
        r_p.resize(0);

    Vec Px = (Pbar.rows() ? Pbar * xbar : Vec());
    Vec ATy = (Abar.rows() ? Abar.transpose() * ybar : Vec());
    Vec sum = Vec::Zero(xbar.size());
    if (Px.size()) sum += Px;
    if (qbar.size()) sum += qbar;
    if (ATy.size()) sum += ATy;
    r_d = (1.0 / S.c) * sum.cwiseQuotient(S.Dx);

    return {inf_norm(r_p), inf_norm(r_d)};
}

}  // namespace solvers::ruiz
