#pragma once

// PDLP — restarted adaptive PDHG for Linear Programming.
//
// C++ port of https://github.com/mmaaz-git/pdlp (PyTorch), which follows
// https://arxiv.org/abs/2106.04756 (PDLP) and https://arxiv.org/abs/2311.12180
// (cuPDLP). CPU backend lives here (header-only, Eigen sparse); the CUDA
// backend (src/pdlp/pdlp_cuda.cu) reuses the problem setup / rescaling below
// and mirrors the same iteration.
//
// Problem form:
//     minimize    c^T x
//     subject to  G x >= h   (m1 inequality rows)
//                 A x  = b   (m2 equality rows)
//                 l <= x <= u
//
// Duals: y[0:m1] >= 0 for G, y[m1:m] free for A.

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace solvers::pdlp {

using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix< double, Eigen::RowMajor, int >;  // CSR
using SpMatCsc = Eigen::SparseMatrix< double, Eigen::ColMajor, int >;

inline constexpr double kEpsZero = 1e-12;
inline constexpr int kTerminationCheckFrequency = 50;  // iters between checks
inline constexpr int kMaxInnerIters = 1000;            // max iters between restarts
inline constexpr double kBetaSufficient = 0.2;
inline constexpr double kBetaNecessary = 0.8;
inline constexpr double kBetaArtificial = 0.36;

struct Settings {
    double iteration_limit = 10000;
    double time_sec_limit = std::numeric_limits< double >::infinity();
    double primal_weight_update_smoothing = 0.5;
    int ruiz_iterations = 10;
    double pock_chambolle_alpha = 1.0;
    double eps_tol = 1e-4;
    double eps_primal_infeasible = 1e-8;
    double eps_dual_infeasible = 1e-8;
    bool verbose = false;
};

struct Result {
    Vec x;  // primal solution (n)
    Vec y;  // dual solution (m1 + m2)
    std::string status;  // optimal | primal_infeasible | dual_infeasible |
                         // iteration_limit | time_limit
    long iterations = 0;
    double solve_time_sec = 0.0;

    // populated for optimal / iteration_limit / time_limit
    double primal_obj = std::numeric_limits< double >::quiet_NaN();
    double dual_obj = std::numeric_limits< double >::quiet_NaN();
    double duality_gap = std::numeric_limits< double >::quiet_NaN();
    double relative_gap = std::numeric_limits< double >::quiet_NaN();
    double primal_residual = std::numeric_limits< double >::quiet_NaN();
    double dual_residual = std::numeric_limits< double >::quiet_NaN();
    double kkt_error_sq = std::numeric_limits< double >::quiet_NaN();

    // populated for primal_infeasible / dual_infeasible
    Vec ray;
    double certificate_quality = std::numeric_limits< double >::quiet_NaN();
    double dual_ray_obj = std::numeric_limits< double >::quiet_NaN();
    double primal_ray_obj = std::numeric_limits< double >::quiet_NaN();
    double max_primal_residual = std::numeric_limits< double >::quiet_NaN();
};

// ── Problem data after stacking K = [G; A] and rescaling ───────────────────

struct Data {
    int m1 = 0, m2 = 0, n = 0, m = 0;
    SpMat K, Kt;            // rescaled K and K^T (both CSR)
    Vec c, q, l, u;         // rescaled
    SpMat K_orig, Kt_orig;  // original problem (for termination checks)
    Vec c_orig, q_orig, l_orig, u_orig;
    Vec constraint_rescaling, variable_rescaling;  // x = x_scaled / var, y = y_scaled / con
};

namespace detail {

inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration< double >(clock::now().time_since_epoch()).count();
}

// row/col rescale vector from accumulated max or sum: sqrt where > 0, else 1
inline Vec rescale_from(const Vec& acc) {
    Vec r(acc.size());
    for (Eigen::Index i = 0; i < acc.size(); ++i) {
        r(i) = acc(i) > kEpsZero ? std::max(std::sqrt(acc(i)), kEpsZero) : 1.0;
    }
    return r;
}

// scale K in place: K(i,j) /= row(i) * col(j)
inline void scale_matrix(SpMat& K, const Vec& row, const Vec& col) {
    for (int i = 0; i < K.outerSize(); ++i) {
        for (SpMat::InnerIterator it(K, i); it; ++it) {
            it.valueRef() /= row(it.row()) * col(it.col());
        }
    }
}

inline void apply_rescaling(Data& d, const Vec& row_rescale, const Vec& col_rescale) {
    d.c = d.c.cwiseQuotient(col_rescale);
    d.l = d.l.cwiseProduct(col_rescale);
    d.u = d.u.cwiseProduct(col_rescale);
    d.q = d.q.cwiseQuotient(row_rescale);
    scale_matrix(d.K, row_rescale, col_rescale);
    d.constraint_rescaling = d.constraint_rescaling.cwiseProduct(row_rescale);
    d.variable_rescaling = d.variable_rescaling.cwiseProduct(col_rescale);
}

}  // namespace detail

// Stack K = [G; A], q = [h; b] and run Ruiz + Pock-Chambolle rescaling.
inline Data build_data(const Vec& c, const SpMatCsc& G, const Vec& h, const SpMatCsc& A,
                       const Vec& b, const Vec& l, const Vec& u, const Settings& s) {
    Data d;
    d.m1 = static_cast< int >(G.rows());
    d.m2 = static_cast< int >(A.rows());
    d.n = static_cast< int >(c.size());
    d.m = d.m1 + d.m2;

    std::vector< Eigen::Triplet< double > > trips;
    trips.reserve(G.nonZeros() + A.nonZeros());
    for (int j = 0; j < G.outerSize(); ++j) {
        for (SpMatCsc::InnerIterator it(G, j); it; ++it) {
            trips.emplace_back(static_cast< int >(it.row()), j, it.value());
        }
    }
    for (int j = 0; j < A.outerSize(); ++j) {
        for (SpMatCsc::InnerIterator it(A, j); it; ++it) {
            trips.emplace_back(d.m1 + static_cast< int >(it.row()), j, it.value());
        }
    }
    d.K.resize(d.m, d.n);
    d.K.setFromTriplets(trips.begin(), trips.end());

    d.q.resize(d.m);
    d.q.head(d.m1) = h;
    d.q.tail(d.m2) = b;
    d.c = c;
    d.l = l;
    d.u = u;

    d.K_orig = d.K;
    d.Kt_orig = SpMat(d.K.transpose());
    d.c_orig = c;
    d.q_orig = d.q;
    d.l_orig = l;
    d.u_orig = u;

    d.constraint_rescaling = Vec::Ones(d.m);
    d.variable_rescaling = Vec::Ones(d.n);

    // Ruiz rescaling (L-infinity equilibration)
    for (int iter = 0; iter < s.ruiz_iterations; ++iter) {
        Vec col_max = Vec::Zero(d.n);
        Vec row_max = Vec::Zero(d.m);
        for (int i = 0; i < d.K.outerSize(); ++i) {
            for (SpMat::InnerIterator it(d.K, i); it; ++it) {
                const double a = std::abs(it.value());
                row_max(it.row()) = std::max(row_max(it.row()), a);
                col_max(it.col()) = std::max(col_max(it.col()), a);
            }
        }
        col_max = col_max.cwiseMax(d.c.cwiseAbs());
        detail::apply_rescaling(d, detail::rescale_from(row_max), detail::rescale_from(col_max));
    }

    // Pock-Chambolle rescaling (operator norm <= 1)
    if (s.pock_chambolle_alpha > 0) {
        const double alpha = s.pock_chambolle_alpha;
        Vec col_sum = Vec::Zero(d.n);
        Vec row_sum = Vec::Zero(d.m);
        for (int i = 0; i < d.K.outerSize(); ++i) {
            for (SpMat::InnerIterator it(d.K, i); it; ++it) {
                const double a = std::abs(it.value());
                row_sum(it.row()) += std::pow(a, alpha);
                col_sum(it.col()) += std::pow(a, 2.0 - alpha);
            }
        }
        detail::apply_rescaling(d, detail::rescale_from(row_sum), detail::rescale_from(col_sum));
    }

    d.Kt = SpMat(d.K.transpose());
    return d;
}

namespace detail {

// Normal-cone component for box constraints at x; g = c - K^T y.
inline Vec lambda_for_box(const Vec& x, const Vec& g, const Vec& lower, const Vec& upper,
                          double eps_tol) {
    Vec lam = Vec::Zero(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        bool at_l = std::isfinite(lower(i)) && x(i) <= lower(i) + eps_tol;
        bool at_u = std::isfinite(upper(i)) && x(i) >= upper(i) - eps_tol;
        if (at_l && at_u) {  // numerically-tight box: pick the closer bound
            const bool closer_l = std::abs(x(i) - lower(i)) <= std::abs(upper(i) - x(i));
            at_l = closer_l;
            at_u = !closer_l;
        }
        if (at_l) {
            lam(i) = std::max(g(i), 0.0);
        } else if (at_u) {
            lam(i) = std::min(g(i), 0.0);
        }
    }
    return lam;
}

// sum of values[i] * multipliers[i] over finite values[i]
inline double sum_finite_products(const Vec& values, const Vec& multipliers) {
    double s = 0.0;
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (std::isfinite(values(i))) s += values(i) * multipliers(i);
    }
    return s;
}

// Dual objective q^T y + l^T lam+ - u^T lam- on the original problem.
inline double dual_objective(const Data& d, const Vec& x, const Vec& y, const Vec& KTy_orig,
                             double eps_tol) {
    const Vec g = d.c_orig - KTy_orig;
    const Vec lam = lambda_for_box(x, g, d.l_orig, d.u_orig, eps_tol);
    const double l_term = sum_finite_products(d.l_orig, lam.cwiseMax(0.0));
    const double u_term = sum_finite_products(d.u_orig, (-lam).cwiseMax(0.0));
    return d.q_orig.dot(y) + l_term - u_term;
}

// Squared KKT error, equation (5) of the PDLP paper, on the rescaled problem.
inline double kkt_error_sq(const Data& d, const Vec& x, const Vec& y, double w, const Vec& Kx,
                           const Vec& KTy, double eps_tol) {
    w = std::max(w, kEpsZero);

    double pri = 0.0;
    for (int i = 0; i < d.m1; ++i) {
        const double r = std::max(d.q(i) - Kx(i), 0.0);
        pri += r * r;
    }
    for (int i = d.m1; i < d.m; ++i) {
        const double r = Kx(i) - d.q(i);
        pri += r * r;
    }
    const double term1 = w * w * pri;

    const Vec g = d.c - KTy;
    const Vec lam = lambda_for_box(x, g, d.l, d.u, eps_tol);
    const double term2 = (g - lam).squaredNorm() / (w * w);

    const double l_term = sum_finite_products(d.l, lam.cwiseMax(0.0));
    const double u_term = sum_finite_products(d.u, (-lam).cwiseMax(0.0));
    const double scalar = d.q.dot(y) + l_term - u_term - d.c.dot(x);

    return term1 + term2 + scalar * scalar;
}

struct TerminationOutcome {
    std::string status;  // "" if continuing
    double primal_obj = 0.0;
    double dual_obj = 0.0;
    // infeasibility certificate data (when status set accordingly)
    Vec ray;
    double certificate_quality = 0.0;
    double dual_ray_obj = 0.0;
    double primal_ray_obj = 0.0;
    double dual_residual = 0.0;
    double max_primal_residual = 0.0;
};

// Termination check on the original (unscaled) problem.
inline TerminationOutcome termination_criteria(const Data& d, const Vec& x_unscaled,
                                               const Vec& y_unscaled, const Vec& Kx_orig,
                                               const Vec& KTy_orig, const Settings& s) {
    TerminationOutcome out;
    out.dual_obj = dual_objective(d, x_unscaled, y_unscaled, KTy_orig, s.eps_tol);
    out.primal_obj = d.c_orig.dot(x_unscaled);

    // primal infeasibility: Farkas certificate via dual ray
    const double dual_norm_inf = y_unscaled.size() ? y_unscaled.lpNorm< Eigen::Infinity >() : 0.0;
    if (dual_norm_inf > kEpsZero) {
        const Vec y_ray = y_unscaled / dual_norm_inf;
        const double dual_ray_obj = d.q_orig.dot(y_ray);
        if (dual_ray_obj > 0) {
            const double dual_residual =
                (KTy_orig / dual_norm_inf).lpNorm< Eigen::Infinity >();
            const double relative_infeas = dual_residual / dual_ray_obj;
            if (relative_infeas < s.eps_primal_infeasible) {
                out.status = "primal_infeasible";
                out.ray = y_ray;
                out.certificate_quality = relative_infeas;
                out.dual_ray_obj = dual_ray_obj;
                out.dual_residual = dual_residual;
                return out;
            }
        }
    }

    // dual infeasibility: primal unbounded via primal ray
    const double primal_norm_inf = x_unscaled.size() ? x_unscaled.lpNorm< Eigen::Infinity >() : 0.0;
    if (primal_norm_inf > kEpsZero) {
        const Vec x_ray = x_unscaled / primal_norm_inf;
        const double primal_ray_obj = d.c_orig.dot(x_ray);
        if (primal_ray_obj < 0) {
            const Vec Kx_ray = Kx_orig / primal_norm_inf;
            double max_primal_residual = 0.0;
            for (int i = 0; i < d.m1; ++i) {
                max_primal_residual = std::max(max_primal_residual, std::max(-Kx_ray(i), 0.0));
            }
            for (int i = d.m1; i < d.m; ++i) {
                max_primal_residual = std::max(max_primal_residual, std::abs(Kx_ray(i)));
            }
            const double relative_infeas = max_primal_residual / (-primal_ray_obj);
            if (relative_infeas < s.eps_dual_infeasible) {
                out.status = "dual_infeasible";
                out.ray = x_ray;
                out.certificate_quality = relative_infeas;
                out.primal_ray_obj = primal_ray_obj;
                out.max_primal_residual = max_primal_residual;
                return out;
            }
        }
    }

    // (1) relative duality gap
    const bool gap_ok = std::abs(out.dual_obj - out.primal_obj) <=
                        s.eps_tol * (1.0 + std::abs(out.dual_obj) + std::abs(out.primal_obj));

    // (2) primal feasibility
    double feas_sq = 0.0;
    for (int i = 0; i < d.m1; ++i) {
        const double r = std::max(d.q_orig(i) - Kx_orig(i), 0.0);
        feas_sq += r * r;
    }
    for (int i = d.m1; i < d.m; ++i) {
        const double r = d.q_orig(i) - Kx_orig(i);
        feas_sq += r * r;
    }
    const bool feas_ok = std::sqrt(feas_sq) <= s.eps_tol * (1.0 + d.q_orig.norm());

    // (3) stationarity (dual feasibility)
    const Vec g_orig = d.c_orig - KTy_orig;
    const Vec lam = lambda_for_box(x_unscaled, g_orig, d.l_orig, d.u_orig, s.eps_tol);
    const bool stat_ok = (g_orig - lam).norm() <= s.eps_tol * (1.0 + d.c_orig.norm());

    if (gap_ok && feas_ok && stat_ok) out.status = "optimal";
    return out;
}

// Fill status-dependent Result fields from the final iterates. Shared by the
// CPU and CUDA backends. res.status/x/y/iterations/solve_time_sec are set by
// the caller; x_scaled/y_scaled are the final iterates on the rescaled problem
// (used only for the kkt_error_sq statistic).
inline void finalize_result(const Data& d, const Settings& settings, Result& res,
                            const Vec& x_scaled, const Vec& y_scaled, double w,
                            const TerminationOutcome& term) {
    const std::string& status = res.status;
    if (status == "primal_infeasible" || status == "dual_infeasible") {
        res.ray = term.ray;
        res.certificate_quality = term.certificate_quality;
        res.primal_obj = term.primal_obj;
        res.dual_obj = term.dual_obj;
        if (status == "primal_infeasible") {
            res.dual_ray_obj = term.dual_ray_obj;
            res.dual_residual = term.dual_residual;
        } else {
            res.primal_ray_obj = term.primal_ray_obj;
            res.max_primal_residual = term.max_primal_residual;
        }
    } else {
        // optimal / iteration_limit / time_limit: final quality metrics on the
        // returned (unscaled) iterate
        const Vec Kx_orig = d.K_orig * res.x;
        const Vec KTy_orig = d.Kt_orig * res.y;
        res.primal_obj = d.c_orig.dot(res.x);
        res.dual_obj = dual_objective(d, res.x, res.y, KTy_orig, settings.eps_tol);
        {
            const Vec Kx = d.K * x_scaled, KTy = d.Kt * y_scaled;
            res.kkt_error_sq =
                kkt_error_sq(d, x_scaled, y_scaled, w, Kx, KTy, settings.eps_tol);
        }
        double feas_sq = 0.0;
        for (int i = 0; i < d.m1; ++i) {
            const double r = std::max(d.q_orig(i) - Kx_orig(i), 0.0);
            feas_sq += r * r;
        }
        for (int i = d.m1; i < d.m; ++i) {
            const double r = d.q_orig(i) - Kx_orig(i);
            feas_sq += r * r;
        }
        res.primal_residual = std::sqrt(feas_sq);
        const Vec g_orig = d.c_orig - KTy_orig;
        const Vec lam = lambda_for_box(res.x, g_orig, d.l_orig, d.u_orig, settings.eps_tol);
        res.dual_residual = (g_orig - lam).norm();
        res.duality_gap = std::abs(res.primal_obj - res.dual_obj);
        res.relative_gap =
            res.duality_gap / (1.0 + std::abs(res.primal_obj) + std::abs(res.dual_obj));
    }

    if (settings.verbose) {
        std::printf("\n  Status: %s after %ld iterations in %.3fs\n", status.c_str(),
                    res.iterations, res.solve_time_sec);
        if (status == "optimal" || status == "iteration_limit" || status == "time_limit") {
            std::printf("  Primal objective: %.6e\n  Dual objective: %.6e\n  Duality gap: %.6e\n",
                        res.primal_obj, res.dual_obj, res.duality_gap);
        }
    }
}

}  // namespace detail

// ── CPU solver ──────────────────────────────────────────────────────────────

inline Result solve(const Vec& c, const SpMatCsc& G, const Vec& h, const SpMatCsc& A,
                    const Vec& b, const Vec& l, const Vec& u, const Settings& settings = {}) {
    const double start_time = detail::now_sec();
    const int m1 = static_cast< int >(G.rows());
    const int m2 = static_cast< int >(A.rows());
    const int n = static_cast< int >(c.size());
    const int m = m1 + m2;
    const auto elapsed = [&] { return detail::now_sec() - start_time; };

    if (settings.verbose) {
        std::printf("\nPDLP Solver\n  Problem: %d inequalities, %d equalities, %d variables\n",
                    m1, m2, n);
    }

    Result res;

    // trivial: no variables — feasible iff h <= 0 and b = 0
    if (n == 0) {
        bool feasible = (m1 == 0 || (h.array() <= kEpsZero).all()) &&
                        (m2 == 0 || (b.cwiseAbs().array() <= kEpsZero).all());
        res.x = Vec::Zero(0);
        res.y = Vec::Zero(m);
        res.iterations = 0;
        if (feasible) {
            res.status = "optimal";
            res.primal_obj = res.dual_obj = 0.0;
            res.duality_gap = res.relative_gap = 0.0;
            res.primal_residual = res.dual_residual = res.kkt_error_sq = 0.0;
        } else {
            Vec y_ray(m);
            for (int i = 0; i < m1; ++i) y_ray(i) = h(i) > kEpsZero ? 1.0 : 0.0;
            for (int i = 0; i < m2; ++i) y_ray(m1 + i) = std::abs(b(i)) > kEpsZero ? 1.0 : 0.0;
            y_ray /= std::max(y_ray.lpNorm< Eigen::Infinity >(), kEpsZero);
            res.status = "primal_infeasible";
            res.ray = y_ray;
            Vec q(m);
            q.head(m1) = h;
            q.tail(m2) = b;
            res.dual_ray_obj = q.dot(y_ray);
            res.dual_residual = 0.0;
            res.certificate_quality = 0.0;
        }
        res.solve_time_sec = elapsed();
        return res;
    }

    // trivial: no constraints — go to the bound in the objective's direction
    if (m == 0) {
        Vec x_sol(n);
        for (int i = 0; i < n; ++i) {
            if (c(i) < -kEpsZero) {
                x_sol(i) = u(i);
            } else if (c(i) > kEpsZero) {
                x_sol(i) = l(i);
            } else {
                x_sol(i) = std::min(std::max(0.0, l(i)), u(i));
            }
        }
        int unbounded_idx = -1;
        for (int i = 0; i < n; ++i) {
            if ((c(i) < -kEpsZero && std::isinf(u(i))) || (c(i) > kEpsZero && std::isinf(l(i)))) {
                unbounded_idx = i;
                break;
            }
        }
        res.y = Vec::Zero(0);
        res.iterations = 0;
        if (unbounded_idx >= 0) {
            Vec x_ray = Vec::Zero(n);
            x_ray(unbounded_idx) = c(unbounded_idx) < 0 ? 1.0 : -1.0;
            res.status = "dual_infeasible";
            res.x = x_sol;
            res.ray = x_ray;
            res.primal_ray_obj = c.dot(x_ray);
            res.max_primal_residual = 0.0;
            res.certificate_quality = 0.0;
        } else {
            res.status = "optimal";
            res.x = x_sol;
            res.primal_obj = res.dual_obj = c.dot(x_sol);
            res.duality_gap = res.relative_gap = 0.0;
            res.primal_residual = res.dual_residual = res.kkt_error_sq = 0.0;
        }
        res.solve_time_sec = elapsed();
        return res;
    }

    Data d = build_data(c, G, h, A, b, l, u, settings);

    if (settings.verbose) {
        std::printf("  Rescaling: Ruiz iters=%d, Pock-Chambolle alpha=%g\n",
                    settings.ruiz_iterations, settings.pock_chambolle_alpha);
    }

    const auto proj_X = [&](Vec& x) { x = x.cwiseMax(d.l).cwiseMin(d.u); };
    const auto proj_Y = [&](Vec& y) { y.head(d.m1) = y.head(d.m1).cwiseMax(0.0); };

    // initialization
    Vec x = Vec::Zero(n);
    proj_X(x);
    Vec y = Vec::Zero(m);

    // step size 1/||K||_inf (max row abs-sum)
    Vec row_sums = Vec::Zero(m);
    for (int i = 0; i < d.K.outerSize(); ++i) {
        for (SpMat::InnerIterator it(d.K, i); it; ++it) {
            row_sums(it.row()) += std::abs(it.value());
        }
    }
    double eta_hat = 1.0 / std::max(row_sums.maxCoeff(), kEpsZero);

    const double c_norm = d.c.norm(), q_norm = d.q.norm();
    double w = (c_norm > kEpsZero && q_norm > kEpsZero) ? c_norm / q_norm : 1.0;

    Vec x_prev = x, y_prev = y;
    Vec x_c = x, y_c = y;
    Vec x_unscaled_last = x.cwiseQuotient(d.variable_rescaling);
    Vec y_unscaled_last = y.cwiseQuotient(d.constraint_rescaling);

    long n_iterations = 0;
    std::string status;
    detail::TerminationOutcome term;

    // one adaptive PDHG step; updates x, y in place, returns eta actually used
    const auto adaptive_step = [&](double& eta_hat_io, long k) -> double {
        double eta = std::max(eta_hat_io, kEpsZero);
        const double kp1 = static_cast< double >(k + 1);
        const double fac1 = k == 0 ? 1.0 : 1.0 - std::pow(kp1, -0.3);
        const double fac2 = 1.0 + std::pow(kp1, -0.6);

        const Vec KTy = d.Kt * y;
        const Vec Kx = d.K * x;
        const Vec grad = d.c - KTy;

        while (true) {
            Vec x_p = x - (eta / w) * grad;
            proj_X(x_p);
            const Vec Kxp = d.K * x_p;
            Vec y_p = y + (eta * w) * (d.q - (2.0 * Kxp - Kx));
            proj_Y(y_p);

            const Vec dx = x_p - x;
            const Vec dy = y_p - y;

            const double num = w * dx.squaredNorm() + dy.squaredNorm() / w;
            const double denom = 2.0 * std::abs(dy.dot(Kxp - Kx));
            const double bar_eta =
                denom <= kEpsZero ? std::numeric_limits< double >::infinity() : num / denom;

            const double eta_p = std::max(std::min(fac1 * bar_eta, fac2 * eta), kEpsZero);

            if (eta <= bar_eta) {
                x = x_p;
                y = y_p;
                eta_hat_io = eta_p;
                return eta;
            }
            eta = eta_p;
        }
    };

    while (true) {
        // KKT of the restart point with the current primal weight
        double kkt_last_restart;
        {
            const Vec Kx = d.K * x, KTy = d.Kt * y;
            kkt_last_restart = detail::kkt_error_sq(d, x, y, w, Kx, KTy, settings.eps_tol);
        }

        x_unscaled_last = x.cwiseQuotient(d.variable_rescaling);
        y_unscaled_last = y.cwiseQuotient(d.constraint_rescaling);

        double eta_sum = 0.0;
        Vec x_bar = x, y_bar = y;
        double kkt_c_prev = kkt_last_restart;
        Vec x_c_new = x, y_c_new = y;
        bool inner_broke = false;

        for (int t = 0; t < kMaxInnerIters; ++t) {
            const double eta_used = adaptive_step(eta_hat, n_iterations);

            // step-size-weighted average of iterates
            eta_sum += eta_used;
            const double alpha = eta_used / eta_sum;
            x_bar += alpha * (x - x_bar);
            y_bar += alpha * (y - y_bar);

            ++n_iterations;

            if (static_cast< double >(n_iterations) >= settings.iteration_limit) {
                status = "iteration_limit";
                x_unscaled_last = x.cwiseQuotient(d.variable_rescaling);
                y_unscaled_last = y.cwiseQuotient(d.constraint_rescaling);
                inner_broke = true;
                break;
            }
            if (elapsed() >= settings.time_sec_limit) {
                status = "time_limit";
                x_unscaled_last = x.cwiseQuotient(d.variable_rescaling);
                y_unscaled_last = y.cwiseQuotient(d.constraint_rescaling);
                inner_broke = true;
                break;
            }

            // termination + restart checks: first 10 iters, then every frequency
            if (n_iterations <= 10 || n_iterations % kTerminationCheckFrequency == 0) {
                const Vec Kx = d.K * x, KTy = d.Kt * y;
                const Vec Kx_bar = d.K * x_bar, KTy_bar = d.Kt * y_bar;
                const Vec x_unscaled = x.cwiseQuotient(d.variable_rescaling);
                const Vec y_unscaled = y.cwiseQuotient(d.constraint_rescaling);
                const Vec Kx_orig = d.K_orig * x_unscaled;
                const Vec KTy_orig = d.Kt_orig * y_unscaled;

                // restart candidate: current vs averaged, lower KKT wins
                const double kkt_current =
                    detail::kkt_error_sq(d, x, y, w, Kx, KTy, settings.eps_tol);
                const double kkt_averaged =
                    detail::kkt_error_sq(d, x_bar, y_bar, w, Kx_bar, KTy_bar, settings.eps_tol);
                double kkt_c_new;
                if (kkt_current < kkt_averaged) {
                    x_c_new = x;
                    y_c_new = y;
                    kkt_c_new = kkt_current;
                } else {
                    x_c_new = x_bar;
                    y_c_new = y_bar;
                    kkt_c_new = kkt_averaged;
                }

                term = detail::termination_criteria(d, x_unscaled, y_unscaled, Kx_orig, KTy_orig,
                                                    settings);
                status = term.status;
                if (settings.verbose) {
                    std::printf(
                        "  Iter %5ld: primal_obj = %+.6e, dual_obj = %+.6e, gap = %.3e, KKT = "
                        "%.3e\n",
                        n_iterations, term.primal_obj, term.dual_obj,
                        std::abs(term.primal_obj - term.dual_obj), std::sqrt(kkt_current));
                }
                if (!status.empty()) {
                    // ignore infeasibility detections before iteration 10 (early false positives)
                    if (n_iterations < 10 && status != "optimal") {
                        status.clear();
                    } else {
                        x_unscaled_last = x_unscaled;
                        y_unscaled_last = y_unscaled;
                        inner_broke = true;
                        break;
                    }
                }

                const bool cond_i = kkt_c_new <= kBetaSufficient * kBetaSufficient * kkt_last_restart;
                const bool cond_ii = kkt_c_new <= kBetaNecessary * kBetaNecessary * kkt_last_restart &&
                                     t > 0 && kkt_c_new > kkt_c_prev;
                const bool cond_iii =
                    t >= kBetaArtificial * static_cast< double >(n_iterations);

                if (cond_i || cond_ii || cond_iii) {
                    x_c = x_c_new;
                    y_c = y_c_new;
                    inner_broke = true;
                    break;
                }
                kkt_c_prev = kkt_c_new;
            }
        }
        if (!inner_broke) {  // inner loop exhausted without break
            x_c = x_c_new;
            y_c = y_c_new;
        }

        if (!status.empty()) break;

        // restart from candidate
        x = x_c;
        y = y_c;

        // primal weight update (exponential moving average of ||dy||/||dx||)
        {
            const double dx = (x - x_prev).norm();
            const double dy = (y - y_prev).norm();
            if (dx > kEpsZero && dy > kEpsZero) {
                const double theta = settings.primal_weight_update_smoothing;
                w = std::pow(dy / dx, theta) * std::pow(w, 1.0 - theta);
            }
        }
        x_prev = x;
        y_prev = y;
    }

    res.status = status;
    res.x = x_unscaled_last;
    res.y = y_unscaled_last;
    res.iterations = n_iterations;
    res.solve_time_sec = elapsed();
    detail::finalize_result(d, settings, res, x, y, w, term);
    return res;
}

}  // namespace solvers::pdlp
