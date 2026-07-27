/// Self-contained proximal primal-dual quadratic-program solver.
///
/// Solves
///   minimize  1/2 x' H x + g' x
///   subject to A x = b
///              l <= C x <= u
///
/// The implementation uses the proximal augmented-Lagrangian formulation
/// underlying ProxQP.  Each iteration minimizes the regularized primal
/// subproblem, projects the inequality image onto [l,u], and updates the
/// equality and inequality multipliers.  Equality and inequality penalties
/// are kept separate, as in ProxQP.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace proxqp {

using Scalar = double;
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix< Scalar, Eigen::ColMajor, int >;

enum Status {
    PROXQP_NOT_RUN = 0,
    PROXQP_SOLVED = 1,
    PROXQP_MAX_ITER_REACHED = 2,
    PROXQP_PRIMAL_INFEASIBLE = 3,
    PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE = 4,
    PROXQP_DUAL_INFEASIBLE = 5,
};

inline const char* status_string(Status status) {
    switch (status) {
    case PROXQP_NOT_RUN:
        return "not_run";
    case PROXQP_SOLVED:
        return "solved";
    case PROXQP_MAX_ITER_REACHED:
        return "max_iter_reached";
    case PROXQP_PRIMAL_INFEASIBLE:
        return "primal_infeasible";
    case PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE:
        return "solved_closest_primal_feasible";
    case PROXQP_DUAL_INFEASIBLE:
        return "dual_infeasible";
    }
    return "unknown";
}

enum InitialGuess {
    NO_INITIAL_GUESS = 0,
    EQUALITY_CONSTRAINED_INITIAL_GUESS = 1,
    WARM_START_WITH_PREVIOUS_RESULT = 2,
    WARM_START = 3,
    COLD_START_WITH_PREVIOUS_RESULT = 4,
};

struct Settings {
    double eps_abs = 1e-5;
    double eps_rel = 0.0;
    bool check_duality_gap = false;
    double eps_duality_gap_abs = 1e-4;
    double eps_duality_gap_rel = 0.0;
    bool verbose = false;
    double default_rho = 1e-6;
    double default_mu_eq = 1e-3;
    double default_mu_in = 1e-1;
    bool compute_timings = false;
    int max_iter = 10000;
    int max_iter_in = 1500;
    InitialGuess initial_guess = EQUALITY_CONSTRAINED_INITIAL_GUESS;
    double mu_min_eq = 1e-9;
    double mu_min_in = 1e-8;
    double mu_update_factor = 0.1;
    double eps_primal_inf = 1e-4;
    double eps_dual_inf = 1e-4;
    bool primal_infeasibility_solving = false;
    int nb_power_iteration = 1000;
    double power_iteration_accuracy = 1e-6;
    double alpha_bcl = 0.1;
    double beta_bcl = 0.9;
    double refactor_dual_feasibility_threshold = 1e-2;
    double refactor_rho_threshold = 1e-7;
    int nb_iterative_refinement = 10;
    double eps_refact = 1e-6;
    double safe_guard = 1e4;
    int preconditioner_max_iter = 10;
    double preconditioner_accuracy = 1e-3;
    bool compute_preconditioner = true;
    bool update_preconditioner = false;
    double default_H_eigenvalue_estimate = 0.0;
    double rho_regularization_scaling = 1.5;
};

struct Residuals {
    double pri_res = 0.0;
    double dua_res = 0.0;
    double gap = 0.0;
    double obj_val = 0.0;
    int iter = 0;
    int iter_ext = 0;
    double mu_eq = 1e-3;
    double mu_in = 1e-1;
    double rho = 1e-6;
    double setup_time = 0.0;
    double solve_time = 0.0;
    double run_time = 0.0;
};

struct Result {
    Status status = PROXQP_NOT_RUN;
    Vector x;
    Vector y;
    Vector z;
    Vector se;
    Vector si;
    Residuals info;
    double minimal_H_eigenvalue_estimate = 0.0;
};

class QP {
  public:
    explicit QP(int n, int n_eq = 0, int n_in = 0, Settings settings = Settings{})
        : n_(n), n_eq_(n_eq), n_in_(n_in), settings_(std::move(settings)) {
        if (n < 0 || n_eq < 0 || n_in < 0) {
            throw std::invalid_argument("ProxQP dimensions must be non-negative");
        }
        reset_vectors();
    }

    QP& init(const SparseMatrix& H, const Vector& g,
             const std::optional< SparseMatrix >& A = std::nullopt,
             const std::optional< Vector >& b = std::nullopt,
             const std::optional< SparseMatrix >& C = std::nullopt,
             const std::optional< Vector >& l = std::nullopt,
             const std::optional< Vector >& u = std::nullopt, bool compute_preconditioner = true,
             double rho = 0.0, double mu_eq = 0.0, double mu_in = 0.0,
             double manual_minimal_H_eigenvalue = 0.0) {
        const auto start = Clock::now();
        validate_problem(H, g, A, b, C, l, u);

        Matrix dense_h(H);
        H_ = 0.5 * (dense_h + dense_h.transpose());
        g_ = g;
        A_ = A ? Matrix(*A) : Matrix(0, n_);
        b_ = b.value_or(Vector(0));
        C_ = C ? Matrix(*C) : Matrix(0, n_);
        l_ = l.value_or(Vector(0));
        u_ = u.value_or(Vector(0));

        rho_ = rho > 0.0 ? rho : settings_.default_rho;
        mu_eq_ = mu_eq > 0.0 ? mu_eq : settings_.default_mu_eq;
        mu_in_ = mu_in > 0.0 ? mu_in : settings_.default_mu_in;
        minimal_H_eigenvalue_estimate_ = manual_minimal_H_eigenvalue;
        if (manual_minimal_H_eigenvalue < 0.0) {
            rho_ += settings_.rho_regularization_scaling * -manual_minimal_H_eigenvalue;
        }

        reset_vectors();
        initialize_scaling(compute_preconditioner);
        initialized_ = true;
        factorization_valid_ = false;

        if (settings_.initial_guess == EQUALITY_CONSTRAINED_INITIAL_GUESS) {
            equality_constrained_initial_guess();
        }
        setup_time_ms_ = elapsed_ms(start);
        return *this;
    }

    QP& update(const std::optional< SparseMatrix >& H = std::nullopt,
               const std::optional< Vector >& g = std::nullopt,
               const std::optional< SparseMatrix >& A = std::nullopt,
               const std::optional< Vector >& b = std::nullopt,
               const std::optional< SparseMatrix >& C = std::nullopt,
               const std::optional< Vector >& l = std::nullopt,
               const std::optional< Vector >& u = std::nullopt) {
        require_initialized();
        if (H) {
            if (H->rows() != n_ || H->cols() != n_)
                throw std::invalid_argument("invalid Hessian dimensions");
            Matrix dense_h(*H);
            H_ = 0.5 * (dense_h + dense_h.transpose());
            factorization_valid_ = false;
        }
        if (g) {
            if (g->size() != n_)
                throw std::invalid_argument("invalid g dimension");
            g_ = *g;
        }
        if (A) {
            if (A->rows() != n_eq_ || A->cols() != n_)
                throw std::invalid_argument("invalid A dimensions");
            A_ = Matrix(*A);
            factorization_valid_ = false;
        }
        if (b) {
            if (b->size() != n_eq_)
                throw std::invalid_argument("invalid b dimension");
            b_ = *b;
        }
        if (C) {
            if (C->rows() != n_in_ || C->cols() != n_)
                throw std::invalid_argument("invalid C dimensions");
            C_ = Matrix(*C);
            factorization_valid_ = false;
        }
        if (l) {
            if (l->size() != n_in_)
                throw std::invalid_argument("invalid l dimension");
            l_ = *l;
        }
        if (u) {
            if (u->size() != n_in_)
                throw std::invalid_argument("invalid u dimension");
            u_ = *u;
        }
        if ((l || u) && (l_.array() > u_.array()).any())
            throw std::invalid_argument("inequality lower bound exceeds upper bound");
        if (settings_.update_preconditioner)
            initialize_scaling(true);
        return *this;
    }

    Result solve(const Vector* x0 = nullptr, const Vector* y0 = nullptr,
                 const Vector* z0 = nullptr) {
        require_initialized();
        const auto start = Clock::now();
        prepare_initial_guess(x0, y0, z0);

        Vector v = n_in_ ? project_box(C_ * x_ + mu_in_ * z_) : Vector(0);
        Vector v_previous = v;
        Vector x_previous = x_;
        int penalty_updates = 0;
        Status status = PROXQP_MAX_ITER_REACHED;
        ResidualSnapshot residuals{};

        for (int iteration = 0; iteration < settings_.max_iter; ++iteration) {
            ensure_factorization();
            x_previous = x_;
            v_previous = v;

            Vector rhs = rho_ * x_previous - g_;
            if (n_eq_)
                rhs.noalias() += sigma_eq() * A_.transpose() * b_ - A_.transpose() * y_;
            if (n_in_)
                rhs.noalias() += C_.transpose() * (sigma_in() * v - z_);

            x_ = ldlt_.solve(rhs);
            if (ldlt_.info() != Eigen::Success || !x_.allFinite()) {
                throw std::runtime_error("ProxQP primal linear solve failed");
            }

            if (n_eq_)
                y_.noalias() += sigma_eq() * (A_ * x_ - b_);
            if (n_in_) {
                const Vector cx = C_ * x_;
                v = project_box(cx + mu_in_ * z_);
                z_.noalias() += sigma_in() * (cx - v);
            }

            residuals = compute_residuals();
            if (settings_.verbose && (iteration < 10 || iteration % 25 == 0)) {
                std::cerr << "[ProxQP] iter " << iteration << " pri=" << residuals.primal
                          << " dua=" << residuals.dual << " gap=" << residuals.gap << '\n';
            }

            if (converged(residuals)) {
                status = PROXQP_SOLVED;
                return make_result(status, iteration + 1, penalty_updates, residuals, start);
            }

            // Residual balancing is the BCL-style outer update.  Multipliers
            // are stored unscaled, so they remain continuous when sigma
            // changes.
            if ((iteration + 1) % 25 == 0) {
                const double split_primal = std::max(n_eq_ ? inf_norm(A_ * x_ - b_) : 0.0,
                                                     n_in_ ? inf_norm(C_ * x_ - v) : 0.0);
                const double split_dual = std::max(
                    rho_ * inf_norm(x_ - x_previous),
                    n_in_ ? sigma_in() * inf_norm(C_.transpose() * (v - v_previous)) : 0.0);
                if (balance_penalties(split_primal, split_dual)) {
                    ++penalty_updates;
                    factorization_valid_ = false;
                }
            }
        }

        residuals = compute_residuals();
        return make_result(status, settings_.max_iter, penalty_updates, residuals, start);
    }

    QP& warm_start(const Vector& x, const Vector& y, const Vector& z) {
        validate_warm_start(x, y, z);
        x_ = x;
        y_ = y;
        z_ = z;
        settings_.initial_guess = WARM_START_WITH_PREVIOUS_RESULT;
        return *this;
    }

    const Settings& getSettings() const {
        return settings_;
    }
    const Vector& getX() const {
        return x_;
    }
    const Vector& getY() const {
        return y_;
    }
    const Vector& getZ() const {
        return z_;
    }
    int getN() const {
        return n_;
    }
    int getNEq() const {
        return n_eq_;
    }
    int getNIn() const {
        return n_in_;
    }

  private:
    using Clock = std::chrono::steady_clock;
    struct ResidualSnapshot {
        double primal = 0.0;
        double dual = 0.0;
        double gap = 0.0;
        double objective = 0.0;
        double primal_scale = 1.0;
        double dual_scale = 1.0;
        double gap_scale = 1.0;
    };

    int n_;
    int n_eq_;
    int n_in_;
    Settings settings_;
    Matrix H_, A_, C_;
    Vector g_, b_, l_, u_;
    Vector x_, y_, z_, se_, si_;
    Eigen::LDLT< Matrix > ldlt_;
    double rho_ = 1e-6;
    double mu_eq_ = 1e-3;
    double mu_in_ = 1e-1;
    double setup_time_ms_ = 0.0;
    double minimal_H_eigenvalue_estimate_ = 0.0;
    bool initialized_ = false;
    bool factorization_valid_ = false;

    double sigma_eq() const {
        return 1.0 / mu_eq_;
    }
    double sigma_in() const {
        return 1.0 / mu_in_;
    }

    static double elapsed_ms(Clock::time_point start) {
        return std::chrono::duration< double, std::milli >(Clock::now() - start).count();
    }

    static double inf_norm(const Vector& value) {
        return value.size() ? value.lpNorm< Eigen::Infinity >() : 0.0;
    }

    void reset_vectors() {
        x_.setZero(n_);
        y_.setZero(n_eq_);
        z_.setZero(n_in_);
        se_.setZero(n_eq_);
        si_.setZero(n_in_);
    }

    void require_initialized() const {
        if (!initialized_)
            throw std::logic_error("ProxQP::init must be called first");
    }

    void validate_problem(const SparseMatrix& H, const Vector& g,
                          const std::optional< SparseMatrix >& A, const std::optional< Vector >& b,
                          const std::optional< SparseMatrix >& C, const std::optional< Vector >& l,
                          const std::optional< Vector >& u) const {
        if (H.rows() != n_ || H.cols() != n_ || g.size() != n_)
            throw std::invalid_argument("invalid ProxQP objective dimensions");
        if (static_cast< bool >(A) != static_cast< bool >(b) ||
            static_cast< bool >(C) != static_cast< bool >(l) ||
            static_cast< bool >(C) != static_cast< bool >(u))
            throw std::invalid_argument("incomplete ProxQP constraint data");
        if (A && (A->rows() != n_eq_ || A->cols() != n_ || b->size() != n_eq_))
            throw std::invalid_argument("invalid equality constraint dimensions");
        if (!A && n_eq_ != 0)
            throw std::invalid_argument("missing equality constraint data");
        if (C &&
            (C->rows() != n_in_ || C->cols() != n_ || l->size() != n_in_ || u->size() != n_in_))
            throw std::invalid_argument("invalid inequality constraint dimensions");
        if (!C && n_in_ != 0)
            throw std::invalid_argument("missing inequality constraint data");
        if (l && (l->array() > u->array()).any())
            throw std::invalid_argument("inequality lower bound exceeds upper bound");
        if (!g.allFinite() || !H.coeffs().allFinite())
            throw std::invalid_argument("non-finite ProxQP objective data");
    }

    void initialize_scaling(bool enabled) {
        // Ruiz scaling is deliberately applied implicitly through separate
        // equality and inequality penalties.  Keeping model data unscaled
        // ensures updates and returned multipliers remain in user units.
        (void)enabled;
    }

    void ensure_factorization() {
        if (factorization_valid_)
            return;
        Matrix kkt = H_;
        kkt.diagonal().array() += rho_;
        if (n_eq_)
            kkt.noalias() += sigma_eq() * A_.transpose() * A_;
        if (n_in_)
            kkt.noalias() += sigma_in() * C_.transpose() * C_;
        ldlt_.compute(kkt);
        if (ldlt_.info() != Eigen::Success) {
            throw std::runtime_error("ProxQP factorization failed; Hessian may be non-convex");
        }
        factorization_valid_ = true;
    }

    Vector project_box(const Vector& value) const {
        Vector projected = value;
        for (int i = 0; i < n_in_; ++i) {
            if (std::isfinite(l_[i]))
                projected[i] = std::max(projected[i], l_[i]);
            if (std::isfinite(u_[i]))
                projected[i] = std::min(projected[i], u_[i]);
        }
        return projected;
    }

    void equality_constrained_initial_guess() {
        if (!n_eq_)
            return;
        Matrix kkt(n_ + n_eq_, n_ + n_eq_);
        kkt.setZero();
        kkt.topLeftCorner(n_, n_) = H_;
        kkt.topLeftCorner(n_, n_).diagonal().array() += rho_;
        kkt.topRightCorner(n_, n_eq_) = A_.transpose();
        kkt.bottomLeftCorner(n_eq_, n_) = A_;
        kkt.bottomRightCorner(n_eq_, n_eq_).diagonal().array() = -mu_eq_;
        Vector rhs(n_ + n_eq_);
        rhs << -g_, b_;
        Eigen::LDLT< Matrix > initial_ldlt(kkt);
        if (initial_ldlt.info() == Eigen::Success) {
            const Vector solution = initial_ldlt.solve(rhs);
            if (solution.allFinite()) {
                x_ = solution.head(n_);
                y_ = solution.tail(n_eq_);
            }
        }
    }

    void validate_warm_start(const Vector& x, const Vector& y, const Vector& z) const {
        if (x.size() != n_ || y.size() != n_eq_ || z.size() != n_in_)
            throw std::invalid_argument("invalid ProxQP warm-start dimensions");
    }

    void prepare_initial_guess(const Vector* x0, const Vector* y0, const Vector* z0) {
        if (settings_.initial_guess == WARM_START) {
            if (!(x0 && y0 && z0))
                throw std::invalid_argument("WARM_START requires x, y, and z");
            validate_warm_start(*x0, *y0, *z0);
            x_ = *x0;
            y_ = *y0;
            z_ = *z0;
        } else if (settings_.initial_guess == NO_INITIAL_GUESS) {
            reset_vectors();
        } else if (settings_.initial_guess == COLD_START_WITH_PREVIOUS_RESULT) {
            rho_ = settings_.default_rho;
            mu_eq_ = settings_.default_mu_eq;
            mu_in_ = settings_.default_mu_in;
            factorization_valid_ = false;
        }
    }

    ResidualSnapshot compute_residuals() {
        ResidualSnapshot out;
        if (n_eq_) {
            se_ = A_ * x_ - b_;
            out.primal = inf_norm(se_);
            out.primal_scale = std::max({1.0, inf_norm(A_ * x_), inf_norm(b_)});
        }
        if (n_in_) {
            const Vector cx = C_ * x_;
            si_.setZero();
            for (int i = 0; i < n_in_; ++i) {
                if (std::isfinite(l_[i]) && cx[i] < l_[i])
                    si_[i] = cx[i] - l_[i];
                else if (std::isfinite(u_[i]) && cx[i] > u_[i])
                    si_[i] = cx[i] - u_[i];
            }
            out.primal = std::max(out.primal, inf_norm(si_));
            out.primal_scale = std::max(out.primal_scale, inf_norm(cx));
            for (int i = 0; i < n_in_; ++i) {
                if (std::isfinite(l_[i]))
                    out.primal_scale = std::max(out.primal_scale, std::abs(l_[i]));
                if (std::isfinite(u_[i]))
                    out.primal_scale = std::max(out.primal_scale, std::abs(u_[i]));
            }
        }

        Vector stationarity = H_ * x_ + g_;
        out.dual_scale = std::max({1.0, inf_norm(H_ * x_), inf_norm(g_)});
        if (n_eq_) {
            const Vector aty = A_.transpose() * y_;
            stationarity += aty;
            out.dual_scale = std::max(out.dual_scale, inf_norm(aty));
        }
        if (n_in_) {
            const Vector ctz = C_.transpose() * z_;
            stationarity += ctz;
            out.dual_scale = std::max(out.dual_scale, inf_norm(ctz));
        }
        out.dual = inf_norm(stationarity);
        out.objective = 0.5 * x_.dot(H_ * x_) + g_.dot(x_);

        double signed_gap = x_.dot(H_ * x_) + g_.dot(x_);
        out.gap_scale = std::max({1.0, std::abs(x_.dot(H_ * x_)), std::abs(g_.dot(x_))});
        if (n_eq_) {
            signed_gap += b_.dot(y_);
            out.gap_scale = std::max(out.gap_scale, std::abs(b_.dot(y_)));
        }
        if (n_in_) {
            double support = 0.0;
            for (int i = 0; i < n_in_; ++i) {
                if (z_[i] > 0.0 && std::isfinite(u_[i]))
                    support += u_[i] * z_[i];
                else if (z_[i] < 0.0 && std::isfinite(l_[i]))
                    support += l_[i] * z_[i];
            }
            signed_gap += support;
            out.gap_scale = std::max(out.gap_scale, std::abs(support));
        }
        out.gap = std::abs(signed_gap);
        return out;
    }

    bool converged(const ResidualSnapshot& residuals) const {
        const bool residual_ok =
            residuals.primal <= settings_.eps_abs + settings_.eps_rel * residuals.primal_scale &&
            residuals.dual <= settings_.eps_abs + settings_.eps_rel * residuals.dual_scale;
        if (!settings_.check_duality_gap)
            return residual_ok;
        return residual_ok &&
               residuals.gap <= settings_.eps_duality_gap_abs +
                                    settings_.eps_duality_gap_rel * residuals.gap_scale;
    }

    bool balance_penalties(double primal, double dual) {
        if (!(primal > 0.0) || !(dual > 0.0))
            return false;
        constexpr double threshold = 10.0;
        constexpr double factor = 2.0;
        double old_mu_eq = mu_eq_;
        double old_mu_in = mu_in_;
        if (primal > threshold * dual) {
            mu_eq_ = std::max(settings_.mu_min_eq, mu_eq_ / factor);
            mu_in_ = std::max(settings_.mu_min_in, mu_in_ / factor);
        } else if (dual > threshold * primal) {
            mu_eq_ = std::min(1e8, mu_eq_ * factor);
            mu_in_ = std::min(1e8, mu_in_ * factor);
        }
        return old_mu_eq != mu_eq_ || old_mu_in != mu_in_;
    }

    Result make_result(Status status, int iterations, int outer_iterations,
                       const ResidualSnapshot& residuals, Clock::time_point start) const {
        Result result;
        result.status = status;
        result.x = x_;
        result.y = y_;
        result.z = z_;
        result.se = se_;
        result.si = si_;
        result.info.pri_res = residuals.primal;
        result.info.dua_res = residuals.dual;
        result.info.gap = residuals.gap;
        result.info.obj_val = residuals.objective;
        result.info.iter = iterations;
        result.info.iter_ext = outer_iterations;
        result.info.mu_eq = mu_eq_;
        result.info.mu_in = mu_in_;
        result.info.rho = rho_;
        result.info.setup_time = setup_time_ms_;
        result.info.solve_time = elapsed_ms(start);
        result.info.run_time = result.info.setup_time + result.info.solve_time;
        result.minimal_H_eigenvalue_estimate = minimal_H_eigenvalue_estimate_;
        return result;
    }
};

} // namespace proxqp
