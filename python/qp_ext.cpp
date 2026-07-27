#include "python/bindings_utils.h"
#include "qp/osqp.h"
#include "qp/piqp.h"
#include "qp/proxqp.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace {

using DenseMatrix = solvers_py::DenseMatrix;
using SparseMatrix = solvers_py::SparseMatrix;
using Vector = solvers_py::Vector;

// ── Convert Python input to Eigen SparseMatrix ──────────────────────────────

inline SparseMatrix to_sparse_from_python(nb::handle input) {
    if (nb::isinstance<DenseMatrix>(input)) {
        return nb::cast<DenseMatrix>(input).sparseView();
    }
    nb::module_ np = nb::module_::import_("numpy");
    nb::module_ scipy_sparse = nb::module_::import_("scipy.sparse");
    nb::object dense_obj = np.attr("asarray")(
        scipy_sparse.attr("coo_matrix")(input).attr("toarray")());
    return nb::cast<DenseMatrix>(dense_obj).sparseView();
}

// ── Unified result ──────────────────────────────────────────────────────────

struct QPResult {
    std::string status;
    double obj_val = 0.0;
    Vector x;
    Vector y;
    Vector z;
    Vector slack;
    std::map<std::string, double> residuals;
    std::optional<Vector> x_polish;
};

nb::dict result_to_dict(const QPResult& r) {
    nb::dict out;
    out["status"] = r.status;
    out["obj_val"] = r.obj_val;
    out["x"] = r.x;
    out["y"] = r.y;
    out["z"] = r.z;
    out["slack"] = r.slack;
    nb::dict res;
    for (auto& [k, v] : r.residuals) {
        res[k.c_str()] = v;
    }
    out["residuals"] = res;
    if (r.x_polish) {
        out["x_polish"] = *r.x_polish;
    } else {
        out["x_polish"] = nb::none();
    }
    return out;
}

// ── Per-solver settings parsers ─────────────────────────────────────────────

void parse_osqp_settings(const nb::dict& d, sosqp::Settings& def) {
    for (auto [key_h, val] : d) {
        std::string k = nb::cast<std::string>(key_h);
        auto set = [&](auto* field, auto&& val2) { *field = nb::cast<std::decay_t<decltype(*field)>>(val2); };
        if (k == "sigma")          set(&def.sigma, val);
        else if (k == "alpha")    set(&def.alpha, val);
        else if (k == "rho0")     set(&def.rho0, val);
        else if (k == "rho")      set(&def.rho, val);
        else if (k == "rho_eq_scale") set(&def.rho_eq_scale, val);
        else if (k == "adaptive_rho") set(&def.adaptive_rho, val);
        else if (k == "eps_abs")  set(&def.eps_abs, val);
        else if (k == "eps_rel")  set(&def.eps_rel, val);
        else if (k == "eps_pinf") set(&def.eps_pinf, val);
        else if (k == "eps_dinf") set(&def.eps_dinf, val);
        else if (k == "max_iter") set(&def.max_iter, val);
        else if (k == "check_every") set(&def.check_every, val);
        else if (k == "diag_reg") set(&def.diag_reg, val);
        else if (k == "eq_tol")   set(&def.eq_tol, val);
        else if (k == "verbose")  set(&def.verbose, val);
        else if (k == "polish")   set(&def.polish, val);
        else if (k == "polish_delta") set(&def.polish_delta, val);
        else if (k == "polish_refine_steps") set(&def.polish_refine_steps, val);
        else if (k == "rho_min")  set(&def.rho_min, val);
        else if (k == "rho_max")  set(&def.rho_max, val);
        else if (k == "explode_refactor") set(&def.explode_refactor, val);
        else if (k == "max_refactor") set(&def.max_refactor, val);
        else if (k == "enable_ruiz") set(&def.enable_ruiz, val);
        else if (k == "ruiz_max_iter") set(&def.ruiz_max_iter, val);
        else if (k == "ruiz_tol") set(&def.ruiz_tol, val);
        else if (k == "check_unscaled") set(&def.check_unscaled, val);
    }
}

void parse_piqp_settings(const nb::dict& d, piqp::PIQPSettings& def) {
    for (auto [key_h, val] : d) {
        std::string k = nb::cast<std::string>(key_h);
        auto set = [&](auto* field, auto&& val2) { *field = nb::cast<std::decay_t<decltype(*field)>>(val2); };
        if (k == "eps_abs")      set(&def.eps_abs, val);
        else if (k == "eps_rel") set(&def.eps_rel, val);
        else if (k == "max_iter") set(&def.max_iter, val);
        else if (k == "rho_init") set(&def.rho_init, val);
        else if (k == "delta_init") set(&def.delta_init, val);
        else if (k == "rho_floor") set(&def.rho_floor, val);
        else if (k == "delta_floor") set(&def.delta_floor, val);
        else if (k == "tau")     set(&def.tau, val);
        else if (k == "reg_eps") set(&def.reg_eps, val);
        else if (k == "verbose") set(&def.verbose, val);
        else if (k == "min_slack") set(&def.min_slack, val);
        else if (k == "scale")   set(&def.scale, val);
        else if (k == "ruiz_iters") set(&def.ruiz_iters, val);
        else if (k == "scale_eps") set(&def.scale_eps, val);
        else if (k == "cost_scaling") set(&def.cost_scaling, val);
    }
}

void parse_proxqp_settings(const nb::dict& d, proxqp::Settings& def) {
    for (auto [key_h, val] : d) {
        std::string k = nb::cast<std::string>(key_h);
        auto set = [&](auto* field, auto&& val2) { *field = nb::cast<std::decay_t<decltype(*field)>>(val2); };
        if (k == "eps_abs")                      set(&def.eps_abs, val);
        else if (k == "eps_rel")                 set(&def.eps_rel, val);
        else if (k == "check_duality_gap")       set(&def.check_duality_gap, val);
        else if (k == "eps_duality_gap_abs")     set(&def.eps_duality_gap_abs, val);
        else if (k == "eps_duality_gap_rel")     set(&def.eps_duality_gap_rel, val);
        else if (k == "verbose")                 set(&def.verbose, val);
        else if (k == "default_rho")             set(&def.default_rho, val);
        else if (k == "default_mu_eq")           set(&def.default_mu_eq, val);
        else if (k == "default_mu_in")           set(&def.default_mu_in, val);
        else if (k == "compute_timings")         set(&def.compute_timings, val);
        else if (k == "max_iter")                set(&def.max_iter, val);
        else if (k == "max_iter_in")             set(&def.max_iter_in, val);
        else if (k == "initial_guess") {
            std::string gs = nb::cast<std::string>(val);
            if (gs == "NO_INITIAL_GUESS")            def.initial_guess = proxqp::NO_INITIAL_GUESS;
            else if (gs == "EQUALITY_CONSTRAINED_INITIAL_GUESS")
                def.initial_guess = proxqp::EQUALITY_CONSTRAINED_INITIAL_GUESS;
            else if (gs == "WARM_START_WITH_PREVIOUS_RESULT")
                def.initial_guess = proxqp::WARM_START_WITH_PREVIOUS_RESULT;
            else if (gs == "WARM_START")
                def.initial_guess = proxqp::WARM_START;
            else if (gs == "COLD_START_WITH_PREVIOUS_RESULT")
                def.initial_guess = proxqp::COLD_START_WITH_PREVIOUS_RESULT;
        }
        else if (k == "mu_min_eq")               set(&def.mu_min_eq, val);
        else if (k == "mu_min_in")               set(&def.mu_min_in, val);
        else if (k == "mu_update_factor")        set(&def.mu_update_factor, val);
        else if (k == "eps_primal_inf")          set(&def.eps_primal_inf, val);
        else if (k == "eps_dual_inf")            set(&def.eps_dual_inf, val);
        else if (k == "primal_infeasibility_solving")
            set(&def.primal_infeasibility_solving, val);
        else if (k == "nb_power_iteration")      set(&def.nb_power_iteration, val);
        else if (k == "power_iteration_accuracy")
            set(&def.power_iteration_accuracy, val);
        else if (k == "alpha_bcl")               set(&def.alpha_bcl, val);
        else if (k == "beta_bcl")                set(&def.beta_bcl, val);
        else if (k == "refactor_dual_feasibility_threshold")
            set(&def.refactor_dual_feasibility_threshold, val);
        else if (k == "refactor_rho_threshold")  set(&def.refactor_rho_threshold, val);
        else if (k == "nb_iterative_refinement") set(&def.nb_iterative_refinement, val);
        else if (k == "eps_refact")              set(&def.eps_refact, val);
        else if (k == "safe_guard")              set(&def.safe_guard, val);
        else if (k == "preconditioner_max_iter") set(&def.preconditioner_max_iter, val);
        else if (k == "preconditioner_accuracy") set(&def.preconditioner_accuracy, val);
        else if (k == "compute_preconditioner")  set(&def.compute_preconditioner, val);
        else if (k == "update_preconditioner")   set(&def.update_preconditioner, val);
        else if (k == "default_H_eigenvalue_estimate")
            set(&def.default_H_eigenvalue_estimate, val);
        else if (k == "rho_regularization_scaling")
            set(&def.rho_regularization_scaling, val);
    }
}

// ── OSQP dispatch ───────────────────────────────────────────────────────────

QPResult solve_osqp(
    const SparseMatrix& P,
    const Vector& q,
    const SparseMatrix& A,
    const Vector& l,
    const Vector& u,
    const nb::dict& settings_dict) {

    sosqp::Settings s = sosqp::Settings{};
    parse_osqp_settings(settings_dict, s);
    sosqp::sparse_osqp_solver solver(s);
    sosqp::Result raw = solver.solve(P, q, A, l, u);

    QPResult r;
    r.status = raw.status;
    r.obj_val = raw.obj_val;
    r.x = raw.x;
    r.y = raw.y;
    r.z = raw.z;
    r.residuals["pri_inf"] = raw.res.pri_inf;
    r.residuals["dua_inf"] = raw.res.dua_inf;
    if (raw.x_polish) {
        r.x_polish = *raw.x_polish;
    }
    return r;
}

// ── PIQP dispatch ───────────────────────────────────────────────────────────

QPResult solve_piqp(
    const SparseMatrix& P,
    const Vector& q,
    const SparseMatrix& A,
    const Vector& l,
    const Vector& u,
    const nb::dict& settings_dict) {

    piqp::PIQPSettings s = piqp::PIQPSettings{};
    parse_piqp_settings(settings_dict, s);

    // Convert l <= Ax <= u to PIQP form: Ax = b, Gx <= h
    // G = [A; -A], h = [u; -l]  =>  Ax <= u, -Ax <= -l  =>  l <= Ax <= u

    SparseMatrix G_sparse(A.rows() * 2, A.cols());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() * 2);
    for (int j = 0; j < A.outerSize(); ++j) {
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            double val = it.value();
            int row = it.row();
            int col = it.col();
            triplets.emplace_back(row, col, val);
            triplets.emplace_back(row + A.rows(), col, -val);
        }
    }
    G_sparse.setFromTriplets(triplets.begin(), triplets.end());

    // l <= Ax <= u  =>  no equality, G = [A; -A], h = [u; -l]
    Vector h = Vector(A.rows() * 2);
    h.head(A.rows()) = u;
    h.tail(A.rows()) = -l;

    piqp::piqp_solver solver(s);
    solver.setup(P, q, std::nullopt, std::nullopt,
                 std::optional<SparseMatrix>(G_sparse),
                 std::optional<Vector>(h));
    piqp::PIQPResult raw = solver.solve();

    QPResult r;
    r.status = raw.status;
    r.obj_val = raw.obj_val;
    r.x = raw.x;
    r.y = raw.y;
    r.z = raw.z;
    r.residuals["eq_inf"] = raw.residuals.eq_inf;
    r.residuals["ineq_inf"] = raw.residuals.ineq_inf;
    r.residuals["stat_inf"] = raw.residuals.stat_inf;
    r.residuals["gap"] = raw.residuals.gap;
    return r;
}

// ── ProxQP dispatch ─────────────────────────────────────────────────────────

QPResult solve_proxqp(
    const SparseMatrix& P,
    const Vector& q,
    const SparseMatrix& A,
    const Vector& l,
    const Vector& u,
    const nb::dict& settings_dict) {

    proxqp::Settings s = proxqp::Settings{};
    parse_proxqp_settings(settings_dict, s);

    // ProxQP form: min 0.5 x^T H x + g^T x,  Ax = b,  l <= Cx <= u
    // Map OSQP l <= Ax <= u: A_eq = A, b = u, C = A, l_in = l, u_in = u

    int n = P.rows();
    int n_eq = A.rows();
    int n_in = A.rows();

    // l <= Ax <= u  =>  no equality, C = A, l_in = l, u_in = u
    proxqp::QP qp(n, 0, A.rows(), s);
    qp.init(P, q, std::nullopt, std::nullopt,
            std::optional<SparseMatrix>(A),
            std::optional<Vector>(l), std::optional<Vector>(u));

    proxqp::Result raw = qp.solve();

    QPResult r;
    r.status = proxqp::status_string(raw.status);
    r.obj_val = raw.info.obj_val;
    r.x = raw.x;
    r.y = raw.y;
    r.z = raw.z;
    r.slack = raw.si;
    r.residuals["pri_res"] = raw.info.pri_res;
    r.residuals["dua_res"] = raw.info.dua_res;
    r.residuals["gap"] = raw.info.gap;
    return r;
}

// ── Dispatcher ──────────────────────────────────────────────────────────────

QPResult solve_qp(
    const std::string& solver,
    nb::handle P_in,
    nb::handle q_in,
    nb::handle A_in,
    nb::handle l_in,
    nb::handle u_in,
    const nb::dict& settings) {

    SparseMatrix P = to_sparse_from_python(P_in);
    Vector q = nb::cast<Vector>(q_in);
    SparseMatrix A = to_sparse_from_python(A_in);
    Vector l = nb::cast<Vector>(l_in);
    Vector u = nb::cast<Vector>(u_in);

    if (solver == "osqp") {
        return solve_osqp(P, q, A, l, u, settings);
    } else if (solver == "piqp") {
        return solve_piqp(P, q, A, l, u, settings);
    } else if (solver == "proxqp") {
        return solve_proxqp(P, q, A, l, u, settings);
    } else {
        throw std::invalid_argument(
            "Unknown QP solver '" + solver +
            "'. Available: osqp, piqp, proxqp");
    }
}

}  // namespace

NB_MODULE(qp, m) {
    m.doc() = "Unified QP solver — dispatch to osqp / piqp / proxqp by name";

    m.def(
        "solve",
        [](const std::string& solver, nb::handle P_in, nb::handle q_in,
           nb::handle A_in, nb::handle l_in, nb::handle u_in,
           const nb::dict& settings) {
            return result_to_dict(solve_qp(solver, P_in, q_in, A_in, l_in, u_in, settings));
        },
        nb::arg("solver"),
        nb::arg("P"),
        nb::arg("q"),
        nb::arg("A"),
        nb::arg("l"),
        nb::arg("u"),
        nb::arg("settings") = nb::dict(),
        R"pbdoc(
        Solve a QP:  min  0.5 * x^T P x + q^T x
                     s.t.  l <= A x <= u

        Parameters:
            solver: One of "osqp", "piqp", "proxqp".
            P:      Symmetric positive-semidefinite quadratic matrix (n x n).
                    Accepts dense numpy or scipy sparse.
            q:      Linear objective vector (n,).
            A:      Constraint matrix (m x n).
                    Accepts dense numpy or scipy sparse.
            l:      Lower bounds (m,).
            u:      Upper bounds (m,).
            settings: Dict of solver-specific options.

        Returns:
            Dict with keys: status, obj_val, x, y, z, slack, residuals, x_polish.
        )pbdoc");

    m.def(
        "available_solvers",
        []() -> std::vector<std::string> {
            return {"osqp", "piqp", "proxqp"};
        },
        R"pbdoc(Return names of all compiled QP solvers.)pbdoc");
}
