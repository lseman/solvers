#include "python/bindings/common/bindings_utils.h"
#include "qp/proxqp.h"

#include <nanobind/stl/optional.h>

namespace nb = nanobind;

namespace {

proxqp::SparseMatrix to_spmat(const Eigen::Ref< const solvers_py::DenseMatrix >& value) {
    return value.sparseView();
}

nb::dict residuals_to_dict(const proxqp::Residuals& r) {
    nb::dict out;
    out["pri_res"] = r.pri_res;
    out["dua_res"] = r.dua_res;
    out["gap"] = r.gap;
    out["obj_val"] = r.obj_val;
    out["iter"] = r.iter;
    out["iter_ext"] = r.iter_ext;
    out["mu_eq"] = r.mu_eq;
    out["mu_in"] = r.mu_in;
    out["rho"] = r.rho;
    out["setup_time"] = r.setup_time;
    out["solve_time"] = r.solve_time;
    out["run_time"] = r.run_time;
    return out;
}

nb::dict result_to_dict(const proxqp::Result& result) {
    nb::dict out;
    out["status"] = proxqp::status_string(result.status);
    out["x"] = result.x;
    out["y"] = result.y;
    out["z"] = result.z;
    out["se"] = result.se;
    out["si"] = result.si;
    out["info"] = residuals_to_dict(result.info);
    out["minimal_H_eigenvalue_estimate"] = result.minimal_H_eigenvalue_estimate;
    return out;
}

} // namespace

NB_MODULE(proxqp, m) {
    m.doc() = "nanobind wrappers for the ProxQP proximal QP solver";

    // Status enum
    nb::enum_< proxqp::Status >(m, "Status")
        .value("NOT_RUN", proxqp::PROXQP_NOT_RUN)
        .value("SOLVED", proxqp::PROXQP_SOLVED)
        .value("MAX_ITER_REACHED", proxqp::PROXQP_MAX_ITER_REACHED)
        .value("PRIMAL_INFEASIBLE", proxqp::PROXQP_PRIMAL_INFEASIBLE)
        .value("SOLVED_CLOSEST_PRIMAL_FEASIBLE", proxqp::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE)
        .value("DUAL_INFEASIBLE", proxqp::PROXQP_DUAL_INFEASIBLE)
        .export_values();

    // InitialGuess enum
    nb::enum_< proxqp::InitialGuess >(m, "InitialGuess")
        .value("NO_INITIAL_GUESS", proxqp::NO_INITIAL_GUESS)
        .value("EQUALITY_CONSTRAINED_INITIAL_GUESS", proxqp::EQUALITY_CONSTRAINED_INITIAL_GUESS)
        .value("WARM_START_WITH_PREVIOUS_RESULT", proxqp::WARM_START_WITH_PREVIOUS_RESULT)
        .value("WARM_START", proxqp::WARM_START)
        .value("COLD_START_WITH_PREVIOUS_RESULT", proxqp::COLD_START_WITH_PREVIOUS_RESULT)
        .export_values();

    // Settings
    nb::class_< proxqp::Settings >(m, "Settings")
        .def(nb::init<>())
        .def_rw("eps_abs", &proxqp::Settings::eps_abs)
        .def_rw("eps_rel", &proxqp::Settings::eps_rel)
        .def_rw("check_duality_gap", &proxqp::Settings::check_duality_gap)
        .def_rw("eps_duality_gap_abs", &proxqp::Settings::eps_duality_gap_abs)
        .def_rw("eps_duality_gap_rel", &proxqp::Settings::eps_duality_gap_rel)
        .def_rw("verbose", &proxqp::Settings::verbose)
        .def_rw("default_rho", &proxqp::Settings::default_rho)
        .def_rw("default_mu_eq", &proxqp::Settings::default_mu_eq)
        .def_rw("default_mu_in", &proxqp::Settings::default_mu_in)
        .def_rw("compute_timings", &proxqp::Settings::compute_timings)
        .def_rw("max_iter", &proxqp::Settings::max_iter)
        .def_rw("max_iter_in", &proxqp::Settings::max_iter_in)
        .def_rw("initial_guess", &proxqp::Settings::initial_guess)
        .def_rw("mu_min_eq", &proxqp::Settings::mu_min_eq)
        .def_rw("mu_min_in", &proxqp::Settings::mu_min_in)
        .def_rw("mu_update_factor", &proxqp::Settings::mu_update_factor)
        .def_rw("eps_primal_inf", &proxqp::Settings::eps_primal_inf)
        .def_rw("eps_dual_inf", &proxqp::Settings::eps_dual_inf)
        .def_rw("primal_infeasibility_solving", &proxqp::Settings::primal_infeasibility_solving)
        .def_rw("nb_power_iteration", &proxqp::Settings::nb_power_iteration)
        .def_rw("power_iteration_accuracy", &proxqp::Settings::power_iteration_accuracy)
        .def_rw("alpha_bcl", &proxqp::Settings::alpha_bcl)
        .def_rw("beta_bcl", &proxqp::Settings::beta_bcl)
        .def_rw("refactor_dual_feasibility_threshold",
                &proxqp::Settings::refactor_dual_feasibility_threshold)
        .def_rw("refactor_rho_threshold", &proxqp::Settings::refactor_rho_threshold)
        .def_rw("nb_iterative_refinement", &proxqp::Settings::nb_iterative_refinement)
        .def_rw("eps_refact", &proxqp::Settings::eps_refact)
        .def_rw("safe_guard", &proxqp::Settings::safe_guard)
        .def_rw("preconditioner_max_iter", &proxqp::Settings::preconditioner_max_iter)
        .def_rw("preconditioner_accuracy", &proxqp::Settings::preconditioner_accuracy)
        .def_rw("compute_preconditioner", &proxqp::Settings::compute_preconditioner)
        .def_rw("update_preconditioner", &proxqp::Settings::update_preconditioner)
        .def_rw("default_H_eigenvalue_estimate", &proxqp::Settings::default_H_eigenvalue_estimate)
        .def_rw("rho_regularization_scaling", &proxqp::Settings::rho_regularization_scaling);

    // QP Solver class
    nb::class_< proxqp::QP >(m, "QP")
        .def(nb::init< int, int, int, proxqp::Settings >(), nb::arg("n"), nb::arg("n_eq") = 0,
             nb::arg("n_in") = 0, nb::arg("settings") = proxqp::Settings{})
        .def(
            "init",
            [](proxqp::QP& qp, const Eigen::Ref< const solvers_py::DenseMatrix >& H,
               const Eigen::Ref< const proxqp::Vector >& g,
               const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& A,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& b,
               const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& C,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& l,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& u,
               bool compute_preconditioner = true, double rho = 0.0, double mu_eq = 0.0,
               double mu_in = 0.0, double manual_minimal_H_eigenvalue = 0.0) -> proxqp::QP& {
                auto A_sparse =
                    A ? std::optional< proxqp::SparseMatrix >((*A).sparseView()) : std::nullopt;
                auto C_sparse =
                    C ? std::optional< proxqp::SparseMatrix >((*C).sparseView()) : std::nullopt;
                auto b_vec = b ? std::optional< proxqp::Vector >(*b) : std::nullopt;
                auto l_vec = l ? std::optional< proxqp::Vector >(*l) : std::nullopt;
                auto u_vec = u ? std::optional< proxqp::Vector >(*u) : std::nullopt;
                return qp.init(to_spmat(H), g, A_sparse, b_vec, C_sparse, l_vec, u_vec,
                               compute_preconditioner, rho, mu_eq, mu_in,
                               manual_minimal_H_eigenvalue);
            },
            nb::arg("H"), nb::arg("g"), nb::arg("A") = nb::none(), nb::arg("b") = nb::none(),
            nb::arg("C") = nb::none(), nb::arg("l") = nb::none(), nb::arg("u") = nb::none(),
            nb::arg("compute_preconditioner") = true, nb::arg("rho") = 0.0, nb::arg("mu_eq") = 0.0,
            nb::arg("mu_in") = 0.0, nb::arg("manual_minimal_H_eigenvalue") = 0.0,
            nb::rv_policy::reference_internal)
        .def(
            "update",
            [](proxqp::QP& qp,
               const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& H,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& g,
               const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& A,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& b,
               const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& C,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& l,
               const std::optional< Eigen::Ref< const proxqp::Vector > >& u) -> proxqp::QP& {
                auto H_sparse =
                    H ? std::optional< proxqp::SparseMatrix >((*H).sparseView()) : std::nullopt;
                auto g_vec = g ? std::optional< proxqp::Vector >(*g) : std::nullopt;
                auto A_sparse =
                    A ? std::optional< proxqp::SparseMatrix >((*A).sparseView()) : std::nullopt;
                auto b_vec = b ? std::optional< proxqp::Vector >(*b) : std::nullopt;
                auto C_sparse =
                    C ? std::optional< proxqp::SparseMatrix >((*C).sparseView()) : std::nullopt;
                auto l_vec = l ? std::optional< proxqp::Vector >(*l) : std::nullopt;
                auto u_vec = u ? std::optional< proxqp::Vector >(*u) : std::nullopt;
                return qp.update(H_sparse, g_vec, A_sparse, b_vec, C_sparse, l_vec, u_vec);
            },
            nb::arg("H") = nb::none(), nb::arg("g") = nb::none(), nb::arg("A") = nb::none(),
            nb::arg("b") = nb::none(), nb::arg("C") = nb::none(), nb::arg("l") = nb::none(),
            nb::arg("u") = nb::none(), nb::rv_policy::reference_internal)
        .def("solve", [](proxqp::QP& qp) { return result_to_dict(qp.solve()); })
        .def(
            "warm_start",
            [](proxqp::QP& qp, const Eigen::Ref< const proxqp::Vector >& x,
               const Eigen::Ref< const proxqp::Vector >& y,
               const Eigen::Ref< const proxqp::Vector >& z) -> proxqp::QP& {
                return qp.warm_start(x, y, z);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::rv_policy::reference_internal)
        .def("get_n", &proxqp::QP::getN)
        .def("get_n_eq", &proxqp::QP::getNEq)
        .def("get_n_in", &proxqp::QP::getNIn);

    // Convenience free function
    m.def(
        "solve",
        [](const Eigen::Ref< const solvers_py::DenseMatrix >& H,
           const Eigen::Ref< const proxqp::Vector >& g,
           const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& A,
           const std::optional< Eigen::Ref< const proxqp::Vector > >& b,
           const std::optional< Eigen::Ref< const solvers_py::DenseMatrix > >& C,
           const std::optional< Eigen::Ref< const proxqp::Vector > >& l,
           const std::optional< Eigen::Ref< const proxqp::Vector > >& u,
           const proxqp::Settings& settings) {
            int n = static_cast< int >(H.rows());
            int n_eq = A ? static_cast< int >((*A).rows()) : 0;
            int n_in = C ? static_cast< int >((*C).rows()) : 0;
            proxqp::QP qp(n, n_eq, n_in, settings);
            auto A_sparse =
                A ? std::optional< proxqp::SparseMatrix >((*A).sparseView()) : std::nullopt;
            auto C_sparse =
                C ? std::optional< proxqp::SparseMatrix >((*C).sparseView()) : std::nullopt;
            auto b_vec = b ? std::optional< proxqp::Vector >(*b) : std::nullopt;
            auto l_vec = l ? std::optional< proxqp::Vector >(*l) : std::nullopt;
            auto u_vec = u ? std::optional< proxqp::Vector >(*u) : std::nullopt;
            qp.init(to_spmat(H), g, A_sparse, b_vec, C_sparse, l_vec, u_vec);
            return result_to_dict(qp.solve());
        },
        nb::arg("H"), nb::arg("g"), nb::arg("A") = nb::none(), nb::arg("b") = nb::none(),
        nb::arg("C") = nb::none(), nb::arg("l") = nb::none(), nb::arg("u") = nb::none(),
        nb::arg("settings") = proxqp::Settings{});
}
