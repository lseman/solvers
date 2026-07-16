#include "python/bindings_utils.h"
#include "qp/osqp.h"

#include <nanobind/stl/optional.h>

namespace nb = nanobind;

namespace {

sosqp::SpMat to_spmat(const Eigen::Ref<const solvers_py::DenseMatrix>& value) {
    return value.sparseView();
}

nb::dict residuals_to_dict(const sosqp::Residuals& residuals) {
    nb::dict out;
    out["pri_inf"] = residuals.pri_inf;
    out["dua_inf"] = residuals.dua_inf;
    return out;
}

nb::dict result_to_dict(const sosqp::Result& result) {
    nb::dict out;
    out["status"] = result.status;
    out["iters"] = result.iters;
    out["obj_val"] = result.obj_val;
    out["x"] = result.x;
    out["z"] = result.z;
    out["y"] = result.y;
    out["residuals"] = residuals_to_dict(result.res);
    out["primal_infeasible"] = result.primal_infeasible;
    out["dual_infeasible"] = result.dual_infeasible;
    out["y_cert"] = result.y_cert;
    out["x_cert"] = result.x_cert;
    out["x_polish"] = result.x_polish ? nb::cast(*result.x_polish) : nb::none();
    return out;
}

}  // namespace

NB_MODULE(osqp, m) {
    m.doc() = "nanobind wrappers for the sparse OSQP-style solver";

    nb::class_<sosqp::Settings>(m, "Settings")
        .def(nb::init<>())
        .def_rw("sigma", &sosqp::Settings::sigma)
        .def_rw("alpha", &sosqp::Settings::alpha)
        .def_rw("rho0", &sosqp::Settings::rho0)
        .def_rw("rho", &sosqp::Settings::rho)
        .def_rw("rho_eq_scale", &sosqp::Settings::rho_eq_scale)
        .def_rw("adaptive_rho", &sosqp::Settings::adaptive_rho)
        .def_rw("eps_abs", &sosqp::Settings::eps_abs)
        .def_rw("eps_rel", &sosqp::Settings::eps_rel)
        .def_rw("eps_pinf", &sosqp::Settings::eps_pinf)
        .def_rw("eps_dinf", &sosqp::Settings::eps_dinf)
        .def_rw("max_iter", &sosqp::Settings::max_iter)
        .def_rw("check_every", &sosqp::Settings::check_every)
        .def_rw("diag_reg", &sosqp::Settings::diag_reg)
        .def_rw("eq_tol", &sosqp::Settings::eq_tol)
        .def_rw("verbose", &sosqp::Settings::verbose)
        .def_rw("polish", &sosqp::Settings::polish)
        .def_rw("polish_delta", &sosqp::Settings::polish_delta)
        .def_rw("polish_refine_steps", &sosqp::Settings::polish_refine_steps)
        .def_rw("rho_min", &sosqp::Settings::rho_min)
        .def_rw("rho_max", &sosqp::Settings::rho_max)
        .def_rw("explode_refactor", &sosqp::Settings::explode_refactor)
        .def_rw("max_refactor", &sosqp::Settings::max_refactor)
        .def_rw("enable_ruiz", &sosqp::Settings::enable_ruiz)
        .def_rw("ruiz_max_iter", &sosqp::Settings::ruiz_max_iter)
        .def_rw("ruiz_tol", &sosqp::Settings::ruiz_tol)
        .def_rw("check_unscaled", &sosqp::Settings::check_unscaled);

    nb::class_<sosqp::sparse_osqp_solver>(m, "sparse_osqp_solver")
        .def(nb::init<sosqp::Settings>(), nb::arg("settings") = sosqp::Settings{})
        .def(
            "solve",
            [](sosqp::sparse_osqp_solver& solver,
               const Eigen::Ref<const solvers_py::DenseMatrix>& P,
               const Eigen::Ref<const sosqp::Vec>& q,
               const Eigen::Ref<const solvers_py::DenseMatrix>& A,
               const Eigen::Ref<const sosqp::Vec>& l,
               const Eigen::Ref<const sosqp::Vec>& u) {
                return result_to_dict(solver.solve(to_spmat(P), q, to_spmat(A), l, u));
            },
            nb::arg("P"), nb::arg("q"), nb::arg("A"), nb::arg("l"), nb::arg("u"));

    m.def(
        "solve",
        [](const Eigen::Ref<const solvers_py::DenseMatrix>& P,
           const Eigen::Ref<const sosqp::Vec>& q,
           const Eigen::Ref<const solvers_py::DenseMatrix>& A,
           const Eigen::Ref<const sosqp::Vec>& l,
           const Eigen::Ref<const sosqp::Vec>& u,
           const sosqp::Settings& settings) {
            sosqp::sparse_osqp_solver solver(settings);
            return result_to_dict(solver.solve(to_spmat(P), q, to_spmat(A), l, u));
        },
        nb::arg("P"), nb::arg("q"), nb::arg("A"), nb::arg("l"), nb::arg("u"),
        nb::arg("settings") = sosqp::Settings{});
}
