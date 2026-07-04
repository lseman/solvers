#include "python/bindings_utils.h"
#include "qp/piqp.h"

#include <nanobind/stl/optional.h>

namespace nb = nanobind;

namespace {

piqp::SparseMatrix to_spmat(const Eigen::Ref<const solvers_py::DenseMatrix>& value) {
    return value.sparseView();
}

nb::dict residuals_to_dict(const piqp::PIQPResiduals& residuals) {
    nb::dict out;
    out["eq_inf"] = residuals.eq_inf;
    out["ineq_inf"] = residuals.ineq_inf;
    out["stat_inf"] = residuals.stat_inf;
    out["gap"] = residuals.gap;
    return out;
}

nb::dict result_to_dict(const piqp::PIQPResult& result) {
    nb::dict out;
    out["status"] = result.status;
    out["iterations"] = result.iterations;
    out["x"] = result.x;
    out["s"] = result.s;
    out["y"] = result.y;
    out["z"] = result.z;
    out["obj_val"] = result.obj_val;
    out["residuals"] = residuals_to_dict(result.residuals);
    return out;
}

}  // namespace

NB_MODULE(piqp, m) {
    m.doc() = "nanobind wrappers for the sparse PIQP solver";

    nb::class_<piqp::PIQPSettings>(m, "PIQPSettings")
        .def(nb::init<>())
        .def_rw("eps_abs", &piqp::PIQPSettings::eps_abs)
        .def_rw("eps_rel", &piqp::PIQPSettings::eps_rel)
        .def_rw("max_iter", &piqp::PIQPSettings::max_iter)
        .def_rw("rho_init", &piqp::PIQPSettings::rho_init)
        .def_rw("delta_init", &piqp::PIQPSettings::delta_init)
        .def_rw("rho_floor", &piqp::PIQPSettings::rho_floor)
        .def_rw("delta_floor", &piqp::PIQPSettings::delta_floor)
        .def_rw("tau", &piqp::PIQPSettings::tau)
        .def_rw("reg_eps", &piqp::PIQPSettings::reg_eps)
        .def_rw("verbose", &piqp::PIQPSettings::verbose)
        .def_rw("min_slack", &piqp::PIQPSettings::min_slack)
        .def_rw("scale", &piqp::PIQPSettings::scale)
        .def_rw("ruiz_iters", &piqp::PIQPSettings::ruiz_iters)
        .def_rw("scale_eps", &piqp::PIQPSettings::scale_eps)
        .def_rw("cost_scaling", &piqp::PIQPSettings::cost_scaling);

    nb::class_<piqp::PIQPSolver>(m, "PIQPSolver")
        .def(nb::init<piqp::PIQPSettings>(),
             nb::arg("settings") = piqp::PIQPSettings{})
        .def(
            "setup",
            [](piqp::PIQPSolver& solver,
               const Eigen::Ref<const solvers_py::DenseMatrix>& P,
               const Eigen::Ref<const piqp::Vector>& q,
               const std::optional<Eigen::Ref<const solvers_py::DenseMatrix>>& A,
               const std::optional<Eigen::Ref<const piqp::Vector>>& b,
               const std::optional<Eigen::Ref<const solvers_py::DenseMatrix>>& G,
               const std::optional<Eigen::Ref<const piqp::Vector>>& h) -> piqp::PIQPSolver& {
                auto A_sparse = A ? std::optional<piqp::SparseMatrix>((*A).sparseView()) : std::nullopt;
                auto G_sparse = G ? std::optional<piqp::SparseMatrix>((*G).sparseView()) : std::nullopt;
                auto b_vec = b ? std::optional<piqp::Vector>(*b) : std::nullopt;
                auto h_vec = h ? std::optional<piqp::Vector>(*h) : std::nullopt;
                return solver.setup(to_spmat(P), q, A_sparse, b_vec, G_sparse, h_vec);
            },
            nb::arg("P"), nb::arg("q"), nb::arg("A") = nb::none(),
            nb::arg("b") = nb::none(), nb::arg("G") = nb::none(),
            nb::arg("h") = nb::none(), nb::rv_policy::reference_internal)
        .def("solve", [](piqp::PIQPSolver& solver) { return result_to_dict(solver.solve()); });

    m.def(
        "solve",
        [](const Eigen::Ref<const solvers_py::DenseMatrix>& P,
           const Eigen::Ref<const piqp::Vector>& q,
           const std::optional<Eigen::Ref<const solvers_py::DenseMatrix>>& A,
           const std::optional<Eigen::Ref<const piqp::Vector>>& b,
           const std::optional<Eigen::Ref<const solvers_py::DenseMatrix>>& G,
           const std::optional<Eigen::Ref<const piqp::Vector>>& h,
           const piqp::PIQPSettings& settings) {
            piqp::PIQPSolver solver(settings);
            auto A_sparse = A ? std::optional<piqp::SparseMatrix>((*A).sparseView()) : std::nullopt;
            auto G_sparse = G ? std::optional<piqp::SparseMatrix>((*G).sparseView()) : std::nullopt;
            auto b_vec = b ? std::optional<piqp::Vector>(*b) : std::nullopt;
            auto h_vec = h ? std::optional<piqp::Vector>(*h) : std::nullopt;
            solver.setup(to_spmat(P), q, A_sparse, b_vec, G_sparse, h_vec);
            return result_to_dict(solver.solve());
        },
        nb::arg("P"), nb::arg("q"), nb::arg("A") = nb::none(),
        nb::arg("b") = nb::none(), nb::arg("G") = nb::none(),
        nb::arg("h") = nb::none(), nb::arg("settings") = piqp::PIQPSettings{});
}
