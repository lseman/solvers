// pdlp_ext.cpp — PDLP (restarted PDHG for LP) Python bindings.
//
// Mirrors the API of https://github.com/mmaaz-git/pdlp:
//     x, y, status, info = pdlp.solve(c, G, h, A, b, l, u, ...)
// with an extra `device` argument ("cpu" or "cuda").

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "../pdlp/pdlp.h"
#ifdef PDLP_WITH_CUDA
#include "../pdlp/pdlp_cuda.h"
#endif

namespace nb = nanobind;
using namespace nb::literals;
using solvers::pdlp::Result;
using solvers::pdlp::Settings;
using solvers::pdlp::SpMatCsc;
using solvers::pdlp::Vec;

namespace {

nb::dict make_info(const Result& res) {
    nb::dict info;
    info["solve_time_sec"] = res.solve_time_sec;
    info["iterations"] = res.iterations;
    if (res.status == "primal_infeasible" || res.status == "dual_infeasible") {
        info["ray"] = res.ray;
        info["certificate_quality"] = res.certificate_quality;
        if (res.status == "primal_infeasible") {
            info["dual_ray_obj"] = res.dual_ray_obj;
            info["dual_residual"] = res.dual_residual;
        } else {
            info["primal_ray_obj"] = res.primal_ray_obj;
            info["max_primal_residual"] = res.max_primal_residual;
        }
        if (!std::isnan(res.primal_obj)) info["primal_obj"] = res.primal_obj;
        if (!std::isnan(res.dual_obj)) info["dual_obj"] = res.dual_obj;
    } else {
        info["primal_obj"] = res.primal_obj;
        info["dual_obj"] = res.dual_obj;
        info["duality_gap"] = res.duality_gap;
        info["relative_gap"] = res.relative_gap;
        info["primal_residual"] = res.primal_residual;
        info["dual_residual"] = res.dual_residual;
        info["kkt_error_sq"] = res.kkt_error_sq;
    }
    return info;
}

nb::tuple solve_py(const Vec& c, const SpMatCsc& G, const Vec& h, const SpMatCsc& A,
                   const Vec& b, const Vec& l, const Vec& u, double iteration_limit,
                   double time_sec_limit, double primal_weight_update_smoothing,
                   int ruiz_iterations, double pock_chambolle_alpha, double eps_tol,
                   double eps_primal_infeasible, double eps_dual_infeasible, bool verbose,
                   const std::string& device) {
    if (G.rows() != h.size() || A.rows() != b.size()) {
        throw std::invalid_argument("pdlp: G/h or A/b row mismatch");
    }
    if (G.cols() != c.size() || A.cols() != c.size() || l.size() != c.size() ||
        u.size() != c.size()) {
        throw std::invalid_argument("pdlp: G.cols, A.cols, l, u must match c.size");
    }
    if (std::isinf(iteration_limit) && std::isinf(time_sec_limit)) {
        throw std::invalid_argument(
            "pdlp: at least one of iteration_limit or time_sec_limit must be finite");
    }

    Settings s;
    s.iteration_limit = iteration_limit;
    s.time_sec_limit = time_sec_limit;
    s.primal_weight_update_smoothing = primal_weight_update_smoothing;
    s.ruiz_iterations = ruiz_iterations;
    s.pock_chambolle_alpha = pock_chambolle_alpha;
    s.eps_tol = eps_tol;
    s.eps_primal_infeasible = eps_primal_infeasible;
    s.eps_dual_infeasible = eps_dual_infeasible;
    s.verbose = verbose;

    Result res;
    if (device == "cpu") {
        nb::gil_scoped_release release;
        res = solvers::pdlp::solve(c, G, h, A, b, l, u, s);
    } else if (device == "cuda") {
#ifdef PDLP_WITH_CUDA
        nb::gil_scoped_release release;
        res = solvers::pdlp::solve_cuda(c, G, h, A, b, l, u, s);
#else
        throw std::runtime_error("pdlp: built without CUDA support (configure with -DWITH_CUDA=ON)");
#endif
    } else {
        throw std::invalid_argument("pdlp: device must be 'cpu' or 'cuda'");
    }

    return nb::make_tuple(res.x, res.y, res.status, make_info(res));
}

bool cuda_available_py() {
#ifdef PDLP_WITH_CUDA
    return solvers::pdlp::cuda_available();
#else
    return false;
#endif
}

}  // namespace

NB_MODULE(pdlp, m) {
    m.doc() = "PDLP — restarted adaptive PDHG solver for linear programming (CPU/CUDA)";

    m.def("solve", &solve_py, "c"_a, "G"_a, "h"_a, "A"_a, "b"_a, "l"_a, "u"_a,
          "iteration_limit"_a = 10000.0,
          "time_sec_limit"_a = std::numeric_limits< double >::infinity(),
          "primal_weight_update_smoothing"_a = 0.5, "ruiz_iterations"_a = 10,
          "pock_chambolle_alpha"_a = 1.0, "eps_tol"_a = 1e-4,
          "eps_primal_infeasible"_a = 1e-8, "eps_dual_infeasible"_a = 1e-8, "verbose"_a = false,
          "device"_a = "cpu",
          "Solve  min c'x  s.t. Gx >= h, Ax = b, l <= x <= u.\n\n"
          "G and A must be scipy.sparse CSC matrices (use shape (0, n) when absent).\n"
          "Returns (x, y, status, info) where y[:m1] are inequality duals and\n"
          "y[m1:] equality duals; status is one of 'optimal', 'primal_infeasible',\n"
          "'dual_infeasible', 'iteration_limit', 'time_limit'.");

    m.def("cuda_available", &cuda_available_py,
          "True if the module was built with CUDA and a device is present.");
}
