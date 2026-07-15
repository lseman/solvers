// CUDA backend for the PDLP solver. Mirrors the CPU iteration in pdlp/pdlp.h:
// problem setup / rescaling run on the host (build_data), the restarted
// adaptive PDHG loop runs on the device. SpMV via cuSPARSE (CSR, K and K^T
// stored explicitly), dot products / norms via cuBLAS, elementwise updates via
// fused kernels, odd reductions via thrust.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "../../pdlp/pdlp_cuda.h"

namespace solvers::pdlp {

namespace {

#define PDLP_CUDA_CHECK(call)                                                              \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            throw std::runtime_error(std::string("pdlp cuda error: ") +                    \
                                     cudaGetErrorString(err_));                            \
        }                                                                                  \
    } while (0)

#define PDLP_CUSPARSE_CHECK(call)                                                          \
    do {                                                                                   \
        cusparseStatus_t st_ = (call);                                                     \
        if (st_ != CUSPARSE_STATUS_SUCCESS) {                                              \
            throw std::runtime_error(std::string("pdlp cusparse error: ") +                \
                                     cusparseGetErrorString(st_));                         \
        }                                                                                  \
    } while (0)

#define PDLP_CUBLAS_CHECK(call)                                                            \
    do {                                                                                   \
        cublasStatus_t st_ = (call);                                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                \
            throw std::runtime_error("pdlp cublas error: status " + std::to_string(st_));  \
        }                                                                                  \
    } while (0)

constexpr int kBlock = 256;
inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// ── kernels ─────────────────────────────────────────────────────────────────

__global__ void k_clamp_box(int n, double* x, const double* l, const double* u) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = fmin(fmax(x[i], l[i]), u[i]);
    }
}

// xp = clamp(x - s * grad, l, u)
__global__ void k_primal_step(int n, const double* x, const double* grad, double s,
                              const double* l, const double* u, double* xp) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        xp[i] = fmin(fmax(x[i] - s * grad[i], l[i]), u[i]);
    }
}

// yp = proj_Y(y + s * (q - (2 Kxp - Kx))), proj: first m1 entries >= 0
__global__ void k_dual_step(int m, int m1, const double* y, const double* q, const double* Kxp,
                            const double* Kx, double s, double* yp) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        double v = y[i] + s * (q[i] - (2.0 * Kxp[i] - Kx[i]));
        yp[i] = i < m1 ? fmax(v, 0.0) : v;
    }
}

// bar += alpha * (v - bar)
__global__ void k_avg(int n, double alpha, const double* v, double* bar) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        bar[i] += alpha * (v[i] - bar[i]);
    }
}

__global__ void k_diff(int n, const double* a, const double* b, double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = a[i] - b[i];
    }
}

__global__ void k_unscale(int n, const double* v, const double* scale, double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = v[i] / scale[i];
    }
}

// normal-cone component for box constraints (see detail::lambda_for_box)
__global__ void k_lambda(int n, const double* x, const double* g, const double* l,
                         const double* u, double eps_tol, double* lam) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        bool at_l = isfinite(l[i]) && x[i] <= l[i] + eps_tol;
        bool at_u = isfinite(u[i]) && x[i] >= u[i] - eps_tol;
        if (at_l && at_u) {
            const bool closer_l = fabs(x[i] - l[i]) <= fabs(u[i] - x[i]);
            at_l = closer_l;
            at_u = !closer_l;
        }
        lam[i] = at_l ? fmax(g[i], 0.0) : (at_u ? fmin(g[i], 0.0) : 0.0);
    }
}

// primal residual vector: max(q - Kx, 0) on inequality rows, Kx - q on equality rows
__global__ void k_pri_res(int m, int m1, const double* Kx, const double* q, double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        out[i] = i < m1 ? fmax(q[i] - Kx[i], 0.0) : Kx[i] - q[i];
    }
}

// primal-ray residual: max(-Kx, 0) on inequality rows, |Kx| on equality rows
__global__ void k_ray_res(int m, int m1, const double* Kx, double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        out[i] = i < m1 ? fmax(-Kx[i], 0.0) : fabs(Kx[i]);
    }
}

// out = isfinite(vals) ? vals * max(sign * lam, 0) : 0
__global__ void k_finite_prod(int n, const double* vals, const double* lam, double sign,
                              double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = isfinite(vals[i]) ? vals[i] * fmax(sign * lam[i], 0.0) : 0.0;
    }
}

// ── device memory / library RAII ────────────────────────────────────────────

class DVec {
public:
    DVec() = default;
    explicit DVec(int n) : n_(n) {
        if (n_ > 0) PDLP_CUDA_CHECK(cudaMalloc(&p_, sizeof(double) * n_));
    }
    explicit DVec(const Vec& h) : DVec(static_cast< int >(h.size())) { upload(h); }
    DVec(const DVec&) = delete;
    DVec& operator=(const DVec&) = delete;
    ~DVec() {
        if (p_) cudaFree(p_);
    }
    void upload(const Vec& h) {
        if (n_ > 0)
            PDLP_CUDA_CHECK(
                cudaMemcpy(p_, h.data(), sizeof(double) * n_, cudaMemcpyHostToDevice));
    }
    Vec download() const {
        Vec h(n_);
        if (n_ > 0)
            PDLP_CUDA_CHECK(
                cudaMemcpy(h.data(), p_, sizeof(double) * n_, cudaMemcpyDeviceToHost));
        return h;
    }
    void copy_from(const DVec& o) {
        PDLP_CUDA_CHECK(cudaMemcpy(p_, o.p_, sizeof(double) * n_, cudaMemcpyDeviceToDevice));
    }
    void zero() {
        if (n_ > 0) PDLP_CUDA_CHECK(cudaMemset(p_, 0, sizeof(double) * n_));
    }
    double* p() const { return p_; }
    int n() const { return n_; }

private:
    double* p_ = nullptr;
    int n_ = 0;
};

class CsrGpu {
public:
    CsrGpu(cusparseHandle_t hs, const SpMat& M) : rows_((int)M.rows()), cols_((int)M.cols()) {
        const int nnz = (int)M.nonZeros();
        PDLP_CUDA_CHECK(cudaMalloc(&rowptr_, sizeof(int) * (rows_ + 1)));
        PDLP_CUDA_CHECK(cudaMalloc(&colind_, sizeof(int) * std::max(nnz, 1)));
        PDLP_CUDA_CHECK(cudaMalloc(&vals_, sizeof(double) * std::max(nnz, 1)));
        PDLP_CUDA_CHECK(cudaMemcpy(rowptr_, M.outerIndexPtr(), sizeof(int) * (rows_ + 1),
                                   cudaMemcpyHostToDevice));
        if (nnz > 0) {
            PDLP_CUDA_CHECK(cudaMemcpy(colind_, M.innerIndexPtr(), sizeof(int) * nnz,
                                       cudaMemcpyHostToDevice));
            PDLP_CUDA_CHECK(
                cudaMemcpy(vals_, M.valuePtr(), sizeof(double) * nnz, cudaMemcpyHostToDevice));
        }
        PDLP_CUSPARSE_CHECK(cusparseCreateCsr(&mat_, rows_, cols_, nnz, rowptr_, colind_, vals_,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        // workspace query with throwaway vectors
        DVec tx(cols_), ty(rows_);
        cusparseDnVecDescr_t vx, vy;
        PDLP_CUSPARSE_CHECK(cusparseCreateDnVec(&vx, cols_, tx.p(), CUDA_R_64F));
        PDLP_CUSPARSE_CHECK(cusparseCreateDnVec(&vy, rows_, ty.p(), CUDA_R_64F));
        const double one = 1.0, zero = 0.0;
        PDLP_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            hs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, vx, &zero, vy, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufsize_));
        PDLP_CUSPARSE_CHECK(cusparseDestroyDnVec(vx));
        PDLP_CUSPARSE_CHECK(cusparseDestroyDnVec(vy));
        if (bufsize_ > 0) PDLP_CUDA_CHECK(cudaMalloc(&buf_, bufsize_));
    }
    CsrGpu(const CsrGpu&) = delete;
    CsrGpu& operator=(const CsrGpu&) = delete;
    ~CsrGpu() {
        if (mat_) cusparseDestroySpMat(mat_);
        if (buf_) cudaFree(buf_);
        if (rowptr_) cudaFree(rowptr_);
        if (colind_) cudaFree(colind_);
        if (vals_) cudaFree(vals_);
    }

    // out = M * in
    void spmv(cusparseHandle_t hs, const DVec& in, DVec& out) const {
        cusparseDnVecDescr_t vx, vy;
        PDLP_CUSPARSE_CHECK(cusparseCreateDnVec(&vx, cols_, in.p(), CUDA_R_64F));
        PDLP_CUSPARSE_CHECK(cusparseCreateDnVec(&vy, rows_, out.p(), CUDA_R_64F));
        const double one = 1.0, zero = 0.0;
        PDLP_CUSPARSE_CHECK(cusparseSpMV(hs, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, vx,
                                         &zero, vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf_));
        PDLP_CUSPARSE_CHECK(cusparseDestroyDnVec(vx));
        PDLP_CUSPARSE_CHECK(cusparseDestroyDnVec(vy));
    }

private:
    int rows_ = 0, cols_ = 0;
    int* rowptr_ = nullptr;
    int* colind_ = nullptr;
    double* vals_ = nullptr;
    cusparseSpMatDescr_t mat_ = nullptr;
    void* buf_ = nullptr;
    size_t bufsize_ = 0;
};

struct Handles {
    cusparseHandle_t sp = nullptr;
    cublasHandle_t bl = nullptr;
    Handles() {
        PDLP_CUSPARSE_CHECK(cusparseCreate(&sp));
        PDLP_CUBLAS_CHECK(cublasCreate(&bl));
        PDLP_CUBLAS_CHECK(cublasSetPointerMode(bl, CUBLAS_POINTER_MODE_HOST));
    }
    ~Handles() {
        if (bl) cublasDestroy(bl);
        if (sp) cusparseDestroy(sp);
    }
};

// ── reductions ──────────────────────────────────────────────────────────────

double ddot(cublasHandle_t h, const DVec& a, const DVec& b) {
    if (a.n() == 0) return 0.0;
    double r = 0.0;
    PDLP_CUBLAS_CHECK(cublasDdot(h, a.n(), a.p(), 1, b.p(), 1, &r));
    return r;
}

double dnrm2(cublasHandle_t h, const DVec& a) {
    if (a.n() == 0) return 0.0;
    double r = 0.0;
    PDLP_CUBLAS_CHECK(cublasDnrm2(h, a.n(), a.p(), 1, &r));
    return r;
}

double dinf_norm(cublasHandle_t h, const DVec& a) {
    if (a.n() == 0) return 0.0;
    int idx = 0;
    PDLP_CUBLAS_CHECK(cublasIdamax(h, a.n(), a.p(), 1, &idx));  // 1-based
    double v = 0.0;
    PDLP_CUDA_CHECK(cudaMemcpy(&v, a.p() + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost));
    return std::abs(v);
}

double dsum(const DVec& a) {
    if (a.n() == 0) return 0.0;
    const thrust::device_ptr< const double > p(a.p());
    return thrust::reduce(thrust::device, p, p + a.n(), 0.0);
}

// ── device-side solver state ────────────────────────────────────────────────

struct Dev {
    Handles h;
    int m1, m2, n, m;
    CsrGpu K, Kt, K_orig, Kt_orig;
    DVec c, q, l, u;              // rescaled
    DVec c_o, q_o, l_o, u_o;      // original
    DVec var_resc, con_resc;
    // scratch
    DVec scr_m, scr_n, gvec, lam;

    Dev(const Data& d)
        : m1(d.m1),
          m2(d.m2),
          n(d.n),
          m(d.m),
          K(h.sp, d.K),
          Kt(h.sp, d.Kt),
          K_orig(h.sp, d.K_orig),
          Kt_orig(h.sp, d.Kt_orig),
          c(d.c),
          q(d.q),
          l(d.l),
          u(d.u),
          c_o(d.c_orig),
          q_o(d.q_orig),
          l_o(d.l_orig),
          u_o(d.u_orig),
          var_resc(d.variable_rescaling),
          con_resc(d.constraint_rescaling),
          scr_m(d.m),
          scr_n(d.n),
          gvec(d.n),
          lam(d.n) {}

    // squared KKT error on the rescaled problem (mirrors detail::kkt_error_sq)
    double kkt_error_sq(const DVec& x, const DVec& y, double w, const DVec& Kx, const DVec& KTy,
                        double eps_tol) {
        k_pri_res<<< grid_for(m), kBlock >>>(m, m1, Kx.p(), q.p(), scr_m.p());
        const double pri = ddot(h.bl, scr_m, scr_m);

        k_diff<<< grid_for(n), kBlock >>>(n, c.p(), KTy.p(), gvec.p());
        k_lambda<<< grid_for(n), kBlock >>>(n, x.p(), gvec.p(), l.p(), u.p(), eps_tol, lam.p());
        k_diff<<< grid_for(n), kBlock >>>(n, gvec.p(), lam.p(), scr_n.p());
        const double rs2 = ddot(h.bl, scr_n, scr_n);

        k_finite_prod<<< grid_for(n), kBlock >>>(n, l.p(), lam.p(), 1.0, scr_n.p());
        const double l_term = dsum(scr_n);
        k_finite_prod<<< grid_for(n), kBlock >>>(n, u.p(), lam.p(), -1.0, scr_n.p());
        const double u_term = dsum(scr_n);

        const double scalar = ddot(h.bl, q, y) + l_term - u_term - ddot(h.bl, c, x);
        w = std::max(w, kEpsZero);
        return w * w * pri + rs2 / (w * w) + scalar * scalar;
    }

    // dual objective on the original problem (mirrors detail::dual_objective)
    double dual_objective(const DVec& xu, const DVec& yu, const DVec& KTy_orig, double eps_tol) {
        k_diff<<< grid_for(n), kBlock >>>(n, c_o.p(), KTy_orig.p(), gvec.p());
        k_lambda<<< grid_for(n), kBlock >>>(n, xu.p(), gvec.p(), l_o.p(), u_o.p(), eps_tol,
                                            lam.p());
        k_finite_prod<<< grid_for(n), kBlock >>>(n, l_o.p(), lam.p(), 1.0, scr_n.p());
        const double l_term = dsum(scr_n);
        k_finite_prod<<< grid_for(n), kBlock >>>(n, u_o.p(), lam.p(), -1.0, scr_n.p());
        const double u_term = dsum(scr_n);
        return ddot(h.bl, q_o, yu) + l_term - u_term;
    }

    // termination check on the original problem (mirrors detail::termination_criteria)
    detail::TerminationOutcome termination_criteria(const DVec& xu, const DVec& yu,
                                                    const DVec& Kx_orig, const DVec& KTy_orig,
                                                    double c_orig_norm, double q_orig_norm,
                                                    const Settings& s) {
        detail::TerminationOutcome out;
        out.dual_obj = dual_objective(xu, yu, KTy_orig, s.eps_tol);
        out.primal_obj = ddot(h.bl, c_o, xu);

        const double dual_norm_inf = dinf_norm(h.bl, yu);
        if (dual_norm_inf > kEpsZero) {
            const double dual_ray_obj = ddot(h.bl, q_o, yu) / dual_norm_inf;
            if (dual_ray_obj > 0) {
                const double dual_residual = dinf_norm(h.bl, KTy_orig) / dual_norm_inf;
                const double relative_infeas = dual_residual / dual_ray_obj;
                if (relative_infeas < s.eps_primal_infeasible) {
                    out.status = "primal_infeasible";
                    out.ray = yu.download() / dual_norm_inf;
                    out.certificate_quality = relative_infeas;
                    out.dual_ray_obj = dual_ray_obj;
                    out.dual_residual = dual_residual;
                    return out;
                }
            }
        }

        const double primal_norm_inf = dinf_norm(h.bl, xu);
        if (primal_norm_inf > kEpsZero) {
            const double primal_ray_obj = ddot(h.bl, c_o, xu) / primal_norm_inf;
            if (primal_ray_obj < 0) {
                k_ray_res<<< grid_for(m), kBlock >>>(m, m1, Kx_orig.p(), scr_m.p());
                const double max_primal_residual = dinf_norm(h.bl, scr_m) / primal_norm_inf;
                const double relative_infeas = max_primal_residual / (-primal_ray_obj);
                if (relative_infeas < s.eps_dual_infeasible) {
                    out.status = "dual_infeasible";
                    out.ray = xu.download() / primal_norm_inf;
                    out.certificate_quality = relative_infeas;
                    out.primal_ray_obj = primal_ray_obj;
                    out.max_primal_residual = max_primal_residual;
                    return out;
                }
            }
        }

        const bool gap_ok = std::abs(out.dual_obj - out.primal_obj) <=
                            s.eps_tol * (1.0 + std::abs(out.dual_obj) + std::abs(out.primal_obj));

        k_pri_res<<< grid_for(m), kBlock >>>(m, m1, Kx_orig.p(), q_o.p(), scr_m.p());
        const double feas = std::sqrt(ddot(h.bl, scr_m, scr_m));
        const bool feas_ok = feas <= s.eps_tol * (1.0 + q_orig_norm);

        k_diff<<< grid_for(n), kBlock >>>(n, c_o.p(), KTy_orig.p(), gvec.p());
        k_lambda<<< grid_for(n), kBlock >>>(n, xu.p(), gvec.p(), l_o.p(), u_o.p(), s.eps_tol,
                                            lam.p());
        k_diff<<< grid_for(n), kBlock >>>(n, gvec.p(), lam.p(), scr_n.p());
        const bool stat_ok = dnrm2(h.bl, scr_n) <= s.eps_tol * (1.0 + c_orig_norm);

        if (gap_ok && feas_ok && stat_ok) out.status = "optimal";
        return out;
    }
};

}  // namespace

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Result solve_cuda(const Vec& c, const SpMatCsc& G, const Vec& h_rhs, const SpMatCsc& A,
                  const Vec& b, const Vec& l, const Vec& u, const Settings& settings) {
    // trivial cases don't need a GPU
    if (c.size() == 0 || G.rows() + A.rows() == 0) {
        return solve(c, G, h_rhs, A, b, l, u, settings);
    }

    const double start_time = detail::now_sec();
    const auto elapsed = [&] { return detail::now_sec() - start_time; };

    if (settings.verbose) {
        std::printf("\nPDLP Solver (CUDA)\n  Problem: %d inequalities, %d equalities, %d "
                    "variables\n",
                    (int)G.rows(), (int)A.rows(), (int)c.size());
    }

    Data d = build_data(c, G, h_rhs, A, b, l, u, settings);

    if (settings.verbose) {
        std::printf("  Rescaling: Ruiz iters=%d, Pock-Chambolle alpha=%g\n",
                    settings.ruiz_iterations, settings.pock_chambolle_alpha);
    }

    Dev dev(d);
    const int n = d.n, m = d.m;

    // iterates and per-iteration buffers
    DVec x(n), y(m), xp(n), yp(m);
    DVec x_prev(n), y_prev(m), x_bar(n), y_bar(m), x_c(n), y_c(m), x_cn(n), y_cn(m);
    DVec Kx(m), KTy(n), Kxp(m), grad(n), dxv(n), dyv(m), Kdx(m);
    DVec Kx_bar(m), KTy_bar(n);
    DVec xu(n), yu(m), Kx_orig(m), KTy_orig(n);
    DVec x_unscaled_last(n), y_unscaled_last(m);

    // x0 = proj_X(0), y0 = 0
    x.zero();
    k_clamp_box<<< grid_for(n), kBlock >>>(n, x.p(), dev.l.p(), dev.u.p());
    y.zero();

    // step size 1/||K||_inf (max row abs-sum), computed host-side from d.K
    double max_row_sum = 0.0;
    {
        Vec row_sums = Vec::Zero(m);
        for (int i = 0; i < d.K.outerSize(); ++i) {
            for (SpMat::InnerIterator it(d.K, i); it; ++it) {
                row_sums(it.row()) += std::abs(it.value());
            }
        }
        max_row_sum = row_sums.maxCoeff();
    }
    double eta_hat = 1.0 / std::max(max_row_sum, kEpsZero);

    const double c_norm = d.c.norm(), q_norm = d.q.norm();
    double w = (c_norm > kEpsZero && q_norm > kEpsZero) ? c_norm / q_norm : 1.0;
    const double c_orig_norm = d.c_orig.norm(), q_orig_norm = d.q_orig.norm();

    x_prev.copy_from(x);
    y_prev.copy_from(y);
    x_c.copy_from(x);
    y_c.copy_from(y);
    k_unscale<<< grid_for(n), kBlock >>>(n, x.p(), dev.var_resc.p(), x_unscaled_last.p());
    k_unscale<<< grid_for(m), kBlock >>>(m, y.p(), dev.con_resc.p(), y_unscaled_last.p());

    long n_iterations = 0;
    std::string status;
    detail::TerminationOutcome term;

    // one adaptive PDHG step: x, y updated in place, returns eta actually used
    const auto adaptive_step = [&](double& eta_hat_io, long k) -> double {
        double eta = std::max(eta_hat_io, kEpsZero);
        const double kp1 = static_cast< double >(k + 1);
        const double fac1 = k == 0 ? 1.0 : 1.0 - std::pow(kp1, -0.3);
        const double fac2 = 1.0 + std::pow(kp1, -0.6);

        dev.Kt.spmv(dev.h.sp, y, KTy);
        dev.K.spmv(dev.h.sp, x, Kx);
        k_diff<<< grid_for(n), kBlock >>>(n, dev.c.p(), KTy.p(), grad.p());

        while (true) {
            k_primal_step<<< grid_for(n), kBlock >>>(n, x.p(), grad.p(), eta / w, dev.l.p(),
                                                     dev.u.p(), xp.p());
            dev.K.spmv(dev.h.sp, xp, Kxp);
            k_dual_step<<< grid_for(m), kBlock >>>(m, d.m1, y.p(), dev.q.p(), Kxp.p(), Kx.p(),
                                                   eta * w, yp.p());

            k_diff<<< grid_for(n), kBlock >>>(n, xp.p(), x.p(), dxv.p());
            k_diff<<< grid_for(m), kBlock >>>(m, yp.p(), y.p(), dyv.p());
            k_diff<<< grid_for(m), kBlock >>>(m, Kxp.p(), Kx.p(), Kdx.p());

            const double num = w * ddot(dev.h.bl, dxv, dxv) + ddot(dev.h.bl, dyv, dyv) / w;
            const double denom = 2.0 * std::abs(ddot(dev.h.bl, dyv, Kdx));
            const double bar_eta =
                denom <= kEpsZero ? std::numeric_limits< double >::infinity() : num / denom;

            const double eta_p = std::max(std::min(fac1 * bar_eta, fac2 * eta), kEpsZero);

            if (eta <= bar_eta) {
                x.copy_from(xp);
                y.copy_from(yp);
                eta_hat_io = eta_p;
                return eta;
            }
            eta = eta_p;
        }
    };

    while (true) {
        // KKT of the restart point with the current primal weight
        dev.K.spmv(dev.h.sp, x, Kx);
        dev.Kt.spmv(dev.h.sp, y, KTy);
        const double kkt_last_restart = dev.kkt_error_sq(x, y, w, Kx, KTy, settings.eps_tol);

        k_unscale<<< grid_for(n), kBlock >>>(n, x.p(), dev.var_resc.p(), x_unscaled_last.p());
        k_unscale<<< grid_for(m), kBlock >>>(m, y.p(), dev.con_resc.p(), y_unscaled_last.p());

        double eta_sum = 0.0;
        x_bar.copy_from(x);
        y_bar.copy_from(y);
        double kkt_c_prev = kkt_last_restart;
        x_cn.copy_from(x);
        y_cn.copy_from(y);
        bool inner_broke = false;

        for (int t = 0; t < kMaxInnerIters; ++t) {
            const double eta_used = adaptive_step(eta_hat, n_iterations);

            eta_sum += eta_used;
            const double alpha = eta_used / eta_sum;
            k_avg<<< grid_for(n), kBlock >>>(n, alpha, x.p(), x_bar.p());
            k_avg<<< grid_for(m), kBlock >>>(m, alpha, y.p(), y_bar.p());

            ++n_iterations;

            if (static_cast< double >(n_iterations) >= settings.iteration_limit ||
                elapsed() >= settings.time_sec_limit) {
                status = static_cast< double >(n_iterations) >= settings.iteration_limit
                             ? "iteration_limit"
                             : "time_limit";
                k_unscale<<< grid_for(n), kBlock >>>(n, x.p(), dev.var_resc.p(),
                                                     x_unscaled_last.p());
                k_unscale<<< grid_for(m), kBlock >>>(m, y.p(), dev.con_resc.p(),
                                                     y_unscaled_last.p());
                inner_broke = true;
                break;
            }

            if (n_iterations <= 10 || n_iterations % kTerminationCheckFrequency == 0) {
                dev.K.spmv(dev.h.sp, x, Kx);
                dev.Kt.spmv(dev.h.sp, y, KTy);
                dev.K.spmv(dev.h.sp, x_bar, Kx_bar);
                dev.Kt.spmv(dev.h.sp, y_bar, KTy_bar);
                k_unscale<<< grid_for(n), kBlock >>>(n, x.p(), dev.var_resc.p(), xu.p());
                k_unscale<<< grid_for(m), kBlock >>>(m, y.p(), dev.con_resc.p(), yu.p());
                dev.K_orig.spmv(dev.h.sp, xu, Kx_orig);
                dev.Kt_orig.spmv(dev.h.sp, yu, KTy_orig);

                const double kkt_current =
                    dev.kkt_error_sq(x, y, w, Kx, KTy, settings.eps_tol);
                const double kkt_averaged =
                    dev.kkt_error_sq(x_bar, y_bar, w, Kx_bar, KTy_bar, settings.eps_tol);
                double kkt_c_new;
                if (kkt_current < kkt_averaged) {
                    x_cn.copy_from(x);
                    y_cn.copy_from(y);
                    kkt_c_new = kkt_current;
                } else {
                    x_cn.copy_from(x_bar);
                    y_cn.copy_from(y_bar);
                    kkt_c_new = kkt_averaged;
                }

                term = dev.termination_criteria(xu, yu, Kx_orig, KTy_orig, c_orig_norm,
                                                q_orig_norm, settings);
                status = term.status;
                if (settings.verbose) {
                    std::printf(
                        "  Iter %5ld: primal_obj = %+.6e, dual_obj = %+.6e, gap = %.3e, KKT = "
                        "%.3e\n",
                        n_iterations, term.primal_obj, term.dual_obj,
                        std::abs(term.primal_obj - term.dual_obj), std::sqrt(kkt_current));
                }
                if (!status.empty()) {
                    if (n_iterations < 10 && status != "optimal") {
                        status.clear();
                    } else {
                        x_unscaled_last.copy_from(xu);
                        y_unscaled_last.copy_from(yu);
                        inner_broke = true;
                        break;
                    }
                }

                const bool cond_i =
                    kkt_c_new <= kBetaSufficient * kBetaSufficient * kkt_last_restart;
                const bool cond_ii =
                    kkt_c_new <= kBetaNecessary * kBetaNecessary * kkt_last_restart && t > 0 &&
                    kkt_c_new > kkt_c_prev;
                const bool cond_iii = t >= kBetaArtificial * static_cast< double >(n_iterations);

                if (cond_i || cond_ii || cond_iii) {
                    x_c.copy_from(x_cn);
                    y_c.copy_from(y_cn);
                    inner_broke = true;
                    break;
                }
                kkt_c_prev = kkt_c_new;
            }
        }
        if (!inner_broke) {
            x_c.copy_from(x_cn);
            y_c.copy_from(y_cn);
        }

        if (!status.empty()) break;

        x.copy_from(x_c);
        y.copy_from(y_c);

        // primal weight update
        k_diff<<< grid_for(n), kBlock >>>(n, x.p(), x_prev.p(), dxv.p());
        k_diff<<< grid_for(m), kBlock >>>(m, y.p(), y_prev.p(), dyv.p());
        const double dxn = dnrm2(dev.h.bl, dxv);
        const double dyn = dnrm2(dev.h.bl, dyv);
        if (dxn > kEpsZero && dyn > kEpsZero) {
            const double theta = settings.primal_weight_update_smoothing;
            w = std::pow(dyn / dxn, theta) * std::pow(w, 1.0 - theta);
        }
        x_prev.copy_from(x);
        y_prev.copy_from(y);
    }

    PDLP_CUDA_CHECK(cudaDeviceSynchronize());

    Result res;
    res.status = status;
    res.x = x_unscaled_last.download();
    res.y = y_unscaled_last.download();
    res.iterations = n_iterations;
    res.solve_time_sec = elapsed();
    detail::finalize_result(d, settings, res, x.download(), y.download(), w, term);
    return res;
}

}  // namespace solvers::pdlp
