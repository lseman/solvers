#pragma once

// CUDA backend for the PDLP solver (see pdlp/pdlp.h for the algorithm and the
// CPU reference implementation). Compiled from src/pdlp/pdlp_cuda.cu when the
// project is configured with -DWITH_CUDA=ON.

#include "pdlp.h"

namespace solvers::pdlp {

// true if at least one CUDA device is usable
bool cuda_available();

// Same contract as solve(), executed on the GPU. Problem setup / rescaling
// happens on the host; the PDHG iteration runs on the device (cuSPARSE SpMV +
// fused kernels). Throws std::runtime_error on CUDA/cuSPARSE/cuBLAS failure.
Result solve_cuda(const Vec& c, const SpMatCsc& G, const Vec& h, const SpMatCsc& A, const Vec& b,
                  const Vec& l, const Vec& u, const Settings& settings = {});

}  // namespace solvers::pdlp
