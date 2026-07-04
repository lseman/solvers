# Solver examples

Build the Python modules first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

Then open the notebooks from this directory or the repository root.

- `01_qdldl_vs_numpy_scipy.ipynb`: compares the QDLDL wrapper against NumPy and SciPy sparse solves.
- `02_osqp_vs_scipy_unconstrained_qp.ipynb`: compares the OSQP-style solver against the analytic and SciPy solution of an unconstrained QP.
- `03_piqp_vs_scipy_constrained_qp.ipynb`: compares PIQP against SciPy SLSQP on a small constrained QP.
