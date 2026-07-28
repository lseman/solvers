"""HiGHS backend — LP, QP, and MIP solver (HiGHS, MIT licensed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_highs():
    import highspy as _highs
    return _highs


class HiGHSBackend:
    """HiGHS solver backend — LP, QP, and MIP solver.

    Solves: min c^T x s.t. l <= Ax <= u, variable_type in {continuous, integer, binary}

    Supports:
    - LP (linear programming)
    - MIP (mixed-integer programming)
    - QP (quadratic programming via Hessian)
    """

    def __init__(self) -> None:
        self._highs = _import_highs()

    def solve(
        self,
        P: sp.spmatrix | None,
        c: np.ndarray,
        A: sp.spmatrix,
        l: np.ndarray,
        u: np.ndarray,
        var_types: np.ndarray,
        **options,
    ) -> dict:
        """Solve a MIP/LP/QP problem.

        Parameters:
            P: Quadratic objective matrix (0.5 x^T P x). None for LP.
            c: Linear objective coefficients.
            A: Constraint matrix (m x n).
            l: Lower bounds on constraints.
            u: Upper bounds on constraints.
            var_types: Array of 0 (continuous), 1 (integer), 2 (binary).
            **options: HiGHS solver options (time_limit, log_level, etc.).
                ``x0``, if given, seeds an initial (primal) solution — for
                MIPs this can let HiGHS validate it as an incumbent via
                presolve and skip most of branch-and-bound.

        Returns:
            dict with keys: x, y, obj_val, status, iter, residuals
        """
        import numpy as np

        highs = self._highs.Highs()
        n_vars = len(c)
        n_rows = len(l)

        # Set solver options
        if "time_limit" in options:
            highs.setOptionValue("time_limit", float(options["time_limit"]))
        if "log_level" in options:
            highs.setOptionValue("output_flag", options["log_level"] > 0)
        else:
            highs.setOptionValue("output_flag", False)  # silent

        # Add columns (variables) first — HiGHS requires this before setting costs/types
        if n_vars > 0:
            # addCols: num_cols, costs, col_lower, col_upper,
            #          num_col_nz, col_start, col_ind, col_value
            # Variable bounds are handled via constraints (l, u), so we use
            # -inf/inf here — the actual variable bounds are encoded as
            # extra rows in A. No column nonzeros at column level — constraints add rows
            highs.addCols(
                n_vars,
                c.astype(np.float64),
                np.full(n_vars, -float("inf"), dtype=np.float64),
                np.full(n_vars, float("inf"), dtype=np.float64),
                0,
                np.array([0], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )

        # Add rows (constraints)
        if A.nnz > 0:
            # HiGHS uses CSR format for addRows
            A_csr = A.tocsr()
            row_lower = l.astype(np.float64)
            row_upper = u.astype(np.float64)
            num_nz = A_csr.nnz
            row_start = A_csr.indptr.astype(np.int32)
            col_indices = A_csr.indices.astype(np.int32)
            col_values = A_csr.data.astype(np.float64)

            highs.addRows(
                n_rows,
                row_lower,
                row_upper,
                num_nz,
                row_start,
                col_indices,
                col_values,
            )

        # Set variable types
        if n_vars > 0:
            vtype_indices = list(range(n_vars))
            vtype_values = [
                self._highs.HighsVarType.kContinuous if vt == 0
                else self._highs.HighsVarType.kInteger  # binary: integer with ub=1
                for vt in var_types
            ]
            highs.changeColsIntegrality(n_vars, vtype_indices, vtype_values)

        # Set QP Hessian if provided
        if P is not None and P.nnz > 0:
            P_csc = P.tocsc()
            # HiGHS expects the lower triangular part of P in CSC format,
            # assigned onto HighsHessian's raw start_/index_/value_ fields
            # (there is no setDimensions/setValues/setHessian in this
            # highspy build — passHessian takes the struct as-is).
            starts = [0]
            indices: list[int] = []
            values: list[float] = []
            for j in range(n_vars):
                for idx in range(P_csc.indptr[j], P_csc.indptr[j + 1]):
                    i = P_csc.indices[idx]
                    if i >= j:  # lower triangular only
                        indices.append(i)
                        values.append(P_csc.data[idx])
                starts.append(len(indices))

            hessian = self._highs.HighsHessian()
            hessian.dim_ = n_vars
            hessian.format_ = self._highs.HessianFormat.kTriangular
            hessian.start_ = starts
            hessian.index_ = indices
            hessian.value_ = values
            highs.passHessian(hessian)

        # Warm start: seed an initial primal solution, if provided.
        x0 = options.get("x0")
        if x0 is not None:
            warm = self._highs.HighsSolution()
            warm.col_value = np.asarray(x0, dtype=np.float64).tolist()
            highs.setSolution(warm)

        # Solve
        highs.run()

        # Extract results
        model_status = highs.getModelStatus()
        status_map = {
            self._highs.HighsModelStatus.kNotset: "notset",
            self._highs.HighsModelStatus.kLoadError: "load_error",
            self._highs.HighsModelStatus.kModelError: "model_error",
            self._highs.HighsModelStatus.kPresolveError: "presolve_error",
            self._highs.HighsModelStatus.kSolveError: "solve_error",
            self._highs.HighsModelStatus.kPostsolveError: "postsolve_error",
            self._highs.HighsModelStatus.kUnbounded: "unbounded",
            self._highs.HighsModelStatus.kUnboundedOrInfeasible: "unbounded_or_infeasible",
            self._highs.HighsModelStatus.kInfeasible: "infeasible",
            self._highs.HighsModelStatus.kSolutionLimit: "solution_limit",
            self._highs.HighsModelStatus.kTimeLimit: "time_limit",
            self._highs.HighsModelStatus.kOptimal: "solved",
            self._highs.HighsModelStatus.kObjectiveBound: "objective_bound",
            self._highs.HighsModelStatus.kObjectiveTarget: "objective_target",
            self._highs.HighsModelStatus.kIterationLimit: "iteration_limit",
            self._highs.HighsModelStatus.kUnknown: "unknown",
            self._highs.HighsModelStatus.kInterrupt: "interrupt",
        }
        status = status_map.get(model_status, str(model_status))

        sol = highs.getSolution()
        x = sol.col_value

        # Get duals (shadow prices)
        try:
            y = sol.row_dual
        except Exception:
            y = np.zeros(n_rows)

        info = highs.getInfo()
        return {
            "x": x,
            "y": y,
            "obj_val": float(highs.getObjectiveValue()),
            "status": status,
            "iter": info.simplex_iteration_count
                    or info.ipm_iteration_count
                    or info.qp_iteration_count
                    or 0,
            "residuals": {
                "mip_gap": float(info.mip_gap),
                "prim_inf": float(info.max_primal_infeasibility),
                "dua_inf": float(info.max_dual_infeasibility),
            },
        }
