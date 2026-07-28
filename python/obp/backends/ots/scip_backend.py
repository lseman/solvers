"""SCIP backend — LP and MIP solver via pyscipopt."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_scip():
    import pyscipopt as _scip
    return _scip


class SCIPBackend:
    """SCIP solver backend (via pyscipopt) — LP and MIP solver.

    Solves: min c^T x s.t. l <= Ax <= u, variable_type in {continuous, integer, binary}

    Quadratic objectives are not supported (pyscipopt's Model.setObjective
    rejects nonlinear objectives) — use ``osqp``, ``piqp``, ``highs``, or
    ``gurobi`` for QP.

    Dual values (``y``) are only computed for pure LPs (no integer/binary
    variables) — duals are undefined for MIPs and return zeros instead.
    """

    def __init__(self) -> None:
        self._scip = _import_scip()

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
        """Solve a MIP/LP problem.

        Parameters:
            P: Quadratic objective matrix. Must be None or empty — QP is
                not supported by this backend.
            c: Linear objective coefficients.
            A: Constraint matrix (m x n), rows already include variable bounds.
            l: Lower bounds on Ax.
            u: Upper bounds on Ax.
            var_types: Array of 0 (continuous), 1 (integer), 2 (binary).
            **options: time_limit, gap_rel, gap_abs, threads, log_level.

        Returns:
            dict with keys: x, y, obj_val, status, residuals

        Raises:
            NotImplementedError: If P encodes a nonzero quadratic objective.
        """
        import numpy as np

        if P is not None and P.nnz > 0:
            raise NotImplementedError(
                "QP via SCIP: quadratic objectives are not supported by "
                "pyscipopt's Model.setObjective. Use solver='osqp', 'piqp', "
                "'highs', or 'gurobi' for QP problems."
            )

        n_vars = len(c)
        is_lp = bool(np.all(var_types == 0))

        model = self._scip.Model()
        model.hideOutput(quiet=not options.get("log_level", 0))
        if is_lp:
            # Duals (getDualsolLinear) are only retrievable when presolve
            # doesn't transform/delete the original constraints.
            model.setPresolve(self._scip.SCIP_PARAMSETTING.OFF)
            model.setHeuristics(self._scip.SCIP_PARAMSETTING.OFF)
            model.disablePropagation()

        if "time_limit" in options:
            model.setParam("limits/time", float(options["time_limit"]))
        if "gap_rel" in options:
            model.setParam("limits/gap", float(options["gap_rel"]))
        if "gap_abs" in options:
            model.setParam("limits/absgap", float(options["gap_abs"]))
        if "threads" in options and int(options["threads"]) > 1:
            import warnings
            warnings.warn("SCIP runs single-threaded; ignoring threads option")

        scip_vars = []
        for j in range(n_vars):
            vtype = "I" if var_types[j] == 1 else "B" if var_types[j] == 2 else "C"
            v = model.addVar(
                name=f"x{j}", vtype=vtype, lb=None, ub=None,
            )
            scip_vars.append(v)

        # One constraint per row (ranged where needed) so each row maps to
        # a single dual value via getDualsolLinear.
        constrs = []
        A_csr = A.tocsr()
        for i in range(A.shape[0]):
            row = A_csr.getrow(i)
            expr = self._scip.quicksum(
                float(v) * scip_vars[j] for j, v in zip(row.indices, row.data)
            )
            lo, hi = l[i], u[i]
            if lo == hi:
                constrs.append(model.addCons(expr == float(lo)))
            elif np.isfinite(lo) and np.isfinite(hi):
                constrs.append(model.addCons(float(lo) <= (expr <= float(hi))))
            elif np.isfinite(lo):
                constrs.append(model.addCons(expr >= float(lo)))
            elif np.isfinite(hi):
                constrs.append(model.addCons(expr <= float(hi)))
            else:
                constrs.append(None)

        obj_expr = self._scip.quicksum(float(ci) * xi for ci, xi in zip(c, scip_vars))
        model.setObjective(obj_expr, sense="minimize")

        model.optimize()

        status_map = {
            "optimal": "solved",
            "infeasible": "infeasible",
            "unbounded": "unbounded",
            "timelimit": "time_limit",
            "userinterrupt": "interrupt",
        }
        scip_status = model.getStatus()
        status = status_map.get(scip_status, scip_status)

        if model.getNSols() > 0:
            sol = model.getBestSol()
            x = np.array([sol[v] for v in scip_vars])
            obj_val = model.getObjVal()
        else:
            x = np.full(n_vars, np.nan)
            obj_val = float("nan")

        # Duals are only well-defined (and retrievable) for a pure LP
        # solved with presolve/heuristics/propagation disabled above.
        if is_lp and status == "solved":
            y = np.array([
                model.getDualsolLinear(c) if c is not None else 0.0
                for c in constrs
            ])
        else:
            y = np.zeros(A.shape[0])

        return {
            "x": x,
            "y": y,
            "obj_val": float(obj_val),
            "status": status,
            "residuals": {},
        }
