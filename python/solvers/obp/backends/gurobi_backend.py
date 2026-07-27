"""Gurobi backend — LP, QP, and MIP solver via gurobipy (requires a license)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_gurobi():
    import gurobipy as _gurobi
    return _gurobi


class GurobiBackend:
    """Gurobi solver backend (via gurobipy) — LP, QP, and MIP solver.

    Solves: min c^T x s.t. l <= Ax <= u, variable_type in {continuous, integer, binary}
    Requires a valid Gurobi license.
    """

    def __init__(self) -> None:
        self._gp = _import_gurobi()

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
            A: Constraint matrix (m x n), rows already include variable bounds.
            l: Lower bounds on Ax.
            u: Upper bounds on Ax.
            var_types: Array of 0 (continuous), 1 (integer), 2 (binary).
            **options: time_limit, gap_rel, threads, log_level.

        Returns:
            dict with keys: x, y, obj_val, status, iter, residuals
        """
        import numpy as np

        gp = self._gp
        model = gp.Model()
        if not options.get("log_level", 0):
            model.setParam("OutputFlag", 0)
        if "time_limit" in options:
            model.setParam("TimeLimit", float(options["time_limit"]))
        if "gap_rel" in options:
            model.setParam("MIPGap", float(options["gap_rel"]))
        if "threads" in options:
            model.setParam("Threads", int(options["threads"]))

        n_vars = len(c)
        vtype_map = {0: gp.GRB.CONTINUOUS, 1: gp.GRB.INTEGER, 2: gp.GRB.BINARY}
        gvars = model.addMVar(
            n_vars,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
            vtype=[vtype_map[int(t)] for t in var_types],
        )

        if A.shape[0] > 0:
            A_csr = A.tocsr()
            model.addMConstr(A_csr, gvars, "<=", u)
            model.addMConstr(A_csr, gvars, ">=", l)

        if P is not None and P.nnz > 0:
            model.setObjective(0.5 * gvars @ P.tocsr() @ gvars + c @ gvars, gp.GRB.MINIMIZE)
        else:
            model.setObjective(c @ gvars, gp.GRB.MINIMIZE)

        model.optimize()

        GRB = gp.GRB
        status_map = {
            GRB.OPTIMAL: "solved",
            GRB.INFEASIBLE: "infeasible",
            GRB.UNBOUNDED: "unbounded",
            GRB.TIME_LIMIT: "time_limit",
            GRB.INTERRUPTED: "interrupt",
        }
        status = status_map.get(model.Status, str(model.Status))

        if model.SolCount >= 1:
            x = np.array(gvars.X)
            obj_val = model.ObjVal
        else:
            x = np.full(n_vars, np.nan)
            obj_val = float("nan")

        return {
            "x": x,
            "y": np.zeros(A.shape[0]),
            "obj_val": float(obj_val),
            "status": status,
            "residuals": {},
        }
