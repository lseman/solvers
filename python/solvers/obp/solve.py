"""solve — dispatcher and Solution class.

Entry point: ``result = solve(problem, solver="auto", **options)``

Auto-detect logic:
- Quadratic objective + linear constraints → osqp (default QP solver)
- Linear objective + general senses → ipm (LP/SOCP)
- Integer/binary variables, or SOS1/SOS2 constraints → highs (MIP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from .expression import Expression
from .matrices import build_symmetric_P, build_constraints_matrix, split_constraints_for_piqp
from .backends import get_backend, available_solvers
from .sos.expand import expand_sos_constraints

if TYPE_CHECKING:
    from .model import Problem, Variable


@dataclass
class Solution:
    """Result of a solved optimization problem.

    Attributes:
        x: Optimal primal solution ( NumPy array).
        obj_val: Optimal objective value.
        status: Solver status string (``"solved"``, ``"max_iter"``, ...).
        solver: Name of the solver used.
        y: Dual variables (constraint multipliers), if available.
        slack: Slack values, if available.
        residuals: Solver residuals dict.
        problem: The solved :class:`Problem` (for debugging).
    """

    x: np.ndarray
    obj_val: float
    status: str
    solver: str
    problem: Problem
    y: np.ndarray | None = None
    slack: np.ndarray | None = None
    residuals: dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if the solver reports a successful termination."""
        return self.status in ("solved", "optimal", "OPT")

    def __repr__(self) -> str:
        return (
            f"Solution(status={self.status!r}, obj={self.obj_val:.6g}, "
            f"solver={self.solver!r})"
        )


def _detect_solver(
    obj: Expression,
    constraints: list,
    soc_constraints: list,
    variables: list["Variable"],
    sos_constraints: list | None = None,
) -> str:
    """Auto-detect the best solver from problem structure.

    Returns:
        Solver name: "highs", "osqp", "piqp", "proxqp", "ipm".
    """
    # MIP check — HiGHS handles MIP
    has_mip = any(v.vtype in ("integer", "binary") for v in variables)

    # SOCP check
    if soc_constraints:
        return "highs"  # HiGHS supports SOCP via conic

    # SOS1/SOS2 → HiGHS (reformulated as MIP)
    if sos_constraints:
        return "highs"

    # MIP → HiGHS
    if has_mip:
        return "highs"

    # QP vs LP
    if obj.is_quadratic:
        # Both OSQP and PIQP handle QP; default to OSQP for speed
        return "osqp"

    # LP — IPM handles LP with general senses
    return "ipm"


def _expand_soc_for_ipm(
    soc_constraints,
    variables,
    n_vars: int,
) -> tuple[sp.coo_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand SOCP constraints into IPM conic matrix form.

    For each SOC ||x_cone + b||_2 <= c^T x + d, the IPM expects
    the exponential cone representation. For now, we return
    the SOC as a placeholder — the IPM backend needs cone-aware
    assembly to handle this properly.

    Returns:
        (A_cone, b_cone, sense_cone, cone_types, cone_groups)
    """
    import numpy as np

    cone_A_rows: list[int] = []
    cone_A_cols: list[int] = []
    cone_A_vals: list[float] = []
    cone_b: list[float] = []
    cone_sense: list[float] = []

    for soc in soc_constraints:
        # For simplicity, SOCP constraints are handled by the IPM
        # through its cone projection. We append a placeholder row.
        # The actual expansion happens in the C++ IPM solver.
        pass

    A_cone = sp.coo_matrix((0, n_vars))
    b_cone = np.array([], dtype=np.float64)
    sense_cone = np.array([], dtype=np.float64)

    return A_cone, b_cone, sense_cone


def solve(
    problem: Problem,
    solver: str | None = None,
    **options,
) -> Solution:
    """Solve an optimization problem.

    Parameters:
        problem: A populated :class:`Problem`.
        solver: Solver name — ``"osqp"``, ``"piqp"``, ``"ipm"``, ``"highs"``,
            ``"scip"``, ``"gurobi"``, or ``None`` for auto-detect.
        **options: Solver-specific options passed through.

    Returns:
        A :class:`Solution` with the optimal result.

    Raises:
        ValueError: If the problem is malformed, or if SOS constraints are
            used with a solver that doesn't support them.

    Example::

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(x[0]**2 + x[1]**2 + x[2]**2)
        pb.add_constraint(x[0] + x[1] + x[2] >= 1)
        result = solve(pb)
        print(result.x)
    """
    obj, sense, constraints, soc_constraints, variables = problem.assemble()
    sos_constraints = problem._sos_constraints

    # Auto-detect solver
    if solver is None:
        solver = _detect_solver(obj, constraints, soc_constraints, variables, sos_constraints)

    if sos_constraints and solver not in ("highs", "scip", "gurobi"):
        raise ValueError(
            f"SOS1/SOS2 constraints require solver='highs', 'scip', or 'gurobi' "
            f"(got {solver!r})"
        )

    if sos_constraints:
        extra_vars, extra_constraints = expand_sos_constraints(sos_constraints, variables)
        variables = list(variables) + extra_vars
        constraints = list(constraints) + extra_constraints

    n_vars = len(variables)

    # Handle maximize: negate objective (obj_val is un-negated before returning)
    obj_sign = -1.0 if sense == "maximize" else 1.0
    if sense == "maximize":
        obj = obj * (-1)

    # Extract variable bounds and types
    lb = np.array([v.lb for v in variables], dtype=np.float64)
    ub = np.array([v.ub for v in variables], dtype=np.float64)
    var_types = np.array(
        [0 if v.vtype == "continuous" else 1 if v.vtype == "integer" else 2
         for v in variables],
        dtype=np.int64,
    )

    # Extract objective coefficients
    if obj.is_linear:
        obj_rows, obj_cols, obj_vals = obj.to_linear_arrays(n_vars)
        c = np.zeros(n_vars, dtype=np.float64)
        for col, val in zip(obj_cols, obj_vals):
            c[col] = val
        P = None  # No quadratic part
    else:
        # Quadratic objective
        q_rows, q_cols, q_vals = obj.to_linear_arrays(n_vars)
        c = np.zeros(n_vars, dtype=np.float64)
        for col, val in zip(q_cols, q_vals):
            c[col] = val
        P = build_symmetric_P(*obj.to_quadratic_arrays(n_vars), n_vars)

    # Extract constraint matrix
    A_constr, l, u, sense_vec = build_constraints_matrix(constraints, n_vars)

    # Variable bounds are separate from constraint bounds.
    # For OSQP/PIQP: we encode variable bounds as extra rows in the constraint matrix.
    # For IPM: variable bounds are passed directly to the solver.

    # Dispatch to solver
    backend = get_backend(solver)

    if solver == "osqp":
        # OSQP: min 0.5 x^T P x + q^T x, l <= Ax <= u
        if P is None:
            P = sp.coo_matrix((n_vars, n_vars))

        has_bounds = any(v.lb > -np.inf or v.ub < np.inf for v in variables)

        if A_constr.nnz > 0 and has_bounds:
            lb_rows = np.eye(n_vars)
            ub_rows = -np.eye(n_vars)
            A_full = sp.vstack([A_constr.tocsr(), lb_rows, ub_rows]).tocsc()
            l_full = np.concatenate([l, -ub])
            u_full = np.concatenate([u, -lb])
        elif has_bounds:
            lb_rows = np.eye(n_vars)
            ub_rows = -np.eye(n_vars)
            A_full = sp.vstack([lb_rows, ub_rows]).tocsc()
            l_full = -ub
            u_full = -lb
        else:
            A_full = A_constr.tocsc() if A_constr.nnz > 0 else sp.coo_matrix((0, n_vars))
            l_full = np.array([], dtype=np.float64)
            u_full = np.array([], dtype=np.float64)

        result = backend.solve(  # type: ignore[union-attr]
            P.tocsc(), c, A_full.tocsc(), l_full, u_full, **options
        )

        return Solution(
            x=np.array(result["x"]),
            obj_val=obj_sign * float(result["obj_val"]),
            status=result["status"],
            solver=solver,
            problem=problem,
            y=np.array(result.get("y", [])),
            residuals=result.get("residuals", {}),
        )

    elif solver == "piqp":
        # PIQP: min 0.5 x^T P x + q^T x, Ax = b, Gx <= h
        if P is None:
            P = sp.coo_matrix((n_vars, n_vars))

        A_eq, b_eq, G_ineq, h_ineq = split_constraints_for_piqp(
            constraints, n_vars
        )

        result = backend.solve(  # type: ignore[union-attr]
            P.tocsc(), c, A_eq, b_eq, G_ineq, h_ineq, **options
        )

        return Solution(
            x=np.array(result["x"]),
            obj_val=obj_sign * float(result["obj_val"]),
            status=result["status"],
            solver=solver,
            problem=problem,
            y=np.array(result.get("y", [])),
            residuals=result.get("residuals", {}),
        )

    elif solver == "proxqp":
        # ProxQP: min 0.5 x^T Hx + g^T x, Ax = b, l <= Cx <= u
        if P is None:
            H = sp.coo_matrix((n_vars, n_vars))
        else:
            H = P

        # Split constraints into equality (A, b) and inequality (C, l, u)
        A_eq, b_eq, C_ineq, l_ineq, u_ineq = split_constraints_for_proxqp(
            constraints, n_vars
        )

        result = backend.solve(  # type: ignore[union-attr]
            H.tocsc(), c, A_eq, b_eq, C_ineq, l_ineq, u_ineq, **options
        )

        return Solution(
            x=np.array(result["x"]),
            obj_val=obj_sign * float(result["obj_val"]),
            status=result["status"],
            solver=solver,
            problem=problem,
            y=np.array(result.get("y", [])),
            slack=np.array(result.get("si", [])),
            residuals=result.get("residuals", {}),
        )

    elif solver == "ipm":
        # IPM: min c^T x, lb <= x <= ub, sense-based constraints
        if P is not None:
            # QP via IPM: add P to c (gradient of 0.5 x^T P x is Px)
            # Actually, IPM solver handles QP via its internal P matrix
            # For now, pass P through as part of the problem
            # The IPM backend needs to accept P for QP
            # Re-assemble: IPM solve_ipm accepts A, b, c, lb, ub, sense
            # For QP, we need to pass P separately — check if IPM supports it
            raise NotImplementedError(
                "QP via IPM: P matrix support pending. "
                "Use solver='osqp' for QP problems."
            )

        # Linear part only
        A_full = A_constr.tocsr() if A_constr.nnz > 0 else sp.coo_matrix((0, n_vars))

        # Pad sense_vec if no constraints
        if A_full.nnz == 0:
            sense_full = np.array([], dtype=np.float64)
            b_full = np.array([], dtype=np.float64)
        else:
            sense_full = sense_vec
            b_full = np.where(u != np.inf, np.minimum(ub, u), ub)

        result = backend.solve(  # type: ignore[union-attr]
            A_full, b_full, c, lb, ub, sense_full, **options
        )

        return Solution(
            x=np.array(result["x"]),
            obj_val=obj_sign * float(result["obj_val"]),
            status=result["status"],
            solver=solver,
            problem=problem,
            y=np.array(result.get("y", [])),
            residuals=result.get("residuals", {}),
        )

    elif solver in ("highs", "scip", "gurobi"):
        # HiGHS/SCIP/Gurobi: min c^T x s.t. l <= Ax <= u, variable types
        # Variable bounds are encoded as extra constraint rows
        has_lb = any(v.lb > -np.inf for v in variables)
        has_ub = any(v.ub < np.inf for v in variables)

        if A_constr.nnz > 0 and (has_lb or has_ub):
            rows_list = [A_constr.tocsr()]
            l_parts = list(l)
            u_parts = list(u)

            if has_lb:
                # lb <= x  ->  x >= lb  ->  lower=lb, upper=inf
                rows_list.append(sp.csr_matrix(np.eye(n_vars)))
                l_parts.extend(lb.tolist())
                u_parts.extend([np.inf] * n_vars)

            if has_ub:
                # x <= ub  ->  -x >= -ub  ->  lower=-ub, upper=inf
                rows_list.append(sp.csr_matrix(-np.eye(n_vars)))
                l_parts.extend((-ub).tolist())
                u_parts.extend([np.inf] * n_vars)

            A_full = sp.vstack(rows_list).tocsc()
            l_full = np.array(l_parts, dtype=np.float64)
            u_full = np.array(u_parts, dtype=np.float64)
        elif has_lb or has_ub:
            rows_list = []
            l_parts = []
            u_parts = []

            if has_lb:
                rows_list.append(sp.csr_matrix(np.eye(n_vars)))
                l_parts.extend(lb.tolist())
                u_parts.extend([np.inf] * n_vars)

            if has_ub:
                rows_list.append(sp.csr_matrix(-np.eye(n_vars)))
                l_parts.extend((-ub).tolist())
                u_parts.extend([np.inf] * n_vars)

            A_full = sp.vstack(rows_list).tocsc()
            l_full = np.array(l_parts, dtype=np.float64)
            u_full = np.array(u_parts, dtype=np.float64)
        else:
            A_full = A_constr.tocsc() if A_constr.nnz > 0 else sp.coo_matrix((0, n_vars))
            l_full = np.array([], dtype=np.float64)
            u_full = np.array([], dtype=np.float64)

        # HiGHS P matrix (quadratic objective)
        P_full: sp.spmatrix | None = None
        if P is not None and P.nnz > 0:
            P_full = P

        result = backend.solve(  # type: ignore[union-attr]
            P_full, c, A_full, l_full, u_full, var_types, **options
        )

        x_full = np.array(result["x"])
        x_out = x_full[: problem.n_vars] if sos_constraints else x_full

        return Solution(
            x=x_out,
            obj_val=obj_sign * float(result["obj_val"]),
            status=result["status"],
            solver=solver,
            problem=problem,
            y=np.array(result.get("y", [])),
            residuals=result.get("residuals", {}),
        )

    else:
        raise ValueError(f"Unknown solver: {solver}")
