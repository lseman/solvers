"""export — write a Problem to standard .lp / .mps files via HiGHS.

Usage::

    from obp import Problem
    pb = Problem("my_model")
    ...
    pb.write_lp("model.lp")
    pb.write_mps("model.mps")

Uses HiGHS's model writer so the emitted file matches exactly what the
``highs``/``gurobi``/``scip`` backends would solve. SOS1/SOS2 constraints
are expanded to their MIP reformulation before writing (no native SOS
support in highspy's writer); SOC constraints cannot be represented in
LP/MPS (a linear/MIP-only format) and raise an error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .matrices import build_constraints_matrix, build_symmetric_P
from .sos.expand import expand_sos_constraints

if TYPE_CHECKING:
    from .model import Problem


def _write_via_highs(problem: "Problem", path: str) -> None:
    import numpy as np

    obj, sense, constraints, soc_constraints, variables = problem.assemble()
    sos_constraints = problem._sos_constraints

    if soc_constraints:
        raise ValueError(
            "Cannot export to LP/MPS: problem has second-order cone "
            "constraints, which are not representable in the LP/MPS "
            "linear/MIP format."
        )

    if sos_constraints:
        extra_vars, extra_constraints = expand_sos_constraints(sos_constraints, variables)
        variables = list(variables) + extra_vars
        constraints = list(constraints) + extra_constraints

    n_vars = len(variables)

    lb = np.array([v.lb for v in variables], dtype=np.float64)
    ub = np.array([v.ub for v in variables], dtype=np.float64)
    var_types = np.array(
        [0 if v.vtype == "continuous" else 1 if v.vtype == "integer" else 2
         for v in variables],
        dtype=np.int64,
    )

    if obj.is_linear:
        obj_cols, obj_vals = obj.to_linear_arrays(n_vars)[1:]
        c = np.zeros(n_vars, dtype=np.float64)
        for col, val in zip(obj_cols, obj_vals):
            c[col] = val
        P = None
    else:
        q_cols, q_vals = obj.to_linear_arrays(n_vars)[1:]
        c = np.zeros(n_vars, dtype=np.float64)
        for col, val in zip(q_cols, q_vals):
            c[col] = val
        P = build_symmetric_P(*obj.to_quadratic_arrays(n_vars), n_vars)

    A_constr, l, u, _sense_vec, row_constraints = build_constraints_matrix(constraints, n_vars)

    import highspy

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)

    if sense == "maximize":
        highs.changeObjectiveSense(highspy.ObjSense.kMaximize)

    if n_vars > 0:
        highs.addCols(
            n_vars,
            c.astype(np.float64),
            lb,
            ub,
            0,
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    if A_constr.nnz > 0 or A_constr.shape[0] > 0:
        A_csr = A_constr.tocsr()
        highs.addRows(
            A_csr.shape[0],
            l.astype(np.float64),
            u.astype(np.float64),
            A_csr.nnz,
            A_csr.indptr.astype(np.int32),
            A_csr.indices.astype(np.int32),
            A_csr.data.astype(np.float64),
        )

    if n_vars > 0:
        vtype_indices = list(range(n_vars))
        vtype_values = [
            highspy.HighsVarType.kContinuous if vt == 0
            else highspy.HighsVarType.kInteger
            for vt in var_types
        ]
        highs.changeColsIntegrality(n_vars, vtype_indices, vtype_values)

    if P is not None and P.nnz > 0:
        P_csc = P.tocsc()
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

        hessian = highspy.HighsHessian()
        hessian.dim_ = n_vars
        hessian.format_ = highspy.HessianFormat.kTriangular
        hessian.start_ = starts
        hessian.index_ = indices
        hessian.value_ = values
        highs.passHessian(hessian)

    for j, v in enumerate(variables):
        if v.name:
            highs.passColName(j, v.name)

    for i, c in enumerate(row_constraints):
        if c.name:
            highs.passRowName(i, c.name)

    status = highs.writeModel(str(path))
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS writeModel({path!r}) failed with status {status}")


def write_lp(problem: "Problem", path: str) -> None:
    """Write *problem* to a CPLEX LP-format file at *path*."""
    _write_via_highs(problem, path)


def write_mps(problem: "Problem", path: str) -> None:
    """Write *problem* to a free-format MPS file at *path*."""
    _write_via_highs(problem, path)
