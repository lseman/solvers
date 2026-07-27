"""IPM backend — conic interior-point solver (LP, QP*, SOCP, sense-based: =, >=, <=)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import import_local_extension

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_ipm():
    return import_local_extension("ipm_solver", __file__)


class IPMBackend:
    """IPM solver backend (conic interior-point for LP/QP/SOCP)."""

    def __init__(self) -> None:
        self._ipm = _import_ipm()

    def solve(
        self,
        A: sp.spmatrix,
        b: np.ndarray,
        c: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        sense: np.ndarray,
        **options,
    ) -> dict:
        import numpy as np

        A_dense = A.to_dense() if hasattr(A, "to_dense") else A.toarray()
        b_np = np.asarray(b, dtype=np.float64)
        c_np = np.asarray(c, dtype=np.float64)
        lb_np = np.asarray(lb, dtype=np.float64)
        ub_np = np.asarray(ub, dtype=np.float64)
        sense_np = np.asarray(sense, dtype=np.float64)

        tol = options.get("tol", 1e-8)

        # Use the standalone solve_ipm function for simplicity
        result = self._ipm.solve_ipm(
            A_dense, b_np, c_np, lb_np, ub_np, sense_np, tol
        )

        return {
            "x": np.array(result.primals),
            "y": np.array(result.duals),
            "obj_val": float(result.objective),
            "status": result.status,
            "residuals": {},
        }
