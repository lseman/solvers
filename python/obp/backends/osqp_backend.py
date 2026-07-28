"""OSQP backend — sparse ADMM QP solver (l <= Ax <= u)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import import_local_extension

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_osqp():
    return import_local_extension("osqp", __file__)


class OSQPBackend:
    """OSQP solver backend (first-order ADMM for QP)."""

    def __init__(self) -> None:
        self._osqp = _import_osqp()

    def solve(
        self,
        P: sp.spmatrix,
        q: np.ndarray,
        A: sp.spmatrix,
        l: np.ndarray,
        u: np.ndarray,
        **options,
    ) -> dict:
        import numpy as np

        # The compiled extension takes dense ndarrays.
        P_dense = np.asarray(P.toarray() if hasattr(P, "toarray") else P, dtype=np.float64)
        A_dense = np.asarray(A.toarray() if hasattr(A, "toarray") else A, dtype=np.float64)
        q_np = np.asarray(q, dtype=np.float64)
        l_np = np.asarray(l, dtype=np.float64)
        u_np = np.asarray(u, dtype=np.float64)

        settings = self._osqp.Settings()
        for k, v in options.items():
            if hasattr(settings, k):
                setattr(settings, k, v)

        result = self._osqp.solve(P_dense, q_np, A_dense, l_np, u_np, settings)

        return {
            "x": np.asarray(result["x"]),
            "y": np.asarray(result["y"]),
            "obj_val": result["obj_val"],
            "status": result["status"],
            "iter": result.get("iters", 0),
            "residuals": result.get("residuals", {}),
        }
