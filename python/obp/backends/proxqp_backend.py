"""ProxQP backend — proximal primal-dual QP solver (Ax = b, l ≤ Cx ≤ u)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import import_local_extension

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_proxqp():
    return import_local_extension("proxqp", __file__)


class ProxQPBackend:
    """ProxQP solver backend (proximal primal-dual algorithm for QP).

    Problem form:
        minimize  ½ xᵀHx + gᵀx
        s.t.      Ax = b
                  l ≤ Cx ≤ u
    """

    def __init__(self) -> None:
        self._proxqp = _import_proxqp()

    def solve(
        self,
        H,  # sp.spmatrix or np.ndarray
        g: np.ndarray,
        A,  # sp.spmatrix or np.ndarray or None
        b: np.ndarray | None,
        C,  # sp.spmatrix or np.ndarray or None
        l: np.ndarray | None,
        u: np.ndarray | None,
        **options,
    ) -> dict:
        import numpy as np

        def to_dense(M):
            """Convert scipy sparse or numpy array to dense numpy."""
            if M is None:
                return None
            if isinstance(M, np.ndarray):
                return M
            return M.toarray()

        H_dense = to_dense(H)
        g_np = np.asarray(g, dtype=np.float64)

        # Build settings
        settings = self._proxqp.Settings()
        for k, v in options.items():
            if hasattr(settings, k):
                setattr(settings, k, v)

        # Create and initialize QP
        n = H_dense.shape[0]
        n_eq = 0 if A is None else A.shape[0]
        n_in = 0 if C is None else C.shape[0]

        qp = self._proxqp.QP(n, n_eq, n_in, settings)

        A_dense = to_dense(A)
        C_dense = to_dense(C)

        qp.init(
            H_dense,
            g_np,
            A_dense,
            np.asarray(b, dtype=np.float64) if b is not None else None,
            C_dense,
            np.asarray(l, dtype=np.float64) if l is not None else None,
            np.asarray(u, dtype=np.float64) if u is not None else None,
        )

        result = qp.solve()

        return {
            "x": np.array(result["x"]),
            "y": np.array(result["y"]),
            "z": np.array(result["z"]),
            "se": np.array(result.get("se", [])),
            "si": np.array(result.get("si", [])),
            "obj_val": float(result["info"]["obj_val"]),
            "status": result["status"],
            "iter": result["info"]["iter"],
            "residuals": {
                "pri_res": float(result["info"]["pri_res"]),
                "dua_res": float(result["info"]["dua_res"]),
                "gap": float(result["info"]["gap"]),
            },
            "info": {
                "iter_ext": result["info"]["iter_ext"],
                "mu_eq": float(result["info"]["mu_eq"]),
                "mu_in": float(result["info"]["mu_in"]),
                "rho": float(result["info"]["rho"]),
                "setup_time": float(result["info"]["setup_time"]),
                "solve_time": float(result["info"]["solve_time"]),
                "run_time": float(result["info"]["run_time"]),
            },
        }
