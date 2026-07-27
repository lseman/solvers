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

        P_csc = P.tocsc()
        A_csc = A.tocsc()
        q_np = np.asarray(q, dtype=np.float64)
        l_np = np.asarray(l, dtype=np.float64)
        u_np = np.asarray(u, dtype=np.float64)

        # Build OSQP settings
        settings = self._osqp.settings()
        for k, v in options.items():
            if hasattr(settings, k):
                setattr(settings, k, v)

        # Setup and solve
        prob = self._osqp.Problem(P_csc, q_np, A_csc, l_np, u_np)
        prob.setup(verbose=False, **{k: v for k, v in options.items()
                                     if k not in ("eps_abs", "eps_rel",
                                                   "eps_pinf", "eps_dinf",
                                                   "max_iter", "rho",
                                                   "adaptive_rho")})
        # Pass tolerance options separately
        for key in ("eps_abs", "eps_rel", "eps_pinf", "eps_dinf", "max_iter"):
            if key in options:
                prob.settings()[key] = options[key]

        result = prob.solve()

        return {
            "x": result.x,
            "y": result.y,
            "obj_val": result.info.obj_val,
            "status": result.info.status,
            "status_val": result.info.statusVal,
            "iter": result.info.iter,
            "residuals": {
                "pri_inf": result.info.pri_res,
                "dua_inf": result.info.dua_res,
            },
        }
