"""PIQP backend — interior-point QP solver (Ax = b, Gx <= h)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import import_local_extension

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


def _import_piqp():
    return import_local_extension("piqp", __file__)


class PIQPBackend:
    """PIQP solver backend (interior-point for QP)."""

    def __init__(self) -> None:
        self._piqp = _import_piqp()

    def solve(
        self,
        P: sp.spmatrix,
        q: np.ndarray,
        A: sp.spmatrix | None,
        b: np.ndarray | None,
        G: sp.spmatrix | None,
        h: np.ndarray | None,
        **options,
    ) -> dict:
        import numpy as np

        P_csc = P.tocsc()
        solver = self._piqp.SparseQP()

        # Setup problem
        solver.setup(
            P_csc,
            np.asarray(q, dtype=np.float64),
            A.tocsc() if A is not None else None,
            np.asarray(b, dtype=np.float64) if b is not None else None,
            G.tocsc() if G is not None else None,
            np.asarray(h, dtype=np.float64) if h is not None else None,
        )

        # Apply settings
        settings = self._piqp.PIQPSettings()
        for k, v in options.items():
            if hasattr(settings, k):
                setattr(settings, k, v)
        solver.setSettings(settings)

        solver.solve()

        return {
            "x": solver.x(),
            "y": solver.y(),
            "obj_val": float(solver.objValue()),
            "status": solver.status(),
            "iter": solver.iterations(),
            "residuals": {
                "eq_inf": float(solver.residuals().eq_inf),
                "ineq_inf": float(solver.residuals().ineq_inf),
                "stat_inf": float(solver.residuals().stat_inf),
                "gap": float(solver.residuals().gap),
            },
        }
