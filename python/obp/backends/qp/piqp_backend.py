"""PIQP backend — interior-point QP solver (Ax = b, Gx <= h)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._common import import_local_extension

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

        # The compiled extension takes dense ndarrays.
        P_dense = np.asarray(P.toarray() if hasattr(P, "toarray") else P, dtype=np.float64)
        q_np = np.asarray(q, dtype=np.float64)
        A_dense = np.asarray(A.toarray(), dtype=np.float64) if A is not None else None
        b_np = np.asarray(b, dtype=np.float64) if b is not None else None
        G_dense = np.asarray(G.toarray(), dtype=np.float64) if G is not None else None
        h_np = np.asarray(h, dtype=np.float64) if h is not None else None

        settings = self._piqp.PIQPSettings()
        for k, v in options.items():
            if hasattr(settings, k):
                setattr(settings, k, v)

        result = self._piqp.solve(P_dense, q_np, A_dense, b_np, G_dense, h_np, settings)

        return {
            "x": np.asarray(result["x"]),
            "y": np.asarray(result["y"]),
            "obj_val": result["obj_val"],
            "status": result["status"],
            "iter": result.get("iterations", 0),
            "residuals": result.get("residuals", {}),
        }
