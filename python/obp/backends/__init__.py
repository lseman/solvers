"""backends — thin wrappers around the solver backends.

Grouped by kind:
- ots (off-the-shelf): gurobi, highs, scip — general MIP/LP/QP solvers
- qp: osqp, piqp, proxqp — dedicated QP solvers
- ip: ipm_solver — conic interior-point (sense-based: =, >=, <=)

Each backend implements ``solve(problem_data, **options) -> dict``
with a consistent interface.
"""

from __future__ import annotations

from typing import Protocol

from .ip.ipm_backend import IPMBackend
from .ots.gurobi_backend import GurobiBackend
from .ots.highs_backend import HiGHSBackend
from .ots.scip_backend import SCIPBackend
from .qp.osqp_backend import OSQPBackend
from .qp.piqp_backend import PIQPBackend
from .qp.proxqp_backend import ProxQPBackend


class _BackendProtocol(Protocol):
    """Protocol for solver backends."""

    def solve(self, *args, **kwargs) -> dict:
        """Solve the problem and return results dict."""
        ...


# Backend registry (lazy initialization)
_BACKEND_CACHE: dict[str, _BackendProtocol] = {}
_BACKEND_CLASSES: dict[str, type] = {
    "osqp": OSQPBackend,
    "piqp": PIQPBackend,
    "proxqp": ProxQPBackend,
    "ipm": IPMBackend,
    "highs": HiGHSBackend,
    "scip": SCIPBackend,
    "gurobi": GurobiBackend,
}


def get_backend(name: str) -> _BackendProtocol:
    """Return the named backend, creating it lazily. Raises if unavailable."""
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    cls = _BACKEND_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(_BACKEND_CLASSES.keys())}"
        )
    try:
        backend: _BackendProtocol = cls()
        _BACKEND_CACHE[name] = backend
        return backend
    except ImportError as e:
        raise RuntimeError(
            f"Solver '{name}' not available — "
            f"build the C++ extensions (cmake -B build && make -C build) "
            f"or pip-install the solver package. "
            f"Details: {e}"
        ) from e


def available_solvers() -> list[str]:
    """Return names of all registered backend classes."""
    return list(_BACKEND_CLASSES.keys())


__all__ = [
    "OSQPBackend",
    "PIQPBackend",
    "ProxQPBackend",
    "IPMBackend",
    "HiGHSBackend",
    "SCIPBackend",
    "GurobiBackend",
    "get_backend",
    "available_solvers",
]
