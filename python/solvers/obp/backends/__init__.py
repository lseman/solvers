"""backends — thin wrappers around the solver backends.

Wraps:
- osqp:  sparse ADMM QP solver (l <= Ax <= u)
- piqp:  interior-point QP solver (Ax = b, Gx <= h)
- ipm_solver: conic IPM (sense-based: =, >=, <=)
- highs: MIP/LP/QP solver (HiGHS, MIT licensed)
- scip: MIP/LP solver via pyscipopt
- gurobi: MIP/LP/QP solver via gurobipy (requires a license)

Each backend implements ``solve(problem_data, **options) -> dict``
with a consistent interface.
"""

from __future__ import annotations

from .gurobi_backend import GurobiBackend
from .highs_backend import HiGHSBackend
from .ipm_backend import IPMBackend
from .osqp_backend import OSQPBackend
from .piqp_backend import PIQPBackend
from .proxqp_backend import ProxQPBackend
from .scip_backend import SCIPBackend

# Backend registry (lazy initialization)
_BACKEND_CACHE: dict[str, object] = {}
_BACKEND_CLASSES: dict[str, type] = {
    "osqp": OSQPBackend,
    "piqp": PIQPBackend,
    "proxqp": ProxQPBackend,
    "ipm": IPMBackend,
    "highs": HiGHSBackend,
    "scip": SCIPBackend,
    "gurobi": GurobiBackend,
}


def get_backend(name: str) -> object:
    """Return the named backend, creating it lazily. Raises if unavailable."""
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    cls = _BACKEND_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(_BACKEND_CLASSES.keys())}"
        )
    try:
        backend = cls()
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
