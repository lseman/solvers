"""qp — dedicated QP solver backends (OSQP, PIQP, ProxQP)."""

from __future__ import annotations

from .osqp_backend import OSQPBackend
from .piqp_backend import PIQPBackend
from .proxqp_backend import ProxQPBackend

__all__ = ["OSQPBackend", "PIQPBackend", "ProxQPBackend"]
