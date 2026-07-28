"""ots — off-the-shelf solver backends (Gurobi, HiGHS, SCIP)."""

from __future__ import annotations

from .gurobi_backend import GurobiBackend
from .highs_backend import HiGHSBackend
from .scip_backend import SCIPBackend

__all__ = ["GurobiBackend", "HiGHSBackend", "SCIPBackend"]
