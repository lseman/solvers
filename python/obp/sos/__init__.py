"""sos — special-ordered-set (SOS1/SOS2) constraints and their MIP reformulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..model import Variable

from .expand import expand_sos_constraints

__all__ = ["SOSConstraint", "expand_sos_constraints"]


@dataclass
class SOSConstraint:
    """A special-ordered-set constraint (SOS1 or SOS2).

    SOS1: at most one variable in the set may be nonzero.
    SOS2: at most two variables may be nonzero, and if two are nonzero
    they must be consecutive in the order given by *weights*.

    Reformulated as a MIP (binary indicators) and solved via HiGHS, SCIP,
    or Gurobi; not supported by any other backend. Each variable must have
    finite lb and ub for the reformulation's big-M bounds to hold.

    Parameters:
        variables: Variables in the set, ordered.
        type: 1 (SOS1) or 2 (SOS2).
        weights: Optional ordering weights (defaults to 0..n-1); only
            the induced order matters, used for SOS2 adjacency.
        method: Reformulation to use for SOS2 (ignored for SOS1, which
            always uses the direct per-variable indicator encoding). Names
            match PiecewiseLinearOpt.jl's SOS2 formulations:

            - ``"cc"`` (default): convex-combination encoding — ``k-1``
              unary window indicators, each x_i bounded directly by the
              windows touching it. Simplest, most robust.
            - ``"dlog"``: disaggregated logarithmic — per-segment
              disaggregated share variables (as in ``"cc"``-style
              disaggregation), but the active segment is selected via
              ``ceil(log2(k-1))`` binaries instead of ``k-1`` unary ones.
            - ``"log"``: logarithmic (Vielma-Nemhauser) encoding using
              reflected Gray codes — only ``ceil(log2(k-1))`` binaries.
        name: Optional label.
    """

    variables: list["Variable"]
    type: Literal[1, 2]
    weights: list[float] | None = None
    method: Literal["cc", "dlog", "log"] = "cc"
    name: str = ""

    def __post_init__(self) -> None:
        if self.type not in (1, 2):
            raise ValueError(f"SOS type must be 1 or 2, got {self.type}")
        if len(self.variables) < 2:
            raise ValueError("SOS constraint requires at least 2 variables")
        if self.weights is not None and len(self.weights) != len(self.variables):
            raise ValueError("weights must match variables length")
        if self.method not in ("cc", "dlog", "log"):
            raise ValueError(
                f"SOS method must be 'cc', 'dlog', or 'log', got {self.method!r}"
            )
        for v in self.variables:
            if v.lb == -float("inf") or v.ub == float("inf"):
                raise ValueError(
                    f"SOS constraint requires finite bounds on all variables "
                    f"(variable {v.name or v.index!r} has lb={v.lb}, ub={v.ub})"
                )
