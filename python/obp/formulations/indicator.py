"""indicator — big-M indicator constraints and exact absolute value.

``b == 1 => (expr <= bound)`` (and the ``>=``/``==`` variants) reformulated
as a single big-M inequality per direction, using the tightest M derivable
from the expression's own variable bounds (not an arbitrary large constant).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..expression import Expression

if TYPE_CHECKING:
    from ..model import Constraint, Problem, Variable


def _expr_bounds(expr: Expression) -> tuple[float, float]:
    """Return (lo, hi) — the tightest interval implied by expr's own
    variable bounds, i.e. min/max of expr over the variables' box.
    """
    lo = expr.resolved_constant
    hi = expr.resolved_constant
    for v, c in expr.linear_coeffs.items():
        if not math.isfinite(v.lb) or not math.isfinite(v.ub):
            raise ValueError(
                f"indicator constraint requires finite bounds on every "
                f"variable in the expression (got {v.name or v.index!r} "
                f"with lb={v.lb}, ub={v.ub})"
            )
        lo += c * v.lb if c >= 0 else c * v.ub
        hi += c * v.ub if c >= 0 else c * v.lb
    return lo, hi


def add_indicator_constraint(
    problem: "Problem",
    binary: "Variable",
    body: Expression,
    sense: str,
    bound: float,
    name: str = "",
    activate_on: int = 1,
) -> list["Constraint"]:
    """Enforce ``binary == activate_on  =>  (body sense bound)`` via big-M.

    Only the implied direction is enforced -- when the indicator is off,
    ``body`` is unconstrained by this call. All variables referenced in
    ``body`` must have finite bounds (used to derive the tightest big-M).

    Parameters:
        problem: The :class:`Problem` to add constraints to.
        binary: The indicator variable (must be binary).
        body: Left-hand side expression.
        sense: ``"<="``, ``">="``, or ``"=="``.
        bound: Right-hand side scalar.
        name: Optional name prefix.
        activate_on: ``1`` (default) to trigger when ``binary == 1``, or
            ``0`` to trigger when ``binary == 0``.

    Returns:
        The list of added :class:`Constraint` objects.

    Raises:
        ValueError: If ``binary`` isn't binary, ``sense`` is invalid, or
            any variable in ``body`` has an infinite bound.
    """
    from ..model import Constraint

    if binary.vtype != "binary":
        raise ValueError(f"binary must be a binary variable (got vtype={binary.vtype!r})")
    if sense not in ("<=", ">=", "=="):
        raise ValueError(f"sense must be '<=', '>=', or '==' (got {sense!r})")
    if activate_on not in (0, 1):
        raise ValueError(f"activate_on must be 0 or 1 (got {activate_on!r})")

    lo, hi = _expr_bounds(body)
    prefix = name or f"_indicator_b{binary.index}"
    added: list["Constraint"] = []

    # "off" is the complement of the triggering value: how far binary is
    # from activate_on, scaled into {0, 1}. off == 0 when triggered (the
    # big-M term vanishes); off == 1 when not triggered (constraint slack).
    bx = Expression.from_variable(binary)
    off = (Expression.constant(1.0) - bx) if activate_on == 1 else bx

    if sense in ("<=", "=="):
        M = hi - bound
        if M > 0:
            # body <= bound + M*off  <=>  body - M*off <= bound
            added.append(
                problem.add_constraint(body - M * off <= bound, f"{prefix}_le")
            )
    if sense in (">=", "=="):
        M = bound - lo
        if M > 0:
            # body >= bound - M*off  <=>  body + M*off >= bound
            added.append(
                problem.add_constraint(body + M * off >= bound, f"{prefix}_ge")
            )

    return added


def abs_value(
    problem: "Problem",
    x: "Variable",
    name: str = "",
) -> "Variable":
    """Create an auxiliary variable equal to ``|x|``.

    ``x`` must have finite bounds. ``z >= x`` and ``z >= -x`` always hold;
    a binary selects which side is tight (``z <= x + M1*(1-b)`` and
    ``z <= -x + M2*b``), so ``z == |x|`` exactly at every feasible MIP
    solution -- not just a lower-bounding relaxation. ``M1``/``M2`` are
    the tightest big-Ms consistent with ``z``'s and ``x``'s own bounds
    (``z``'s upper bound minus ``x``'s lower/negated-upper bound).
    """
    if not math.isfinite(x.lb) or not math.isfinite(x.ub):
        raise ValueError(
            f"abs_value requires finite bounds on x (got {x.name or x.index!r} "
            f"with lb={x.lb}, ub={x.ub})"
        )

    prefix = name or f"_abs_x{x.index}"
    z_ub = max(abs(x.lb), abs(x.ub))
    z = problem.add_variables(prefix, lb=0.0, ub=z_ub)
    b = problem.add_variables(f"{prefix}_sign", vtype="binary")

    zx = Expression.from_variable(z)
    xx = Expression.from_variable(x)

    M1 = z_ub - x.lb  # bounds (z - x) over the box
    M2 = z_ub + x.ub  # bounds (z + x) over the box

    problem.add_constraint(zx - xx >= 0.0, f"{prefix}_ge_x")
    problem.add_constraint(zx + xx >= 0.0, f"{prefix}_ge_negx")
    # b == 1 selects the x >= 0 branch (z == x); b == 0 selects x <= 0 (z == -x).
    # z - x <= M1*(1-b)  <=>  z - x + M1*b <= M1
    problem.add_constraint(zx - xx + M1 * b <= M1, f"{prefix}_le_x_on")
    # z + x <= M2*b  <=>  z + x - M2*b <= 0
    problem.add_constraint(zx + xx - M2 * b <= 0.0, f"{prefix}_le_negx_on")

    return z
