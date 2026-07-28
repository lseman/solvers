"""Exact linearizations for products involving binary variables."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Problem, Variable


def linearize_binary_product(
    problem: "Problem",
    x: "Variable",
    y: "Variable",
    name: str = "",
) -> "Variable":
    """Create an auxiliary variable equal to the product of two binaries.

    The returned continuous variable ``z`` is exact at every feasible MIP
    solution through ``z <= x``, ``z <= y``, and ``z >= x + y - 1``.
    """
    _require_binary(x, "x")
    _require_binary(y, "y")

    prefix = name or f"_binary_product_{x.index}_{y.index}"
    z = problem.add_variables(prefix, lb=0.0, ub=1.0)
    problem.add_constraint(z - x <= 0.0, f"{prefix}_le_x")
    problem.add_constraint(z - y <= 0.0, f"{prefix}_le_y")
    problem.add_constraint(z - x - y >= -1.0, f"{prefix}_ge_sum_minus_one")
    return z


def linearize_binary_continuous_product(
    problem: "Problem",
    binary: "Variable",
    continuous: "Variable",
    name: str = "",
) -> "Variable":
    """Create an auxiliary variable equal to ``binary * continuous``.

    ``continuous`` must have finite lower and upper bounds ``L`` and ``U``.
    The four standard inequalities produce the exact convex-hull
    formulation, including when the bounds are negative or cross zero.
    """
    _require_binary(binary, "binary")
    if continuous.vtype != "continuous":
        raise ValueError(
            "linearize_binary_continuous_product requires a continuous "
            f"second factor (got vtype={continuous.vtype!r})"
        )
    if not math.isfinite(continuous.lb) or not math.isfinite(continuous.ub):
        raise ValueError(
            "linearize_binary_continuous_product requires finite bounds on "
            f"{continuous.name or continuous.index!r}"
        )

    lower = float(continuous.lb)
    upper = float(continuous.ub)
    prefix = name or f"_binary_continuous_product_{binary.index}_{continuous.index}"
    z = problem.add_variables(
        prefix,
        lb=min(0.0, lower),
        ub=max(0.0, upper),
    )

    problem.add_constraint(z - upper * binary <= 0.0, f"{prefix}_upper_on")
    problem.add_constraint(z - lower * binary >= 0.0, f"{prefix}_lower_on")
    problem.add_constraint(
        z - continuous - lower * binary <= -lower,
        f"{prefix}_upper_off",
    )
    problem.add_constraint(
        z - continuous - upper * binary >= -upper,
        f"{prefix}_lower_off",
    )
    return z


def _require_binary(variable: "Variable", label: str) -> None:
    if variable.vtype != "binary":
        raise ValueError(
            f"{label} must be a binary variable (got vtype={variable.vtype!r})"
        )
