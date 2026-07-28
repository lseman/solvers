"""logic — AND/OR/NOT reformulations for binary variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..expression import Expression

if TYPE_CHECKING:
    from ..model import Problem, Variable


def _require_binary(variable: "Variable", label: str) -> None:
    if variable.vtype != "binary":
        raise ValueError(
            f"{label} must be a binary variable (got vtype={variable.vtype!r})"
        )


def _sum_vars(vars_: list) -> Expression:
    expr = Expression.constant(0.0)
    for v in vars_:
        expr = expr + Expression.from_variable(v)
    return expr


def logical_not(b: "Variable") -> Expression:
    """Return ``NOT b`` as ``1 - b`` -- an Expression, no new variable needed.

    Usable directly in constraints/objectives (e.g. ``x <= 5 * logical_not(b)``).
    """
    _require_binary(b, "b")
    return Expression.constant(1.0) - Expression.from_variable(b)


def logical_and(
    problem: "Problem",
    terms: list["Variable"],
    name: str = "",
) -> "Variable":
    """Create a binary equal to ``AND(terms)`` (1 iff every term is 1).

    ``z <= b_i`` for every term (z can't be 1 unless all terms are) and
    ``z >= sum(b_i) - (n-1)`` (z must be 1 once every term is), giving an
    exact reformulation.

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        terms: Binary variables to AND together, length >= 2.
        name: Optional name prefix.

    Returns:
        The new binary :class:`Variable`.
    """
    if len(terms) < 2:
        raise ValueError(f"logical_and needs at least 2 terms, got {len(terms)}")
    for i, t in enumerate(terms):
        _require_binary(t, f"terms[{i}]")

    prefix = name or f"_and_{'_'.join(str(t.index) for t in terms)}"
    z = problem.add_variables(prefix, vtype="binary")
    zx = Expression.from_variable(z)

    for i, t in enumerate(terms):
        problem.add_constraint(zx - Expression.from_variable(t) <= 0.0, f"{prefix}_le{i}")
    n = len(terms)
    problem.add_constraint(zx - _sum_vars(terms) >= -(n - 1), f"{prefix}_ge")
    return z


def logical_or(
    problem: "Problem",
    terms: list["Variable"],
    name: str = "",
) -> "Variable":
    """Create a binary equal to ``OR(terms)`` (1 iff at least one term is 1).

    ``z >= b_i`` for every term (z must be 1 if any term is) and
    ``z <= sum(b_i)`` (z can't be 1 unless at least one term is), giving
    an exact reformulation.

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        terms: Binary variables to OR together, length >= 2.
        name: Optional name prefix.

    Returns:
        The new binary :class:`Variable`.
    """
    if len(terms) < 2:
        raise ValueError(f"logical_or needs at least 2 terms, got {len(terms)}")
    for i, t in enumerate(terms):
        _require_binary(t, f"terms[{i}]")

    prefix = name or f"_or_{'_'.join(str(t.index) for t in terms)}"
    z = problem.add_variables(prefix, vtype="binary")
    zx = Expression.from_variable(z)

    for i, t in enumerate(terms):
        problem.add_constraint(zx - Expression.from_variable(t) >= 0.0, f"{prefix}_ge{i}")
    problem.add_constraint(zx - _sum_vars(terms) <= 0.0, f"{prefix}_le")
    return z
