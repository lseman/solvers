"""bigm — big-M reformulations for max/min constraints.

z = max(x_1, ..., x_k) and z = min(x_1, ..., x_k) are not linear, but can
be exactly reformulated as a MIP using one binary indicator per term and
a big-M per term derived from the terms' own bounds (the tightest valid
big-M, not an arbitrary large constant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..expression import Expression

if TYPE_CHECKING:
    from ..model import Constraint, Problem, Variable


def _sum_vars(vars_) -> Expression:
    expr = Expression.constant(0.0)
    for v in vars_:
        expr = expr + Expression.from_variable(v)
    return expr


def add_max_constraint(
    problem: "Problem",
    z: "Variable",
    terms: list["Variable"],
    name: str = "",
) -> list["Constraint"]:
    """Enforce ``z == max(terms)`` via a big-M MIP reformulation.

    For each term x_i: ``z >= x_i`` always holds (no big-M needed since
    the max is never less than any term). One binary b_i per term selects
    which term attains the max (``sum(b_i) == 1``), and
    ``z <= x_i + M_i*(1 - b_i)`` where ``M_i = z.ub - x_i.lb`` is the
    tightest big-M consistent with the variables' own bounds — when
    b_i == 0 the constraint is slack by at most that amount; when
    b_i == 1 it forces ``z <= x_i``, which combined with ``z >= x_i``
    pins ``z == x_i`` for the selected term.

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        z: The variable constrained to equal the max. Must have finite ub.
        terms: The variables being maxed over, length >= 2. Each must
            have a finite lb.
        name: Optional name prefix (default derived from z's index).

    Returns:
        The list of added :class:`Constraint` objects.

    Raises:
        ValueError: If fewer than 2 terms are given, z has no finite ub,
            or any term has no finite lb.
    """
    from ..model import BinVar, Constraint

    if len(terms) < 2:
        raise ValueError(f"add_max_constraint needs at least 2 terms, got {len(terms)}")
    if z.ub == float("inf"):
        raise ValueError(
            f"add_max_constraint requires a finite ub on z "
            f"(z={z.name or z.index!r} has ub={z.ub})"
        )
    for t in terms:
        if t.lb == -float("inf"):
            raise ValueError(
                f"add_max_constraint requires finite lb on every term "
                f"(term {t.name or t.index!r} has lb={t.lb})"
            )

    prefix = name or f"_max_z{z.index}"
    added: list["Constraint"] = []

    b_vars = [
        problem.add_variables(f"{prefix}_b{i}", vtype="binary") for i in range(len(terms))
    ]
    added.append(problem.add_constraint(_sum_vars(b_vars) == 1.0, f"{prefix}_pick"))

    for i, x_i in enumerate(terms):
        added.append(
            problem.add_constraint(
                Expression.from_variable(z) - Expression.from_variable(x_i) >= 0.0,
                f"{prefix}_ge{i}",
            )
        )
        M_i = z.ub - x_i.lb
        added.append(
            problem.add_constraint(
                Expression.from_variable(z) - Expression.from_variable(x_i) + M_i * b_vars[i]
                <= M_i,
                f"{prefix}_le{i}",
            )
        )

    return added


def add_min_constraint(
    problem: "Problem",
    z: "Variable",
    terms: list["Variable"],
    name: str = "",
) -> list["Constraint"]:
    """Enforce ``z == min(terms)`` via a big-M MIP reformulation.

    Mirror of :func:`add_max_constraint`: ``z <= x_i`` always holds, one
    binary b_i per term selects the minimizing term (``sum(b_i) == 1``),
    and ``z >= x_i - M_i*(1 - b_i)`` where ``M_i = x_i.ub - z.lb``.

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        z: The variable constrained to equal the min. Must have finite lb.
        terms: The variables being minned over, length >= 2. Each must
            have a finite ub.
        name: Optional name prefix (default derived from z's index).

    Returns:
        The list of added :class:`Constraint` objects.

    Raises:
        ValueError: If fewer than 2 terms are given, z has no finite lb,
            or any term has no finite ub.
    """
    from ..model import BinVar, Constraint

    if len(terms) < 2:
        raise ValueError(f"add_min_constraint needs at least 2 terms, got {len(terms)}")
    if z.lb == -float("inf"):
        raise ValueError(
            f"add_min_constraint requires a finite lb on z "
            f"(z={z.name or z.index!r} has lb={z.lb})"
        )
    for t in terms:
        if t.ub == float("inf"):
            raise ValueError(
                f"add_min_constraint requires finite ub on every term "
                f"(term {t.name or t.index!r} has ub={t.ub})"
            )

    prefix = name or f"_min_z{z.index}"
    added: list["Constraint"] = []

    b_vars = [
        problem.add_variables(f"{prefix}_b{i}", vtype="binary") for i in range(len(terms))
    ]
    added.append(problem.add_constraint(_sum_vars(b_vars) == 1.0, f"{prefix}_pick"))

    for i, x_i in enumerate(terms):
        added.append(
            problem.add_constraint(
                Expression.from_variable(z) - Expression.from_variable(x_i) <= 0.0,
                f"{prefix}_le{i}",
            )
        )
        M_i = x_i.ub - z.lb
        added.append(
            problem.add_constraint(
                Expression.from_variable(z) - Expression.from_variable(x_i) - M_i * b_vars[i]
                >= -M_i,
                f"{prefix}_ge{i}",
            )
        )

    return added
