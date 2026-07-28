"""Parameter — a mutable scalar leaf usable in Expression arithmetic.

Unlike Expression's other terms (numeric coefficients captured at
construction time), a Parameter's contribution is resolved lazily — each
time the Expression is assembled into matrices (i.e. at solve() time). This
lets you build a Problem once, mutate Parameter.value, and re-solve without
reconstructing variables, constraints, or the objective.

Usage::

    budget = Parameter(10.0)
    pb.add_constraint(x[0] + x[1] <= budget)
    solve(pb)
    budget.value = 20.0
    solve(pb)  # re-assembles with the new bound, no rebuild needed
"""

from __future__ import annotations


class Parameter:
    """A named, mutable scalar usable as an additive term or a bound/RHS.

    Combining a Parameter with a Variable or Expression produces an
    Expression that keeps a live reference to the Parameter (via
    ``_param_linear``) instead of baking in its current value — later
    changes to ``.value`` are picked up on the next solve.
    """

    __slots__ = ("value", "name")

    def __init__(self, value: float = 0.0, name: str = "") -> None:
        self.value = float(value)
        self.name = name

    def __add__(self, other: object) -> "Expression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: 1.0}) + other

    def __radd__(self, other: object) -> "Expression":
        return self.__add__(other)

    def __sub__(self, other: object) -> "Expression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: 1.0}) - other

    def __rsub__(self, other: object) -> "Expression":
        from .expression import Expression

        return Expression.constant(0.0) + other - Expression._from_raw(
            {}, [], 0.0, {self: 1.0}
        )

    def __mul__(self, other: object) -> "Expression":
        if isinstance(other, (int, float)):
            from .expression import Expression

            return Expression._from_raw({}, [], 0.0, {self: float(other)})
        return NotImplemented

    def __rmul__(self, other: object) -> "Expression":
        return self.__mul__(other)

    def __neg__(self) -> "Expression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: -1.0})

    def __le__(self, other: object) -> "_BoundExpression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: 1.0}) <= other

    def __ge__(self, other: object) -> "_BoundExpression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: 1.0}) >= other

    def __eq__(self, other: object) -> "_BoundExpression":
        from .expression import Expression

        return Expression._from_raw({}, [], 0.0, {self: 1.0}) == other

    def __repr__(self) -> str:
        return f"Parameter({self.name or '?'}={self.value:.6g})"

    def __hash__(self) -> int:
        return id(self)
