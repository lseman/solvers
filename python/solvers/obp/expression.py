"""Expression — linear and quadratic expressions over optimization variables.

Stores terms in sparse form:
- linear_coeffs: dict[Variable, float]   (non-zero linear coefficients)
- quadratic_terms: list[(Variable, Variable, float)]  (sparse quadratic pairs)
- constant: float offset

Arithmetic produces new Expression objects; no symbolic tree building.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .model import Variable
    from .parameter import Parameter


@dataclass(frozen=True)
class _BoundExpression:
    """Result of comparing an Expression with a scalar.

    Produced when you write ``expr <= 5`` via comparison operators
    monkey-patched onto :class:`Expression`.
    """

    _body: Expression
    _sense: str
    _bound: float


@dataclass(frozen=True)
class Expression:
    """Linear or quadratic expression over :class:`Variable` objects.

    Usage::

        expr = 3*x + 2*y**2 - 1
        # linear_coeffs = {x: 3}
        # quadratic_terms = [(y, y, 2)]
        # constant = -1
    """

    _linear: dict["Variable", float] = field(default_factory=dict)
    _quadratic: list[tuple["Variable", "Variable", float]] = field(default_factory=list)
    _constant: float = 0.0
    _param_linear: dict["Parameter", float] = field(default_factory=dict)

    # -- constructors --------------------------------------------------------

    @staticmethod
    def constant(c: float) -> Expression:
        """Zero-variance expression equal to *c*."""
        return Expression(_linear={}, _quadratic=[], _constant=c)

    @staticmethod
    def from_variable(v: "Variable") -> Expression:
        """Single-variable expression (x ↦ x)."""
        return Expression(_linear={v: 1.0}, _quadratic=[], _constant=0.0)

    @staticmethod
    def _from_raw(
        linear: dict["Variable", float],
        quadratic: list[tuple["Variable", "Variable", float]],
        constant: float = 0.0,
        param_linear: dict["Parameter", float] | None = None,
    ) -> Expression:
        """Direct construction — bypasses normalization. Use only internally."""
        return Expression(
            _linear=linear,
            _quadratic=quadratic,
            _constant=constant,
            _param_linear=param_linear or {},
        )

    @property
    def resolved_constant(self) -> float:
        """Constant term plus the current value of any Parameters."""
        total = self._constant
        for p, c in self._param_linear.items():
            total += c * p.value
        return total

    # -- normalization helpers -----------------------------------------------

    @classmethod
    def _coalesce_linear(cls, d: dict["Variable", float]) -> dict["Variable", float]:
        """Remove zero coefficients and merge duplicates."""
        return {v: c for v, c in d.items() if abs(c) > 1e-15}

    @classmethod
    def _coalesce_quadratic(
        cls, terms: list[tuple["Variable", "Variable", float]]
    ) -> list[tuple["Variable", "Variable", float]]:
        """Sum duplicate (i, j) pairs and remove near-zero entries."""
        merged: dict[tuple[int, int], tuple[float, "Variable", "Variable"]] = {}
        for vi, vj, c in terms:
            key = (vi._index, vj._index)
            if vi._index > vj._index:
                key = (vj._index, vi._index)
            if key in merged:
                merged[key] = (merged[key][0] + c, merged[key][1], merged[key][2])
            else:
                merged[key] = (c, vi, vj)
        result: list[tuple["Variable", "Variable", float]] = []
        for (c, vi, vj) in merged.values():
            if abs(c) > 1e-15:
                result.append((vi, vj, c))
        return result

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression._from_raw(
                self._linear,
                self._quadratic,
                self._constant + float(other),
                self._param_linear,
            )
        elif isinstance(other, Expression):
            new_param = dict(self._param_linear)
            for p, c in other._param_linear.items():
                new_param[p] = new_param.get(p, 0.0) + c
            return Expression._from_raw(
                {**self._linear, **other._linear},
                self._quadratic + other._quadratic,
                self._constant + other._constant,
                new_param,
            )
        # Handle Parameter: bare parameter, additive
        elif hasattr(other, "value") and hasattr(other, "name") and not hasattr(other, "_index"):
            new_param = dict(self._param_linear)
            new_param[other] = new_param.get(other, 0.0) + 1.0
            return Expression._from_raw(
                self._linear,
                self._quadratic,
                self._constant,
                new_param,
            )
        # Handle Variable: extract from expression
        elif hasattr(other, "_index") and hasattr(other, "name"):
            v = other
            new_linear = dict(self._linear)
            new_linear[v] = new_linear.get(v, 0.0) + 1.0
            return Expression._from_raw(
                new_linear,
                self._quadratic,
                self._constant,
                self._param_linear,
            )
        return NotImplemented

    def __radd__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            other = Expression.constant(float(other))
        elif not isinstance(other, Expression):
            return NotImplemented
        return other + self

    def __sub__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression._from_raw(
                self._linear,
                self._quadratic,
                self._constant - float(other),
                self._param_linear,
            )
        if isinstance(other, Expression):
            # Both are Expressions: subtract linear, quadratic, constant directly
            new_linear = {}
            for v, c in self._linear.items():
                new_linear[v] = c
            for v, c in other._linear.items():
                new_linear[v] = new_linear.get(v, 0.0) - c
            # Remove near-zero coefficients
            new_linear = {v: c for v, c in new_linear.items() if abs(c) > 1e-15}
            neg_quad = [(vi, vj, -c) for vi, vj, c in other._quadratic]
            new_param = dict(self._param_linear)
            for p, c in other._param_linear.items():
                new_param[p] = new_param.get(p, 0.0) - c
            return Expression._from_raw(
                new_linear,
                self._quadratic + neg_quad,
                self._constant - other._constant,
                new_param,
            )
        elif hasattr(other, "value") and hasattr(other, "name") and not hasattr(other, "_index"):
            new_param = dict(self._param_linear)
            new_param[other] = new_param.get(other, 0.0) - 1.0
            return Expression._from_raw(
                self._linear,
                self._quadratic,
                self._constant,
                new_param,
            )
        elif hasattr(other, "_index") and hasattr(other, "name"):
            v = other
            new_linear = dict(self._linear)
            new_linear[v] = new_linear.get(v, 0.0) - 1.0
            new_linear = {v2: c for v2, c in new_linear.items() if abs(c) > 1e-15}
            return Expression._from_raw(
                new_linear,
                self._quadratic,
                self._constant,
                self._param_linear,
            )
        return NotImplemented

    def __rsub__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.constant(float(other)) - self
        return NotImplemented

    def __mul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            scale = float(other)
            return Expression._from_raw(
                {v: c * scale for v, c in self._linear.items()},
                [(vi, vj, c * scale) for vi, vj, c in self._quadratic],
                self._constant * scale,
                {p: c * scale for p, c in self._param_linear.items()},
            )
        return NotImplemented

    def __rmul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return self * float(other)
        return NotImplemented

    def __neg__(self) -> Expression:
        return Expression._from_raw(
            {v: -c for v, c in self._linear.items()},
            [(vi, vj, -c) for vi, vj, c in self._quadratic],
            -self._constant,
            {p: -c for p, c in self._param_linear.items()},
        )

    def __pow__(self, power: int) -> Expression:
        if power == 2:
            # (a + b + ...)^2 = sum_i a_i^2 + 2*sum_{i<j} a_i*a_j
            new_linear: dict["Variable", float] = {}
            new_quadratic: list[tuple["Variable", "Variable", float]] = []

            vars_list = list(self._linear.items())
            for i, (vi, ci) in enumerate(vars_list):
                for j, (vj, cj) in enumerate(vars_list):
                    coeff = ci * cj
                    if vi._index == vj._index:
                        new_quadratic.append((vi, vi, coeff))
                    elif vi._index < vj._index:
                        new_quadratic.append((vi, vj, coeff))
                    elif vi._index > vj._index:
                        new_quadratic.append((vj, vi, coeff))

            # Coalesce duplicate quadratic terms
            new_quadratic = self._coalesce_quadratic(new_quadratic)

            return Expression._from_raw(new_linear, new_quadratic, 0.0)
        return NotImplemented

    # -- query helpers -------------------------------------------------------

    @property
    def is_linear(self) -> bool:
        """True if this expression has no quadratic terms."""
        return len(self._quadratic) == 0

    @property
    def is_quadratic(self) -> bool:
        """True if this expression has any quadratic terms."""
        return len(self._quadratic) > 0

    @property
    def variables(self) -> set["Variable"]:
        """All variables appearing in this expression."""
        vars_set = set(self._linear.keys())
        for vi, vj, _ in self._quadratic:
            vars_set.add(vi)
            vars_set.add(vj)
        return vars_set

    @property
    def linear_coeffs(self) -> dict["Variable", float]:
        """Read-only view of linear coefficients."""
        return dict(self._linear)

    @property
    def quadratic_terms(self) -> list[tuple["Variable", "Variable", float]]:
        """Read-only view of quadratic terms."""
        return list(self._quadratic)

    # -- sparse assembly -----------------------------------------------------

    def to_arrays(self, n_vars: int) -> tuple[list[int], list[int], list[float]]:
        """Return COO-style (row, col, val) triplets for sparse matrix assembly."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for v, c in self._linear.items():
            rows.append(0)
            cols.append(v._index)
            vals.append(c)

        for vi, vj, c in self._quadratic:
            rows.append(vi._index)
            cols.append(vj._index)
            vals.append(c)

        return rows, cols, vals

    def to_quadratic_arrays(self, n_vars: int) -> tuple[list[int], list[int], list[float]]:
        """Return COO triplets for the P (quadratic objective) matrix only."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for vi, vj, c in self._quadratic:
            rows.append(vi._index)
            cols.append(vj._index)
            vals.append(c)
        return rows, cols, vals

    def to_linear_arrays(self, n_vars: int) -> tuple[list[int], list[int], list[float]]:
        """Return COO triplets for the linear part (objective c or constraint A row)."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for v, c in self._linear.items():
            rows.append(0)
            cols.append(v._index)
            vals.append(c)
        return rows, cols, vals

    def __repr__(self) -> str:
        parts: list[str] = []
        for v, c in self._linear.items():
            parts.append(f"{c:.4g}*{v.name or '?'}")
        for vi, vj, c in self._quadratic:
            name_i = vi.name or "?"
            name_j = vj.name or "?"
            if vi._index == vj._index:
                parts.append(f"{c:.4g}*{name_i}^2")
            else:
                parts.append(f"{c:.4g}*{name_i}*{name_j}")
        if self._constant != 0:
            parts.append(f"{self._constant:.4g}")
        if not parts:
            return "Expression(constant 0)"
        return f"Expression({', '.join(parts)})"


def _is_parameter(obj: object) -> bool:
    return hasattr(obj, "value") and hasattr(obj, "name") and not hasattr(obj, "_index")


# Monkey-patch Expression to support comparison operators (<=, >=, ==).
# A Parameter on the RHS is moved into the body (self - param <= 0) so its
# value is resolved lazily via Constraint.effective_bound, not baked in here.
def _cmp(self: Expression, other: object, sense: str) -> _BoundExpression | type(NotImplemented):
    if isinstance(other, (int, float)):
        return _BoundExpression(self, sense, float(other))
    if _is_parameter(other):
        return _BoundExpression(self - other, sense, 0.0)
    return NotImplemented


Expression.__le__ = lambda self, other: _cmp(self, other, "<=")
Expression.__ge__ = lambda self, other: _cmp(self, other, ">=")
Expression.__eq__ = lambda self, other: _cmp(self, other, "==")
