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


__all__ = [
    "Expression",
    "_BoundExpression",
]


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

    Comparison operators produce :class:`_BoundExpression` (constraints),
    not bools — ``x + y <= 5`` is a constraint, not a boolean test.
    Use ``Expression(a) == Expression(b)`` to compare two Expression objects for
    equality (e.g. in tests).
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
        """Single-variable expression (x -> x)."""
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

    # -- properties ----------------------------------------------------------

    @property
    def is_linear(self) -> bool:
        """True if the expression has no quadratic terms."""
        return not self._quadratic

    @property
    def is_quadratic(self) -> bool:
        """True if the expression has quadratic terms."""
        return bool(self._quadratic)

    @property
    def variables(self) -> set["Variable"]:
        """All :class:`Variable` objects referenced in this expression."""
        return set(self._linear.keys()) | {vi for vi, vj, c in self._quadratic}

    @property
    def linear_coeffs(self) -> dict["Variable", float]:
        """Non-zero linear coefficients as a dict."""
        return self._linear

    @property
    def quadratic_terms(self) -> list[tuple["Variable", "Variable", float]]:
        """Quadratic terms as a list of (var_i, var_j, coefficient)."""
        return self._quadratic

    @property
    def resolved_constant(self) -> float:
        """Constant term with all Parameter values resolved (current values)."""
        return self._constant + sum(
            p.value * coeff for p, coeff in self._param_linear.items()
        )

    # -- arithmetic ----------------------------------------------------------

    def _coalesce_linear(self) -> dict["Variable", float]:
        """Merge duplicate variable keys in linear coefficients."""
        linear: dict["Variable", float] = {}
        for v, c in self._linear.items():
            linear[v] = linear.get(v, 0.0) + c
        # Remove zero entries
        return {v: c for v, c in linear.items() if c != 0.0}

    def _coalesce_quadratic(self) -> list[tuple["Variable", "Variable", float]]:
        """Merge duplicate quadratic pairs (vi, vj) into single entries."""
        quad: dict[tuple["Variable", "Variable"], float] = {}
        for vi, vj, c in self._quadratic:
            key = (vi, vj)
            quad[key] = quad.get(key, 0.0) + c
        # Remove zero entries
        return [(vi, vj, c) for (vi, vj), c in quad.items() if c != 0.0]

    def __add__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression._from_raw(
                self._linear.copy(),
                self._quadratic.copy(),
                self._constant + float(other),
                self._param_linear.copy(),
            )
        if isinstance(other, Expression):
            # Merge linear terms
            new_linear = self._linear.copy()
            for v, c in other._linear.items():
                new_linear[v] = new_linear.get(v, 0.0) + c
            # Merge quadratic terms
            new_quad = self._quadratic + other._quadratic
            # Merge constant
            new_constant = self._constant + other._constant
            # Merge param terms
            new_param = self._param_linear.copy()
            for p, c in other._param_linear.items():
                new_param[p] = new_param.get(p, 0.0) + c
            return Expression._from_raw(
                new_linear, new_quad, new_constant, new_param
            )
        if hasattr(other, "_index") and hasattr(other, "name") and not hasattr(other, "_param_linear"):
            # Handle bare Variable via lazy import
            from .model import Variable as _Var
            if isinstance(other, _Var):
                return Expression._from_raw(
                    self._linear.copy(),
                    self._quadratic.copy(),
                    self._constant,
                    self._param_linear.copy(),
                ) + Expression.from_variable(other)
        return NotImplemented

    def __radd__(self, other: object) -> Expression:
        return self.__add__(other)

    def __sub__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression._from_raw(
                self._linear.copy(),
                self._quadratic.copy(),
                self._constant - float(other),
                self._param_linear.copy(),
            )
        if isinstance(other, Expression):
            new_linear = self._linear.copy()
            for v, c in other._linear.items():
                new_linear[v] = new_linear.get(v, 0.0) - c
            new_quad = self._quadratic + [(vi, vj, -c) for vi, vj, c in other._quadratic]
            new_constant = self._constant - other._constant
            new_param = self._param_linear.copy()
            for p, c in other._param_linear.items():
                new_param[p] = new_param.get(p, 0.0) - c
            return Expression._from_raw(
                new_linear, new_quad, new_constant, new_param
            )
        if hasattr(other, "_index") and hasattr(other, "name") and not hasattr(other, "_param_linear"):
            # Handle bare Variable via lazy import
            from .model import Variable as _Var
            if isinstance(other, _Var):
                return self - Expression.from_variable(other)
        return NotImplemented

    def __rsub__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression._from_raw(
                {}, [], float(other) - self._constant, {}
            ) - self
        return NotImplemented

    def __mul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            scale = float(other)
            return Expression._from_raw(
                {v: c * scale for v, c in self._linear.items() if c * scale != 0.0},
                [(vi, vj, c * scale) for vi, vj, c in self._quadratic if c * scale != 0.0],
                self._constant * scale,
                {p: c * scale for p, c in self._param_linear.items() if c * scale != 0.0},
            )
        return NotImplemented

    def __rmul__(self, other: object) -> Expression:
        return self.__mul__(other)

    def __pow__(self, power: int) -> Expression:
        if power != 2:
            return NotImplemented
        if self._quadratic or self._constant != 0 or self._param_linear:
            raise TypeError(
                "Expression.__pow__ only supports squaring a purely linear expression "
                "(e.g., x**2 or (x + y)**2). For other cases, use multiplication."
            )
        items = list(self._linear.items())
        quad: list[tuple["Variable", "Variable", float]] = []
        for i, (vi, ci) in enumerate(items):
            for vj, cj in items[i:]:
                coeff = ci * cj if vi is vj else 2.0 * ci * cj
                quad.append((vi, vj, coeff))
        return Expression._from_raw({}, quad)

    def __neg__(self) -> Expression:
        return Expression._from_raw(
            {v: -c for v, c in self._linear.items()},
            [(vi, vj, -c) for vi, vj, c in self._quadratic],
            -self._constant,
            {p: -c for p, c in self._param_linear.items()},
        )

    def __truediv__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            scale = 1.0 / float(other)
            return Expression._from_raw(
                {v: c * scale for v, c in self._linear.items()},
                [(vi, vj, c * scale) for vi, vj, c in self._quadratic],
                self._constant * scale,
                {p: c * scale for p, c in self._param_linear.items()},
            )
        return NotImplemented

    # -- sparse assembly -----------------------------------------------------

    def to_quadratic_arrays(self, n_vars: int) -> tuple[list[int], list[int], list[float]]:
        """Return COO triplets for the quadratic part (P matrix in 0.5 x^T P x)."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for vi, vj, c in self._quadratic:
            rows.append(vi._index)
            cols.append(vj._index)
            vals.append(c)
        return rows, cols, vals

    def to_arrays(self, n_vars: int) -> tuple[list[int], list[int], list[float]]:
        """Return combined COO triplets: linear terms followed by quadratic terms."""
        rows, cols, vals = self.to_linear_arrays(n_vars)
        qrows, qcols, qvals = self.to_quadratic_arrays(n_vars)
        return rows + qrows, cols + qcols, vals + qvals

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
