"""Model — Variable, Constraint, and Problem classes for building optimization models.

Usage::

    pb = Problem("my_model")
    x = pb.add_variables("x", 3, lb=0)
    pb.set_objective(3*x[0] + 2*x[1] + x[2])
    pb.add_constraint(x[0] + x[1] + x[2] <= 10)
    result = solve(pb)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .expression import Expression
from .sos import SOSConstraint

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

    from .expression import _BoundExpression
    from .matrices import SparseMatrixBuilder


class Variable:
    """An optimization variable with optional name, bounds, and type.

    Variables are created through :meth:`Problem.add_variables`, not directly.
    Supports arithmetic operators that produce :class:`Expression` objects.
    """

    __slots__ = ("_id", "_index", "name", "lb", "ub", "vtype")

    _next_id: int = 0

    def __init__(
        self,
        index: int,
        name: str = "",
        lb: float = -float("inf"),
        ub: float = float("inf"),
        vtype: str = "continuous",
    ) -> None:
        self._id = Variable._next_id
        Variable._next_id += 1
        self._index = index
        self.name = name
        self.lb = lb
        self.ub = ub
        self.vtype = vtype

    @property
    def index(self) -> int:
        return self._index

    def __add__(self, other: object) -> Expression:
        from .expression import Expression

        if isinstance(other, (int, float)):
            return Expression.from_variable(self) + Expression.constant(float(other))
        if isinstance(other, Variable):
            return Expression.from_variable(self) + Expression.from_variable(other)
        if isinstance(other, Expression):
            return Expression.from_variable(self) + other
        return NotImplemented

    def __radd__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.constant(float(other)) + Expression.from_variable(self)
        return NotImplemented

    def __sub__(self, other: object) -> Expression:
        from .expression import Expression

        if isinstance(other, (int, float)):
            return Expression.from_variable(self) - Expression.constant(float(other))
        if isinstance(other, Variable):
            return Expression.from_variable(self) - Expression.from_variable(other)
        if isinstance(other, Expression):
            return Expression.from_variable(self) - other
        return NotImplemented

    def __rsub__(self, other: object) -> Expression:
        from .expression import Expression

        if isinstance(other, (int, float)):
            return Expression.constant(float(other)) - Expression.from_variable(self)
        return NotImplemented

    def __mul__(self, other: object) -> Expression:
        from .expression import Expression

        if isinstance(other, (int, float)):
            return Expression.from_variable(self) * float(other)
        if isinstance(other, Variable):
            # x * y → quadratic term
            return Expression._from_raw({}, [(self, other, 1.0)])
        if isinstance(other, Expression):
            return other * self  # __rmul__ on Expression handles scalar
        return NotImplemented

    def __rmul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.from_variable(self) * float(other)
        return NotImplemented

    def __pow__(self, power: int) -> Expression:
        from .expression import Expression

        if power == 2:
            # x^2 → quadratic diagonal term
            return Expression._from_raw({}, [(self, self, 1.0)])
        return NotImplemented

    def __neg__(self) -> Expression:
        from .expression import Expression

        return Expression.from_variable(self) * (-1)

    def __eq__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return self._id == other._id
        elif isinstance(other, (int, float)):
            return Expression.from_variable(self) == other
        return NotImplemented

    def __le__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return Expression.from_variable(self) <= other
        elif isinstance(other, (int, float)):
            return Expression.from_variable(self) <= other
        return NotImplemented

    def __ge__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return Expression.from_variable(self) >= other
        elif isinstance(other, (int, float)):
            return Expression.from_variable(self) >= other
        return NotImplemented

    def __repr__(self) -> str:
        return f"Variable({self.name or '?'}, idx={self._index})"

    def __hash__(self) -> int:
        return hash(self._id)


class IntVar(Variable):
    """Integer variable (lb <= x <= ub, x ∈ ℤ)."""

    def __init__(
        self,
        index: int,
        name: str = "",
        lb: float = 0,
        ub: float = float("inf"),
    ) -> None:
        super().__init__(index, name, lb, ub, vtype="integer")

    def __mul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.from_variable(self) * float(other)
        raise TypeError(
            "IntVar * IntVar/BinVar produces a quadratic term, "
            "not supported in MIP models."
        )


class BinVar(Variable):
    """Binary variable (0 <= x <= 1, x ∈ {0, 1})."""

    def __init__(self, index: int, name: str = "") -> None:
        super().__init__(index, name, lb=0, ub=1, vtype="binary")

    def __mul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.from_variable(self) * float(other)
        raise TypeError(
            "BinVar * IntVar/BinVar produces a quadratic term, "
            "not supported in MIP models."
        )


def check_mip(variables: list) -> list[str]:
    """Return names of integer/binary variables, or [] if all continuous."""
    mip_names: list[str] = []
    for v in variables:
        if v.vtype in ("integer", "binary"):
            mip_names.append(f"{v.name}(idx={v._index})")
    return mip_names


@dataclass
class Constraint:
    """A single constraint: body sense bound.

    Parameters:
        body: Left-hand side :class:`Expression`.
        sense: Comparison operator — ``"<="``, ``">="``, or ``"=="``.
        bound: Right-hand side scalar.
        name: Optional label for the constraint.
    """

    body: Expression
    sense: Literal["<=", ">=", "=="]
    bound: float
    name: str = ""


class Problem:
    """Container for the full optimization model.

    Usage::

        pb = Problem("blend")
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(3*x[0] + 1*x[1] + 2*x[2])
        pb.add_constraint(x[0] + x[1] + x[2] == 4)
        pb.add_constraint(2*x[0] + x[1] <= 5)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._variables: list[Variable] = []
        self._objective_expr: Expression | None = None
        self._objective_sense: Literal["minimize", "maximize"] = "minimize"
        self._constraints: list[Constraint] = []
        self._soc_constraints: list[_SOCPConstraint] = []
        self._sos_constraints: list[SOSConstraint] = []

    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return len(self._variables)

    @property
    def n_constraints(self) -> int:
        """Number of linear constraints."""
        return len(self._constraints)

    def add_variables(
        self,
        name: str | None = None,
        n: int = 1,
        lb: float = -float("inf"),
        ub: float = float("inf"),
        vtype: str = "continuous",
    ) -> Variable | list[Variable]:
        """Add one or more optimization variables.

        Parameters:
            name: Optional prefix for variable names (``name_0``, ``name_1``, ...).
            n: Number of variables to add.
            lb: Lower bound (scalar, applied to all).
            ub: Upper bound (scalar, applied to all).
            vtype: Variable type — ``"continuous"``, ``"integer"``, or ``"binary"``.

        Returns:
            A single :class:`Variable` if ``n==1``, otherwise a list of :class:`Variable`.
        """
        prefix = f"{name}_" if name else ""
        vars_list: list[Variable] = []

        for i in range(n):
            if vtype == "integer":
                v = IntVar(len(self._variables), f"{prefix}{i}", lb, ub)
            elif vtype == "binary":
                v = BinVar(len(self._variables), f"{prefix}{i}")
            else:
                v = Variable(len(self._variables), f"{prefix}{i}", lb, ub)
            vars_list.append(v)
            self._variables.append(v)

        return vars_list[0] if n == 1 else vars_list

    def set_objective(
        self,
        expr: Expression,
        sense: Literal["minimize", "maximize"] = "minimize",
    ) -> None:
        """Set the objective function.

        Parameters:
            expr: Objective expression (linear or quadratic).
            sense: ``"minimize"`` (default) or ``"maximize"``.
        """
        self._objective_expr = expr
        self._objective_sense = sense

    def add_constraint(
        self,
        constraint: Constraint | Expression,
        name: str = "",
    ) -> Constraint:
        """Add a linear constraint.

        Parameters:
            constraint: Either a :class:`Constraint` object or an
                :class:`Expression` representing a bound expression.
                If an ``Expression`` is passed, it must be of the form
                ``expr <= bound``, ``expr >= bound``, or ``expr == bound``,
                created using Python comparison operators on expressions.
            name: Optional label.

        Returns:
            The created :class:`Constraint`.
        """
        if isinstance(constraint, Constraint):
            if name and not constraint.name:
                constraint = Constraint(
                    constraint.body, constraint.sense, constraint.bound, name
                )
            self._constraints.append(constraint)
            return constraint

        # Expression comparison produces a _BoundExpression
        if hasattr(constraint, "_sense") and hasattr(constraint, "_bound"):
            c = Constraint(constraint._body, constraint._sense, constraint._bound, name)
            self._constraints.append(c)
            return c

        raise TypeError(
            "add_constraint expects a Constraint or a comparison expression "
            "(e.g., x[0] + x[1] <= 5)"
        )

    def _add_constrained(
        self,
        body: Expression,
        sense: Literal["<=", ">=", "=="],
        bound: float,
        name: str = "",
    ) -> Constraint:
        """Internal: add a constraint without validation."""
        c = Constraint(body, sense, bound, name)
        self._constraints.append(c)
        return c



    # -- SOCP support --------------------------------------------------------

    def add_soc_constraint(
        self,
        soc: _SOCPConstraint,
        name: str = "",
    ) -> _SOCPConstraint:
        """Add a second-order cone constraint.

        Parameters:
            soc: A :class:`_SOCPConstraint` (use :func:`SOC` to construct).
            name: Optional label.
        """
        if name:
            soc.name = name
        self._soc_constraints.append(soc)
        return soc

    # -- SOS support -----------------------------------------------------------

    def add_sos_constraint(
        self,
        variables: list[Variable],
        type: Literal[1, 2],
        weights: list[float] | None = None,
        method: Literal["cc", "dlog", "log"] = "cc",
        name: str = "",
    ) -> SOSConstraint:
        """Add a special-ordered-set (SOS1/SOS2) constraint.

        Parameters:
            variables: Variables in the set, ordered.
            type: 1 (at most one nonzero) or 2 (at most two consecutive nonzero).
            weights: Optional ordering weights; defaults to positional order.
            method: SOS2 reformulation — ``"cc"`` (default), ``"dlog"``, or
                ``"log"``. See :class:`SOSConstraint` for details. Ignored
                for SOS1.
            name: Optional label.

        Returns:
            The created :class:`SOSConstraint`.

        Note:
            Only solvable via the ``highs``, ``scip``, or ``gurobi`` backends
            (reformulated as a MIP). All variables must have finite bounds.
        """
        sos = SOSConstraint(
            variables=list(variables), type=type, weights=weights, method=method, name=name
        )
        self._sos_constraints.append(sos)
        return sos

    # -- max/min support -------------------------------------------------------

    def add_max_constraint(
        self,
        z: Variable,
        terms: list[Variable],
        name: str = "",
    ) -> list[Constraint]:
        """Enforce ``z == max(terms)`` via a big-M MIP reformulation.

        Parameters:
            z: The variable constrained to equal the max. Must have finite ub.
            terms: Variables being maxed over, length >= 2. Each must have
                a finite lb.
            name: Optional name prefix.

        Returns:
            The list of added :class:`Constraint` objects.

        Note:
            Only solvable via ``highs``, ``scip``, or ``gurobi`` (reformulated
            as a MIP: one binary indicator per term).
        """
        from .formulations.bigm import add_max_constraint

        return add_max_constraint(self, z, terms, name=name)

    def add_min_constraint(
        self,
        z: Variable,
        terms: list[Variable],
        name: str = "",
    ) -> list[Constraint]:
        """Enforce ``z == min(terms)`` via a big-M MIP reformulation.

        Parameters:
            z: The variable constrained to equal the min. Must have finite lb.
            terms: Variables being minned over, length >= 2. Each must have
                a finite ub.
            name: Optional name prefix.

        Returns:
            The list of added :class:`Constraint` objects.

        Note:
            Only solvable via ``highs``, ``scip``, or ``gurobi`` (reformulated
            as a MIP: one binary indicator per term).
        """
        from .formulations.bigm import add_min_constraint

        return add_min_constraint(self, z, terms, name=name)

    # -- assembly ------------------------------------------------------------

    def assemble(
        self,
    ) -> tuple[
        Expression,
        Literal["minimize", "maximize"],
        list[Constraint],
        list[_SOCPConstraint],
        list[Variable],
    ]:
        """Validate and return the assembled problem data.

        Returns:
            (objective_expr, objective_sense, constraints, soc_constraints, variables)
        """
        if self._objective_expr is None:
            raise ValueError("Problem has no objective — call set_objective() first")

        if not self._variables:
            raise ValueError("Problem has no variables — call add_variables() first")

        # Validate all expressions reference known variables
        obj_vars = self._objective_expr.variables
        bad_vars = obj_vars - set(self._variables)
        if bad_vars:
            names = [v.name for v in bad_vars]
            raise ValueError(f"Objective references unknown variables: {names}")

        for c in self._constraints:
            c_vars = c.body.variables
            bad = c_vars - set(self._variables)
            if bad:
                names = [v.name for v in bad]
                raise ValueError(
                    f"Constraint '{c.name or '?'}' references unknown variables: {names}"
                )

        return (
            self._objective_expr,
            self._objective_sense,
            self._constraints,
            self._soc_constraints,
            self._variables,
        )

    def __repr__(self) -> str:
        return (
            f"Problem({self.name or '?'}, "
            f"vars={self.n_vars}, constraints={self.n_constraints})"
        )





# Monkey-patch Expression to support comparison operators
@dataclass
class _SOCPConstraint:
    """Second-order cone constraint: ||cone_body||_2 <= affine_part.

    Parameters:
        variables: List of variables involved.
        cone_indices: Indices into *variables* forming the cone body.
        b: Offset inside the norm (zeros if identity).
        c: Linear coefficients outside the norm (x[cone[-1]] by default).
        d: Offset outside the norm.
        name: Optional label.
    """

    variables: list[Variable | int]
    cone_indices: list[int]
    b: list[float] = field(default_factory=list)
    c: dict[Variable | int, float] = field(default_factory=dict)
    d: float = 0.0
    name: str = ""
