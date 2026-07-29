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

import numpy as np

from .expression import Expression
from .matrices import build_symmetric_P, build_constraints_matrix
from .sos import SOSConstraint

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

    from .expression import _BoundExpression
    from .matrices import SparseMatrixBuilder


__all__ = [
    "BinVar",
    "Constraint",
    "IntVar",
    "Problem",
    "Variable",
    "_SOCPConstraint",
    "check_mip",
]


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
            # x * y -> quadratic term
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
            # x^2 -> quadratic diagonal term
            return Expression._from_raw({}, [(self, self, 1.0)])
        return NotImplemented

    def __neg__(self) -> Expression:
        from .expression import Expression

        return Expression.from_variable(self) * (-1)

    def __eq__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return self._id == other._id
        elif isinstance(other, (int, float)) or hasattr(other, "value"):
            return Expression.from_variable(self) == other
        return NotImplemented

    def __le__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return Expression.from_variable(self) <= other
        elif isinstance(other, (int, float)) or hasattr(other, "value"):
            return Expression.from_variable(self) <= other
        return NotImplemented

    def __ge__(self, other: object) -> Expression | bool:
        if isinstance(other, Variable):
            return Expression.from_variable(self) >= other
        elif isinstance(other, (int, float)) or hasattr(other, "value"):
            return Expression.from_variable(self) >= other
        return NotImplemented

    def __repr__(self) -> str:
        return f"Variable({self.name or '?'}, idx={self._index})"

    def __hash__(self) -> int:
        return hash(self._id)

    # -- NumPy ufunc dispatch (array broadcasting) --------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Dispatch NumPy universal functions to element-wise Python ops.

        Enables ``np.array([x, y, z]) + 3`` to broadcast over the object array,
        returning an object array of :class:`Expression` objects.
        """
        import numpy as np

        if method == "__call__":
            if len(inputs) == 1:
                if ufunc is np.negative:
                    return -self
            elif len(inputs) == 2:
                a, b = inputs
                if isinstance(a, (int, float)) and isinstance(b, Variable):
                    if ufunc is np.add:
                        return Expression.from_variable(b) + Expression.constant(float(a))
                    elif ufunc is np.subtract:
                        return Expression.constant(float(a)) - b
                    elif ufunc is np.multiply:
                        return Expression.from_variable(b) * float(a)
                elif isinstance(a, Variable) and isinstance(b, (int, float)):
                    if ufunc is np.add:
                        return Expression.from_variable(a) + Expression.constant(float(b))
                    elif ufunc is np.subtract:
                        return Expression.from_variable(a) - Expression.constant(float(b))
                    elif ufunc is np.multiply:
                        return Expression.from_variable(a) * float(b)
                elif isinstance(a, Variable) and isinstance(b, Variable):
                    if ufunc is np.add:
                        return Expression.from_variable(a) + Expression.from_variable(b)
                    elif ufunc is np.subtract:
                        return Expression.from_variable(a) - Expression.from_variable(b)
                    elif ufunc is np.multiply:
                        return Expression._from_raw({}, [(a, b, 1.0)])
        if "out" in kwargs:
            return NotImplemented
        return NotImplemented


class IntVar(Variable):
    """Integer variable (lb <= x <= ub, x in Z)."""

    def __init__(
        self,
        index: int,
        name: str = "",
        lb: float = 0,
        ub: float = float("inf"),
    ) -> None:
        if lb > ub:
            raise ValueError(
                f"IntVar '{name or index}': lb={lb} > ub={ub}"
            )
        super().__init__(index, name, lb, ub, vtype="integer")

    def __mul__(self, other: object) -> Expression:
        if isinstance(other, (int, float)):
            return Expression.from_variable(self) * float(other)
        raise TypeError(
            "IntVar * IntVar/BinVar produces a quadratic term, "
            "not supported in MIP models."
        )


class BinVar(Variable):
    """Binary variable (0 <= x <= 1, x in {0, 1})."""

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
        sense: Comparison operator -- ``"<="``, ``">="``, or ``"=="``.
        bound: Right-hand side scalar.
        name: Optional label for the constraint.
    """

    body: Expression
    sense: Literal["<=", ">=", "=="]
    bound: float
    name: str = ""

    @property
    def effective_bound(self) -> float:
        """RHS with the body's constant term (and any Parameter values)
        moved across: for ``body <= bound``, solving only the linear part
        requires comparing against ``bound - body.resolved_constant``.
        """
        return self.bound - self.body.resolved_constant


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
        # --- DPP cache: keyed by (solver, param_fingerprint) → assembled matrices ---
        self._cache: dict[tuple, dict] = {}
        self._cache_enabled: bool = True  # default on, toggle with enable_caching()

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
            vtype: Variable type -- ``"continuous"``, ``"integer"``, or ``"binary"``.

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

    def add_constraints(
        self,
        exprs,
        sense: Literal["<=", ">=", "=="],
        bound,
        name: str = "",
    ) -> list[Constraint]:
        """Add one constraint per row of a batch of expressions.

        Built for the matrix-form pattern where ``exprs`` comes from
        ``A @ x`` on a NumPy object-array of Variables (matmul dispatches
        through Expression arithmetic and yields one Expression per row) --
        elementwise ``<=``/``>=``/``==`` on that result does not work
        because NumPy coerces the comparison to booleans, so build the
        per-row Constraint objects explicitly here instead.

        Usage::

            x = np.array(pb.add_variables("x", 3, lb=0), dtype=object)
            pb.add_constraints(A @ x, "<=", b)   # one row of A per constraint

        Parameters:
            exprs: Sequence of :class:`Expression`, one per row.
            sense: ``"<="``, ``">="``, or ``"=="``, applied to every row.
            bound: Scalar (broadcast to all rows) or a sequence with one
                value per row.
            name: Optional label prefix -- rows are named ``{name}_0``, etc.

        Returns:
            The list of created :class:`Constraint` objects, one per row.
        """
        exprs = list(exprs)
        if hasattr(bound, "__len__") and not isinstance(bound, (str, bytes)):
            bounds = list(bound)
            if len(bounds) != len(exprs):
                raise ValueError(
                    f"add_constraints: {len(exprs)} expressions but "
                    f"{len(bounds)} bounds"
                )
        else:
            bounds = [bound] * len(exprs)

        constraints: list[Constraint] = []
        for i, (expr, b) in enumerate(zip(exprs, bounds)):
            row_name = f"{name}_{i}" if name else ""
            c = Constraint(expr, sense, float(b), row_name)
            self._constraints.append(c)
            constraints.append(c)
        return constraints

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
            method: SOS2 reformulation -- ``"cc"`` (default), ``"dlog"``, or
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

    # -- indicator / logic support ----------------------------------------------

    def add_indicator_constraint(
        self,
        binary: Variable,
        body: Expression,
        sense: Literal["<=", ">=", "=="],
        bound: float,
        name: str = "",
        activate_on: int = 1,
    ) -> list[Constraint]:
        """Enforce ``binary == activate_on  =>  (body sense bound)`` via big-M.

        Parameters:
            binary: The indicator variable (must be binary).
            body: Left-hand side expression. Every variable it references
                must have finite bounds (used to derive the tightest big-M).
            sense: ``"<="``, ``">="``, or ``"=="``.
            bound: Right-hand side scalar.
            name: Optional name prefix.
            activate_on: ``1`` (default) to trigger when ``binary == 1``,
                or ``0`` to trigger when ``binary == 0``.

        Returns:
            The list of added :class:`Constraint` objects.

        Note:
            Only solvable via ``highs``, ``scip``, or ``gurobi`` (uses a
            binary indicator).
        """
        from .formulations.indicator import add_indicator_constraint

        return add_indicator_constraint(
            self, binary, body, sense, bound, name=name, activate_on=activate_on
        )

    def add_abs_value(self, x: Variable, name: str = "") -> Variable:
        """Create an auxiliary variable equal to ``|x|``. ``x`` must have
        finite bounds.

        Note:
            Only solvable via ``highs``, ``scip``, or ``gurobi`` (uses a
            binary indicator internally).
        """
        from .formulations.indicator import abs_value

        return abs_value(self, x, name=name)

    def add_logical_and(self, terms: list[Variable], name: str = "") -> Variable:
        """Create a binary equal to ``AND(terms)``. ``terms`` must be binary,
        length >= 2.
        """
        from .formulations.logic import logical_and

        return logical_and(self, terms, name=name)

    def add_logical_or(self, terms: list[Variable], name: str = "") -> Variable:
        """Create a binary equal to ``OR(terms)``. ``terms`` must be binary,
        length >= 2.
        """
        from .formulations.logic import logical_or

        return logical_or(self, terms, name=name)

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
            raise ValueError("Problem has no objective -- call set_objective() first")

        if not self._variables:
            raise ValueError("Problem has no variables -- call add_variables() first")

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

    # -- file export -----------------------------------------------------------

    def write_lp(self, path: str) -> None:
        """Write this problem to a CPLEX LP-format file at *path*."""
        from .export import write_lp

        write_lp(self, path)

    def write_mps(self, path: str) -> None:
        """Write this problem to a free-format MPS file at *path*."""
        from .export import write_mps

        write_mps(self, path)

    # -- serialization / debugging -------------------------------------------

    def to_dict(self) -> dict:
        """Return a serializable dict of the problem structure.

        Useful for debugging and inspection without solving.
        """
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "n_constraints": self.n_constraints,
            "n_soc_constraints": len(self._soc_constraints),
            "n_sos_constraints": len(self._sos_constraints),
            "objective_sense": self._objective_sense,
            "objective_is_quadratic": (
                self._objective_expr.is_quadratic
                if self._objective_expr
                else None
            ),
            "variables": [
                {
                    "name": v.name,
                    "index": v.index,
                    "lb": v.lb,
                    "ub": v.ub,
                    "vtype": v.vtype,
                }
                for v in self._variables
            ],
            "constraints": [
                {
                    "name": c.name,
                    "sense": c.sense,
                    "bound": c.bound,
                }
                for c in self._constraints
            ],
        }

    # --- DPP cache control ---------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the DPP assemble cache.

        Call after mutating Parameters if you want to force a rebuild
        on the next :func:`~obp.solve()`.
        """
        self._cache.clear()

    def enable_caching(self, enabled: bool = True) -> None:
        """Enable or disable DPP caching for this Problem.

        Parameters:
            enabled: Whether to cache assembled matrices between solves.
        """
        self._cache_enabled = enabled

    def get_cache_stats(self) -> dict:
        """Return cache statistics: size, enabled.

        Returns:
            Dict with ``'size'`` (number of cached entries) and
            ``'enabled'`` (boolean).
        """
        return {"size": len(self._cache), "enabled": self._cache_enabled}

    # --- get_problem_data (CVXPY-style standard form extraction) ------------

    def get_problem_data(
        self,
        solver: str | None = None,
    ) -> tuple[dict, dict]:
        """Extract canonical problem data for external solvers.

        Returns a (data, inverse_data) tuple where:
        - **data** is a dict with keys matching each solver's expected
          problem data (see :func:`~obp.solve` dispatch logic).
        - **inverse_data** is a dict with keys needed to unpack a
          solution back into :class:`Solution` format.

        The data dict always contains the canonical keys::

            data = {
                "P": scipy_sparse | None,   # 0.5 x^T P x in objective
                "c": np.ndarray,            # linear objective coefficients
                "A": scipy_sparse,          # constraint matrix
                "l": np.ndarray,            # lower bounds on Ax
                "u": np.ndarray,            # upper bounds on Ax
                "sense": np.ndarray,        # sense vector (=1, >=-1, <=0)
                "lb": np.ndarray,           # variable lower bounds
                "ub": np.ndarray,           # variable upper bounds
                "var_types": np.ndarray,    # 0=cont, 1=int, 2=binary
                "obj_offset": float,        # constant in objective
                "obj_sense": str,           # "minimize" or "maximize"
                "row_constraints": list,    # Constraint objects per row
                "soc_constraints": list,    # SOC constraints
                "sos_constraints": list,    # SOS constraints
                "n_vars": int,
                "n_constraints": int,
            }

        Parameters:
            solver: Solver name (optional; affects formatting in the
                returned data dict when solver-specific transforms are
                applied). Pass ``None`` for canonical form.

        Returns:
            (data, inverse_data) tuple.

        Example::

            pb = Problem()
            x = pb.add_variables("x", 3, lb=0)
            pb.set_objective(x[0]**2 + x[1]**2 + x[2]**2)
            pb.add_constraint(x[0] + x[1] + x[2] == 4)

            data, inv = pb.get_problem_data()
            # data["A"], data["c"], data["P"], ... are ready to pass to
            # a solver that speaks the canonical form.
        """
        obj, sense, constraints, soc_constraints, variables = self.assemble()
        sos_constraints = self._sos_constraints

        if solver is not None:
            from .solve import _detect_solver, _compute_cache_key, _param_fingerprint
            # Compute cache key to get assembled matrices (same path as solve())
            cache_key = _compute_cache_key(
                self, solver, obj, constraints, soc_constraints, sos_constraints
            )
            cached = self._cache.get(cache_key) if self._cache_enabled else None
            if cached is None:
                # Assemble once to get the canonical data
                n_vars = len(variables)
                obj_sign = -1.0 if sense == "maximize" else 1.0
                obj_offset = obj.resolved_constant
                lb = np.array([v.lb for v in variables], dtype=np.float64)
                ub = np.array([v.ub for v in variables], dtype=np.float64)
                var_types = np.array(
                    [0 if v.vtype == "continuous" else 1 if v.vtype == "integer" else 2
                     for v in variables],
                    dtype=np.int64,
                )
                if obj.is_linear:
                    obj_rows, obj_cols, obj_vals = obj.to_linear_arrays(n_vars)
                    c = np.zeros(n_vars, dtype=np.float64)
                    for col, val in zip(obj_cols, obj_vals):
                        c[col] = val
                    P = None
                else:
                    q_rows, q_cols, q_vals = obj.to_linear_arrays(n_vars)
                    c = np.zeros(n_vars, dtype=np.float64)
                    for col, val in zip(q_cols, q_vals):
                        c[col] = val
                    P = build_symmetric_P(*obj.to_quadratic_arrays(n_vars), n_vars)
                A_constr, l, u, sense_vec, row_constraints = build_constraints_matrix(
                    constraints, n_vars
                )
            else:
                c = cached["c"]
                P = cached["P"]
                A_constr = cached["A_constr"]
                l = cached["l"]
                u = cached["u"]
                sense_vec = cached["sense_vec"]
                row_constraints = cached["row_constraints"]
                lb = cached["lb"]
                ub = cached["ub"]
                var_types = cached["var_types"]
                obj_offset = cached["obj_offset"]
                n_vars = len(variables)
        else:
            # Canonical form: assemble fresh
            n_vars = len(variables)
            obj_sign = -1.0 if sense == "maximize" else 1.0
            obj_offset = obj.resolved_constant
            lb = np.array([v.lb for v in variables], dtype=np.float64)
            ub = np.array([v.ub for v in variables], dtype=np.float64)
            var_types = np.array(
                [0 if v.vtype == "continuous" else 1 if v.vtype == "integer" else 2
                 for v in variables],
                dtype=np.int64,
            )
            if obj.is_linear:
                obj_rows, obj_cols, obj_vals = obj.to_linear_arrays(n_vars)
                c = np.zeros(n_vars, dtype=np.float64)
                for col, val in zip(obj_cols, obj_vals):
                    c[col] = val
                P = None
            else:
                q_rows, q_cols, q_vals = obj.to_linear_arrays(n_vars)
                c = np.zeros(n_vars, dtype=np.float64)
                for col, val in zip(q_cols, q_vals):
                    c[col] = val
                P = build_symmetric_P(*obj.to_quadratic_arrays(n_vars), n_vars)
            A_constr, l, u, sense_vec, row_constraints = build_constraints_matrix(
                constraints, n_vars
            )

        data = {
            "P": P,
            "c": c,
            "A": A_constr,
            "l": l,
            "u": u,
            "sense": sense_vec,
            "lb": lb,
            "ub": ub,
            "var_types": var_types,
            "obj_offset": obj_offset,
            "obj_sense": sense,
            "row_constraints": row_constraints,
            "soc_constraints": soc_constraints,
            "sos_constraints": sos_constraints,
            "n_vars": n_vars,
            "n_constraints": len(constraints),
        }

        inverse_data = {
            "obj_sense": sense,
            "obj_offset": obj_offset,
            "variables": variables,
            "row_constraints": row_constraints,
        }

        return data, inverse_data

    # --- clone (deep-copy with fresh variables) -----------------------------

    def clone(
        self,
        name: str | None = None,
        x0: np.ndarray | None = None,
    ) -> "Problem":
        """Create a deep copy of this Problem with fresh Variable instances.

        The cloned problem has identical structure (objective, constraints,
        SOC/SOS) but all Variables are new objects with re-indexed positions.
        This is useful for:

        - **Warm-start**: solve the original, then clone and warm-start from
          the original solution.
        - **Scenario analysis**: create multiple copies of a problem with
          different Parameter values or bounds.
        - **Multi-start optimization**: clone and solve from different x0.

        Parameters:
            name: Optional name override for the clone. Defaults to
                ``f"{self.name}_clone"`` (or just ``"clone"`` if self
                has no name).
            x0: Optional initial primal solution (passed to :func:`~obp.solve`
                as a warm start). Not stored on the clone itself — it is
                passed through to :func:`~obp.solve` when you call
                ``solve(clone_pb, x0=x0)``.

        Returns:
            A new :class:`Problem` instance with identical structure.

        Example::

            pb = Problem("original")
            x = pb.add_variables("x", 3, lb=0)
            pb.set_objective(x[0]**2 + x[1]**2 + x[2]**2)
            pb.add_constraint(x[0] + x[1] + x[2] == 4)

            r1 = solve(pb, solver="osqp")
            pb2 = pb.clone("warm_start")
            r2 = solve(pb2, solver="osqp", x0=r1.x)  # warm start
        """
        import copy

        clone_name = name or (f"{self.name}_clone" if self.name else "clone")
        clone = Problem(name=clone_name)

        # Build variable index mapping: old_variable -> new_variable
        var_map: dict[Variable, Variable] = {}
        for i, v in enumerate(self._variables):
            new_v = Variable(
                index=i,
                name=v.name,
                lb=v.lb,
                ub=v.ub,
                vtype=v.vtype,
            )
            var_map[v] = new_v
            clone._variables.append(new_v)

        # Deep-copy objective expression with variable mapping
        if self._objective_expr is not None:
            clone._objective_expr = _reindex_expression(
                self._objective_expr, var_map
            )
        clone._objective_sense = self._objective_sense

        # Deep-copy constraints with variable mapping
        for c in self._constraints:
            new_c = Constraint(
                body=_reindex_expression(c.body, var_map),
                sense=c.sense,
                bound=c.bound,
                name=c.name,
            )
            clone._constraints.append(new_c)

        # Deep-copy SOC constraints with variable mapping
        for soc in self._soc_constraints:
            new_variables = [var_map.get(v, v) for v in soc.variables]
            new_c = _SOCPConstraint(
                variables=new_variables,
                cone_indices=list(soc.cone_indices),
                b=list(soc.b),
                c={var_map.get(k, k): v for k, v in soc.c.items()},
                d=soc.d,
                name=soc.name,
            )
            clone._soc_constraints.append(new_c)

        # Deep-copy SOS constraints with variable mapping
        for sos in self._sos_constraints:
            new_variables = [var_map.get(v, v) for v in sos.variables]
            new_sos = SOSConstraint(
                variables=new_variables,
                type=sos.type,
                weights=list(sos.weights) if sos.weights is not None else None,
                method=sos.method,
                name=sos.name,
            )
            clone._sos_constraints.append(new_sos)

        # Clone does not inherit cache (fresh problem = fresh cache)
        clone._cache = {}

        return clone


# --- Internal helpers --------------------------------------------------------

def _reindex_expression(
    expr: "Expression",
    var_map: dict["Variable", "Variable"],
) -> "Expression":
    """Create a new Expression with Variables replaced according to *var_map*."""
    from .expression import Expression

    new_linear = {var_map.get(v, v): c for v, c in expr._linear.items()}
    new_quad = [
        (var_map.get(vi, vi), var_map.get(vj, vj), c)
        for vi, vj, c in expr._quadratic
    ]
    new_param = {p: c for p, c in expr._param_linear.items()}
    return Expression._from_raw(
        new_linear, new_quad, expr._constant, new_param
    )
