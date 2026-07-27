"""pwl — piecewise-linear interpolation of f(x) and f(x, y).

Given breakpoints and function values, adds a new variable z constrained
to equal the piecewise-linear interpolant of the input variable(s) — not
to be confused with :mod:`solvers.obp.sos` (raw "at most k adjacent
nonzero" constraints on existing variables) or :mod:`solvers.obp.mccormick`
(relaxation of a bilinear term, not exact interpolation of an arbitrary
function).

1D uses the standard V-formulation (convex-combination weights lambda_i
over the breakpoints, reusing the verified SOS2 machinery in
:mod:`solvers.obp.sos` to enforce "at most two adjacent lambda nonzero").
2D triangulates the (x, y) breakpoint grid and applies the same
convex-combination idea over triangle vertices, following
PiecewiseLinearOpt.jl's ``BivariatePWLFunction`` / ``:CC`` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..expression import Expression

if TYPE_CHECKING:
    from ..model import Problem, Variable


def _sum_vars(vars_) -> Expression:
    expr = Expression.constant(0.0)
    for v in vars_:
        expr = expr + Expression.from_variable(v)
    return expr


def piecewise_linear_1d(
    problem: "Problem",
    x: "Variable",
    breakpoints: list[float],
    values: list[float],
    method: Literal["cc", "dlog", "log"] = "cc",
    name: str = "",
) -> "Variable":
    """Piecewise-linear interpolation of z = f(x) given breakpoints.

    Adds convex-combination weights lambda_0..lambda_{n-1} (one per
    breakpoint), links x and the returned z to weighted sums of the
    breakpoints/values, and enforces "at most two adjacent lambda
    nonzero" via the requested SOS2 formulation (see
    :class:`solvers.obp.sos.SOSConstraint` for what each method means).

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        x: The input variable. Not otherwise bounded by this function —
            add your own bounds/constraints on x if you need x to stay
            within [breakpoints[0], breakpoints[-1]].
        breakpoints: Strictly increasing x-coordinates, length >= 2.
        values: f(breakpoints[i]) for each i; same length as breakpoints.
        method: SOS2 formulation for the lambda weights — ``"cc"``
            (default), ``"dlog"``, or ``"log"``.
        name: Optional name prefix (default derived from x's index).

    Returns:
        The new auxiliary :class:`Variable` z = f(x) (piecewise-linear
        interpolant).

    Raises:
        ValueError: If fewer than 2 breakpoints are given, breakpoints
            aren't strictly increasing, or breakpoints/values lengths differ.
    """
    n = len(breakpoints)
    if n < 2:
        raise ValueError(f"breakpoints needs at least 2 points, got {n}")
    if len(values) != n:
        raise ValueError(
            f"breakpoints ({n}) and values ({len(values)}) must have the same length"
        )
    if any(breakpoints[i] >= breakpoints[i + 1] for i in range(n - 1)):
        raise ValueError("breakpoints must be strictly increasing")

    prefix = name or f"_pwl1d_z{x.index}"

    lam = [
        problem.add_variables(f"{prefix}_lam{i}", lb=0.0, ub=1.0)
        for i in range(n)
    ]

    z = problem.add_variables(prefix, lb=min(values), ub=max(values))

    problem.add_constraint(_sum_vars(lam) == 1.0, f"{prefix}_lam_sum")

    x_body = Expression.constant(0.0)
    z_body = Expression.constant(0.0)
    for i in range(n):
        x_body = x_body + float(breakpoints[i]) * lam[i]
        z_body = z_body + float(values[i]) * lam[i]
    problem.add_constraint(Expression.from_variable(x) - x_body == 0.0, f"{prefix}_x_link")
    problem.add_constraint(Expression.from_variable(z) - z_body == 0.0, f"{prefix}_z_link")

    problem.add_sos_constraint(lam, type=2, method=method, name=f"{prefix}_sos2")

    return z


def piecewise_linear_2d(
    problem: "Problem",
    x: "Variable",
    y: "Variable",
    x_breaks: list[float],
    y_breaks: list[float],
    z_grid: list[list[float]],
    name: str = "",
) -> "Variable":
    """Piecewise-linear interpolation of z = f(x, y) over a breakpoint grid.

    Splits the ``(len(x_breaks)-1) x (len(y_breaks)-1)`` grid of rectangular
    cells into two triangles each, then applies the convex-combination
    ("CC") formulation over the triangulation: one binary per triangle
    selects the active triangle, each grid-vertex weight lambda is bounded
    by the sum of the (up to 6) triangles touching it, and z/x/y are
    weighted sums of the triangle-vertex breakpoints/values. This matches
    PiecewiseLinearOpt.jl's ``BivariatePWLFunction`` + ``:CC`` method
    (:UnionJack-style diagonal split).

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        x: First input variable.
        y: Second input variable.
        x_breaks: Strictly increasing x-coordinates, length >= 2.
        y_breaks: Strictly increasing y-coordinates, length >= 2.
        z_grid: f(x_breaks[i], y_breaks[j]) as z_grid[i][j]; shape
            ``(len(x_breaks), len(y_breaks))``.
        name: Optional name prefix (default derived from x, y indices).

    Returns:
        The new auxiliary :class:`Variable` z = f(x, y) (piecewise-linear
        interpolant over the triangulated grid).

    Raises:
        ValueError: If fewer than 2 breakpoints are given on either axis,
            breakpoints aren't strictly increasing, or z_grid's shape
            doesn't match (len(x_breaks), len(y_breaks)).
    """
    from ..model import BinVar

    nx = len(x_breaks)
    ny = len(y_breaks)
    if nx < 2:
        raise ValueError(f"x_breaks needs at least 2 points, got {nx}")
    if ny < 2:
        raise ValueError(f"y_breaks needs at least 2 points, got {ny}")
    if any(x_breaks[i] >= x_breaks[i + 1] for i in range(nx - 1)):
        raise ValueError("x_breaks must be strictly increasing")
    if any(y_breaks[i] >= y_breaks[i + 1] for i in range(ny - 1)):
        raise ValueError("y_breaks must be strictly increasing")
    if len(z_grid) != nx or any(len(row) != ny for row in z_grid):
        raise ValueError(
            f"z_grid must have shape ({nx}, {ny}), got "
            f"({len(z_grid)}, {len(z_grid[0]) if z_grid else 0})"
        )

    prefix = name or f"_pwl2d_z{x.index}_{y.index}"

    # lambda[i][j]: convex-combination weight of grid vertex (i, j)
    lam = [
        [problem.add_variables(f"{prefix}_lam_{i}_{j}", lb=0.0, ub=1.0) for j in range(ny)]
        for i in range(nx)
    ]

    all_z = [z_grid[i][j] for i in range(nx) for j in range(ny)]
    z = problem.add_variables(prefix, lb=min(all_z), ub=max(all_z))

    all_lam = [lam[i][j] for i in range(nx) for j in range(ny)]
    problem.add_constraint(_sum_vars(all_lam) == 1.0, f"{prefix}_lam_sum")

    x_body = Expression.constant(0.0)
    y_body = Expression.constant(0.0)
    z_body = Expression.constant(0.0)
    for i in range(nx):
        for j in range(ny):
            x_body = x_body + float(x_breaks[i]) * lam[i][j]
            y_body = y_body + float(y_breaks[j]) * lam[i][j]
            z_body = z_body + float(z_grid[i][j]) * lam[i][j]
    problem.add_constraint(Expression.from_variable(x) - x_body == 0.0, f"{prefix}_x_link")
    problem.add_constraint(Expression.from_variable(y) - y_body == 0.0, f"{prefix}_y_link")
    problem.add_constraint(Expression.from_variable(z) - z_body == 0.0, f"{prefix}_z_link")

    # Triangulate each cell [i,i+1] x [j,j+1] into two triangles along the
    # (i,j)-(i+1,j+1) diagonal: {SW,NW,NE} and {SW,NE,SE}.
    triangles: list[list[tuple[int, int]]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            sw, nw, ne, se = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)
            triangles.append([sw, nw, ne])
            triangles.append([sw, ne, se])

    y_tri: list["Variable"] = [
        problem.add_variables(f"{prefix}_tri{t}", vtype="binary") for t in range(len(triangles))
    ]
    problem.add_constraint(_sum_vars(y_tri) == 1.0, f"{prefix}_tri_card")

    touching: dict[tuple[int, int], list["Variable"]] = {
        (i, j): [] for i in range(nx) for j in range(ny)
    }
    for t, verts in enumerate(triangles):
        for v in verts:
            touching[v].append(y_tri[t])

    for i in range(nx):
        for j in range(ny):
            problem.add_constraint(
                Expression.from_variable(lam[i][j]) - _sum_vars(touching[(i, j)]) <= 0.0,
                f"{prefix}_lam_{i}_{j}_bound",
            )

    return z
