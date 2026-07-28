"""mccormick — McCormick envelope relaxation for bilinear terms.

For w = x*y with x in [xL, xU] and y in [yL, yU], the McCormick envelope
is the tightest possible linear relaxation using only the box bounds:

    w >= xL*y + yL*x - xL*yL   (underestimator 1)
    w >= xU*y + yU*x - xU*yU   (underestimator 2)
    w <= xU*y + yL*x - xU*yL   (overestimator 1)
    w <= xL*y + yU*x - xL*yU   (overestimator 2)

This is exact when x or y is fixed (one bound is tight) and is the convex/
concave envelope of x*y over the box [xL,xU] x [yL,yU] in general. Useful
for relaxing bilinear terms into an LP/MIP-solvable form (e.g. for solvers
that don't support quadratic objectives/constraints, such as SCIP via
this library, or when a global bound on a bilinear term is needed).

Usage::

    pb = Problem()
    x = pb.add_variables("x", lb=0, ub=10)
    y = pb.add_variables("y", lb=0, ub=5)
    w = mccormick_envelope(pb, x, y, name="w_xy")
    pb.set_objective(w, sense="maximize")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..expression import Expression
    from ..model import Problem, Variable


def mccormick_envelope(
    problem: "Problem",
    x: "Variable",
    y: "Variable",
    name: str = "",
) -> "Variable":
    """Relax the bilinear term ``w = x * y`` via its McCormick envelope.

    Adds a new continuous variable ``w`` to *problem*, bounded to
    ``[min(products), max(products)]`` over the box corners, plus the four
    McCormick linear inequalities relating it to *x* and *y*. Returns
    ``w`` — use it in place of ``x * y`` in the objective/constraints.

    Parameters:
        problem: The :class:`Problem` to add the variable and constraints to.
        x: First factor. Must have finite lb and ub.
        y: Second factor. Must have finite lb and ub.
        name: Optional name prefix for the new variable (default ``"w"``).

    Returns:
        The new auxiliary :class:`Variable` representing the relaxed ``x*y``.

    Raises:
        ValueError: If *x* or *y* has a non-finite bound.

    Note:
        This is a relaxation, not an exact reformulation — ``w`` may not
        equal ``x*y`` exactly at every feasible point unless *x* or *y* is
        at one of its bounds. It is exact for the specific corner points of
        the box, and it is the convex hull of ``{(x,y,xy)}`` over the box,
        so it is the tightest possible bound using only box constraints.
    """
    for v, label in ((x, "x"), (y, "y")):
        if v.lb == -float("inf") or v.ub == float("inf"):
            raise ValueError(
                f"mccormick_envelope requires finite bounds on both variables "
                f"({label}={v.name or v.index!r} has lb={v.lb}, ub={v.ub})"
            )

    xL, xU = x.lb, x.ub
    yL, yU = y.lb, y.ub

    corner_products = [xL * yL, xL * yU, xU * yL, xU * yU]
    w_lb = min(corner_products)
    w_ub = max(corner_products)

    prefix = name or f"_mccormick_w{x.index}_{y.index}"
    w = problem.add_variables(prefix, n=1, lb=w_lb, ub=w_ub)

    # w >= xL*y + yL*x - xL*yL  <=>  w - yL*x - xL*y >= -xL*yL
    problem.add_constraint(w - yL * x - xL * y >= -xL * yL, f"{prefix}_under1")
    # w >= xU*y + yU*x - xU*yU
    problem.add_constraint(w - yU * x - xU * y >= -xU * yU, f"{prefix}_under2")
    # w <= xU*y + yL*x - xU*yL
    problem.add_constraint(w - yL * x - xU * y <= -xU * yL, f"{prefix}_over1")
    # w <= xL*y + yU*x - xL*yU
    problem.add_constraint(w - yU * x - xL * y <= -xL * yU, f"{prefix}_over2")

    return w


def mccormick_envelope_grid(
    problem: "Problem",
    x: "Variable",
    y: "Variable",
    x_breaks: list[float],
    y_breaks: list[float],
    name: str = "",
) -> tuple["Variable", dict[tuple[int, int], "Variable"]]:
    """Piecewise McCormick relaxation of ``z = x * y`` over a breakpoint grid.

    Partitions ``[xL,xU] x [yL,yU]`` into a ``(len(x_breaks)-1) x
    (len(y_breaks)-1)`` grid of cells, picks exactly one active cell via
    binary indicators, and applies the plain McCormick envelope within that
    cell only. Refining the grid (more breakpoints) shrinks the relaxation
    gap toward the true bilinear surface, at the cost of one binary per
    cell — this is the standard MIP-based tightening of the single-cell
    envelope (see :func:`mccormick_envelope`), used e.g. in
    Misener & Floudas' piecewise-McCormick global optimization approach.

    Parameters:
        problem: The :class:`Problem` to add variables/constraints to.
        x: First factor. Must have finite lb and ub.
        y: Second factor. Must have finite lb and ub.
        x_breaks: Increasing breakpoints partitioning [x.lb, x.ub];
            must start at x.lb and end at x.ub.
        y_breaks: Increasing breakpoints partitioning [y.lb, y.ub];
            must start at y.lb and end at y.ub.
        name: Optional name prefix (default derived from x, y indices).

    Returns:
        (z, delta) where ``z`` is the new auxiliary variable representing
        the relaxed ``x*y``, and ``delta`` maps ``(i, j)`` cell indices to
        the binary indicator variable for that cell (1 iff the grid cell
        ``[x_breaks[i], x_breaks[i+1]] x [y_breaks[j], y_breaks[j+1]]`` is
        the active one).

    Raises:
        ValueError: If bounds are non-finite, fewer than 2 breakpoints are
            given on either axis, breakpoints aren't strictly increasing,
            or they don't span [lb, ub] exactly.
    """
    from ..model import BinVar, Variable

    for v, label in ((x, "x"), (y, "y")):
        if v.lb == -float("inf") or v.ub == float("inf"):
            raise ValueError(
                f"mccormick_envelope_grid requires finite bounds on both variables "
                f"({label}={v.name or v.index!r} has lb={v.lb}, ub={v.ub})"
            )

    def _check_breaks(breaks: list[float], lo: float, hi: float, label: str) -> None:
        if len(breaks) < 2:
            raise ValueError(f"{label}_breaks needs at least 2 points, got {len(breaks)}")
        if any(breaks[i] >= breaks[i + 1] for i in range(len(breaks) - 1)):
            raise ValueError(f"{label}_breaks must be strictly increasing")
        if breaks[0] != lo or breaks[-1] != hi:
            raise ValueError(
                f"{label}_breaks must span [{lo}, {hi}] exactly "
                f"(got [{breaks[0]}, {breaks[-1]}])"
            )

    _check_breaks(x_breaks, x.lb, x.ub, "x")
    _check_breaks(y_breaks, y.lb, y.ub, "y")

    Kx = len(x_breaks) - 1
    Ky = len(y_breaks) - 1
    prefix = name or f"_gridmc_z{x.index}_{y.index}"

    delta: dict[tuple[int, int], "Variable"] = {}
    xij: dict[tuple[int, int], "Variable"] = {}
    yij: dict[tuple[int, int], "Variable"] = {}
    zij: dict[tuple[int, int], "Variable"] = {}

    for i in range(Kx):
        a, b = float(x_breaks[i]), float(x_breaks[i + 1])
        for j in range(Ky):
            c, d = float(y_breaks[j]), float(y_breaks[j + 1])
            delta[i, j] = problem.add_variables(f"{prefix}_delta_{i}_{j}", vtype="binary")
            xij[i, j] = problem.add_variables(f"{prefix}_x_{i}_{j}", lb=min(0.0, a), ub=max(0.0, b))
            yij[i, j] = problem.add_variables(f"{prefix}_y_{i}_{j}", lb=min(0.0, c), ub=max(0.0, d))
            corner_products = [a * c, a * d, b * c, b * d]
            zij[i, j] = problem.add_variables(
                f"{prefix}_z_{i}_{j}",
                lb=min(0.0, min(corner_products)),
                ub=max(0.0, max(corner_products)),
            )

    z = problem.add_variables(prefix, lb=min(0.0, x.lb * y.lb, x.lb * y.ub, x.ub * y.lb, x.ub * y.ub),
                               ub=max(0.0, x.lb * y.lb, x.lb * y.ub, x.ub * y.lb, x.ub * y.ub))

    problem.add_constraint(_sum_vars(delta.values()) == 1.0, f"{prefix}_one_cell")
    problem.add_constraint(x - _sum_vars(xij.values()) == 0.0, f"{prefix}_x_disagg")
    problem.add_constraint(y - _sum_vars(yij.values()) == 0.0, f"{prefix}_y_disagg")
    problem.add_constraint(z - _sum_vars(zij.values()) == 0.0, f"{prefix}_z_disagg")

    for i in range(Kx):
        a, b = float(x_breaks[i]), float(x_breaks[i + 1])
        for j in range(Ky):
            c, d = float(y_breaks[j]), float(y_breaks[j + 1])
            Dij = delta[i, j]
            Xij = xij[i, j]
            Yij = yij[i, j]
            Zij = zij[i, j]

            problem.add_constraint(Xij - a * Dij >= 0.0, f"{prefix}_x_lb_{i}_{j}")
            problem.add_constraint(Xij - b * Dij <= 0.0, f"{prefix}_x_ub_{i}_{j}")
            problem.add_constraint(Yij - c * Dij >= 0.0, f"{prefix}_y_lb_{i}_{j}")
            problem.add_constraint(Yij - d * Dij <= 0.0, f"{prefix}_y_ub_{i}_{j}")

            problem.add_constraint(
                Zij - a * Yij - c * Xij + a * c * Dij >= 0.0, f"{prefix}_mc_lb1_{i}_{j}"
            )
            problem.add_constraint(
                Zij - b * Yij - d * Xij + b * d * Dij >= 0.0, f"{prefix}_mc_lb2_{i}_{j}"
            )
            problem.add_constraint(
                Zij - b * Yij - c * Xij + b * c * Dij <= 0.0, f"{prefix}_mc_ub1_{i}_{j}"
            )
            problem.add_constraint(
                Zij - a * Yij - d * Xij + a * d * Dij <= 0.0, f"{prefix}_mc_ub2_{i}_{j}"
            )

    return z, delta


def _sum_vars(vars_) -> "Expression":
    from ..expression import Expression

    expr = Expression.constant(0.0)
    for v in vars_:
        expr = expr + Expression.from_variable(v)
    return expr
