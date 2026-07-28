#!/usr/bin/env python3
"""optBlocks Modeling Language (obp) — Quickstart Examples.

Demonstrates the lightweight DSL for building optimization models.
"""

import sys
from pathlib import Path

# Ensure we can import from the local python/ directory
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "python"))

from obp import (
    Problem,
    Variable,
    solve,
    Expression,
    SOC,
    soc,
    IntVar,
    BinVar,
)


def example_lp():
    """Linear programming: minimize 3x + y + 2z subject to constraints."""
    print("=" * 60)
    print("Example 1: Linear Programming")
    print("=" * 60)

    pb = Problem("blend")
    x = pb.add_variables("x", 3, lb=0)
    pb.set_objective(3 * x[0] + 1 * x[1] + 2 * x[2])
    pb.add_constraint(x[0] + x[1] + x[2] == 4, "sum_eq")
    pb.add_constraint(2 * x[0] + 0.5 * x[1] + x[2] <= 5, "mix_limit")
    pb.add_constraint(x[0] <= 3, "x0_cap")

    print(f"Model: {pb}")
    print(f"Objective: 3x₀ + x₁ + 2x₂")
    print(f"Constraints: x₀+x₁+x₂=4, 2x₀+0.5x₁+x₂≤5, x₀≤3")

    try:
        result = solve(pb)
        print(f"Status: {result.status}")
        print(f"Objective value: {result.obj_val:.6f}")
        print(f"Solution: x = {result.x}")
    except RuntimeError as e:
        print(f"Solver not available: {e}")
    print()


def example_qp():
    """Quadratic programming: minimize x² + y² subject to x + y ≤ 1, 0 ≤ x,y ≤ 1."""
    print("=" * 60)
    print("Example 2: Quadratic Programming")
    print("=" * 60)

    pb = Problem("qp_demo")
    x = pb.add_variables("x", 2, lb=0, ub=1)
    pb.set_objective(x[0] ** 2 + x[1] ** 2)
    pb.add_constraint(x[0] + x[1] <= 1, "sum_limit")

    print(f"Model: {pb}")
    print(f"Objective: x₀² + x₁²")
    print(f"Constraints: x₀+x₁≤1, 0≤x,y≤1")

    try:
        result = solve(pb, solver="osqp")
        print(f"Status: {result.status}")
        print(f"Objective value: {result.obj_val:.6f}")
        print(f"Solution: x = {result.x}")
        print("Expected: x = [0.5, 0.5], obj = 0.5")
    except RuntimeError as e:
        print(f"Solver not available: {e}")
    print()


def example_qp2():
    """QP with equality constraints via PIQP."""
    print("=" * 60)
    print("Example 3: QP with Equality Constraints (PIQP)")
    print("=" * 60)

    pb = Problem("qp_eq")
    x = pb.add_variables("x", 3)
    pb.set_objective(2 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2 - x[0] - x[1] - x[2])
    pb.add_constraint(x[0] + x[1] <= 1, "ineq")
    pb.add_constraint(x[1] + x[2] == 1, "eq")

    print(f"Model: {pb}")
    print(f"Objective: 2x₀²+2x₁²+2x₂²-x₀-x₁-x₂")

    try:
        result = solve(pb, solver="piqp")
        print(f"Status: {result.status}")
        print(f"Objective value: {result.obj_val:.6f}")
        print(f"Solution: x = {result.x}")
    except RuntimeError as e:
        print(f"Solver not available: {e}")
    print()


def example_socp():
    """Second-order cone programming."""
    print("=" * 60)
    print("Example 4: Second-Order Cone Programming (SOCP)")
    print("=" * 60)

    pb = Problem("socp")
    x = pb.add_variables("x", 3, lb=0)
    pb.set_objective(x[0] + x[1] + x[2])
    # ||x[1], x[2]||₂ ≤ x[0]  (rotated SOC)
    pb.add_soc_constraint(soc(x, cone_indices=[1, 2]), "risk_limit")

    print(f"Model: {pb}")
    print(f"Objective: x₀+x₁+x₂")
    print(f"SOC: ||x[1],x[2]||₂ ≤ x[0]")

    try:
        result = solve(pb, solver="ipm")
        print(f"Status: {result.status}")
        print(f"Objective value: {result.obj_val:.6f}")
        print(f"Solution: x = {result.x}")
    except RuntimeError as e:
        print(f"Solver not available: {e}")
    print()


def example_mip():
    """MIP: integer and binary variables via HiGHS."""
    print("=" * 60)
    print("Example 5: MIP Variables (HiGHS)")
    print("=" * 60)

    pb = Problem("mip_demo")
    x = pb.add_variables("x", 2, vtype="integer", lb=0)
    b = pb.add_variables("b", 1, vtype="binary")  # single BinVar, not list
    pb.set_objective(-x[0] - 2*x[1] + b)  # use b directly, not b[0]
    pb.add_constraint(x[0] + x[1] + b <= 3, "sum_limit")
    pb.add_constraint(2*x[0] + x[1] <= 4, "mix_limit")
    pb.add_constraint(x[0] >= 1, "x0_lower")

    print(f"Model: {pb}")
    print(f"Objective: -x0 - 2*x1 + b (minimize)")
    print(f"Variables: 2 integer + 1 binary")

    result = solve(pb)  # auto-detects MIP → highs
    print(f"Status: {result.status}")
    print(f"Objective: {result.obj_val}")
    print(f"x0={result.x[0]:.0f}, x1={result.x[1]:.0f}, b={result.x[2]:.0f}")
    print("Expected: x0=1, x1=2, b=0, obj=-5")
    print()


def example_expression_api():
    """Demonstrates the Expression arithmetic API."""
    print("=" * 60)
    print("Example 6: Expression API")
    print("=" * 60)

    from obp.expression import Expression

    x = Expression.from_variable(Variable(0, "x"))
    y = Expression.from_variable(Variable(1, "y"))

    # Linear arithmetic
    e1 = 3 * x + 2 * y - 5
    print(f"3x + 2y - 5 = {e1}")

    # Quadratic: (x + y)²
    e2 = (x + y) ** 2
    print(f"(x + y)² = {e2}")

    # Constraint expressions
    pb = Problem()
    xs = pb.add_variables("xs", 2)
    pb.add_constraint(xs[0] + xs[1] <= 10, "budget")
    pb.add_constraint(2 * xs[0] + xs[1] >= 5, "demand")
    pb.add_constraint(xs[0] == 3, "fixed")

    print(f"Model with {pb.n_constraints} constraints: {pb}")
    print()


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  optBlocks Modeling Language (obp) — Quickstart Examples")
    print("█" * 60 + "\n")

    example_expression_api()
    example_lp()
    example_qp()
    example_qp2()
    example_socp()
    example_mip()

    print("=" * 60)
    print("Done. All examples completed.")
    print("=" * 60)
