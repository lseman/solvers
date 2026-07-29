"""optBlocks Modeling Language (obp) — lightweight DSL for optimization.

A PuLP-like modeling layer on top of optBlocks' fast C++ solvers.

Quick start::

    from obp import Problem, Variable, solve

    pb = Problem("my_model")
    x = pb.add_variables("x", 3, lb=0)
    pb.set_objective(3*x[0] + 2*x[1] + x[2])
    pb.add_constraint(x[0] + x[1] + x[2] == 4)
    pb.add_constraint(2*x[0] + x[1] <= 5)

    result = solve(pb)
    print(result.x)       # optimal solution
    print(result.obj_val) # objective value

Supported problem types:
- LP (linear programming) via IPM
- QP (quadratic programming) via OSQP or PIQP
- SOCP (second-order cone programming) via IPM
- MIP (mixed-integer programming) via HiGHS, SCIP, or Gurobi, including SOS1/SOS2
- Piecewise-linear interpolation (1D and 2D) and McCormick bilinear relaxation

Other features:
- ``Parameter``: a mutable scalar or ND-tensor (numpy array) usable in
  expressions/bounds -- change ``.value`` and re-solve without rebuilding.
  Use ``ParameterVector(n, name=...)`` to create independent scalar
  Parameters from a single factory.
- **DPP caching**: assembled matrices are cached per-solver keyed on
  Parameter values. ``solve()`` skips expression → matrix conversion
  when Parameters haven't changed, giving ~100x speedup for repeated
  solves (e.g. MPC, trajectory optimization).
  Toggle with ``pb.enable_caching(False)``; clear with ``pb.clear_cache()``.
- **NumPy broadcasting**: ``np.array(pb.add_variables("x", 3)) + 3``
  broadcasts element-wise, returning an object array of Expressions.
- **``Problem.get_problem_data(solver)``**: extract canonical problem
  data (A, b, c, P, lb, ub, var_types) for external solvers.
  Returns ``(data, inverse_data)`` tuple like CVXPY.
- **``Problem.clone()``**: deep-copy problem with fresh Variables.
  Use for warm-start, scenario analysis, or multi-start optimization.
- ``Problem.add_constraints(A @ x, sense, b)``: batch constraints from a
  NumPy object-array of Variables (``A @ x`` already dispatches through
  Expression arithmetic; this adds one row per constraint).
- ``Solution.dual(constraint)``: per-constraint dual value, for the
  ``osqp``, ``highs``, and ``ipm`` backends.
- ``solve(pb, solver="highs", x0=[...])``: warm-start HiGHS with an initial
  primal solution (seeds the MIP incumbent via presolve).
- ``pb.write_lp(path)`` / ``pb.write_mps(path)``: export the problem to a
  standard LP or MPS file (via HiGHS; SOS constraints are expanded first,
  SOC constraints are not representable and raise).
- ``pb.add_indicator_constraint(b, body, sense, bound)``: big-M
  ``b == 1 => (body sense bound)``. ``pb.add_abs_value(x)``: exact
  ``z == |x|``. ``logical_and``/``logical_or``/``logical_not``: AND/OR/NOT
  reformulations for binary variables.

Solver backends:
- ``osqp``: Sparse ADMM QP solver (first-order, large-scale)
- ``piqp``: Interior-point QP solver (predictor-corrector)
- ``ipm``: Conic IPM (LP, QP*, SOCP, conic)
  (* QP via IPM pending — use osqp for QP)
- ``highs``: LP/MIP/QP solver (HiGHS, MIT licensed, default for MIP)
- ``scip``: LP/MIP/QP solver via pyscipopt (ZIB Academic / permissive license)
- ``gurobi``: LP/MIP/QP solver via gurobipy (requires a Gurobi license)
"""

from .export import write_lp, write_mps
from .expression import Expression
from .matrices import SparseMatrixBuilder, build_matrix_from_expr, build_symmetric_P
from .model import (
    BinVar,
    Constraint,
    IntVar,
    Problem,
    Variable,
)
from .parameter import Parameter, ParameterVector
from .sos import SOSConstraint
from .solve import Solution, solve
from .conic import (
    ExpCone,
    PowerCone,
    PowerCone05,
    SOC,
    SOCConstraint,
    exp_cone,
    soc,
    power_cone,
)
from .model import check_mip
from .formulations import (
    mccormick_envelope,
    mccormick_envelope_grid,
    linearize_binary_continuous_product,
    linearize_binary_product,
    piecewise_linear_1d,
    piecewise_linear_2d,
    add_indicator_constraint,
    abs_value,
    logical_and,
    logical_or,
    logical_not,
)

# Re-export key types at the package level
__all__ = [
    # Core
    "Problem",
    "Variable",
    "IntVar",
    "BinVar",
    "Parameter",
    "ParameterVector",
    "Expression",
    "Constraint",
    "SOSConstraint",
    "Solution",
    # Solver
    "solve",
    # Conic
    "SOC",
    "ExpCone",
    "PowerCone",
    "PowerCone05",
    "soc",
    "exp_cone",
    "power_cone",
    "SOCConstraint",
    # Utilities
    "SparseMatrixBuilder",
    "build_matrix_from_expr",
    "build_symmetric_P",
    "write_lp",
    "write_mps",
    # MIP
    "check_mip",
    # Bilinear relaxation
    "mccormick_envelope",
    "mccormick_envelope_grid",
    "linearize_binary_product",
    "linearize_binary_continuous_product",
    # Piecewise-linear interpolation
    "piecewise_linear_1d",
    "piecewise_linear_2d",
    # Indicator / logic
    "add_indicator_constraint",
    "abs_value",
    "logical_and",
    "logical_or",
    "logical_not",
]
