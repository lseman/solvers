# %% [markdown]
# # IPM vs HiGHS and PuLP
#
# This notebook compares the `ipm_solver` nanobind module against SciPy HiGHS and, when installed, PuLP on small linear programs.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT.parent

BUILD_SOLVERS = ROOT / "build"
if str(BUILD_SOLVERS) not in sys.path:
    sys.path.insert(0, str(BUILD_SOLVERS))

import ipm_solver
import numpy as np
from ipm_solver import SolverType
from scipy.optimize import linprog

try:
    import pulp
except ImportError:
    pulp = None

print(f"using {BUILD_SOLVERS}")
print("PuLP available:", pulp is not None)

# %%
problems = [
    {
        "name": "simplex edge",
        "c": np.array([1.0, 2.0]),
        "A_eq": np.array([[1.0, 1.0]]),
        "b_eq": np.array([1.0]),
        "bounds": [(0.0, None), (0.0, None)],
    },
    {
        "name": "bounded blend",
        "c": np.array([3.0, 1.0, 2.0]),
        "A_eq": np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 1.0]]),
        "b_eq": np.array([4.0, 5.0]),
        "bounds": [(0.0, 3.0), (0.0, 4.0), (0.0, None)],
    },
]


def bounds_to_vectors(bounds):
    lb = np.array([0.0 if lo is None else lo for lo, _ in bounds], dtype=float)
    ub = np.array([np.inf if hi is None else hi for _, hi in bounds], dtype=float)
    return lb, ub


def residual(A, b, x):
    if x is None:
        return np.nan
    return float(np.linalg.norm(A @ np.asarray(x) - b, ord=np.inf))


# %%
def solve_highs(problem):
    res = linprog(
        problem["c"],
        A_eq=problem["A_eq"],
        b_eq=problem["b_eq"],
        bounds=problem["bounds"],
        method="highs",
    )
    return {
        "solver": "HiGHS",
        "status": res.message,
        "objective": float(res.fun) if res.success else np.nan,
        "x": res.x if res.success else None,
    }


def solve_ipm(problem, solver_type=SolverType.LDLT):
    lb, ub = bounds_to_vectors(problem["bounds"])
    try:
        s = ipm_solver.IPSolver()
        s.set_solver_type(solver_type)
        sol = s.solve(
            problem["A_eq"],
            problem["b_eq"],
            problem["c"],
            lb,
            ub,
            ["="] * len(problem["b_eq"]),
            1e-7,
        )
        return {
            "solver": f"IPM ({solver_type.name})",
            "status": sol.status,
            "objective": float(sol.objective),
            "x": np.array(sol.x, dtype=float),
        }
    except Exception as exc:
        return {
            "solver": f"IPM ({solver_type.name})",
            "status": repr(exc),
            "objective": np.nan,
            "x": None,
        }


def solve_pulp(problem):
    if pulp is None:
        return {
            "solver": "PuLP",
            "status": "not installed",
            "objective": np.nan,
            "x": None,
        }
    model = pulp.LpProblem(problem["name"], pulp.LpMinimize)
    xs = []
    for j, (lo, hi) in enumerate(problem["bounds"]):
        xs.append(pulp.LpVariable(f"x_{j}", lowBound=lo, upBound=hi))
    model += pulp.lpSum(float(cj) * xj for cj, xj in zip(problem["c"], xs))
    for i, row in enumerate(problem["A_eq"]):
        model += pulp.lpSum(float(aij) * xj for aij, xj in zip(row, xs)) == float(
            problem["b_eq"][i]
        )
    try:
        status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
        x = np.array([pulp.value(xj) for xj in xs], dtype=float)
        return {
            "solver": "PuLP",
            "status": pulp.LpStatus[status_code],
            "objective": float(pulp.value(model.objective)),
            "x": x,
        }
    except Exception as exc:
        return {"solver": "PuLP", "status": repr(exc), "objective": np.nan, "x": None}


# %%
rows = []
for problem in problems:
    highs = solve_highs(problem)
    baseline = highs["objective"]
    for result in [
        highs,
        solve_pulp(problem),
        solve_ipm(problem, SolverType.LDLT),
        solve_ipm(problem, SolverType.FRONTAL),
        solve_ipm(problem, SolverType.SUPERNOODAL),
        solve_ipm(problem, SolverType.QD_LDLT),
    ]:
        x = result["x"]
        rows.append({
            "problem": problem["name"],
            "solver": result["solver"],
            "status": result["status"],
            "objective": result["objective"],
            "obj_gap_vs_highs": result["objective"] - baseline,
            "eq_residual_inf": residual(problem["A_eq"], problem["b_eq"], x),
            "x": None if x is None else np.round(x, 8).tolist(),
        })

for row in rows:
    print(row)
