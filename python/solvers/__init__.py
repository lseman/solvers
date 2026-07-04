"""Python bindings for the solvers headers."""

try:
    from .ipm_solver import IPSolver, IPMSolution, SolverType, solve_ipm
except ImportError:
    IPSolver = None
    IPMSolution = None
    SolverType = None
    solve_ipm = None

try:
    from .ldlt import LDLTSolver, solve as solve_ldlt
except ImportError:
    LDLTSolver = None
    solve_ldlt = None

__all__ = [
    "IPSolver",
    "IPMSolution",
    "SolverType",
    "solve_ipm",
    "LDLTSolver",
    "solve_ldlt",
]
