"""pytest fixtures for obp tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow end-to-end tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow (skip unless --run-slow)")


def skip_if_no_solver(solver_name: str) -> bool:
    """Check if a solver backend is available."""
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[2] / "obp"
    root = pkg
    while not (root / "build").is_dir() and root.parent != root:
        root = root.parent
    build = root / "build"

    # Check local binding
    sol_map = {
        "osqp": "osqp.*.so",
        "piqp": "piqp.*.so",
        "ipm": "ipm_solver.*.so",
    }
    pattern = sol_map.get(solver_name)
    if pattern and list(build.glob(pattern)):
        return False  # binding found, don't skip

    # Check pip-installed
    try:
        if solver_name == "osqp":
            import osqp  # noqa: F401
        elif solver_name == "piqp":
            import piqp  # noqa: F401
        elif solver_name == "ipm":
            import ipm_solver  # noqa: F401
        return False  # import succeeded
    except ImportError:
        pass

    return True  # no solver available
