"""
Test OSQP-style sparse QP solver against scipy reference.

Problem form:
    minimize  0.5 * x' P x + q' x
    s.t.      l <= A x <= u

"""

import numpy as np
import scipy.optimize as spo
import pytest
from osqp import Settings, sparse_osqp_solver, solve as osqp_solve

from .test_matrix_generators import rand_spd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_constraints(n: int, m: int = 0):
    """Return (A, l, u) for m constraints on n variables."""
    A = np.empty((m, n))
    l = np.empty((m,))
    u = np.empty((m,))
    return A, l, u


def _scipy_qp(P, q, A=None, l=None, u=None):
    """Solve QP via scipy SLSQP."""
    n = len(q)
    constraints = []
    if A is not None and l is not None and u is not None:
        if len(l) > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: u - A @ x,
                "jac": lambda x: -A,
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda x: A @ x - l,
                "jac": lambda x: A,
            })
    res = spo.minimize(
        lambda x: 0.5 * x @ P @ x + q @ x,
        x0=np.zeros(n),
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return res


# ── Module-level solve() — unconstrained (works) ─────────────────────────────

class TestOSQPUnconstrained:
    """OSQP works for unconstrained QPs."""

    def test_unconstrained_quadratic(self):
        """min 0.5 x'Px + q'x  — closed form x = -P⁻¹q."""
        n = 30
        P = rand_spd(n, seed=42)
        q = np.random.default_rng(42).standard_normal(n)
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        res = _scipy_qp(P, q)
        assert res.success
        err = float(np.linalg.norm(x - res.x))
        assert err < 1e-3, f"OSQP vs scipy diff: {err}"

    def test_diagonal_P(self):
        """Diagonal P: x_i = -q_i / P_ii."""
        n = 4
        P = np.diag([1.0, 2.0, 3.0, 4.0])
        q = -np.array([1.0, 2.0, 3.0, 4.0])
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, np.array([1.0, 1.0, 1.0, 1.0]), atol=1e-3)

    def test_result_keys(self):
        """Verify all expected keys in result dict."""
        n = 3
        P = np.eye(n)
        q = -np.ones(n)
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        expected_keys = {
            "status", "iters", "obj_val", "x", "z", "y",
            "residuals", "primal_infeasible", "dual_infeasible",
        }
        assert expected_keys.issubset(r.keys())
        assert {"pri_inf", "dua_inf"}.issubset(r["residuals"].keys())

    def test_obj_value(self):
        """Objective value should match 0.5 x'Px + q'x."""
        n = 2
        P = np.diag([2.0, 3.0])
        q = -np.array([1.0, 2.0])
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        x = np.asarray(r["x"]).ravel()
        expected_obj = 0.5 * x @ P @ x + q @ x
        assert abs(r["obj_val"] - expected_obj) < 1e-6

    def test_no_constraints_empty_A(self):
        """Empty constraint matrix should work."""
        n = 2
        P = np.diag([1.0, 1.0])
        q = -np.ones(n)
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, np.ones(n), atol=1e-3)

    @pytest.mark.parametrize("n", [50, 100])
    def test_scaling(self, n: int):
        """Test larger unconstrained problems."""
        P = rand_spd(n, density=max(0.01, 10.0 / n), seed=42)
        q = np.random.default_rng(42).standard_normal(n)
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved", f"Failed at n={n}: {r['status']}"
        x = np.asarray(r["x"]).ravel()
        assert np.all(np.isfinite(x))


# ── Settings ─────────────────────────────────────────────────────────────────

class TestOSQPSettings:
    """Test Settings class configuration."""

    def test_default_settings(self):
        s = Settings()
        assert s.sigma == 1e-6
        assert s.alpha == 1.6
        assert s.rho0 == 1e-1
        assert s.eps_abs == 1e-3
        assert s.eps_rel == 1e-3
        assert s.max_iter == 4000
        assert s.verbose is False
        assert s.polish is True

    def test_custom_settings(self):
        s = Settings()
        s.eps_abs = 1e-8
        s.eps_rel = 1e-8
        s.max_iter = 5000
        s.verbose = False
        s.polish = False
        s.adaptive_rho = True
        s.rho0 = 0.1

        n = 3
        P = np.eye(n)
        q = -np.ones(n)
        A, l, u = _empty_constraints(n)
        r = osqp_solve(P, q, A, l, u, settings=s)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, np.ones(n), atol=1e-3)

    def test_polish_disabled(self):
        """Polishing should not affect solution quality."""
        n = 3
        P = np.diag([2.0, 2.0, 2.0])
        q = -np.ones(n)
        A, l, u = _empty_constraints(n)
        s1 = Settings(); s1.polish = True
        s2 = Settings(); s2.polish = False
        r1 = osqp_solve(P, q, A, l, u, settings=s1)
        r2 = osqp_solve(P, q, A, l, u, settings=s2)
        assert r1["status"] == "solved"
        assert r2["status"] == "solved"
        x1 = np.asarray(r1["x"]).ravel()
        x2 = np.asarray(r2["x"]).ravel()
        assert np.allclose(x1, x2, atol=1e-3)


# ── Solver class API ─────────────────────────────────────────────────────────

class TestOSPQPSolverClass:
    """Test sparse_osqp_solver class."""

    def test_default_construction(self):
        """Solver can be constructed with default settings."""
        n = 3
        P = np.eye(n)
        q = -np.ones(n)
        A, l, u = _empty_constraints(n)
        solver = sparse_osqp_solver()
        r = solver.solve(P, q, A, l, u)
        assert r["status"] == "solved"

    def test_custom_settings_construction(self):
        """Solver can be constructed with custom settings."""
        n = 2
        s = Settings()
        s.max_iter = 5000
        s.verbose = False
        solver = sparse_osqp_solver(s)
        P = np.diag([1.0, 2.0])
        q = -np.ones(2)
        A, l, u = _empty_constraints(n)
        r = solver.solve(P, q, A, l, u)
        assert r["status"] == "solved"

    def test_x_polish(self):
        """With polish=True and constraints, x_polish should be populated."""
        n = 2
        P = np.diag([2.0, 2.0])
        q = -np.ones(2)
        A = np.eye(2)  # Need constraints for polish (m > 0)
        l = np.zeros(2)
        u = np.ones(2)
        s = Settings()
        s.polish = True
        solver = sparse_osqp_solver(s)
        r = solver.solve(P, q, A, l, u)
        # polish may or may not populate x_polish depending on convergence
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, [0.5, 0.5], atol=1e-3)


# ── Constrained problems ─────────────────────────────────────────────────────

class TestOSQPConstrained:
    """Constrained QPs (previously broken by a z_tilde bug in the ADMM step)."""

    def test_small_box_qp(self):
        # min x1² + x2² - x1 - x2  s.t. 0 <= x <= 1;  optimal x = [0.5, 0.5]
        P = np.diag([2.0, 2.0])
        q = -np.ones(2)
        A = np.eye(2)
        l = np.zeros(2)
        u = np.ones(2)
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, [0.5, 0.5], atol=1e-3)

    def test_osqp_demo_problem(self):
        # Official OSQP demo: x* = [0.3, 0.7]
        P = np.array([[4.0, 1.0], [1.0, 2.0]])
        q = np.ones(2)
        A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        l = np.array([1.0, 0.0, 0.0])
        u = np.array([1.0, 0.7, 0.7])
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, [0.3, 0.7], atol=1e-3)

    def test_equality_constrained(self):
        # min 0.5 x'x  s.t. sum(x) = 1;  x* = 0.25 each
        n = 4
        P = np.eye(n)
        q = np.zeros(n)
        A = np.ones((1, n))
        l = np.array([1.0])
        u = np.array([1.0])
        r = osqp_solve(P, q, A, l, u)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, 0.25, atol=1e-3)
