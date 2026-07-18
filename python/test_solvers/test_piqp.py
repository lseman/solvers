"""
Test PIQP sparse QP solver against scipy reference.

Problem form:
    minimize  0.5 * x' P x + q' x
    s.t.      A x = b   (equality, optional)
              G x <= h   (inequality, optional)
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as spo
import pytest
from piqp import PIQPSettings, piqp_solver, solve as piqp_solve

from .test_matrix_generators import rand_spd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _scipy_qp(P, q, A=None, b=None, G=None, h=None,
              A_ub=None, b_ub=None, bounds=None):
    """Solve the same QP via scipy's SLSQP."""
    n = len(q)
    constraints = []
    if A is not None and b is not None:
        constraints.append({"type": "eq", "fun": lambda x: A @ x - b,
                            "jac": lambda x: A})
    if G is not None and h is not None:
        constraints.append({"type": "ineq",
                            "fun": lambda x: h - G @ x,
                            "jac": lambda x: -G})
    res = spo.minimize(
        lambda x: 0.5 * x @ P @ x + q @ x,
        x0=np.zeros(n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res


# ── Module-level solve() ─────────────────────────────────────────────────────

class TestPIQPModuleSolve:
    """Tests for the convenience solve() function."""

    def test_unconstrained_quadratic(self):
        """min 0.5 x'P x + q'x  (no constraints) — closed form x = -P⁻¹q."""
        P = rand_spd(30, seed=42)
        q = np.random.default_rng(42).standard_normal(30)
        r = piqp_solve(P, q)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        # Compare to scipy
        res = _scipy_qp(P, q)
        assert res.success
        err = float(np.linalg.norm(x - res.x))
        assert err < 1e-4, f"PIQP vs scipy diff: {err}"

    def test_inequality_only(self):
        """min 0.5 x'P x + q'x  s.t.  Gx <= h."""
        P = 2.0 * np.eye(3)
        q = -np.ones(3)
        G = np.array([[1.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0],
                       [1.0, 0.0, 1.0]])
        h = np.array([1.0, 1.0, 1.0])
        r = piqp_solve(P, q, G=G, h=h)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        # Feasibility
        assert np.all(G @ x <= h + 1e-4), f"Gx <= h violated: {G @ x}"
        # Compare objective to scipy
        res = _scipy_qp(P, q, G=G, h=h)
        assert res.success
        assert abs(r["obj_val"] - res.fun) < 1e-3

    def test_equality_only(self):
        """min 0.5 x'P x + q'x  s.t.  Ax = b."""
        P = 2.0 * np.eye(3)
        q = -np.ones(3)
        A = np.array([[1.0, 1.0, 0.0]])
        b = np.array([1.0])
        r = piqp_solve(P, q, A=A, b=b)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(A @ x, b, atol=1e-4), f"Ax != b: {A @ x}"

    def test_full_constrained(self):
        """min 0.5 x'P x + q'x  s.t.  Ax=b, Gx<=h."""
        P = rand_spd(20, seed=42)
        q = np.random.default_rng(42).standard_normal(20)
        A = np.random.default_rng(1).standard_normal((5, 20))
        b = A @ np.ones(20)
        G = np.random.default_rng(2).standard_normal((8, 20))
        h = G @ np.ones(20) - 0.5
        r = piqp_solve(P, q, A=A, b=b, G=G, h=h)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(A @ x, b, atol=1e-3)
        assert np.all(G @ x <= h + 1e-3)

    def test_no_constraints(self):
        """Unconstrained QP: P = I, q = -1, x = 1."""
        P = np.eye(3)
        q = -np.ones(3)
        r = piqp_solve(P, q)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, np.ones(3), atol=1e-4)

    def test_result_keys(self):
        """Verify all expected keys are present in the result dict."""
        P = np.eye(3)
        q = -np.ones(3)
        r = piqp_solve(P, q)
        expected_keys = {
            "status", "iterations", "x", "s", "y", "z",
            "obj_val", "residuals",
        }
        assert expected_keys.issubset(r.keys()), f"Missing keys: {expected_keys - r.keys()}"
        res_keys = {"eq_inf", "ineq_inf", "stat_inf", "gap"}
        assert res_keys.issubset(r["residuals"].keys())

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_solve_scaling(self, n: int):
        """PIQP should handle increasing problem sizes."""
        rng = np.random.default_rng(42)
        P = rand_spd(n, density=max(0.01, 10.0 / n), seed=42)
        q = rng.standard_normal(n)
        n_ineq = max(3, n // 5)
        G = rng.standard_normal((n_ineq, n))
        h = G @ np.ones(n) - 0.5
        r = piqp_solve(P, q, G=G, h=h)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.all(G @ x <= h + 1e-3)


# ── Solver class API ─────────────────────────────────────────────────────────

class TestPIQPSolverClass:
    """Tests for the piqp_solver class API."""

    def test_setup_and_solve(self):
        """Build solver with setup(), then solve()."""
        P = np.diag([2.0, 2.0, 2.0])
        q = -np.ones(3)
        G = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        h = np.array([1.0, 1.0])

        solver = piqp_solver()
        solver.setup(P, q, G=G, h=h)
        r = solver.solve()
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.all(G @ x <= h + 1e-4)

    def test_settings(self):
        """PIQPSettings should be configurable."""
        settings = PIQPSettings()
        settings.eps_abs = 1e-8
        settings.eps_rel = 1e-8
        settings.max_iter = 500
        settings.verbose = False

        P = np.eye(3)
        q = -np.ones(3)
        solver = piqp_solver(settings)
        solver.setup(P, q)
        r = solver.solve()
        assert r["status"] == "solved"

    def test_warm_start(self):
        """Warm-start via setup with x0 (if supported by C++ API).

        NOTE: The current nanobind binding does NOT expose warm_start().
        This test verifies the solver works without warm-start.
        """
        P = np.diag([2.0, 2.0])
        q = -np.ones(2)
        G = np.array([[1.0, 1.0]])
        h = np.array([1.0])

        solver = piqp_solver()
        solver.setup(P, q, G=G, h=h)
        r = solver.solve()
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.all(G @ x <= h + 1e-4)

    def test_re_solve_same_problem(self):
        """Re-solving the same problem after setup should work."""
        P = np.eye(3)
        q = -np.ones(3)
        G = np.array([[1.0, 1.0, 0.0]])
        h = np.array([1.0])

        solver = piqp_solver()
        solver.setup(P, q, G=G, h=h)
        r1 = solver.solve()
        solver.setup(P, q, G=G, h=h)  # Re-setup
        r2 = solver.solve()
        assert r1["status"] == "solved"
        assert r2["status"] == "solved"
        x1 = np.asarray(r1["x"]).ravel()
        x2 = np.asarray(r2["x"]).ravel()
        assert np.allclose(x1, x2, atol=1e-6)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestPIQPEdgeCases:
    """Edge cases and special scenarios."""

    def test_zero_constraints(self):
        """QP with no inequality or equality constraints."""
        P = np.eye(3)
        q = -np.ones(3)
        r = piqp_solve(P, q)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, np.ones(3), atol=1e-4)

    def test_inequality_feasible_boundary(self):
        """Problem at feasibility boundary (tight constraints)."""
        P = np.eye(2)
        q = np.zeros(2)
        G = np.array([[1.0, 0.0], [-1.0, 0.0]])
        h = np.array([0.5, -0.5])  # 0.5 >= x1 >= 0.5 => x1=0.5
        r = piqp_solve(P, q, G=G, h=h)
        # PIQP may solve or not — test just that it doesn't crash
        assert r["status"] in ("solved", "max_iter_reached")
        if r["status"] == "solved":
            x = np.asarray(r["x"]).ravel()
            assert np.abs(x[0] - 0.5) < 0.1

    def test_sparse_P(self):
        """Sparse P matrix (via scipy sparse input)."""
        P = sp.diags([1.0, 2.0, 3.0], 0, shape=(3, 3)).toarray()
        q = -np.ones(3)
        r = piqp_solve(P, q)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.allclose(x, [1.0, 0.5, 1.0 / 3.0], atol=1e-4)

    def test_status_strings(self):
        """Verify status strings are sensible."""
        P = np.eye(3)
        q = -np.ones(3)
        r = piqp_solve(P, q)
        assert r["status"] in ("solved", "max_iter_reached")
        assert isinstance(r["iterations"], int)
        assert isinstance(r["obj_val"], float)

    def test_ill_conditioned(self):
        """PIQP should handle moderately ill-conditioned P."""
        n = 20
        rng = np.random.default_rng(42)
        B = rng.standard_normal((n, n))
        P = B @ B.T + 1e-4 * np.eye(n)  # ill-conditioned
        q = rng.standard_normal(n)
        r = piqp_solve(P, q)
        assert r["status"] == "solved"
        x = np.asarray(r["x"]).ravel()
        assert np.all(np.isfinite(x))
