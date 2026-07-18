"""Tests for the PDLP first-order LP solver (pdlp extension)."""

import numpy as np
import pytest
import scipy.sparse as sp

pdlp = pytest.importorskip("pdlp")
linprog = pytest.importorskip("scipy.optimize").linprog


def random_feasible_lp(seed, n=20, m1=15, m2=5):
    """Random LP with a known interior feasible point (so it's always feasible)."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 1.0, n)
    G = sp.csc_matrix(rng.normal(size=(m1, n)))
    h = G @ x0 - rng.uniform(0.1, 1.0, m1)  # G x0 >= h with margin
    A = sp.csc_matrix(rng.normal(size=(m2, n)))
    b = A @ x0
    l = np.full(n, -1.0)
    u = np.full(n, 2.0)
    c = rng.normal(size=n)
    return c, G, h, A, b, l, u


def reference_objective(c, G, h, A, b, l, u):
    res = linprog(
        c,
        A_ub=-G.toarray() if G.shape[0] else None,
        b_ub=-h if G.shape[0] else None,
        A_eq=A.toarray() if A.shape[0] else None,
        b_eq=b if A.shape[0] else None,
        bounds=list(zip(l, u)),
        method="highs",
    )
    assert res.status == 0
    return res.fun


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimal_matches_linprog(seed):
    c, G, h, A, b, l, u = random_feasible_lp(seed)
    x, y, status, info = pdlp.solve(
        c, G, h, A, b, l, u, eps_tol=1e-6, iteration_limit=50000
    )
    assert status == "optimal"
    obj_ref = reference_objective(c, G, h, A, b, l, u)
    assert info["primal_obj"] == pytest.approx(obj_ref, abs=1e-4, rel=1e-4)
    # returned x is feasible for the original problem
    assert np.all(G @ x >= h - 1e-4)
    assert np.allclose(A @ x, b, atol=1e-4)
    assert np.all(x >= l - 1e-6) and np.all(x <= u + 1e-6)
    # inequality duals are nonnegative
    assert np.all(y[: G.shape[0]] >= -1e-8)


def test_inequalities_only():
    # min -x1 - x2  s.t. x1 + x2 >= -inf handled via: x1 + 2 x2 <= 4, 3 x1 + x2 <= 6
    # expressed as >=: -x1 - 2 x2 >= -4, -3 x1 - x2 >= -6, x >= 0
    c = np.array([-1.0, -1.0])
    G = sp.csc_matrix(np.array([[-1.0, -2.0], [-3.0, -1.0]]))
    h = np.array([-4.0, -6.0])
    A = sp.csc_matrix((0, 2))
    b = np.zeros(0)
    l = np.zeros(2)
    u = np.full(2, np.inf)
    x, y, status, info = pdlp.solve(c, G, h, A, b, l, u, eps_tol=1e-7)
    assert status == "optimal"
    # optimum at intersection: x = (8/5, 6/5), obj = -14/5
    assert info["primal_obj"] == pytest.approx(-2.8, abs=1e-4)
    assert np.allclose(x, [1.6, 1.2], atol=1e-3)


def test_bounds_only():
    # no constraints: minimize goes to the bound in the objective's direction
    c = np.array([1.0, -2.0, 0.0])
    G = sp.csc_matrix((0, 3))
    A = sp.csc_matrix((0, 3))
    l = np.array([-1.0, -1.0, -1.0])
    u = np.array([2.0, 3.0, 4.0])
    x, y, status, info = pdlp.solve(c, G, np.zeros(0), A, np.zeros(0), l, u)
    assert status == "optimal"
    assert x[0] == -1.0 and x[1] == 3.0
    assert info["primal_obj"] == pytest.approx(-7.0)


def test_primal_infeasible():
    # x1 + x2 = 1 and x1 + x2 = 2 simultaneously
    c = np.zeros(2)
    G = sp.csc_matrix((0, 2))
    h = np.zeros(0)
    A = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    b = np.array([1.0, 2.0])
    l = np.full(2, -np.inf)
    u = np.full(2, np.inf)
    x, y, status, info = pdlp.solve(c, G, h, A, b, l, u, iteration_limit=20000)
    assert status == "primal_infeasible"
    assert "ray" in info
    ray = np.asarray(info["ray"])
    # Farkas: K^T ray ~ 0, q^T ray > 0
    assert abs(A.T @ ray).max() <= 1e-6 * max(1.0, abs(ray).max())
    assert b @ ray > 0


def test_dual_infeasible_unbounded():
    # min -x  s.t. x >= 1, x >= 0 (no upper bound) — unbounded below
    c = np.array([-1.0])
    G = sp.csc_matrix(np.array([[1.0]]))
    h = np.array([1.0])
    A = sp.csc_matrix((0, 1))
    b = np.zeros(0)
    l = np.array([0.0])
    u = np.array([np.inf])
    x, y, status, info = pdlp.solve(c, G, h, A, b, l, u, iteration_limit=20000)
    assert status == "dual_infeasible"
    ray = np.asarray(info["ray"])
    assert c @ ray < 0


def test_iteration_limit():
    c, G, h, A, b, l, u = random_feasible_lp(3)
    x, y, status, info = pdlp.solve(c, G, h, A, b, l, u, iteration_limit=3)
    assert status == "iteration_limit"
    assert info["iterations"] == 3


@pytest.mark.skipif(True, reason="CUDA not available")
@pytest.mark.parametrize("seed", [0, 1])
def test_cuda_matches_cpu(seed):
    c, G, h, A, b, l, u = random_feasible_lp(seed, n=50, m1=40, m2=10)
    kwargs = dict(eps_tol=1e-6, iteration_limit=50000)
    _, _, status_cpu, info_cpu = pdlp.solve(c, G, h, A, b, l, u, **kwargs)
    x, y, status_gpu, info_gpu = pdlp.solve(c, G, h, A, b, l, u, device="cuda", **kwargs)
    assert status_cpu == "optimal"
    assert status_gpu == "optimal"
    assert info_gpu["primal_obj"] == pytest.approx(info_cpu["primal_obj"], abs=1e-4, rel=1e-4)
    assert np.all(G @ x >= h - 1e-4)
    assert np.allclose(A @ x, b, atol=1e-4)


@pytest.mark.skipif(True, reason="CUDA not available")
def test_cuda_infeasible():
    c = np.zeros(2)
    G = sp.csc_matrix((0, 2))
    A = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    b = np.array([1.0, 2.0])
    l = np.full(2, -np.inf)
    u = np.full(2, np.inf)
    _, _, status, info = pdlp.solve(
        c, G, np.zeros(0), A, b, l, u, iteration_limit=20000, device="cuda"
    )
    assert status == "primal_infeasible"
