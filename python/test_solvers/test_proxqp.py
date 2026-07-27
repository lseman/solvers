"""Regression tests for the self-contained ProxQP implementation."""

import numpy as np
import pytest
from scipy.optimize import minimize

import proxqp


def _settings(tolerance=1e-8):
    settings = proxqp.Settings()
    settings.eps_abs = tolerance
    settings.max_iter = 20_000
    return settings


def test_unconstrained_matches_closed_form():
    hessian = np.diag([2.0, 4.0])
    gradient = np.array([-2.0, -8.0])
    result = proxqp.solve(hessian, gradient, settings=_settings())

    assert result["status"] == "solved"
    np.testing.assert_allclose(result["x"], [1.0, 2.0], atol=1e-8)
    assert result["info"]["dua_res"] <= 1e-8


@pytest.mark.parametrize(
    ("gradient", "lower", "upper", "expected_x", "expected_z"),
    [
        (0.0, 2.0, np.inf, 2.0, -2.0),
        (-3.0, -np.inf, 1.0, 1.0, 2.0),
    ],
)
def test_signed_multiplier_for_both_box_faces(
    gradient, lower, upper, expected_x, expected_z
):
    result = proxqp.solve(
        np.array([[1.0]]),
        np.array([gradient]),
        C=np.array([[1.0]]),
        l=np.array([lower]),
        u=np.array([upper]),
        settings=_settings(),
    )

    assert result["status"] == "solved"
    np.testing.assert_allclose(result["x"], [expected_x], atol=2e-8)
    np.testing.assert_allclose(result["z"], [expected_z], atol=2e-8)
    assert result["info"]["pri_res"] <= 1e-8
    assert result["info"]["dua_res"] <= 1e-8


def test_equality_and_two_sided_inequalities_match_scipy():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((8, 8))
    hessian = raw.T @ raw + 0.5 * np.eye(8)
    gradient = rng.standard_normal(8)
    equality = rng.standard_normal((2, 8))
    feasible = rng.uniform(-0.4, 0.4, 8)
    equality_rhs = equality @ feasible
    inequalities = rng.standard_normal((5, 8))
    center = inequalities @ feasible
    lower = center - rng.uniform(0.2, 0.8, 5)
    upper = center + rng.uniform(0.2, 0.8, 5)

    result = proxqp.solve(
        hessian,
        gradient,
        equality,
        equality_rhs,
        inequalities,
        lower,
        upper,
        _settings(1e-7),
    )
    reference = minimize(
        lambda x: 0.5 * x @ hessian @ x + gradient @ x,
        feasible,
        jac=lambda x: hessian @ x + gradient,
        method="SLSQP",
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: equality @ x - equality_rhs,
                "jac": lambda x: equality,
            },
            {
                "type": "ineq",
                "fun": lambda x: inequalities @ x - lower,
                "jac": lambda x: inequalities,
            },
            {
                "type": "ineq",
                "fun": lambda x: upper - inequalities @ x,
                "jac": lambda x: -inequalities,
            },
        ],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )

    assert reference.success
    assert result["status"] == "solved"
    x = np.asarray(result["x"])
    np.testing.assert_allclose(equality @ x, equality_rhs, atol=2e-7)
    assert np.all(inequalities @ x >= lower - 2e-7)
    assert np.all(inequalities @ x <= upper + 2e-7)
    assert abs(result["info"]["obj_val"] - reference.fun) <= 2e-6


def test_rejects_incomplete_constraint_data():
    qp = proxqp.QP(2, 1, 0)
    with pytest.raises(ValueError):
        qp.init(np.eye(2), np.zeros(2), A=np.ones((1, 2)))
