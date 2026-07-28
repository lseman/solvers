"""Tests for the solve() dispatcher: maximize sign, constraint assembly."""

import numpy as np
import pytest

from obp import Problem, solve, Parameter
from obp.matrices import build_constraints_matrix

from .conftest import skip_if_no_solver


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestMultiVarSolveAgreement:
    """Regression: min sum(x_i^2) s.t. sum(x_i) == 4 must split equally
    across backends. Exercises the multi-coefficient constraint-row path
    that split_constraints_for_{piqp,proxqp} previously mis-assembled.
    """

    @pytest.mark.parametrize("backend", ["osqp", "piqp", "proxqp"])
    def test_qp_equal_split(self, backend):
        if skip_if_no_solver(backend):
            pytest.skip(f"{backend} not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        pb.add_constraint(x[0] + x[1] + x[2] == 4)

        result = solve(pb, solver=backend)
        np.testing.assert_allclose(result.x, [4 / 3] * 3, atol=1e-2)
        assert result.obj_val == pytest.approx(3 * (4 / 3) ** 2, abs=1e-2)

    def test_ipm_lp_sense_mix(self):
        if skip_if_no_solver("ipm"):
            pytest.skip("ipm not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(3 * x[0] + 2 * x[1] + x[2])
        pb.add_constraint(x[0] + x[1] + x[2] == 4)
        pb.add_constraint(2 * x[0] + x[1] <= 5)

        result = solve(pb, solver="ipm")
        np.testing.assert_allclose(result.x, [0.0, 0.0, 4.0], atol=1e-4)
        assert result.obj_val == pytest.approx(4.0, abs=1e-4)


class TestMaximizeObjVal:
    def test_maximize_obj_val_sign(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=10)
        pb.set_objective(1 * x, sense="maximize")
        pb.add_constraint(x <= 10)

        result = solve(pb, solver="highs")
        np.testing.assert_allclose(result.x[0], 10.0, atol=1e-6)
        assert result.obj_val == pytest.approx(10.0, abs=1e-6)

    def test_minimize_obj_val_unaffected(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=10)
        pb.set_objective(1 * x)
        pb.add_constraint(x >= 3)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(3.0, abs=1e-6)


class TestBuildConstraintsMatrix:
    def test_multiple_constraints_correct_rows(self):
        pb = Problem()
        x = pb.add_variables("x", 3, lb=0)
        pb.add_constraint(x[0] + x[1] <= 5)
        pb.add_constraint(x[1] + x[2] >= 2)
        pb.add_constraint(x[0] == 1)

        A, l, u, sense, row_constraints = build_constraints_matrix(pb._constraints, 3)
        assert A.shape == (3, 3)
        dense = A.toarray()
        np.testing.assert_allclose(dense[0], [1, 1, 0])
        np.testing.assert_allclose(dense[1], [0, 1, 1])
        np.testing.assert_allclose(dense[2], [1, 0, 0])
        np.testing.assert_allclose(u, [5, np.inf, 1])
        np.testing.assert_allclose(l, [-np.inf, 2, 1])

    def test_empty_constraints(self):
        A, l, u, sense, row_constraints = build_constraints_matrix([], 3)
        assert A.shape == (0, 3)
        assert len(l) == 0


class TestConstantTermsNotDropped:
    """Regression: constraint/objective constant terms used to be silently
    dropped during matrix assembly (only the linear coefficients survived).
    """

    def test_constraint_constant_shifts_bound(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=20)
        pb.set_objective(-x[0] - x[1])  # maximize sum via minimize -sum
        pb.add_constraint(x[0] + x[1] + 5 <= 10)  # effective cap: sum <= 5

        result = solve(pb, solver="highs")
        assert result.x.sum() == pytest.approx(5.0, abs=1e-6)

    def test_objective_constant_included_minimize(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(x[0] + x[1] + 5)
        pb.add_constraint(x[0] + x[1] >= 3)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(8.0, abs=1e-6)

    def test_objective_constant_included_maximize(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(x[0] + x[1] + 5, sense="maximize")
        pb.add_constraint(x[0] + x[1] <= 3)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(8.0, abs=1e-6)


class TestPerConstraintDual:
    def test_highs_dual_matches_shadow_price(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        c1 = pb.add_constraint(x[0] + x[1] <= 10)

        result = solve(pb, solver="highs")
        assert result.dual(c1) == pytest.approx(-1.0, abs=1e-6)

    def test_ipm_dual_matches_shadow_price(self):
        if skip_if_no_solver("ipm"):
            pytest.skip("ipm not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        c1 = pb.add_constraint(x[0] + x[1] <= 10)

        result = solve(pb, solver="ipm")
        assert result.dual(c1) == pytest.approx(-1.0, abs=1e-3)

    def test_unsupported_backend_raises_key_error(self):
        if skip_if_no_solver("piqp"):
            pytest.skip("piqp not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(x[0] ** 2 + x[1] ** 2)
        c1 = pb.add_constraint(x[0] + x[1] == 4)

        result = solve(pb, solver="piqp")
        with pytest.raises(KeyError):
            result.dual(c1)

    def test_multiple_constraints_map_independently(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        c1 = pb.add_constraint(x[0] <= 3)
        c2 = pb.add_constraint(x[1] <= 7)

        result = solve(pb, solver="highs")
        assert result.dual(c1) == pytest.approx(-1.0, abs=1e-6)
        assert result.dual(c2) == pytest.approx(-1.0, abs=1e-6)


class TestParameterResolve:
    """Regression/feature: Parameter values must be resolved fresh at each
    solve() call, so mutating .value and re-solving picks up the change
    without rebuilding the Problem.
    """

    def test_parameter_in_constraint_bound(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        budget = Parameter(10.0, name="budget")
        pb = Problem()
        y = pb.add_variables("y", 2, lb=0, ub=100)
        pb.set_objective(-y[0] - y[1])
        pb.add_constraint(y[0] + y[1] <= budget)

        r1 = solve(pb, solver="highs")
        assert r1.x.sum() == pytest.approx(10.0, abs=1e-6)

        budget.value = 25.0
        r2 = solve(pb, solver="highs")
        assert r2.x.sum() == pytest.approx(25.0, abs=1e-6)

    def test_parameter_in_objective(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        fixed_cost = Parameter(100.0, name="fixed_cost")
        pb = Problem()
        z = pb.add_variables("z", 1, lb=0, ub=10)
        pb.set_objective(z + fixed_cost)
        pb.add_constraint(z >= 4)

        r1 = solve(pb, solver="highs")
        assert r1.obj_val == pytest.approx(104.0, abs=1e-6)

        fixed_cost.value = 50.0
        r2 = solve(pb, solver="highs")
        assert r2.obj_val == pytest.approx(54.0, abs=1e-6)


class TestHiGHSWarmStart:
    def test_x0_does_not_change_optimum(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=1, vtype="binary")
        pb.set_objective(-x[0] - x[1] - x[2])
        pb.add_constraint(x[0] + x[1] + x[2] <= 2)

        cold = solve(pb, solver="highs")
        warm = solve(pb, solver="highs", x0=[1.0, 1.0, 0.0])
        assert cold.obj_val == pytest.approx(warm.obj_val, abs=1e-6)
        assert warm.obj_val == pytest.approx(-2.0, abs=1e-6)

    def test_x0_accepted_for_lp(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= 10)

        result = solve(pb, solver="highs", x0=[5.0, 5.0])
        assert result.x.sum() == pytest.approx(10.0, abs=1e-6)
