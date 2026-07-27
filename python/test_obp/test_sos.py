"""Tests for SOS1/SOS2 (special-ordered-set) constraints."""

import pytest

from solvers.obp import Problem, solve
from solvers.obp.model import SOSConstraint


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestSOSConstraintValidation:
    def test_requires_finite_bounds(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0)  # ub=inf
        with pytest.raises(ValueError, match="finite bounds"):
            pb.add_sos_constraint(x, type=1)

    def test_requires_at_least_two_vars(self):
        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=1)
        with pytest.raises(ValueError, match="at least 2"):
            SOSConstraint(variables=[x], type=1)

    def test_invalid_type(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=1)
        with pytest.raises(ValueError, match="must be 1 or 2"):
            pb.add_sos_constraint(x, type=3)

    def test_weights_length_mismatch(self):
        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=1)
        with pytest.raises(ValueError, match="weights must match"):
            pb.add_sos_constraint(x, type=1, weights=[0, 1])


class TestSOS1Solve:
    def test_sos1_picks_best_single_var(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("sos1")
        x = pb.add_variables("x", 3, lb=0, ub=10)
        pb.set_objective(2 * x[0] + 5 * x[1] + 3 * x[2], sense="maximize")
        pb.add_constraint(x[0] + x[1] + x[2] <= 10)
        pb.add_sos_constraint(x, type=1)

        result = solve(pb)
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(50.0, abs=1e-6)
        nonzero = [xi for xi in result.x if xi > 1e-6]
        assert len(nonzero) <= 1
        assert len(result.x) == 3  # synthetic indicator vars hidden from output

    def test_sos1_requires_highs(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(x[0] + x[1])
        pb.add_constraint(x[0] + x[1] <= 5)
        pb.add_sos_constraint(x, type=1)

        with pytest.raises(ValueError, match="highs"):
            solve(pb, solver="osqp")


SOS2_METHODS = ["cc", "dlog", "log"]


class TestSOS2Solve:
    @pytest.mark.parametrize("method", SOS2_METHODS)
    def test_sos2_allows_adjacent_pair(self, highs_available, method):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("sos2")
        y = pb.add_variables("y", 4, lb=0, ub=10)
        pb.set_objective(y[0] + y[1] + y[2] + y[3], sense="maximize")
        pb.add_constraint(y[0] + y[1] + y[2] + y[3] <= 10)
        pb.add_sos_constraint(y, type=2, method=method)

        result = solve(pb)
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(10.0, abs=1e-6)
        nonzero = [i for i, v in enumerate(result.x) if v > 1e-6]
        assert len(nonzero) <= 2
        if len(nonzero) == 2:
            assert abs(nonzero[0] - nonzero[1]) == 1

    @pytest.mark.parametrize("method", SOS2_METHODS)
    def test_sos2_rejects_nonadjacent_pair(self, highs_available, method):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("sos2_nonadjacent")
        z = pb.add_variables("z", 4, lb=0, ub=10)
        # z0 and z3 both want to be large but are not adjacent
        pb.set_objective(z[0] + 0.1 * z[1] + 0.1 * z[2] + z[3], sense="maximize")
        pb.add_constraint(z[0] + z[1] + z[2] + z[3] <= 20)
        pb.add_sos_constraint(z, type=2, method=method)

        result = solve(pb)
        assert result.status == "solved"
        # z0=10, z3=10 (obj 20) is infeasible under SOS2 since indices 0,3 aren't
        # adjacent — solver must settle for an adjacent pair instead.
        assert result.obj_val < 15.0

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_sos2_methods_agree_with_negative_bounds(self, highs_available, k):
        """Cross-check all 4 formulations against each other on a case with
        negative lower bounds — regression coverage for the HiGHS column-bound
        clamp bug found while validating these formulations."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        results = {}
        for method in SOS2_METHODS:
            pb = Problem()
            x = pb.add_variables(f"x_{method}", k, lb=-100, ub=100)
            for i in range(k):
                x[i].lb = -2.0 - i
                x[i].ub = 3.0 + i
            expr = (i + 1) * x[0]
            for i in range(1, k):
                expr = expr + ((-1) ** i) * (i + 1) * x[i]
            pb.set_objective(expr, sense="maximize")
            total = x[0]
            for i in range(1, k):
                total = total + x[i]
            pb.add_constraint(total <= 5.0)
            pb.add_sos_constraint(x, type=2, method=method)
            results[method] = solve(pb, solver="highs")

        obj_vals = [r.obj_val for r in results.values()]
        for method, r in results.items():
            assert r.status == "solved", f"{method} failed: {r.status}"
        assert max(obj_vals) - min(obj_vals) < 1e-4, results

    def test_invalid_method(self):
        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=1)
        with pytest.raises(ValueError, match="method must be"):
            pb.add_sos_constraint(x, type=2, method="bogus")
