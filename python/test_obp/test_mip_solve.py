"""Tests for MIP solving with HiGHS backend."""

import numpy as np
import pytest

from solvers.obp import Problem, solve
from solvers.obp.model import BinVar, IntVar


@pytest.fixture
def highs_available():
    """Skip tests if HiGHS is not available."""
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestMIPBasic:
    """Basic MIP tests with HiGHS."""

    def test_integer_var(self, highs_available):
        """Test integer variable."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("int_test")
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=10)
        pb.set_objective(1 * x[0])  # use expression, not raw variable
        pb.add_constraint(x[0] >= 5)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val >= 4.99  # should be at least 5
        np.testing.assert_allclose(result.x[0], 5.0, atol=1e-6)

    def test_mixed_integer(self, highs_available):
        """Test mixed integer/continuous variables."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("mixed_test")
        x = pb.add_variables("x", 2, vtype="integer", lb=0)
        y = pb.add_variables("y", 2, vtype="continuous", lb=0, ub=1)
        pb.set_objective(-x[0] - 2*x[1] + y[0])
        pb.add_constraint(x[0] + x[1] + y[0] <= 3)
        pb.add_constraint(2*x[0] + x[1] <= 4)
        pb.add_constraint(x[0] >= 1)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(-5.0, abs=1e-6)
        np.testing.assert_allclose(result.x[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.x[1], 2.0, atol=1e-6)

    def test_binary_variable(self, highs_available):
        """Test binary variables."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("bin_test")
        b = pb.add_variables("b", 3, vtype="binary")
        pb.set_objective(b[0] + 2*b[1] + 3*b[2])
        pb.add_constraint(b[0] + b[1] + b[2] >= 1)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(result.x[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result.x[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.x[2], 0.0, atol=1e-6)

    def test_auto_detect_mip(self, highs_available):
        """Test auto-detection of MIP → HiGHS."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("auto_mip")
        x = pb.add_variables("x", 2, vtype="integer", lb=0)
        pb.set_objective(x[0] + x[1])
        pb.add_constraint(x[0] + x[1] <= 5)

        result = solve(pb)  # no solver specified
        assert result.solver == "highs"
        assert result.status == "solved"
        np.testing.assert_allclose(result.x, [0, 0], atol=1e-6)


class TestMIPInfeasible:
    """Test MIP infeasibility detection."""

    def test_infeasible_mip(self, highs_available):
        """Test detection of infeasible MIP."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("infeasible")
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=1)
        pb.set_objective(1 * x[0])
        pb.add_constraint(x[0] >= 5)  # infeasible with ub=1

        result = solve(pb, solver="highs")
        assert "infeasible" in result.status.lower()


class TestMIPWithBounds:
    """Test MIP with various bound configurations."""

    def test_binary_with_upper_bound(self, highs_available):
        """Test binary variable with explicit upper bound."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("bin_ub")
        b = pb.add_variables("b", 2, vtype="binary")
        pb.set_objective(-b[0] - b[1])
        pb.add_constraint(b[0] + b[1] <= 1)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(-1.0, abs=1e-6)

    def test_integer_with_upper_bound(self, highs_available):
        """Test integer variable with finite upper bound."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("int_ub")
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=5)
        pb.set_objective(1 * x[0])
        pb.add_constraint(x[0] >= 3)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        np.testing.assert_allclose(result.x[0], 3.0, atol=1e-6)
