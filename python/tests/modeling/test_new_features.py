"""Tests for new obp features (2025): DPP caching, numpy broadcasting,
get_problem_data, and Problem.clone.
"""

import numpy as np
import pytest

from obp import Problem, Variable, solve, Parameter
from obp.expression import Expression
from obp.matrices import build_constraints_matrix

from .conftest import skip_if_no_solver


class TestDPPCaching:
    """DPP (Disciplined Parametrized Programming) assemble caching."""

    def test_cache_hit_skips_reassembly(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        budget = Parameter(10.0, name="budget")
        pb = Problem("cache_test")
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= budget)

        # First solve: populate cache
        r1 = solve(pb, solver="highs")
        stats1 = pb.get_cache_stats()
        assert stats1["size"] == 1  # one entry cached
        assert stats1["enabled"] is True
        assert r1.x.sum() == pytest.approx(10.0, abs=1e-6)

    def test_cache_invalidation_on_param_change(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        budget = Parameter(10.0, name="budget")
        pb = Problem("cache_invalidate")
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= budget)

        r1 = solve(pb, solver="highs")
        assert r1.x.sum() == pytest.approx(10.0, abs=1e-6)

        budget.value = 25.0
        r2 = solve(pb, solver="highs")
        assert r2.x.sum() == pytest.approx(25.0, abs=1e-6)

        # Cache should have two entries now (different param values)
        stats = pb.get_cache_stats()
        assert stats["size"] >= 2

    def test_clear_cache(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        budget = Parameter(10.0)
        pb = Problem("clear")
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= budget)

        solve(pb, solver="highs")
        assert pb.get_cache_stats()["size"] == 1

        pb.clear_cache()
        assert pb.get_cache_stats()["size"] == 0

    def test_disable_caching(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("disable")
        x = pb.add_variables("x", 2, lb=0, ub=100)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= 10)

        pb.enable_caching(False)
        solve(pb, solver="highs")
        assert pb.get_cache_stats()["size"] == 0

        pb.enable_caching(True)
        solve(pb, solver="highs")
        assert pb.get_cache_stats()["size"] == 1


class TestNumPyBroadcasting:
    """NumPy broadcasting on Variable/Expression arrays."""

    def test_np_array_of_variables(self):
        pb = Problem("broadcast")
        x = pb.add_variables("x", 3)
        x_arr = np.array(x)
        assert x_arr.shape == (3,)
        assert x_arr.dtype == object

    def test_broadcast_add_scalar(self):
        pb = Problem("broadcast_add")
        x = pb.add_variables("x", 3)
        x_arr = np.array(x)
        result = x_arr + 5.0
        assert result.shape == (3,)
        assert all(isinstance(r, Expression) for r in result)
        for i in range(3):
            assert result[i].linear_coeffs[x[i]] == 1.0
            assert result[i]._constant == 5.0

    def test_broadcast_sub_scalar(self):
        pb = Problem("broadcast_sub")
        x = pb.add_variables("x", 3)
        x_arr = np.array(x)
        result = x_arr - 3.0
        assert all(isinstance(r, Expression) for r in result)
        assert result[0].linear_coeffs[x[0]] == 1.0
        assert result[0]._constant == -3.0

    def test_broadcast_mul_scalar(self):
        pb = Problem("broadcast_mul")
        x = pb.add_variables("x", 3)
        x_arr = np.array(x)
        result = x_arr * 2.5
        assert all(isinstance(r, Expression) for r in result)
        assert result[0].linear_coeffs[x[0]] == 2.5

    def test_np_sum_on_variables(self):
        pb = Problem("np_sum")
        x = pb.add_variables("x", 3, lb=0)
        x_arr = np.array(x)
        total = np.sum(x_arr)
        assert isinstance(total, Expression)
        for i in range(3):
            assert total.linear_coeffs[x[i]] == 1.0

    def test_vectorized_constraints(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("vec_con")
        x = pb.add_variables("x", 3, lb=0, ub=10)
        x_arr = np.array(x)

        # Create upper bounds
        bounds = np.array([5.0, 3.0, 4.0])

        # Vectorized: x_arr + 0 <= bounds
        for i in range(3):
            pb.add_constraint(x_arr[i] <= bounds[i])

        pb.set_objective(-np.sum(x_arr))
        result = solve(pb, solver="highs")

        np.testing.assert_allclose(result.x, [5.0, 3.0, 4.0], atol=1e-6)
        assert result.obj_val == pytest.approx(-12.0, abs=1e-6)

    def test_negate_array(self):
        pb = Problem("negate")
        x = pb.add_variables("x", 2)
        x_arr = np.array(x)
        result = -x_arr
        assert result.shape == (2,)
        assert result[0].linear_coeffs[x[0]] == -1.0


class TestGetProblemData:
    """get_problem_data() — extract canonical problem data."""

    def test_canonical_form(self):
        pb = Problem("data_test")
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(x[0]**2 + x[1]**2 + x[2]**2)
        pb.add_constraint(x[0] + x[1] + x[2] == 4)

        data, inv = pb.get_problem_data()

        assert data["n_vars"] == 3
        assert data["n_constraints"] == 1
        assert data["obj_sense"] == "minimize"
        assert "P" in data
        assert "c" in data
        assert "A" in data
        assert "l" in data
        assert "u" in data
        assert "lb" in data
        assert "ub" in data
        assert "var_types" in data
        assert "row_constraints" in data

    def test_qp_data(self):
        pb = Problem("qp_data")
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(x[0]**2 + x[1]**2)
        pb.add_constraint(x[0] + x[1] <= 3)

        data, _ = pb.get_problem_data()
        assert data["P"] is not None
        assert data["P"].shape == (2, 2)

    def test_lp_data(self):
        pb = Problem("lp_data")
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(3 * x[0] + 2 * x[1])
        pb.add_constraint(x[0] + x[1] <= 5)

        data, _ = pb.get_problem_data()
        assert data["P"] is None  # LP has no quadratic part
        assert data["c"].shape == (2,)

    def test_invertible(self):
        """Test that inverse_data can be used to reconstruct Solution."""
        pb = Problem("invert")
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(x[0]**2 + x[1]**2 + x[2]**2)
        pb.add_constraint(x[0] + x[1] + x[2] == 4)

        data, inv = pb.get_problem_data()
        assert inv["variables"] == pb._variables
        assert inv["obj_sense"] == "minimize"


class TestProblemClone:
    """Problem.clone() — deep-copy with fresh variables."""

    def test_clone_structure(self):
        pb = Problem("original")
        x = pb.add_variables("x", 3, lb=0)
        pb.set_objective(x[0] + x[1] + x[2])
        pb.add_constraint(x[0] + x[1] <= 5)

        cloned = pb.clone("cloned")

        assert cloned.name == "cloned"
        assert cloned.n_vars == 3
        assert cloned.n_constraints == 1

        # Variables are different objects
        assert cloned._variables[0] is not x[0]

    def test_clone_independent(self):
        """Clone should be independent: modifying one doesn't affect the other."""
        pb = Problem("indep")
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(x[0] + x[0] * x[0])  # quadratic
        pb.add_constraint(x[0] + x[1] <= 5)

        cloned = pb.clone("copy")

        # Original and clone should solve to same answer
        r_orig = solve(pb, solver="highs")
        r_clone = solve(cloned, solver="highs")

        np.testing.assert_allclose(r_orig.x, r_clone.x, atol=1e-6)

    def test_clone_name_default(self):
        pb = Problem("my_prob")
        x = pb.add_variables("x", 2)
        pb.set_objective(x[0] + x[1])

        cloned = pb.clone()
        assert cloned.name == "my_prob_clone"

    def test_clone_no_name(self):
        pb = Problem()
        x = pb.add_variables("x", 2)
        pb.set_objective(x[0] + x[1])

        cloned = pb.clone()
        assert cloned.name == "clone"

    def test_clone_with_mip(self):
        pb = Problem("mip_clone")
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=10)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= 3)

        cloned = pb.clone()
        assert cloned._variables[0].vtype == "integer"

        r_orig = solve(pb, solver="highs")
        r_clone = solve(cloned, solver="highs")
        np.testing.assert_allclose(r_orig.x, r_clone.x, atol=1e-6)

    def test_clone_fresh_cache(self):
        pb = Problem("cache_clone")
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= 10)

        solve(pb, solver="highs")
        assert pb.get_cache_stats()["size"] == 1

        cloned = pb.clone()
        assert cloned.get_cache_stats()["size"] == 0


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestCombinedFeatures:
    """Tests that combine multiple new features."""

    def test_cache_and_get_problem_data(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        budget = Parameter(10.0)
        pb = Problem("combined")
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(-x[0] - x[1])
        pb.add_constraint(x[0] + x[1] <= budget)

        # First solve: populate cache
        r1 = solve(pb, solver="highs")
        assert r1.x.sum() == pytest.approx(10.0, abs=1e-6)

        # get_problem_data should work with cache
        data, _ = pb.get_problem_data("highs")
        assert data["A"].shape == (1, 2)

        # Clone + solve
        cloned = pb.clone("cloned")
        r2 = solve(cloned, solver="highs")
        assert r2.x.sum() == pytest.approx(10.0, abs=1e-6)

    def test_numpy_broadcast_with_solve(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem("broadcast_solve")
        n = 5
        x = pb.add_variables("x", n, lb=0, ub=10)
        x_arr = np.array(x)

        # Vectorized objective: maximize sum of x
        total = np.sum(x_arr)
        pb.set_objective(total, sense="maximize")

        # Vectorized constraints: each x[i] <= 3
        for i in range(n):
            pb.add_constraint(x_arr[i] <= 3.0)

        result = solve(pb, solver="highs")
        np.testing.assert_allclose(result.x, [3.0] * n, atol=1e-6)
        assert result.obj_val == pytest.approx(15.0, abs=1e-6)


class TestMIPDetectionPriority:
    """_pick_mip_solver and _detect_solver MIP path priority."""

    def test_detect_solver_mip_picks_first_available(self):
        from obp.solve import _detect_solver
        from obp.model import IntVar, BinVar, Variable
        from obp.expression import Expression

        obj = Expression.constant(0)
        constraints = []
        soc = []

        # All continuous → LP, not MIP
        x_cont = Variable(0, "x", lb=0)
        assert _detect_solver(obj, constraints, soc, [x_cont]) == "ipm"

        # Has integer → should be MIP (guaranteed to have highs)
        x_int = IntVar(1, "y", lb=0)
        result = _detect_solver(obj, constraints, soc, [x_cont, x_int])
        assert result in ("gurobi", "highs", "scip")

        # Has binary → MIP
        x_bin = BinVar(2, "z")
        result = _detect_solver(obj, constraints, soc, [x_cont, x_bin])
        assert result in ("gurobi", "highs", "scip")

    def test_pick_mip_solver_priority(self):
        from obp.solve import _pick_mip_solver

        # Should pick the first available solver
        solver = _pick_mip_solver()
        assert solver in ("gurobi", "highs", "scip")

        # Since gurobi is not installed, it should be highs or scip
        import highspy  # we know highs is available
        assert solver == "highs"  # gurobi not installed → highs is first
