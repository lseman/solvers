"""Tests for the solve() dispatcher: maximize sign, constraint assembly."""

import numpy as np
import pytest

from solvers.obp import Problem, solve
from solvers.obp.matrices import build_constraints_matrix


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


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

        A, l, u, sense = build_constraints_matrix(pb._constraints, 3)
        assert A.shape == (3, 3)
        dense = A.toarray()
        np.testing.assert_allclose(dense[0], [1, 1, 0])
        np.testing.assert_allclose(dense[1], [0, 1, 1])
        np.testing.assert_allclose(dense[2], [1, 0, 0])
        np.testing.assert_allclose(u, [5, np.inf, 1])
        np.testing.assert_allclose(l, [-np.inf, 2, 1])

    def test_empty_constraints(self):
        A, l, u, sense = build_constraints_matrix([], 3)
        assert A.shape == (0, 3)
        assert len(l) == 0
