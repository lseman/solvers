"""Tests for exact binary-product linearizations."""

import itertools

import pytest

from obp import (
    Problem,
    linearize_binary_continuous_product,
    linearize_binary_product,
    solve,
)


def _require_highs() -> None:
    pytest.importorskip("highspy")


class TestBinaryProduct:
    def test_adds_one_continuous_variable_and_three_constraints(self):
        problem = Problem()
        x = problem.add_variables("x", vtype="binary")
        y = problem.add_variables("y", vtype="binary")

        z = linearize_binary_product(problem, x, y, name="xy")

        assert z.name == "xy_0"
        assert z.vtype == "continuous"
        assert (z.lb, z.ub) == (0.0, 1.0)
        assert problem.n_constraints == 3

    @pytest.mark.parametrize(
        "x_value,y_value",
        list(itertools.product((0, 1), repeat=2)),
    )
    def test_is_exact_for_all_binary_assignments(self, x_value, y_value):
        _require_highs()
        problem = Problem()
        x = problem.add_variables("x", vtype="binary")
        y = problem.add_variables("y", vtype="binary")
        z = linearize_binary_product(problem, x, y)
        problem.add_constraint(x == x_value)
        problem.add_constraint(y == y_value)
        problem.set_objective(1 * z)

        result = solve(problem, solver="highs")

        assert result.status == "solved"
        assert result.obj_val == pytest.approx(x_value * y_value)

    def test_rejects_nonbinary_factor(self):
        problem = Problem()
        x = problem.add_variables("x", lb=0, ub=1)
        y = problem.add_variables("y", vtype="binary")

        with pytest.raises(ValueError, match="x must be a binary"):
            linearize_binary_product(problem, x, y)


class TestBinaryContinuousProduct:
    def test_adds_one_variable_and_four_constraints(self):
        problem = Problem()
        binary = problem.add_variables("b", vtype="binary")
        x = problem.add_variables("x", lb=-2, ub=5)

        z = linearize_binary_continuous_product(problem, binary, x, name="bx")

        assert z.name == "bx_0"
        assert (z.lb, z.ub) == (-2.0, 5.0)
        assert problem.n_constraints == 4

    @pytest.mark.parametrize(
        "binary_value,x_value",
        list(itertools.product((0, 1), (-3.0, 0.0, 2.5, 4.0))),
    )
    def test_is_exact_across_bounds(self, binary_value, x_value):
        _require_highs()
        problem = Problem()
        binary = problem.add_variables("b", vtype="binary")
        x = problem.add_variables("x", lb=-3, ub=4)
        z = linearize_binary_continuous_product(problem, binary, x)
        problem.add_constraint(binary == binary_value)
        problem.add_constraint(x == x_value)
        problem.set_objective(1 * z)

        result = solve(problem, solver="highs")

        assert result.status == "solved"
        assert result.obj_val == pytest.approx(binary_value * x_value)

    def test_requires_finite_continuous_bounds(self):
        problem = Problem()
        binary = problem.add_variables("b", vtype="binary")
        x = problem.add_variables("x", lb=0)

        with pytest.raises(ValueError, match="finite bounds"):
            linearize_binary_continuous_product(problem, binary, x)

    def test_requires_continuous_second_factor(self):
        problem = Problem()
        binary = problem.add_variables("b", vtype="binary")
        integer = problem.add_variables("i", lb=0, ub=3, vtype="integer")

        with pytest.raises(ValueError, match="continuous second factor"):
            linearize_binary_continuous_product(problem, binary, integer)
