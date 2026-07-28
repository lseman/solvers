"""Tests for indicator constraints and exact absolute value."""

import pytest

from obp import Problem, solve
from obp.formulations.indicator import abs_value, add_indicator_constraint


def _require_highs() -> None:
    pytest.importorskip("highspy")


class TestIndicatorConstraint:
    @pytest.mark.parametrize("bval,expected", [(0, 10.0), (1, 3.0)])
    def test_le_direction(self, bval, expected):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        add_indicator_constraint(pb, b, 1 * x, "<=", 3.0)
        pb.add_constraint(b == bval)
        pb.set_objective(1 * x, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(expected)

    @pytest.mark.parametrize("bval,expected", [(0, 0.0), (1, 3.0)])
    def test_ge_direction(self, bval, expected):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        add_indicator_constraint(pb, b, 1 * x, ">=", 3.0)
        pb.add_constraint(b == bval)
        pb.set_objective(1 * x)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(expected)

    @pytest.mark.parametrize("bval,expected", [(0, 0.0), (1, 5.0)])
    def test_eq_direction(self, bval, expected):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        add_indicator_constraint(pb, b, 1 * x, "==", 5.0)
        pb.add_constraint(b == bval)
        pb.set_objective(1 * x)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(expected)

    @pytest.mark.parametrize("bval,expected", [(0, 3.0), (1, 10.0)])
    def test_activate_on_zero(self, bval, expected):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        add_indicator_constraint(pb, b, 1 * x, "<=", 3.0, activate_on=0)
        pb.add_constraint(b == bval)
        pb.set_objective(1 * x, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(expected)

    def test_multivariable_body(self):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=20)
        y = pb.add_variables("y", lb=0, ub=20)
        add_indicator_constraint(pb, b, x + 2 * y, "<=", 10.0)
        pb.add_constraint(b == 1)
        pb.set_objective(x + 2 * y, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(10.0)

    def test_rejects_nonbinary_indicator(self):
        pb = Problem()
        b = pb.add_variables("b", lb=0, ub=1)
        x = pb.add_variables("x", lb=0, ub=10)
        with pytest.raises(ValueError, match="binary must be a binary"):
            add_indicator_constraint(pb, b, 1 * x, "<=", 3.0)

    def test_rejects_invalid_sense(self):
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        with pytest.raises(ValueError, match="sense must be"):
            add_indicator_constraint(pb, b, 1 * x, "!=", 3.0)

    def test_rejects_unbounded_body(self):
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0)  # ub = inf
        with pytest.raises(ValueError, match="finite bounds"):
            add_indicator_constraint(pb, b, 1 * x, "<=", 3.0)


class TestAbsValue:
    def test_adds_one_continuous_and_one_binary(self):
        pb = Problem()
        x = pb.add_variables("x", lb=-5, ub=5)
        z = abs_value(pb, x, name="z")

        assert z.name == "z_0"
        assert (z.lb, z.ub) == (0.0, 5.0)
        assert pb.n_vars == 3  # x, z, sign binary

    @pytest.mark.parametrize("xv", [-4.0, -1.5, 0.0, 2.0, 5.0])
    def test_exact_symmetric_bounds(self, xv):
        _require_highs()
        pb = Problem()
        x = pb.add_variables("x", lb=-5, ub=5)
        z = abs_value(pb, x)
        pb.add_constraint(x == xv)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(abs(xv))

    @pytest.mark.parametrize("xv", [-3.0, -0.5, 0.0, 1.0, 7.0])
    def test_exact_asymmetric_bounds(self, xv):
        _require_highs()
        pb = Problem()
        x = pb.add_variables("x", lb=-3, ub=7)
        z = abs_value(pb, x)
        pb.add_constraint(x == xv)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(abs(xv))

    def test_exact_under_maximize(self):
        """z >= x, z >= -x alone would let a maximizing objective push z
        past |x| -- verify the big-M pin keeps it exact either way."""
        _require_highs()
        pb = Problem()
        x = pb.add_variables("x", lb=-5, ub=5)
        z = abs_value(pb, x)
        pb.add_constraint(x == 3.0)
        pb.set_objective(1 * z, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(3.0)

    def test_rejects_unbounded_x(self):
        pb = Problem()
        x = pb.add_variables("x", lb=-float("inf"), ub=5)
        with pytest.raises(ValueError, match="finite bounds"):
            abs_value(pb, x)


class TestProblemMethods:
    def test_add_abs_value_method(self):
        _require_highs()
        pb = Problem()
        x = pb.add_variables("x", lb=-5, ub=5)
        z = pb.add_abs_value(x)
        pb.add_constraint(x == -2.0)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(2.0)

    def test_add_indicator_constraint_method(self):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        x = pb.add_variables("x", lb=0, ub=10)
        pb.add_indicator_constraint(b, 1 * x, "<=", 3.0)
        pb.add_constraint(b == 1)
        pb.set_objective(1 * x, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(3.0)
