"""Tests for Variable, Constraint, and Problem."""

import numpy as np
import pytest

from solvers.obp.expression import Expression, _BoundExpression
from solvers.obp.model import (
    BinVar,
    Constraint,
    IntVar,
    Problem,
    Variable,
)


class TestVariable:
    def test_creation(self):
        v = Variable(0, "x", lb=0, ub=10)
        assert v.index == 0
        assert v.name == "x"
        assert v.lb == 0
        assert v.ub == 10

    def test_hash_and_eq(self):
        v1 = Variable(0, "x")
        v2 = Variable(1, "y")
        assert hash(v1) != hash(v2)
        assert v1 != v2

    def test_add_var(self, x):
        e = x + 5
        assert isinstance(e, Expression)
        assert e.linear_coeffs[x] == 1.0

    def test_add_var_var(self, x):
        y = Variable(1, "y")
        e = x + y
        assert x in e.linear_coeffs
        assert y in e.linear_coeffs

    def test_sub_var(self, x):
        e = x - 3
        assert e.linear_coeffs[x] == 1.0

    def test_mul_scalar(self, x):
        e = 2 * x
        assert e.linear_coeffs[x] == 2.0

    def test_mul_var(self, x):
        y = Variable(1, "y")
        e = x * y
        assert e.is_quadratic is True

    def test_pow2(self, x):
        e = x ** 2
        assert e.is_quadratic is True
        vi, vj, c = e.quadratic_terms[0]
        assert vi._index == x._index
        assert vj._index == x._index
        assert c == 1.0

    def test_neg(self, x):
        e = -x
        assert e.linear_coeffs[x] == -1.0


class TestConstraint:
    def test_creation(self):
        c = Constraint(Expression.constant(0), "<=", 5.0, "my_con")
        assert c.sense == "<="
        assert c.bound == 5.0
        assert c.name == "my_con"


class TestProblem:
    def test_add_single_variable(self):
        pb = Problem("test")
        x = pb.add_variables("x")
        assert isinstance(x, Variable)
        assert pb.n_vars == 1

    def test_add_multiple_variables(self):
        pb = Problem("test")
        xs = pb.add_variables("x", n=3)
        assert isinstance(xs, list)
        assert len(xs) == 3
        assert pb.n_vars == 3

    def test_add_integer_variable(self):
        pb = Problem("test")
        x = pb.add_variables("x", vtype="integer")
        assert isinstance(x, IntVar)
        assert x.vtype == "integer"

    def test_add_binary_variable(self):
        pb = Problem("test")
        x = pb.add_variables("x", vtype="binary")
        assert isinstance(x, BinVar)
        assert x.vtype == "binary"

    def test_set_objective(self, x):
        pb = Problem("test")
        pb.set_objective(3 * x)
        assert pb._objective_expr.linear_coeffs[x] == 3.0

    def test_add_constraint_expression(self, x):
        pb = Problem("test")
        pb.add_constraint(3 * x <= 5)
        assert len(pb._constraints) == 1
        c = pb._constraints[0]
        assert c.sense == "<="
        assert c.bound == 5.0

    def test_add_constraint_ge(self, x):
        pb = Problem("test")
        pb.add_constraint(3 * x >= 5)
        assert pb._constraints[0].sense == ">="

    def test_add_constraint_eq(self, x):
        pb = Problem("test")
        pb.add_constraint(3 * x == 5)
        assert pb._constraints[0].sense == "=="

    def test_add_constraint_object(self, x):
        pb = Problem("test")
        c = Constraint(3 * x, "<=", 5.0, "my_con")
        pb.add_constraint(c)
        assert len(pb._constraints) == 1
        assert pb._constraints[0].name == "my_con"

    def test_assemble_minimize(self):
        pb = Problem("test")
        xs = pb.add_variables("x", n=3, lb=0)
        pb.set_objective(xs[0] + xs[1] + xs[2])
        pb.add_constraint(xs[0] + xs[1] + xs[2] == 4)
        obj, sense, con, soc, vars_list = pb.assemble()
        assert sense == "minimize"
        assert len(con) == 1
        assert len(soc) == 0

    def test_assemble_maximize(self):
        pb = Problem("test")
        xs = pb.add_variables("x", n=2)
        pb.set_objective(xs[0] + xs[1], sense="maximize")
        obj, sense, _, _, _ = pb.assemble()
        assert sense == "maximize"

    def test_assemble_no_objective(self):
        pb = Problem("test")
        pb.add_variables("x")
        with pytest.raises(ValueError, match="objective"):
            pb.assemble()

    def test_assemble_no_variables(self):
        pb = Problem("test")
        pb.set_objective(Expression.constant(0))
        with pytest.raises(ValueError, match="variables"):
            pb.assemble()

    def test_quadratic_objective(self):
        pb = Problem("test")
        xs = pb.add_variables("x", n=2)
        pb.set_objective(xs[0] ** 2 + xs[1] ** 2)
        obj, _, _, _, _ = pb.assemble()
        assert obj.is_quadratic is True

    def test_repr(self):
        pb = Problem("test")
        xs = pb.add_variables("x", n=3)
        pb.set_objective(xs[0])
        r = repr(pb)
        assert "test" in r
        assert "vars=3" in r


class TestBoundExpression:
    def test_le(self, x):
        e = 3 * x
        be = e <= 5
        assert isinstance(be, _BoundExpression)
        assert be._sense == "<="
        assert be._bound == 5.0

    def test_ge(self, x):
        e = 3 * x
        be = e >= 5
        assert be._sense == ">="
        assert be._bound == 5.0

    def test_eq(self, x):
        e = 3 * x
        be = e == 5
        assert be._sense == "=="
        assert be._bound == 5.0


class TestIntVarBinVar:
    def test_intvar_creation(self):
        v = IntVar(0, "x", lb=0, ub=100)
        assert v.vtype == "integer"
        assert v.lb == 0
        assert v.ub == 100

    def test_binvar_creation(self):
        v = BinVar(0, "b")
        assert v.vtype == "binary"
        assert v.lb == 0
        assert v.ub == 1

    def test_intvar_arithmetic(self):
        v = IntVar(0, "x")
        e = 3 * v + 2
        assert e.linear_coeffs[v] == 3.0

    def test_intvar_mul_raises(self):
        v = IntVar(0, "x")
        w = IntVar(1, "y")
        with pytest.raises(TypeError):
            v * w

    def test_binvar_mul_raises(self):
        v = BinVar(0, "b1")
        w = BinVar(1, "b2")
        with pytest.raises(TypeError):
            v * w


class TestAddConstraints:
    def test_matrix_form_row_per_constraint(self):
        pb = Problem()
        x = np.array(pb.add_variables("x", 3, lb=0), dtype=object)
        A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        b = np.array([5.0, 3.0])

        cons = pb.add_constraints(A @ x, "<=", b)
        assert len(cons) == 2
        assert pb.n_constraints == 2
        np.testing.assert_allclose(cons[0].body.linear_coeffs[x[0]], 1.0)
        np.testing.assert_allclose(cons[0].body.linear_coeffs[x[1]], 1.0)
        assert cons[0].bound == 5.0
        assert cons[1].bound == 3.0
        assert cons[0].sense == "<="

    def test_scalar_bound_broadcasts(self):
        pb = Problem()
        x = np.array(pb.add_variables("x", 3, lb=0), dtype=object)
        cons = pb.add_constraints(list(x), "<=", 4.0)
        assert len(cons) == 3
        assert all(c.bound == 4.0 for c in cons)

    def test_bound_length_mismatch_raises(self):
        pb = Problem()
        x = np.array(pb.add_variables("x", 3, lb=0), dtype=object)
        with pytest.raises(ValueError):
            pb.add_constraints(list(x), "<=", [1.0, 2.0])

    def test_name_prefix_applied_per_row(self):
        pb = Problem()
        x = np.array(pb.add_variables("x", 2, lb=0), dtype=object)
        cons = pb.add_constraints(list(x), "<=", 4.0, name="cap")
        assert cons[0].name == "cap_0"
        assert cons[1].name == "cap_1"


@pytest.fixture
def x():
    return Variable(0, "x")
