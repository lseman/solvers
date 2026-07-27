"""Tests for Expression arithmetic and sparse assembly."""

import numpy as np
import pytest

from solvers.obp.expression import Expression
from solvers.obp.model import Variable


@pytest.fixture
def x():
    return Variable(0, "x")


@pytest.fixture
def y():
    return Variable(1, "y")


@pytest.fixture
def z():
    return Variable(2, "z")


class TestExpressionConstructors:
    def test_constant(self):
        e = Expression.constant(5.0)
        assert e.is_linear is True   # constant has no quadratic terms
        assert e.is_quadratic is False

    def test_from_variable(self, x):
        e = Expression.from_variable(x)
        assert x in e.linear_coeffs
        assert e.linear_coeffs[x] == 1.0
        assert e.is_linear is True

    def test_from_variable_coeff(self, x):
        e = Expression._from_raw({x: 3.0}, [])
        assert e.linear_coeffs[x] == 3.0


class TestExpressionArithmetic:
    def test_add_two_variables(self, x, y):
        e = Expression.from_variable(x) + Expression.from_variable(y)
        assert x in e.linear_coeffs
        assert y in e.linear_coeffs
        assert e.linear_coeffs[x] == 1.0
        assert e.linear_coeffs[y] == 1.0
        assert e.is_linear is True

    def test_add_scalar(self, x):
        e = Expression.from_variable(x) + 5.0
        assert x in e.linear_coeffs
        assert e.linear_coeffs[x] == 1.0

    def test_sub_scalar(self, x):
        e = Expression.from_variable(x) - 3.0
        assert e.linear_coeffs[x] == 1.0

    def test_radd_scalar(self, x):
        e = 7.0 + Expression.from_variable(x)
        assert e.linear_coeffs[x] == 1.0

    def test_mul_scalar(self, x):
        e = Expression.from_variable(x) * 4.0
        assert e.linear_coeffs[x] == 4.0

    def test_rmul_scalar(self, x):
        e = 4.0 * Expression.from_variable(x)
        assert e.linear_coeffs[x] == 4.0

    def test_neg(self, x):
        e = -Expression.from_variable(x)
        assert e.linear_coeffs[x] == -1.0

    def test_mul_returns_notimpl_for_expr(self, x):
        with pytest.raises(TypeError):
            Expression.from_variable(x) * Expression.from_variable(x)

    def test_rsub(self, x):
        e = 10.0 - Expression.from_variable(x)
        assert e.linear_coeffs[x] == -1.0


class TestQuadraticExpression:
    def test_pow2_single_var(self, x):
        e = Expression.from_variable(x) ** 2
        assert e.is_quadratic is True
        assert e.is_linear is False
        assert len(e.quadratic_terms) == 1
        vi, vj, c = e.quadratic_terms[0]
        assert vi._index == x._index
        assert vj._index == x._index
        assert c == 1.0

    def test_pow2_cross_terms(self, x, y):
        e = (Expression.from_variable(x) + Expression.from_variable(y)) ** 2
        assert e.is_quadratic is True
        # (x + y)^2 = x^2 + 2xy + y^2
        terms = e.quadratic_terms
        assert len(terms) == 3  # coalesced into 3 terms
        # Check coefficient of xy cross term is 2.0
        xy_terms = [(vi, vj, c) for vi, vj, c in terms
                    if (vi._index == x._index and vj._index == y._index)
                    or (vi._index == y._index and vj._index == x._index)]
        assert len(xy_terms) == 1
        assert xy_terms[0][2] == 2.0
        # Should have x^2, y^2, and 2xy
        has_xx = any(vi._index == x._index and vj._index == x._index for vi, vj, _ in terms)
        has_yy = any(vi._index == y._index and vj._index == y._index for vi, vj, _ in terms)
        has_xy = any(
            (vi._index == x._index and vj._index == y._index) or
            (vi._index == y._index and vj._index == x._index)
            for vi, vj, _ in terms
        )
        assert has_xx and has_yy and has_xy

    def test_quad_from_mul(self, x, y):
        # Expression * Expression not supported; use _from_raw directly
        e = Expression._from_raw({}, [(x, y, 1.0)])
        assert e.is_quadratic is True
        assert len(e.quadratic_terms) == 1


class TestSparseAssembly:
    def test_linear_to_arrays(self, x, y):
        e = 3 * Expression.from_variable(x) + 2 * Expression.from_variable(y)
        rows, cols, vals = e.to_linear_arrays(n_vars=3)
        assert len(rows) == 2
        assert set(cols) == {0, 1}

    def test_quad_to_arrays(self, x, y):
        # Use _from_raw for quadratic terms (Expression * Expression not supported)
        e = (Expression.from_variable(x) ** 2
             + Expression._from_raw({}, [(x, y, 2.0)]))
        rows, cols, vals = e.to_quadratic_arrays(n_vars=3)
        assert len(rows) == 2  # x^2 and xy

    def test_to_arrays_mixed(self, x, y):
        e = 3 * Expression.from_variable(x) + Expression.from_variable(x) ** 2
        rows, cols, vals = e.to_arrays(n_vars=3)
        assert len(rows) == 2  # one linear, one quadratic

    def test_variables_set(self, x, y, z):
        e = 3 * Expression.from_variable(x) + Expression.from_variable(y) ** 2
        vars_in_expr = e.variables
        assert x in vars_in_expr
        assert y in vars_in_expr
        assert z not in vars_in_expr


class TestExpressionRepr:
    def test_empty(self):
        e = Expression.constant(0)
        assert "constant" in repr(e)

    def test_linear(self, x):
        e = 3 * Expression.from_variable(x)
        assert "x" in repr(e)

    def test_quadratic(self, x):
        e = Expression.from_variable(x) ** 2
        assert "x" in repr(e) and "^2" in repr(e)
