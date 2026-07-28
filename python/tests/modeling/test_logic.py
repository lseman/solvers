"""Tests for AND/OR/NOT reformulations on binary variables."""

import itertools

import pytest

from obp import Problem, solve
from obp.formulations.logic import logical_and, logical_not, logical_or


def _require_highs() -> None:
    pytest.importorskip("highspy")


class TestLogicalAnd:
    def test_adds_one_binary_and_constraints(self):
        pb = Problem()
        bs = pb.add_variables("b", 3, vtype="binary")
        z = logical_and(pb, list(bs))

        assert z.vtype == "binary"
        assert pb.n_constraints == 4  # 3 le + 1 ge

    @pytest.mark.parametrize("combo", list(itertools.product((0, 1), repeat=3)))
    def test_exact_for_all_assignments(self, combo):
        _require_highs()
        pb = Problem()
        bs = pb.add_variables("b", 3, vtype="binary")
        z = logical_and(pb, list(bs))
        for i, v in enumerate(combo):
            pb.add_constraint(bs[i] == v)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(float(all(combo)))

    def test_requires_at_least_two_terms(self):
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        with pytest.raises(ValueError, match="at least 2 terms"):
            logical_and(pb, [b])

    def test_rejects_nonbinary_term(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=1)
        b = pb.add_variables("b", vtype="binary")
        with pytest.raises(ValueError, match="must be a binary"):
            logical_and(pb, [x, b])


class TestLogicalOr:
    def test_adds_one_binary_and_constraints(self):
        pb = Problem()
        bs = pb.add_variables("b", 3, vtype="binary")
        z = logical_or(pb, list(bs))

        assert z.vtype == "binary"
        assert pb.n_constraints == 4  # 3 ge + 1 le

    @pytest.mark.parametrize("combo", list(itertools.product((0, 1), repeat=3)))
    def test_exact_for_all_assignments(self, combo):
        _require_highs()
        pb = Problem()
        bs = pb.add_variables("b", 3, vtype="binary")
        z = logical_or(pb, list(bs))
        for i, v in enumerate(combo):
            pb.add_constraint(bs[i] == v)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(float(any(combo)))

    def test_requires_at_least_two_terms(self):
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        with pytest.raises(ValueError, match="at least 2 terms"):
            logical_or(pb, [b])


class TestLogicalNot:
    @pytest.mark.parametrize("v,expected", [(0, 1.0), (1, 0.0)])
    def test_exact(self, v, expected):
        _require_highs()
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        pb.add_constraint(b == v)
        pb.set_objective(1 * logical_not(b) + 0)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(expected)

    def test_returns_expression_no_new_variable(self):
        pb = Problem()
        b = pb.add_variables("b", vtype="binary")
        expr = logical_not(b)
        assert pb.n_vars == 1  # no auxiliary variable added

    def test_rejects_nonbinary(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=1)
        with pytest.raises(ValueError, match="must be a binary"):
            logical_not(x)


class TestProblemMethods:
    def test_add_logical_and_method(self):
        _require_highs()
        pb = Problem()
        bs = pb.add_variables("b", 2, vtype="binary")
        z = pb.add_logical_and(list(bs))
        pb.add_constraint(bs[0] == 1)
        pb.add_constraint(bs[1] == 1)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(1.0)

    def test_add_logical_or_method(self):
        _require_highs()
        pb = Problem()
        bs = pb.add_variables("b", 2, vtype="binary")
        z = pb.add_logical_or(list(bs))
        pb.add_constraint(bs[0] == 0)
        pb.add_constraint(bs[1] == 0)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.obj_val == pytest.approx(0.0)
