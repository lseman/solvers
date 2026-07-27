"""Tests for MIP variable types (IntVar, BinVar) and check_mip."""

import pytest

from solvers.obp.model import BinVar, IntVar, Variable, check_mip


class TestIntVar:
    def test_creation(self):
        v = IntVar(0, "x")
        assert v.index == 0
        assert v.vtype == "integer"

    def test_arithmetic(self):
        v = IntVar(0, "x")
        e = 3 * v + 2
        assert e.linear_coeffs[v] == 3.0

    def test_mul_intvar_raises(self):
        v1 = IntVar(0, "x")
        v2 = IntVar(1, "y")
        with pytest.raises(TypeError):
            v1 * v2


class TestBinVar:
    def test_creation(self):
        v = BinVar(0, "b")
        assert v.index == 0
        assert v.vtype == "binary"
        assert v.lb == 0
        assert v.ub == 1

    def test_arithmetic(self):
        v = BinVar(0, "b")
        e = 5 * v - 1
        assert e.linear_coeffs[v] == 5.0

    def test_mul_binvar_raises(self):
        v1 = BinVar(0, "b1")
        v2 = BinVar(1, "b2")
        with pytest.raises(TypeError):
            v1 * v2


class TestCheckMip:
    def test_all_continuous(self):
        vars_list = [Variable(0, "x"), Variable(1, "y")]
        result = check_mip(vars_list)
        assert result == []

    def test_int_detected(self):
        vars_list = [Variable(0, "x"), IntVar(1, "z")]
        result = check_mip(vars_list)
        assert len(result) == 1
        assert "z" in result[0]

    def test_bin_detected(self):
        vars_list = [Variable(0, "x"), BinVar(1, "b")]
        result = check_mip(vars_list)
        assert len(result) == 1
        assert "b" in result[0]

    def test_mixed(self):
        vars_list = [IntVar(0, "x"), BinVar(1, "b"), Variable(2, "y")]
        result = check_mip(vars_list)
        assert len(result) == 2
