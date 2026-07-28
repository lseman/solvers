"""Tests for conic primitives and SOC constraints."""

import pytest

from obp.conic import (
    ExpCone,
    PowerCone,
    PowerCone05,
    SOC,
    SOCConstraint,
    exp_cone,
    power_cone,
    soc,
)
from obp.model import Variable


class TestConeTypes:
    def test_soc(self):
        assert SOC.name == "SOC"

    def test_exp_cone(self):
        assert ExpCone.name == "ExpCone"

    def test_power_cone(self):
        pc = PowerCone(0.3)
        assert pc.alpha == 0.3
        assert "PowerCone" in str(pc)

    def test_power_cone_05(self):
        assert PowerCone05.alpha == 0.5


class TestSOCConstraint:
    def test_basic_soc(self):
        x = [Variable(i, f"x{i}") for i in range(3)]
        sc = soc(x, cone_indices=[1, 2])
        assert sc.variables == x
        assert sc.cone_indices == [1, 2]
        assert sc.b == [0.0, 0.0]
        assert sc.c == {x[2]: 1.0}

    def test_soc_custom_b(self):
        x = [Variable(i, f"x{i}") for i in range(3)]
        sc = soc(x, cone_indices=[1, 2], b=[1.0, 2.0])
        assert sc.b == [1.0, 2.0]

    def test_soc_custom_c(self):
        x = [Variable(i, f"x{i}") for i in range(4)]
        sc = soc(x, cone_indices=[1, 2], c={x[3]: 2.0})
        assert sc.c == {x[3]: 2.0}

    def test_soc_empty_variables(self):
        with pytest.raises(ValueError, match="at least one"):
            soc([])

    def test_soc_default_cone_indices(self):
        x = [Variable(i, f"x{i}") for i in range(4)]
        sc = soc(x)
        assert sc.cone_indices == [0, 1, 2]


class TestConeFunctions:
    def test_exp_cone_fn(self):
        x, y, z = Variable(0, "x"), Variable(1, "y"), Variable(2, "z")
        result = exp_cone(x, y, z)
        assert result is ExpCone

    def test_power_cone_fn(self):
        x, y, z = Variable(0, "x"), Variable(1, "y"), Variable(2, "z")
        result = power_cone(x, y, z, alpha=0.3)
        assert isinstance(result, PowerCone)
        assert result.alpha == 0.3

    def test_power_cone_fn_default_alpha(self):
        x, y, z = Variable(0, "x"), Variable(1, "y"), Variable(2, "z")
        result = power_cone(x, y, z)
        assert result.alpha == 0.5
