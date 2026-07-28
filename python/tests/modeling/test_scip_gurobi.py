"""Tests for the SCIP and Gurobi MIP backends."""

import numpy as np
import pytest

from obp import Problem, solve


@pytest.fixture
def scip_available():
    try:
        import pyscipopt  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def gurobi_available():
    try:
        import gurobipy  # noqa: F401
        return True
    except ImportError:
        return False


class TestSCIP:
    def test_lp(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(3 * x[0] + 2 * x[1])
        pb.add_constraint(x[0] + x[1] <= 4)
        pb.add_constraint(x[0] >= 1)

        result = solve(pb, solver="scip")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(3.0, abs=1e-5)
        np.testing.assert_allclose(result.x, [1, 0], atol=1e-5)

    def test_mip(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=10)
        pb.set_objective(-x[0] - 2 * x[1])
        pb.add_constraint(x[0] + x[1] <= 3)
        pb.add_constraint(2 * x[0] + x[1] <= 4)

        result = solve(pb, solver="scip")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(-6.0, abs=1e-5)

    def test_binary(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        b = pb.add_variables("b", 3, vtype="binary")
        pb.set_objective(b[0] + 2 * b[1] + 3 * b[2])
        pb.add_constraint(b[0] + b[1] + b[2] >= 1)

        result = solve(pb, solver="scip")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(1.0, abs=1e-5)

    def test_maximize_sign(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=10)
        pb.set_objective(1 * x, sense="maximize")
        pb.add_constraint(x <= 7)

        result = solve(pb, solver="scip")
        assert result.obj_val == pytest.approx(7.0, abs=1e-5)

    def test_qp_raises(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=-10, ub=10)
        pb.set_objective(x[0] ** 2 + x[1] ** 2)
        pb.add_constraint(x[0] + x[1] == 3)

        with pytest.raises(NotImplementedError, match="QP via SCIP"):
            solve(pb, solver="scip")

    def test_sos1(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        pb.set_objective(2 * x[0] + 5 * x[1] + 3 * x[2], sense="maximize")
        pb.add_constraint(x[0] + x[1] + x[2] <= 10)
        pb.add_sos_constraint(x, type=1)

        result = solve(pb, solver="scip")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(50.0, abs=1e-5)
        assert (result.x > 1e-6).sum() <= 1

    def test_sos2(self, scip_available):
        if not scip_available:
            pytest.skip("pyscipopt not available")

        pb = Problem()
        y = pb.add_variables("y", 4, lb=0, ub=10)
        pb.set_objective(y[0] + y[1] + y[2] + y[3], sense="maximize")
        pb.add_constraint(y[0] + y[1] + y[2] + y[3] <= 10)
        pb.add_sos_constraint(y, type=2)

        result = solve(pb, solver="scip")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(10.0, abs=1e-5)
        nonzero = [i for i, v in enumerate(result.x) if v > 1e-6]
        assert len(nonzero) <= 2
        if len(nonzero) == 2:
            assert abs(nonzero[0] - nonzero[1]) == 1


class TestGurobi:
    def test_lp(self, gurobi_available):
        if not gurobi_available:
            pytest.skip("gurobipy not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(3 * x[0] + 2 * x[1])
        pb.add_constraint(x[0] + x[1] <= 4)
        pb.add_constraint(x[0] >= 1)

        result = solve(pb, solver="gurobi")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(3.0, abs=1e-5)

    def test_mip(self, gurobi_available):
        if not gurobi_available:
            pytest.skip("gurobipy not available")

        pb = Problem()
        x = pb.add_variables("x", 2, vtype="integer", lb=0, ub=10)
        pb.set_objective(-x[0] - 2 * x[1])
        pb.add_constraint(x[0] + x[1] <= 3)
        pb.add_constraint(2 * x[0] + x[1] <= 4)

        result = solve(pb, solver="gurobi")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(-6.0, abs=1e-5)

    def test_qp(self, gurobi_available):
        if not gurobi_available:
            pytest.skip("gurobipy not available")

        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=1)
        pb.set_objective(x[0] ** 2 + x[1] ** 2)
        pb.add_constraint(x[0] + x[1] == 1)

        result = solve(pb, solver="gurobi")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(0.5, abs=1e-4)

    def test_sos1(self, gurobi_available):
        if not gurobi_available:
            pytest.skip("gurobipy not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        pb.set_objective(2 * x[0] + 5 * x[1] + 3 * x[2], sense="maximize")
        pb.add_constraint(x[0] + x[1] + x[2] <= 10)
        pb.add_sos_constraint(x, type=1)

        result = solve(pb, solver="gurobi")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(50.0, abs=1e-5)


class TestUnavailableBackendMessage:
    def test_scip_not_installed_message(self, monkeypatch):
        import obp.backends as backends_pkg
        import obp.backends.ots.scip_backend as scip_mod

        def _raise_import_error():
            raise ImportError("no pyscipopt")

        monkeypatch.delitem(backends_pkg._BACKEND_CACHE, "scip", raising=False)
        monkeypatch.setattr(scip_mod, "_import_scip", _raise_import_error)

        with pytest.raises(RuntimeError, match="not available"):
            backends_pkg.get_backend("scip")
