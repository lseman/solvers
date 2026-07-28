"""Tests for Problem.write_lp / Problem.write_mps."""

from __future__ import annotations

import pytest

highspy = pytest.importorskip("highspy")

from obp import Problem, soc, solve


class TestWriteLp:
    def test_writes_readable_file(self, tmp_path):
        pb = Problem("blend")
        x = pb.add_variables("x", 3, lb=0, ub=10)
        pb.set_objective(3 * x[0] + 2 * x[1] + x[2])
        pb.add_constraint(x[0] + x[1] + x[2] == 4, "c_eq")
        pb.add_constraint(2 * x[0] + x[1] <= 5, "c_le")

        path = tmp_path / "model.lp"
        pb.write_lp(str(path))
        assert path.exists()

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getObjectiveValue() == pytest.approx(solve(pb, solver="highs").obj_val)

    def test_preserves_names(self, tmp_path):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0)
        pb.set_objective(x[0] + x[1])
        pb.add_constraint(x[0] + x[1] <= 5, "cap")

        path = tmp_path / "model.lp"
        pb.write_lp(str(path))
        text = path.read_text()
        assert "x_0" in text
        assert "cap" in text

    def test_maximize_sense_preserved(self, tmp_path):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(2 * x[0] + 5 * x[1], sense="maximize")
        pb.add_constraint(x[0] + x[1] <= 10)

        path = tmp_path / "model.lp"
        pb.write_lp(str(path))
        text = path.read_text()
        assert "max" in text.splitlines()[1]

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getObjectiveValue() == pytest.approx(solve(pb, solver="highs").obj_val)


class TestWriteMps:
    def test_writes_readable_file(self, tmp_path):
        pb = Problem("blend")
        x = pb.add_variables("x", 2, lb=0, ub=10)
        pb.set_objective(x[0] + x[1])
        pb.add_constraint(x[0] + x[1] >= 1, "c1")

        path = tmp_path / "model.mps"
        pb.write_mps(str(path))
        assert path.exists()

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getObjectiveValue() == pytest.approx(1.0)

    def test_quadratic_objective(self, tmp_path):
        pb = Problem()
        y = pb.add_variables("y", 2, lb=-10, ub=10)
        pb.set_objective(y[0] ** 2 + y[1] ** 2 + y[0])
        pb.add_constraint(y[0] + y[1] >= 1)

        path = tmp_path / "qp.mps"
        pb.write_mps(str(path))
        text = path.read_text()
        assert "QUADOBJ" in text

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getObjectiveValue() == pytest.approx(solve(pb, solver="highs").obj_val, abs=1e-4)

    def test_mip_variable_types(self, tmp_path):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10, vtype="integer")
        b = pb.add_variables("b", vtype="binary")
        pb.set_objective(2 * x[0] + 5 * x[1] + 3 * b, sense="maximize")
        pb.add_constraint(x[0] + x[1] <= 10)

        path = tmp_path / "mip.mps"
        pb.write_mps(str(path))

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getObjectiveValue() == pytest.approx(solve(pb, solver="highs").obj_val)


class TestExportEdgeCases:
    def test_soc_constraint_raises(self, tmp_path):
        pb = Problem()
        z = pb.add_variables("z", 3, lb=0)
        pb.set_objective(z[0] + 0)
        pb.add_soc_constraint(soc(z, cone_indices=[1, 2]))

        with pytest.raises(ValueError, match="second-order cone"):
            pb.write_lp(str(tmp_path / "soc.lp"))

    def test_sos_constraint_expanded_and_solvable(self, tmp_path):
        pb = Problem()
        s = pb.add_variables("s", 3, lb=0, ub=5)
        pb.set_objective(s[0] + s[1] + s[2])
        pb.add_sos_constraint(s, type=1)

        path = tmp_path / "sos.lp"
        pb.write_lp(str(path))

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(str(path))
        h.run()
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
