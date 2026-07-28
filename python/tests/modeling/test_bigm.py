"""Tests for max/min big-M constraints (Problem.add_max_constraint / add_min_constraint)."""

import pytest

from obp import Problem, solve
from obp.formulations.bigm import add_max_constraint, add_min_constraint


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestMaxConstraintValidation:
    def test_requires_at_least_two_terms(self):
        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        with pytest.raises(ValueError, match="at least 2 terms"):
            pb.add_max_constraint(z, [x])

    def test_requires_finite_ub_on_z(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        z = pb.add_variables("z", lb=0)  # ub=inf
        with pytest.raises(ValueError, match="finite ub on z"):
            pb.add_max_constraint(z, list(x))

    def test_requires_finite_lb_on_terms(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=-float("inf"), ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        with pytest.raises(ValueError, match="finite lb on every term"):
            pb.add_max_constraint(z, list(x))


class TestMinConstraintValidation:
    def test_requires_at_least_two_terms(self):
        pb = Problem()
        x = pb.add_variables("x", 1, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        with pytest.raises(ValueError, match="at least 2 terms"):
            pb.add_min_constraint(z, [x])

    def test_requires_finite_lb_on_z(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=10)
        z = pb.add_variables("z", lb=-float("inf"), ub=10)
        with pytest.raises(ValueError, match="finite lb on z"):
            pb.add_min_constraint(z, list(x))

    def test_requires_finite_ub_on_terms(self):
        pb = Problem()
        x = pb.add_variables("x", 2, lb=0, ub=float("inf"))
        z = pb.add_variables("z", lb=0, ub=10)
        with pytest.raises(ValueError, match="finite ub on every term"):
            pb.add_min_constraint(z, list(x))


class TestMaxConstraintSolve:
    def test_fixed_terms(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        pb.add_max_constraint(z, list(x))
        pb.add_constraint(x[0] == 3)
        pb.add_constraint(x[1] == 7)
        pb.add_constraint(x[2] == 5)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(7.0, abs=1e-6)

    def test_negative_bounds(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=-10, ub=10)
        z = pb.add_variables("z", lb=-10, ub=10)
        pb.add_max_constraint(z, list(x))
        pb.add_constraint(x[0] == -3)
        pb.add_constraint(x[1] == -7)
        pb.add_constraint(x[2] == -1)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(-1.0, abs=1e-6)

    def test_objective_driven_argmax(self, highs_available):
        """Force the solver to actually search for the argmax rather than
        reading it off fixed inputs."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        pb.add_max_constraint(z, list(x))
        pb.add_constraint(x[0] + x[1] + x[2] == 10)
        # minimize z: best strategy spreads mass evenly so the max is as
        # small as possible -> z = 10/3
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(10.0 / 3.0, abs=1e-5)


class TestMinConstraintSolve:
    def test_fixed_terms(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        pb.add_min_constraint(z, list(x))
        pb.add_constraint(x[0] == 3)
        pb.add_constraint(x[1] == 7)
        pb.add_constraint(x[2] == 5)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(3.0, abs=1e-6)

    def test_objective_driven_argmin(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", 3, lb=0, ub=10)
        z = pb.add_variables("z", lb=0, ub=10)
        pb.add_min_constraint(z, list(x))
        pb.add_constraint(x[0] + x[1] + x[2] == 10)
        # maximize z: best strategy spreads mass evenly so the min is as
        # large as possible -> z = 10/3
        pb.set_objective(1 * z, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(10.0 / 3.0, abs=1e-5)


class TestBigMBruteForceAgreement:
    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    @pytest.mark.parametrize("kind", ["max", "min"])
    def test_matches_brute_force_over_random_instances(self, highs_available, k, kind):
        if not highs_available:
            pytest.skip("HiGHS not available")

        import numpy as np
        from scipy.optimize import linprog

        rng = np.random.default_rng(42 + k + (0 if kind == "max" else 100))
        c_x = rng.uniform(-3, 3, size=k)
        c_z = rng.uniform(-3, 3)
        ub = rng.uniform(3, 10, size=k)
        cap = rng.uniform(sum(ub) * 0.2, sum(ub) * 0.8)

        best = None
        for winner in range(k):
            A_extra, b_extra = [], []
            for i in range(k):
                if i == winner:
                    continue
                row = [0.0] * k
                if kind == "max":
                    row[winner], row[i] = -1.0, 1.0
                else:
                    row[winner], row[i] = 1.0, -1.0
                A_extra.append(row)
                b_extra.append(0.0)
            A_ub = [[1.0] * k] + A_extra
            b_ub = [cap] + b_extra
            bounds = [(0, ub[i]) for i in range(k)]
            c_full = list(c_x)
            c_full[winner] += c_z
            res = linprog(c_full, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                obj = float(np.dot(c_x, res.x) + c_z * res.x[winner])
                if best is None or obj < best:
                    best = obj

        pb = Problem()
        x = pb.add_variables("x", k, lb=0, ub=100)
        for i in range(k):
            x[i].ub = float(ub[i])
        z = pb.add_variables("z", lb=0, ub=float(max(ub)))
        if kind == "max":
            pb.add_max_constraint(z, list(x))
        else:
            pb.add_min_constraint(z, list(x))

        expr = c_z * z
        for i in range(k):
            expr = expr + float(c_x[i]) * x[i]
        pb.set_objective(expr)

        total = x[0]
        for i in range(1, k):
            total = total + x[i]
        pb.add_constraint(total <= float(cap))

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(best, abs=1e-4)
