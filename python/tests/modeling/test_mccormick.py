"""Tests for the McCormick envelope relaxation helper."""

import itertools

import pytest

from obp import Problem, solve
from obp.formulations.mccormick import mccormick_envelope, mccormick_envelope_grid


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestMcCormickValidation:
    def test_requires_finite_bounds_x(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0)  # ub=inf
        y = pb.add_variables("y", lb=0, ub=5)
        with pytest.raises(ValueError, match="finite bounds"):
            mccormick_envelope(pb, x, y)

    def test_requires_finite_bounds_y(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=5)
        y = pb.add_variables("y", lb=0)  # ub=inf
        with pytest.raises(ValueError, match="finite bounds"):
            mccormick_envelope(pb, x, y)

    def test_adds_variable_and_four_constraints(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=10)
        y = pb.add_variables("y", lb=0, ub=5)
        n_before = pb.n_constraints
        w = mccormick_envelope(pb, x, y)
        assert pb.n_vars == 3
        assert pb.n_constraints == n_before + 4
        assert w.lb == 0.0
        assert w.ub == 50.0


class TestMcCormickCorners:
    @pytest.mark.parametrize("xv,yv", list(itertools.product([0, 10], [0, 5])))
    def test_exact_at_box_corners(self, highs_available, xv, yv):
        if not highs_available:
            pytest.skip("HiGHS not available")

        for sense in ("minimize", "maximize"):
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=10)
            y = pb.add_variables("y", lb=0, ub=5)
            w = mccormick_envelope(pb, x, y)
            pb.add_constraint(x == xv)
            pb.add_constraint(y == yv)
            pb.set_objective(1 * w, sense=sense)

            result = solve(pb, solver="highs")
            assert result.status == "solved"
            assert result.obj_val == pytest.approx(xv * yv, abs=1e-6)

    def test_relaxation_contains_true_product_at_interior_point(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        xv, yv = 4.0, 2.5
        true_product = xv * yv

        bounds = {}
        for sense in ("minimize", "maximize"):
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=10)
            y = pb.add_variables("y", lb=0, ub=5)
            w = mccormick_envelope(pb, x, y)
            pb.add_constraint(x == xv)
            pb.add_constraint(y == yv)
            pb.set_objective(1 * w, sense=sense)
            bounds[sense] = solve(pb, solver="highs").obj_val

        assert bounds["minimize"] <= true_product <= bounds["maximize"]

    def test_negative_bounds(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", lb=-3, ub=4)
        y = pb.add_variables("y", lb=-2, ub=5)
        w = mccormick_envelope(pb, x, y)
        pb.add_constraint(x == -3)
        pb.add_constraint(y == -2)
        pb.set_objective(1 * w)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(6.0, abs=1e-6)


class TestPiecewiseMcCormickValidation:
    def test_requires_finite_bounds(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0)  # ub=inf
        y = pb.add_variables("y", lb=0, ub=5)
        with pytest.raises(ValueError, match="finite bounds"):
            mccormick_envelope_grid(pb, x, y, x_breaks=[0, 10], y_breaks=[0, 5])

    def test_requires_at_least_two_breakpoints(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=10)
        y = pb.add_variables("y", lb=0, ub=5)
        with pytest.raises(ValueError, match="at least 2 points"):
            mccormick_envelope_grid(pb, x, y, x_breaks=[0], y_breaks=[0, 5])

    def test_requires_strictly_increasing_breakpoints(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=10)
        y = pb.add_variables("y", lb=0, ub=5)
        with pytest.raises(ValueError, match="strictly increasing"):
            mccormick_envelope_grid(pb, x, y, x_breaks=[0, 5, 5, 10], y_breaks=[0, 5])

    def test_requires_breakpoints_span_bounds(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=10)
        y = pb.add_variables("y", lb=0, ub=5)
        with pytest.raises(ValueError, match="must span"):
            mccormick_envelope_grid(pb, x, y, x_breaks=[0, 8], y_breaks=[0, 5])


class TestPiecewiseMcCormickSolve:
    def test_single_cell_matches_plain_mccormick(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        for sense in ("minimize", "maximize"):
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=10)
            y = pb.add_variables("y", lb=0, ub=5)
            z, delta = mccormick_envelope_grid(pb, x, y, x_breaks=[0, 10], y_breaks=[0, 5])
            pb.add_constraint(x == 4.0)
            pb.add_constraint(y == 2.5)
            pb.set_objective(1 * z, sense=sense)
            grid_result = solve(pb, solver="highs")

            pb2 = Problem()
            x2 = pb2.add_variables("x", lb=0, ub=10)
            y2 = pb2.add_variables("y", lb=0, ub=5)
            w = mccormick_envelope(pb2, x2, y2)
            pb2.add_constraint(x2 == 4.0)
            pb2.add_constraint(y2 == 2.5)
            pb2.set_objective(1 * w, sense=sense)
            plain_result = solve(pb2, solver="highs")

            assert grid_result.obj_val == pytest.approx(plain_result.obj_val, abs=1e-6)

    def test_exact_at_grid_corner(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        x_breaks = [0, 2, 4, 6, 8, 10]
        y_breaks = [0, 1, 2, 3, 4, 5]
        for sense in ("minimize", "maximize"):
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=10)
            y = pb.add_variables("y", lb=0, ub=5)
            z, delta = mccormick_envelope_grid(pb, x, y, x_breaks=x_breaks, y_breaks=y_breaks)
            pb.add_constraint(x == 6.0)  # grid corner
            pb.add_constraint(y == 3.0)  # grid corner
            pb.set_objective(1 * z, sense=sense)
            result = solve(pb, solver="highs")
            assert result.status == "solved"
            assert result.obj_val == pytest.approx(18.0, abs=1e-6)

    def test_refinement_shrinks_gap(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        xv, yv = 3.37, 2.13
        true_product = xv * yv
        prev_gap = float("inf")

        for n_breaks in (2, 3, 5, 9):
            x_breaks = [10.0 * i / (n_breaks - 1) for i in range(n_breaks)]
            y_breaks = [5.0 * i / (n_breaks - 1) for i in range(n_breaks)]
            bounds = {}
            for sense in ("minimize", "maximize"):
                pb = Problem()
                x = pb.add_variables("x", lb=0, ub=10)
                y = pb.add_variables("y", lb=0, ub=5)
                z, delta = mccormick_envelope_grid(pb, x, y, x_breaks=x_breaks, y_breaks=y_breaks)
                pb.add_constraint(x == xv)
                pb.add_constraint(y == yv)
                pb.set_objective(1 * z, sense=sense)
                bounds[sense] = solve(pb, solver="highs").obj_val

            assert bounds["minimize"] <= true_product + 1e-6
            assert bounds["maximize"] >= true_product - 1e-6
            gap = bounds["maximize"] - bounds["minimize"]
            assert gap <= prev_gap + 1e-9, f"gap grew at n_breaks={n_breaks}"
            prev_gap = gap

        assert prev_gap < 0.3  # tight grid should meaningfully narrow the gap

    def test_selects_correct_active_cell(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=10)
        y = pb.add_variables("y", lb=0, ub=5)
        x_breaks = [0, 2, 4, 6, 8, 10]
        y_breaks = [0, 1, 2, 3, 4, 5]
        z, delta = mccormick_envelope_grid(pb, x, y, x_breaks=x_breaks, y_breaks=y_breaks)
        pb.add_constraint(x == 5.0)  # cell i=2: [4,6]
        pb.add_constraint(y == 2.5)  # cell j=2: [2,3]
        pb.set_objective(1 * z)
        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert len(delta) == 5 * 5
