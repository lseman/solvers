"""Tests for piecewise-linear interpolation (pwl module)."""

import pytest

from obp import Problem, solve
from obp.formulations.pwl import piecewise_linear_1d, piecewise_linear_2d


@pytest.fixture
def highs_available():
    try:
        import highspy  # noqa: F401
        return True
    except ImportError:
        return False


class TestPWL1DValidation:
    def test_requires_at_least_two_breakpoints(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        with pytest.raises(ValueError, match="at least 2 points"):
            piecewise_linear_1d(pb, x, [0], [0])

    def test_requires_matching_lengths(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        with pytest.raises(ValueError, match="same length"):
            piecewise_linear_1d(pb, x, [0, 1, 2], [0, 1])

    def test_requires_strictly_increasing(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        with pytest.raises(ValueError, match="strictly increasing"):
            piecewise_linear_1d(pb, x, [0, 2, 2, 4], [0, 1, 2, 3])


class TestPWL1DSolve:
    BREAKPOINTS = [0, 1, 2, 3, 4]
    VALUES = [0, 1, 4, 9, 16]  # x^2 at each breakpoint

    @pytest.mark.parametrize("method", ["cc", "dlog", "log"])
    @pytest.mark.parametrize(
        "xv,expected",
        [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (2.5, 6.5), (3.5, 12.5)],
    )
    def test_interpolates_at_breakpoints_and_midpoints(
        self, highs_available, method, xv, expected
    ):
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        z = piecewise_linear_1d(pb, x, self.BREAKPOINTS, self.VALUES, method=method)
        pb.add_constraint(x == xv)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("method", ["cc", "dlog", "log"])
    def test_rejects_nonadjacent_interpolation(self, highs_available, method):
        # zigzag function: maximizing z should hit a true breakpoint peak,
        # not some average of non-adjacent breakpoints.
        if not highs_available:
            pytest.skip("HiGHS not available")

        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        z = piecewise_linear_1d(pb, x, [0, 1, 2, 3, 4], [0, 10, 0, 10, 0], method=method)
        pb.set_objective(1 * z, sense="maximize")

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(10.0, abs=1e-5)


class TestPWL2DValidation:
    def test_requires_at_least_two_breakpoints(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        y = pb.add_variables("y", lb=0, ub=4)
        with pytest.raises(ValueError, match="x_breaks needs at least 2"):
            piecewise_linear_2d(pb, x, y, [0], [0, 1], [[0], [0]])

    def test_requires_z_grid_shape(self):
        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=4)
        y = pb.add_variables("y", lb=0, ub=4)
        with pytest.raises(ValueError, match="z_grid must have shape"):
            piecewise_linear_2d(pb, x, y, [0, 1], [0, 1], [[0, 0]])


class TestPWL2DSolve:
    def test_exact_on_linear_function(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        x_breaks = [0, 2, 4]
        y_breaks = [0, 1, 2]
        z_grid = [[xv + yv for yv in y_breaks] for xv in x_breaks]

        for xv, yv in [(1.0, 0.5), (3.0, 1.5), (2.5, 0.7), (0.0, 0.0), (4.0, 2.0)]:
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=4)
            y = pb.add_variables("y", lb=0, ub=2)
            z = piecewise_linear_2d(pb, x, y, x_breaks, y_breaks, z_grid)
            pb.add_constraint(x == xv)
            pb.add_constraint(y == yv)
            pb.set_objective(1 * z)

            result = solve(pb, solver="highs")
            assert result.status == "solved"
            assert result.obj_val == pytest.approx(xv + yv, abs=1e-5)

    def test_exact_at_grid_corners_bilinear(self, highs_available):
        if not highs_available:
            pytest.skip("HiGHS not available")

        x_breaks = [0, 2, 4]
        y_breaks = [0, 1, 2]
        z_grid = [[xv * yv for yv in y_breaks] for xv in x_breaks]

        for xv, yv in [(0, 0), (2, 1), (4, 2), (2, 0), (0, 2), (4, 0)]:
            pb = Problem()
            x = pb.add_variables("x", lb=0, ub=4)
            y = pb.add_variables("y", lb=0, ub=2)
            z = piecewise_linear_2d(pb, x, y, x_breaks, y_breaks, z_grid)
            pb.add_constraint(x == xv)
            pb.add_constraint(y == yv)
            pb.set_objective(1 * z)

            result = solve(pb, solver="highs")
            assert result.status == "solved"
            assert result.obj_val == pytest.approx(xv * yv, abs=1e-5)

    def test_correct_triangle_interpolation(self, highs_available):
        """Point below the diagonal must use the SW-NE-SE triangle's plane,
        not the SW-NW-NE one — checked against a hand-computed barycentric
        interpolation."""
        if not highs_available:
            pytest.skip("HiGHS not available")

        x_breaks = [0, 2]
        y_breaks = [0, 2]
        z_grid = [[0, 0], [0, 4]]  # x*y at (0,0),(0,2),(2,0),(2,2)

        pb = Problem()
        x = pb.add_variables("x", lb=0, ub=2)
        y = pb.add_variables("y", lb=0, ub=2)
        z = piecewise_linear_2d(pb, x, y, x_breaks, y_breaks, z_grid)
        pb.add_constraint(x == 1.5)
        pb.add_constraint(y == 0.5)
        pb.set_objective(1 * z)

        result = solve(pb, solver="highs")
        assert result.status == "solved"
        assert result.obj_val == pytest.approx(1.0, abs=1e-5)
