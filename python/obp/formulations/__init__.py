"""formulations — MIP/LP reformulation helpers built on top of the core DSL.

- :mod:`.mccormick`: McCormick envelope relaxation for bilinear terms
  (single-cell and piecewise grid).
- :mod:`.pwl`: piecewise-linear interpolation of f(x) and f(x, y).
- :mod:`.bigm`: big-M reformulations for max/min constraints.
- :mod:`.products`: exact linearizations for binary products.
- :mod:`.indicator`: big-M indicator constraints and exact absolute value.
- :mod:`.logic`: AND/OR/NOT reformulations for binary variables.
"""

from __future__ import annotations

from .bigm import add_max_constraint, add_min_constraint
from .indicator import abs_value, add_indicator_constraint
from .logic import logical_and, logical_not, logical_or
from .mccormick import mccormick_envelope, mccormick_envelope_grid
from .products import (
    linearize_binary_continuous_product,
    linearize_binary_product,
)
from .pwl import piecewise_linear_1d, piecewise_linear_2d

__all__ = [
    "mccormick_envelope",
    "mccormick_envelope_grid",
    "piecewise_linear_1d",
    "piecewise_linear_2d",
    "add_max_constraint",
    "add_min_constraint",
    "linearize_binary_product",
    "linearize_binary_continuous_product",
    "add_indicator_constraint",
    "abs_value",
    "logical_and",
    "logical_or",
    "logical_not",
]
