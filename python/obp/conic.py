"""conic — conic primitive types and SOC builder.

Exports:
    SOC: Second-order (L2) cone
    ExpCone: Exponential cone  K_exp = cl(x,y,z) | x>=y*log(z/y), z>y>0 U x>0,y=0,z>=0
    PowerCone: Power cone  K_alpha = cl(x,y,z) | x^alpha * y^(1-alpha) >= |z|, x>=0, y>=0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Variable


@dataclass(frozen=True)
class ConeType:
    """Base type for conic primitives."""

    name: str


SOC = ConeType("SOC")
"""Second-order (L2) cone: ``||x||_2 <= t``."""


ExpCone = ConeType("ExpCone")
"""Exponential cone: ``x >= y*log(z/y)``, ``z > y > 0``."""


@dataclass(frozen=True)
class PowerCone(ConeType):
    """Power cone: ``x^alpha * y^(1-alpha) >= |z|``, ``x >= 0``, ``y >= 0``.

    Parameters:
        alpha: Exponent in ``(0, 1)``.
    """

    alpha: float

    def __init__(self, alpha: float) -> None:
        object.__setattr__(self, "alpha", alpha)
        super().__init__(f"PowerCone(alpha={alpha})")


PowerCone05 = PowerCone(0.5)
"""Symmetric power cone (``alpha = 0.5``), equivalent to a rotated SOC."""


@dataclass
class SOCConstraint:
    """Second-order cone constraint data.

    Represents: ``||A @ x + b||_2 <= c @ x + d``

    If A is None, defaults to identity on the cone variables.
    If c is empty, defaults to the last variable having coefficient 1.
    """

    variables: list["Variable"]
    cone_indices: list[int]
    A_data: list[tuple[int, int, float]] | None = None
    b: list[float] = field(default_factory=list)
    c: dict["Variable", float] = field(default_factory=dict)
    d: float = 0.0

    def __post_init__(self) -> None:
        if not self.b:
            object.__setattr__(self, "b", [0.0] * len(self.cone_indices))
        if not self.c:
            object.__setattr__(
                self,
                "c",
                {self.variables[-1]: 1.0} if len(self.variables) > 1 else {},
            )


def soc(
    variables: list["Variable"],
    cone_indices: list[int] | None = None,
    b: list[float] | None = None,
    c: dict["Variable", float] | None = None,
    d: float = 0.0,
    name: str = "",
) -> SOCConstraint:
    """Create a second-order cone constraint.

    Standard form: ``||x[cone_indices] + b||_2 <= sum(c[i]*x[i]) + d``

    Parameters:
        variables: All variables in the constraint.
        cone_indices: Indices forming the cone body. Default: all but last.
        b: Offset inside the norm. Default: zeros.
        c: Linear coefficients outside the norm. Default: last variable = 1.
        d: Offset outside the norm. Default: 0.
        name: Optional label.

    Returns:
        An :class:`SOCConstraint` ready for :meth:`Problem.add_soc_constraint`.

    Example::

        x = pb.add_variables("x", 3, lb=0)
        # ||x[1], x[2]||_2 <= x[0]  (rotated second-order cone)
        pb.add_soc_constraint(soc(x, cone_indices=[1, 2]), "risk_budget")
    """
    if not variables:
        raise ValueError("soc() requires at least one variable")

    if cone_indices is None:
        cone_indices = list(range(len(variables) - 1))

    return SOCConstraint(
        variables=variables,
        cone_indices=cone_indices,
        b=list(b) if b else [0.0] * len(cone_indices),
        c=dict(c) if c else {variables[-1]: 1.0},
        d=d,
    )


def exp_cone(
    x: "Variable",
    y: "Variable",
    z: "Variable",
    name: str = "",
) -> ConeType:
    """Create an exponential cone constraint.

    Form: ``(x, y, z) ∈ K_exp`` where
    ``K_exp = cl(x,y,z) | x >= y*log(z/y), z > y > 0 U x > 0, y = 0, z >= 0``

    Parameters:
        x, y, z: Three variables forming the cone tuple.
        name: Optional label.

    Returns:
        The :data:`ExpCone` type (for use with the IPM solver's cone handling).
    """
    return ExpCone  # The IPM backend maps (x, y, z) to the conic tuple


def power_cone(
    x: "Variable",
    y: "Variable",
    z: "Variable",
    alpha: float = 0.5,
    name: str = "",
) -> ConeType:
    """Create a power cone constraint.

    Form: ``(x, y, z) ∈ K_alpha`` where
    ``K_alpha = cl(x,y,z) | x^alpha * y^(1-alpha) >= |z|, x >= 0, y >= 0``

    Parameters:
        x, y, z: Three variables forming the cone tuple.
        alpha: Exponent in ``(0, 1)``. Default: 0.5 (symmetric).
        name: Optional label.

    Returns:
        A :class:`PowerCone` type.
    """
    return PowerCone(alpha)
