"""sos.expand — MIP reformulations for SOS1/SOS2 constraints.

Each ``_expand_sos2_*`` function implements one formulation from the PWL/SOS2
literature (Vielma 2015 survey; PiecewiseLinearOpt.jl), adapted to obp's
"free bounded variables" SOS2 semantics (not PWL-breakpoint interpolation
weights): each x_i keeps its own [lb_i, ub_i] range and formulations only
share a window-activity indicator, they don't couple x_i's independently
via a shared convex-combination simplex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..expression import Expression

if TYPE_CHECKING:
    from ..model import Variable
    from . import SOSConstraint


def _sum_expr(vars_: list) -> Expression:
    """Sum a list of Variables into an Expression (avoids Var+Var quadratic-guard)."""
    expr = Expression.constant(0.0)
    for v in vars_:
        expr = expr + Expression.from_variable(v)
    return expr


def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def _log2_ceil(n: int) -> int:
    import math
    return max(1, math.ceil(math.log2(n))) if n > 1 else 1


def _expand_sos1(
    sos, ordered_vars: list["Variable"], next_index: int, prefix: str,
) -> tuple[list["Variable"], list, int]:
    """SOS1: per-variable binary indicators z_i, sum(z_i) <= 1."""
    from ..model import BinVar, Constraint

    extra_vars: list["Variable"] = []
    extra_constraints: list = []
    z_vars: list[BinVar] = []

    for i, v in enumerate(ordered_vars):
        z = BinVar(next_index, f"{prefix}_z{i}")
        next_index += 1
        z_vars.append(z)
        extra_vars.append(z)
        extra_constraints.append(Constraint(v - v.ub * z, "<=", 0.0, f"{z.name}_ub"))
        extra_constraints.append(Constraint(v - v.lb * z, ">=", 0.0, f"{z.name}_lb"))

    extra_constraints.append(Constraint(_sum_expr(z_vars), "<=", 1.0, f"{prefix}_card"))
    return extra_vars, extra_constraints, next_index


def _expand_sos2_cc(
    sos, ordered_vars: list["Variable"], next_index: int, prefix: str,
) -> tuple[list["Variable"], list, int]:
    """SOS2 'cc': convex-combination — x_i bounded directly by adjacent window
    indicators (no separate per-variable z layer). Fewer binaries/constraints
    than 'mip'.
    """
    from ..model import BinVar, Constraint

    extra_vars: list["Variable"] = []
    extra_constraints: list = []
    k = len(ordered_vars)

    y_vars: list[BinVar] = []
    for j in range(k - 1):
        y = BinVar(next_index, f"{prefix}_y{j}")
        next_index += 1
        y_vars.append(y)
        extra_vars.append(y)

    extra_constraints.append(Constraint(_sum_expr(y_vars), "<=", 1.0, f"{prefix}_window_card"))

    windows_touching = [
        [j for j in (i - 1, i) if 0 <= j <= k - 2] for i in range(k)
    ]
    for i, v in enumerate(ordered_vars):
        touching = [y_vars[j] for j in windows_touching[i]]
        w_sum = _sum_expr(touching)
        extra_constraints.append(Constraint(v - v.ub * w_sum, "<=", 0.0, f"{prefix}_x{i}_ub"))
        extra_constraints.append(Constraint(v - v.lb * w_sum, ">=", 0.0, f"{prefix}_x{i}_lb"))

    return extra_vars, extra_constraints, next_index


def _expand_sos2_dlog(
    sos, ordered_vars: list["Variable"], next_index: int, prefix: str,
) -> tuple[list["Variable"], list, int]:
    """SOS2 'dlog': disaggregated logarithmic (Vielma & Nemhauser 2011).

    Disaggregated shares (two nonnegative continuous variables per segment,
    representing the portion of its two endpoint variables attributed to
    that segment being active), but the per-segment activity is selected
    via ceil(log2(k-1)) binaries instead
    of k-1 unary indicators (Vielma & Nemhauser 2011's DisaggLogarithmic
    formulation). Since each segment has exactly one Gray-code word (no
    ambiguity like a vertex touching two segments), the per-bit constraint
    is a direct partition of segments by that bit's value.
    """
    from ..model import BinVar, Constraint, Variable

    extra_vars: list["Variable"] = []
    extra_constraints: list = []
    k = len(ordered_vars)
    n_segments = k - 1
    m = _log2_ceil(n_segments)

    # per-segment activity (relaxed indicator in [0,1]; pinned to 0/1 by the
    # log-bit constraints at any feasible point of the w binaries)
    a_vars: list[Variable] = []
    for j in range(n_segments):
        a = Variable(next_index, f"{prefix}_a{j}", lb=0.0, ub=1.0)
        next_index += 1
        a_vars.append(a)
        extra_vars.append(a)

    extra_constraints.append(Constraint(_sum_expr(a_vars), "==", 1.0, f"{prefix}_segment_card"))

    w_vars: list[BinVar] = []
    for l in range(m):
        w = BinVar(next_index, f"{prefix}_w{l}")
        next_index += 1
        w_vars.append(w)
        extra_vars.append(w)

    codes = _reflected_gray_codes(m)[:n_segments]

    def h(seg: int, l: int) -> int:
        return codes[seg][l]

    for l in range(m):
        one_side = [a_vars[j] for j in range(n_segments) if h(j, l) == 1]
        zero_side = [a_vars[j] for j in range(n_segments) if h(j, l) == 0]
        if one_side:
            extra_constraints.append(
                Constraint(
                    _sum_expr(one_side) - Expression.from_variable(w_vars[l]),
                    "<=", 0.0, f"{prefix}_bit{l}_one",
                )
            )
        if zero_side:
            extra_constraints.append(
                Constraint(
                    _sum_expr(zero_side) + Expression.from_variable(w_vars[l]),
                    "<=", 1.0, f"{prefix}_bit{l}_zero",
                )
            )

    # shares[j] = (share_lo_j, share_hi_j) for segment j touching (ordered_vars[j], ordered_vars[j+1])
    shares: list[tuple[Variable, Variable]] = []
    for j in range(n_segments):
        lb_lo, ub_lo = ordered_vars[j].lb, ordered_vars[j].ub
        lb_hi, ub_hi = ordered_vars[j + 1].lb, ordered_vars[j + 1].ub
        s_lo = Variable(next_index, f"{prefix}_s{j}lo", lb=min(0.0, lb_lo), ub=max(0.0, ub_lo))
        next_index += 1
        s_hi = Variable(next_index, f"{prefix}_s{j}hi", lb=min(0.0, lb_hi), ub=max(0.0, ub_hi))
        next_index += 1
        extra_vars.append(s_lo)
        extra_vars.append(s_hi)
        shares.append((s_lo, s_hi))

        # bound each share by its segment's activity
        extra_constraints.append(
            Constraint(s_lo - ub_lo * a_vars[j], "<=", 0.0, f"{prefix}_s{j}lo_ub")
        )
        extra_constraints.append(
            Constraint(s_lo - lb_lo * a_vars[j], ">=", 0.0, f"{prefix}_s{j}lo_lb")
        )
        extra_constraints.append(
            Constraint(s_hi - ub_hi * a_vars[j], "<=", 0.0, f"{prefix}_s{j}hi_ub")
        )
        extra_constraints.append(
            Constraint(s_hi - lb_hi * a_vars[j], ">=", 0.0, f"{prefix}_s{j}hi_lb")
        )

    # x_i = sum of shares assigned to it from touching segments
    for i, v in enumerate(ordered_vars):
        touching_shares = []
        if i - 1 >= 0:
            touching_shares.append(shares[i - 1][1])  # hi-share of segment i-1
        if i <= k - 2:
            touching_shares.append(shares[i][0])  # lo-share of segment i
        body = Expression.from_variable(v) - _sum_expr(touching_shares)
        extra_constraints.append(Constraint(body, "==", 0.0, f"{prefix}_x{i}_link"))

    return extra_vars, extra_constraints, next_index


def _reflected_gray_codes(k: int) -> list[list[int]]:
    """Standard reflected Gray code sequence of length 2^k, each a k-bit list.

    Matches PiecewiseLinearOpt.jl's ``reflected_gray_codes``: codes'[k-1]
    followed by 0, then codes'[k-1] reversed followed by 1.
    """
    if k == 0:
        return []
    if k == 1:
        return [[0], [1]]
    prev = _reflected_gray_codes(k - 1)
    return [code + [0] for code in prev] + [code + [1] for code in reversed(prev)]


def _expand_sos2_bitcode(
    ordered_vars: list["Variable"], next_index: int, prefix: str, codes: list[list[int]],
) -> tuple[list["Variable"], list, int]:
    """Shared bit-partition core for SOS2 'log': logarithmic window
    encoding using an arbitrary per-segment binary code.

    Note: this construction relies on consecutive segments' codes
    differing in exactly one bit (the reflected-Gray-code adjacency
    property) so that every vertex is fully constrained by at least one
    bit. Codes without that property (e.g. plain zigzag codes) can leave
    an interior vertex unconstrained by any bit, silently breaking
    correctness — do not reuse this helper with non-Gray codes without
    re-deriving and re-verifying.

    Segments j=0..k-2 get code words of length m (m = len(codes[0])).
    Introduces m binaries w_0..w_{m-1} (instead of k-1 unary window
    indicators) and, for each vertex i, a continuous z_i in [0,1] that
    plays the same role as 'cc'-formulation's y_{i-1}+y_i "is vertex i
    inside the active window" quantity, but derived from the code bits:
    z_i is bounded above by 1 on every bit where i's touching segment(s)
    agree with the branch selected by w_l, and forced toward 0 on every
    bit where they disagree. x_i is then bounded by ub_i*z_i / lb_i*z_i
    exactly as in 'cc'.
    """
    from ..model import BinVar, Constraint, Variable

    extra_vars: list["Variable"] = []
    extra_constraints: list = []
    k = len(ordered_vars)
    n_segments = k - 1
    m = len(codes[0])

    z_vars: list[Variable] = []
    for i in range(k):
        z = Variable(next_index, f"{prefix}_z{i}", lb=0.0, ub=1.0)
        next_index += 1
        z_vars.append(z)
        extra_vars.append(z)

    for i, v in enumerate(ordered_vars):
        z = z_vars[i]
        extra_constraints.append(Constraint(v - v.ub * z, "<=", 0.0, f"{prefix}_x{i}_ub"))
        extra_constraints.append(Constraint(v - v.lb * z, ">=", 0.0, f"{prefix}_x{i}_lb"))

    w_vars: list[BinVar] = []
    for l in range(m):
        w = BinVar(next_index, f"{prefix}_w{l}")
        next_index += 1
        w_vars.append(w)
        extra_vars.append(w)

    def h(seg: int, l: int) -> int:
        return codes[seg][l]

    def touching_segments(i: int) -> list[int]:
        return [s for s in (i - 1, i) if 0 <= s <= n_segments - 1]

    # z_i <= 1 on the branch(es) consistent with i's touching segment(s);
    # z_i <= 0 is forced whenever no touching segment agrees with a branch.
    for i in range(k):
        segs = touching_segments(i)
        for l in range(m):
            bits = {h(s, l) for s in segs}
            if bits == {0}:
                # only consistent with w_l == 0: z_i <= 1 - w_l
                extra_constraints.append(
                    Constraint(
                        Expression.from_variable(z_vars[i])
                        + Expression.from_variable(w_vars[l]),
                        "<=", 1.0, f"{prefix}_z{i}_bit{l}",
                    )
                )
            elif bits == {1}:
                # only consistent with w_l == 1: z_i <= w_l
                extra_constraints.append(
                    Constraint(
                        Expression.from_variable(z_vars[i])
                        - Expression.from_variable(w_vars[l]),
                        "<=", 0.0, f"{prefix}_z{i}_bit{l}",
                    )
                )
            # bits == {0, 1}: i's two touching segments disagree on this
            # bit, so neither branch of w_l can rule i out — no constraint.

    return extra_vars, extra_constraints, next_index


def _expand_sos2_log(
    sos, ordered_vars: list["Variable"], next_index: int, prefix: str,
) -> tuple[list["Variable"], list, int]:
    """SOS2 'log': logarithmic window encoding (Vielma & Nemhauser 2011),
    using reflected Gray codes. See :func:`_expand_sos2_bitcode`."""
    k = len(ordered_vars)
    n_segments = k - 1
    m = _log2_ceil(n_segments)
    codes = _reflected_gray_codes(m)[:n_segments]
    return _expand_sos2_bitcode(ordered_vars, next_index, prefix, codes)


def expand_sos_constraints(
    sos_constraints: list["SOSConstraint"],
    variables: list["Variable"],
) -> tuple[list["Variable"], list]:
    """Reformulate SOS1/SOS2 constraints as MIP constraints.

    Dispatches per-constraint to the requested formulation (SOSConstraint.method):
    SOS1 always uses the direct per-variable indicator encoding; SOS2 supports
    "cc" (default), "dlog", and "log" — matching (a subset of) the
    formulations in PiecewiseLinearOpt.jl.

    Returns:
        (extra_variables, extra_constraints) to append to the problem's
        variable/constraint lists before building the solver matrices.
    """
    extra_vars: list["Variable"] = []
    extra_constraints: list = []
    next_index = len(variables)

    for sos in sos_constraints:
        order = list(range(len(sos.variables)))
        if sos.weights is not None:
            order = sorted(order, key=lambda i: sos.weights[i])
        ordered_vars = [sos.variables[i] for i in order]
        prefix = f"_sos{sos.name or id(sos)}"

        if sos.type == 1:
            v, c, next_index = _expand_sos1(sos, ordered_vars, next_index, prefix)
        elif sos.method == "cc":
            v, c, next_index = _expand_sos2_cc(sos, ordered_vars, next_index, prefix)
        elif sos.method == "dlog":
            v, c, next_index = _expand_sos2_dlog(sos, ordered_vars, next_index, prefix)
        elif sos.method == "log":
            v, c, next_index = _expand_sos2_log(sos, ordered_vars, next_index, prefix)
        else:
            raise ValueError(f"Unknown SOS2 method: {sos.method!r}")

        extra_vars.extend(v)
        extra_constraints.extend(c)

    return extra_vars, extra_constraints
