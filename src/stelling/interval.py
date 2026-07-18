# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Outward-rounded interval arithmetic over IEEE-754 doubles. Zero-dep.

This is the one module where paranoia is the design: **a rounding bug here
is a false VERIFIED**, which is the project's own thesis defect. The rules:

* Every arithmetic endpoint is computed in double precision and then
  **bumped one ulp outward** (``math.nextafter``). IEEE basic operations
  (+, -, *, /) are correctly rounded (≤ 0.5 ulp), so the true real result
  always lies inside the bumped bracket. We do not attempt tight rounding;
  we buy soundness with one deliberate ulp of slack per operation.
* ``exp`` endpoints assume a **faithfully-rounded libm** (error ≤ 1 ulp)
  and are bumped one ulp outward — the *same* fidelity demotion the
  supply probe's hand brackets carried (``np.nextafter`` around
  ``np.exp``); it is recorded in every verdict that uses this transfer
  (:data:`EXP_LIBM_ASSUMPTION`). ``pow`` (strictly positive base only)
  makes the same demotion around ``math.pow`` at the monotone corners
  (:data:`POW_LIBM_ASSUMPTION`).
* Endpoints may be ``±inf`` (overflow saturates outward; half-infinite
  sets are representable). Interval multiplication uses the ``0·±inf = 0``
  endpoint convention (sound for closed real intervals, as in IEEE 1788).
* Any ``NaN`` endpoint raises: it means the domain was asked something it
  has no sound answer for, and continuing silently is exactly the failure
  mode this project exists to catch. Division by an interval containing
  zero widens to ⊤ (``[-inf, inf]``) rather than raising — sound, and it
  degrades the verdict to UNKNOWN instead of crashing the walk.

Comparisons return **three-valued** boolean intervals encoded on {0.0, 1.0}
endpoints: ``[1,1]`` definitely true, ``[0,0]`` definitely false, ``[0,1]``
unknown. A definite comparison of outward-rounded operands is sound; an
unknown one is reported as such, never guessed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

_INF = math.inf

EXP_LIBM_ASSUMPTION = (
    "exp endpoints assume a faithfully-rounded libm exp (error <= 1 ulp), "
    "bumped 1 ulp outward — the same demotion as the hand proofs' "
    "np.nextafter brackets"
)

POW_LIBM_ASSUMPTION = (
    "pow endpoints assume a faithfully-rounded libm pow (error <= 1 ulp), "
    "evaluated at the four monotone (base, exponent) corners and bumped "
    "1 ulp outward — the same fidelity demotion as exp's"
)


class IntervalError(ArithmeticError):
    """The domain met a value it has no sound treatment for (e.g. NaN)."""


def _check(x: float) -> float:
    if x != x:  # NaN
        raise IntervalError("NaN endpoint in interval arithmetic")
    return x


def _down(x: float) -> float:
    """Round a computed lower endpoint downward by one ulp (sound bracket)."""
    _check(x)
    return x if x == -_INF else math.nextafter(x, -_INF)


def _up(x: float) -> float:
    """Round a computed upper endpoint upward by one ulp (sound bracket)."""
    _check(x)
    return x if x == _INF else math.nextafter(x, _INF)


def _prod(a: float, b: float) -> float:
    """Endpoint product with the closed-interval convention 0 * ±inf = 0."""
    if (a == 0.0 and (b == _INF or b == -_INF)) or (
        b == 0.0 and (a == _INF or a == -_INF)
    ):
        return 0.0
    return a * b


@dataclass(frozen=True)
class IntervalArray:
    """A box: per-element closed intervals over a fixed shape, flat C-order."""

    shape: tuple[int, ...]
    los: tuple[float, ...]
    his: tuple[float, ...]

    def __post_init__(self) -> None:
        n = 1
        for d in self.shape:
            n *= d
        if not (len(self.los) == len(self.his) == n):
            raise IntervalError(
                f"shape {self.shape} needs {n} elements, got "
                f"{len(self.los)}/{len(self.his)}"
            )
        for lo, hi in zip(self.los, self.his):
            _check(lo)
            _check(hi)
            if lo > hi:
                raise IntervalError(f"empty interval [{lo}, {hi}]")

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def is_scalar(self) -> bool:
        return self.shape == ()


def point(value: float, shape: tuple[int, ...] = ()) -> IntervalArray:
    """A degenerate (exact) interval; no outward bump — the value itself is
    the set."""
    n = 1
    for d in shape:
        n *= d
    v = float(value)
    return IntervalArray(shape=shape, los=(v,) * n, his=(v,) * n)


def from_bounds(shape: tuple[int, ...], lo: float, hi: float) -> IntervalArray:
    n = 1
    for d in shape:
        n *= d
    return IntervalArray(shape=shape, los=(float(lo),) * n, his=(float(hi),) * n)


def from_values(shape: tuple[int, ...], values: list[float]) -> IntervalArray:
    vals = tuple(float(v) for v in values)
    return IntervalArray(shape=shape, los=vals, his=vals)


def top(shape: tuple[int, ...]) -> IntervalArray:
    """⊤: the unbounded interval — sound for any real value."""
    return from_bounds(shape, -_INF, _INF)


BOOL_TRUE = (1.0, 1.0)
BOOL_FALSE = (0.0, 0.0)
BOOL_UNKNOWN = (0.0, 1.0)


# -- elementwise plumbing -----------------------------------------------------


def _broadcast_shape(sa: tuple[int, ...], sb: tuple[int, ...]) -> tuple[int, ...]:
    """numpy-style broadcast of two shapes: trailing axes aligned, size-1
    axes replicate, missing leading axes replicate. Incompatible shapes
    raise :class:`IntervalError` (the transfer declines; never a crash)."""
    out: list[int] = []
    for da, db in itertools.zip_longest(reversed(sa), reversed(sb), fillvalue=1):
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise IntervalError(
                f"shapes {sa} and {sb} do not broadcast "
                f"(axis sizes {da} vs {db})"
            )
    return tuple(reversed(out))


def _bcast_elements(x: IntervalArray, out_shape: tuple[int, ...]):
    """Element (lo, hi) pairs of ``x`` replicated to ``out_shape`` (which
    must be a broadcast target of ``x.shape``), flat C-order."""
    n = 1
    for d in out_shape:
        n *= d
    if n == 0:
        return []
    k, r = len(x.shape), len(out_shape)
    elems = []
    for coord in _coords(out_shape):
        src = tuple(
            0 if x.shape[j] == 1 else coord[r - k + j] for j in range(k)
        )
        i = _flat_index(src, x.shape)
        elems.append((x.los[i], x.his[i]))
    return elems


def _pair_elements(a: IntervalArray, b: IntervalArray):
    """Zip two operands elementwise: equal shapes, the scalar-vs-any fast
    path (jaxprs carry scalar literals as rank-0 operands of elementwise
    eqns), and general numpy-style shape broadcasting (size-1 axes and
    missing leading axes replicate). Incompatible shapes raise
    :class:`IntervalError`, which the propagator converts to a noted
    ⊤-decline."""
    if a.shape == b.shape:
        return a.shape, list(zip(a.los, a.his)), list(zip(b.los, b.his))
    if a.is_scalar():
        n = b.size
        return b.shape, [(a.los[0], a.his[0])] * n, list(zip(b.los, b.his))
    if b.is_scalar():
        n = a.size
        return a.shape, list(zip(a.los, a.his)), [(b.los[0], b.his[0])] * n
    out_shape = _broadcast_shape(a.shape, b.shape)
    return out_shape, _bcast_elements(a, out_shape), _bcast_elements(b, out_shape)


def _binary(a: IntervalArray, b: IntervalArray, f) -> IntervalArray:
    shape, xs, ys = _pair_elements(a, b)
    los, his = [], []
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        lo, hi = f(alo, ahi, blo, bhi)
        los.append(lo)
        his.append(hi)
    return IntervalArray(shape=shape, los=tuple(los), his=tuple(his))


# -- arithmetic ---------------------------------------------------------------


def add(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(a, b, lambda alo, ahi, blo, bhi: (_down(alo + blo), _up(ahi + bhi)))


def sub(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(a, b, lambda alo, ahi, blo, bhi: (_down(alo - bhi), _up(ahi - blo)))


def neg(a: IntervalArray) -> IntervalArray:
    return IntervalArray(
        shape=a.shape,
        los=tuple(-h for h in a.his),
        his=tuple(-l for l in a.los),
    )


def abs_(a: IntervalArray) -> IntervalArray:
    """Piecewise-exact |·|: negation and max of doubles are exact, so the
    endpoints are the true image endpoints — no rounding bump."""
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        if lo >= 0.0:
            los.append(lo)
            his.append(hi)
        elif hi <= 0.0:
            los.append(-hi)
            his.append(-lo)
        else:  # straddles zero: image is [0, max(|lo|, hi)]
            los.append(0.0)
            his.append(max(-lo, hi))
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def mul(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    def f(alo, ahi, blo, bhi):
        products = (_prod(alo, blo), _prod(alo, bhi), _prod(ahi, blo), _prod(ahi, bhi))
        return _down(min(products)), _up(max(products))

    return _binary(a, b, f)


def div(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    def f(alo, ahi, blo, bhi):
        if blo <= 0.0 <= bhi:
            # denominator may vanish: ⊤ is the only sound closed-interval
            # answer here; the verdict degrades to UNKNOWN, never crashes.
            return -_INF, _INF
        quotients = []
        for x in (alo, ahi):
            for y in (blo, bhi):
                if (x == _INF or x == -_INF) and (y == _INF or y == -_INF):
                    # inf/inf is indeterminate; widen fully outward.
                    return -_INF, _INF
                quotients.append(x / y)  # y is nonzero; finite/±inf -> ±0.0
        return _down(min(quotients)), _up(max(quotients))

    return _binary(a, b, f)


def maximum(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    # max is monotone in both args: no rounding, endpoints are real values
    return _binary(a, b, lambda alo, ahi, blo, bhi: (max(alo, blo), max(ahi, bhi)))


def minimum(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(a, b, lambda alo, ahi, blo, bhi: (min(alo, blo), min(ahi, bhi)))


def join(cases: list[IntervalArray]) -> IntervalArray:
    """Interval hull (union) of same-shape boxes — the sound over-approximation
    of a branch whose taken case is not determined."""
    shape = cases[0].shape
    if any(c.shape != shape for c in cases):
        # a larger case would be silently truncated to case 0's element
        # count — refuse rather than mis-join (audit finding 7)
        raise IntervalError(f"join over mismatched shapes {[c.shape for c in cases]}")
    los = tuple(min(c.los[i] for c in cases) for i in range(cases[0].size))
    his = tuple(max(c.his[i] for c in cases) for i in range(cases[0].size))
    return IntervalArray(shape=shape, los=los, his=his)


def select_n(which: IntervalArray, cases: list[IntervalArray]) -> IntervalArray:
    """`select_n(which, *cases)`: elementwise pick of ``cases[which]``.

    ``which`` is a predicate/index interval on {0, 1, …}. Where it is
    **definite** (a single integer at that element) the exact case is
    taken; where it **straddles** (the branch is undetermined) the possible
    cases are joined — sound, and the source of branch imprecision that a
    solver would resolve. An infinite (⊤) selector element joins every
    case rather than crashing on the int conversion (audit finding 5 —
    reachable from any trace whose predicate involves an unregistered
    primitive).

    Out-of-range selectors **clamp** — jax's measured ``lax.select_n``
    semantics (0.11, eager and jit agree: index −1 → case 0), which is NOT
    ``cond``'s convention (measured: index −1 → last branch). Second
    audit, finding 3: the earlier last-case fallback here selected the
    wrong end of that asymmetry.

    Shapes: all cases must agree; ``which`` is either case-shaped
    (elementwise selection) or a **scalar** broadcast across the cases'
    elements (jax permits exactly these two forms). Anything else raises
    :class:`IntervalError` — a decline the propagator notes, not a
    crash."""
    if not cases:
        raise IntervalError("select_n with no cases")
    if any(c.shape != cases[0].shape for c in cases[1:]):
        raise IntervalError(
            f"select_n cases disagree on shape: {[c.shape for c in cases]}"
        )
    scalar_which = which.is_scalar() and not cases[0].is_scalar()
    if which.shape != cases[0].shape and not scalar_which:
        raise IntervalError(
            f"select_n case shapes {[c.shape for c in cases]} != which "
            f"{which.shape} (equal shapes or a scalar selector are the only "
            f"supported forms)"
        )
    n = cases[0].size
    last = len(cases) - 1
    los, his = [], []
    for i in range(n):
        wi = 0 if scalar_which else i
        w_lo, w_hi = which.los[wi], which.his[wi]
        if w_lo == -_INF or w_hi == _INF:
            picks = cases  # ⊤ selector: any case possible
        else:
            lo_idx, hi_idx = int(math.floor(w_lo)), int(math.floor(w_hi))
            possible = set(range(max(0, lo_idx), min(last, hi_idx) + 1))
            if lo_idx < 0:
                possible.add(0)  # below-range mass clamps to the first case
            if hi_idx > last:
                possible.add(last)  # above-range mass clamps to the last case
            picks = [cases[k] for k in sorted(possible)]
        los.append(min(c.los[i] for c in picks))
        his.append(max(c.his[i] for c in picks))
    return IntervalArray(shape=cases[0].shape, los=tuple(los), his=tuple(his))


def exp(a: IntervalArray) -> IntervalArray:
    def e(x: float, up_side: bool) -> float:
        if x == -_INF:
            return 0.0
        if x == _INF:
            return _INF
        try:
            v = math.exp(x)
        except OverflowError:
            return _INF if up_side else math.nextafter(_INF, 0.0)
        return _up(v) if up_side else max(0.0, _down(v))

    return IntervalArray(
        shape=a.shape,
        los=tuple(e(l, False) for l in a.los),
        his=tuple(e(h, True) for h in a.his),
    )


def pow_(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """``base ** exponent`` for a **strictly positive** base interval.

    For base > 0, ``x**y = exp(y·ln x)`` is monotone in ``x`` for every
    fixed ``y`` and monotone in ``y`` for every fixed ``x``, so the
    extremum over the (base, exponent) box lies at one of the four
    corners. Corners are evaluated with ``math.pow`` under the
    faithfully-rounded-libm assumption (:data:`POW_LIBM_ASSUMPTION`) and
    bumped one ulp outward; ``x > 0`` also gives ``x**y > 0``, so lower
    endpoints are floored at 0.

    A base interval reaching 0 or below has no sound rule here
    (``0**negative`` diverges, negative bases alternate sign with the
    exponent's parity): :class:`IntervalError` — the propagator turns it
    into a noted ⊤-decline, never a crash.
    """
    if any(lo <= 0.0 for lo in a.los):
        raise IntervalError(
            f"pow has a sound corner rule only for strictly positive bases; "
            f"base lower bound {min(a.los)} <= 0"
        )

    def f(alo, ahi, blo, bhi):
        lo_bounds, hi_bounds = [], []
        for x in (alo, ahi):
            for y in (blo, bhi):
                try:
                    v = math.pow(x, y)
                except OverflowError:
                    # the true corner value exceeds the double range:
                    # finite but > maxfloat — saturate outward, keeping
                    # maxfloat as a sound finite lower witness (the exp
                    # overflow treatment).
                    lo_bounds.append(math.nextafter(_INF, 0.0))
                    hi_bounds.append(_INF)
                    continue
                except ValueError as e:  # unreachable for x > 0; degrade anyway
                    raise IntervalError(f"math.pow({x}, {y}): {e}") from None
                # v == inf without OverflowError only happens at a corner
                # with an infinite operand endpoint (IEEE pow limits, e.g.
                # pow(inf, 2), pow(0.5, -inf)): the corner's true value is
                # inf itself, not a rounded finite — keep it exact.
                # (CPython math.pow raises OverflowError for finite
                # operands that overflow; it never returns inf silently.)
                lo_bounds.append(v if v == _INF else max(0.0, _down(v)))
                hi_bounds.append(v if v == _INF else _up(v))
        return min(lo_bounds), max(hi_bounds)

    return _binary(a, b, f)


# -- comparisons (three-valued) ----------------------------------------------


def _compare(a: IntervalArray, b: IntervalArray, definite_true, definite_false):
    def f(alo, ahi, blo, bhi):
        if definite_true(alo, ahi, blo, bhi):
            return BOOL_TRUE
        if definite_false(alo, ahi, blo, bhi):
            return BOOL_FALSE
        return BOOL_UNKNOWN

    return _binary(a, b, f)


def lt(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi < blo,
        definite_false=lambda alo, ahi, blo, bhi: alo >= bhi,
    )


def gt(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return lt(b, a)


def le(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi <= blo,
        definite_false=lambda alo, ahi, blo, bhi: alo > bhi,
    )


def ge(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return le(b, a)


def eq(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Three-valued equality. Definitely true **only** when both operands
    are the same single point (a point interval guarantees the true value
    exactly, so the two true values coincide); definitely false when the
    intervals are disjoint (supersets that never meet contain no equal
    pair); everything else — including identical non-point intervals — is
    unknown, never guessed."""
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: alo == ahi == blo == bhi,
        definite_false=lambda alo, ahi, blo, bhi: ahi < blo or bhi < alo,
    )


def ne(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Three-valued inequality: the negation of :func:`eq`'s logic —
    definitely true where eq is definitely false (disjoint), definitely
    false where eq is definitely true (same single point), else unknown."""
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi < blo or bhi < alo,
        definite_false=lambda alo, ahi, blo, bhi: alo == ahi == blo == bhi,
    )


# -- three-valued logic on {0,1}-encoded bool intervals -----------------------


def _bool3(lo: float, hi: float) -> tuple[float, float]:
    """Canonicalize one {0,1}-encoded three-valued element. Anything that
    is not exactly the definite-true or definite-false encoding (including
    a ⊤ interval flowing in from an unregistered producer) reads as
    unknown — sound whenever the true values are booleans, which the
    transfers' bool-dtype guard establishes."""
    if lo == 1.0 and hi == 1.0:
        return BOOL_TRUE
    if lo == 0.0 and hi == 0.0:
        return BOOL_FALSE
    return BOOL_UNKNOWN


def logical_and(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Kleene AND: false ∧ anything = false; true ∧ true = true; else
    unknown. On the {0,1} encoding that is endpoint-wise min of the
    canonicalized operands."""

    def f(alo, ahi, blo, bhi):
        (alo, ahi), (blo, bhi) = _bool3(alo, ahi), _bool3(blo, bhi)
        return min(alo, blo), min(ahi, bhi)

    return _binary(a, b, f)


def logical_or(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Kleene OR: true ∨ anything = true; false ∨ false = false; else
    unknown — endpoint-wise max of the canonicalized operands."""

    def f(alo, ahi, blo, bhi):
        (alo, ahi), (blo, bhi) = _bool3(alo, ahi), _bool3(blo, bhi)
        return max(alo, blo), max(ahi, bhi)

    return _binary(a, b, f)


def reduce_or(a: IntervalArray, axes: tuple[int, ...]) -> IntervalArray:
    """Three-valued OR-fold over ``axes``: output shape is the input shape
    with those axes removed. The fold identity is definite-false (an OR
    over an empty reduction range is false), so empty-range axes reduce to
    ``[0, 0]`` exactly as jax's ``reduce_or`` does."""
    ax = set(axes)
    if any(not (isinstance(d, int) and 0 <= d < len(a.shape)) for d in ax):
        raise IntervalError(
            f"reduce_or axes {axes} out of range for shape {a.shape}"
        )
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in ax)
    out_n = 1
    for d in out_shape:
        out_n *= d
    los = [0.0] * out_n  # OR identity: definitely false
    his = [0.0] * out_n
    if a.size:
        for coord in _coords(a.shape):
            i = _flat_index(coord, a.shape)
            j = _flat_index(
                tuple(c for k, c in enumerate(coord) if k not in ax), out_shape
            )
            lo, hi = _bool3(a.los[i], a.his[i])
            los[j] = max(los[j], lo)
            his[j] = max(his[j], hi)
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


# -- structural ops (exact: no arithmetic, no bump) ---------------------------


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, acc = [], 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


def _flat_index(coord: tuple[int, ...], shape: tuple[int, ...]) -> int:
    return sum(c * s for c, s in zip(coord, _strides(shape)))


def _coords(shape: tuple[int, ...]):
    if shape == ():
        yield ()
        return
    if 0 in shape:  # a zero-size array has no elements — no coordinates
        return  # (audit-gate finding 2: the phantom first coordinate
        # produced an IndexError that bypassed the decline channel)
    idx = [0] * len(shape)
    while True:
        yield tuple(idx)
        for axis in range(len(shape) - 1, -1, -1):
            idx[axis] += 1
            if idx[axis] < shape[axis]:
                break
            idx[axis] = 0
        else:
            return


def slice_(
    a: IntervalArray,
    start_indices: tuple[int, ...],
    limit_indices: tuple[int, ...],
    strides: tuple[int, ...] | None,
) -> IntervalArray:
    steps = strides or (1,) * len(a.shape)
    out_shape = tuple(
        -(-(hi - lo) // st) for lo, hi, st in zip(start_indices, limit_indices, steps)
    )
    los, his = [], []
    for coord in _coords(out_shape):
        src = tuple(lo + c * st for c, lo, st in zip(coord, start_indices, steps))
        i = _flat_index(src, a.shape)
        los.append(a.los[i])
        his.append(a.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def reshape(a: IntervalArray, new_sizes: tuple[int, ...]) -> IntervalArray:
    """Data-preserving shape change: element storage is already flat
    C-order, so a C-order reshape is the identity on the element tuples.
    (Reshapes with a ``dimensions`` permutation are not this function —
    the transfer declines them before calling here.)"""
    n = 1
    for d in new_sizes:
        n *= d
    if n != a.size:
        raise IntervalError(
            f"reshape {a.shape} -> {tuple(new_sizes)} changes element count"
        )
    return IntervalArray(shape=tuple(new_sizes), los=a.los, his=a.his)


def squeeze(a: IntervalArray, dimensions: tuple[int, ...]) -> IntervalArray:
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in set(dimensions))
    return IntervalArray(shape=out_shape, los=a.los, his=a.his)


def broadcast_in_dim(
    a: IntervalArray,
    out_shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
) -> IntervalArray:
    los, his = [], []
    for coord in _coords(out_shape):
        src = tuple(
            coord[out_axis] if a.shape[in_axis] != 1 else 0
            for in_axis, out_axis in enumerate(broadcast_dimensions)
        )
        i = _flat_index(src, a.shape)
        los.append(a.los[i])
        his.append(a.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def concatenate(parts: list[IntervalArray], dimension: int) -> IntervalArray:
    base = parts[0].shape
    out_shape = tuple(
        sum(p.shape[dimension] for p in parts) if ax == dimension else d
        for ax, d in enumerate(base)
    )
    los, his = [], []
    for coord in _coords(out_shape):
        offset = coord[dimension]
        for p in parts:
            if offset < p.shape[dimension]:
                src = tuple(
                    offset if ax == dimension else c for ax, c in enumerate(coord)
                )
                i = _flat_index(src, p.shape)
                los.append(p.los[i])
                his.append(p.his[i])
                break
            offset -= p.shape[dimension]
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))
