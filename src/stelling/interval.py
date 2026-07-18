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
  (:data:`EXP_LIBM_ASSUMPTION`).
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

import math
from dataclasses import dataclass

_INF = math.inf

EXP_LIBM_ASSUMPTION = (
    "exp endpoints assume a faithfully-rounded libm exp (error <= 1 ulp), "
    "bumped 1 ulp outward — the same demotion as the hand proofs' "
    "np.nextafter brackets"
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


def _pair_elements(a: IntervalArray, b: IntervalArray):
    """Zip two operands elementwise, allowing a scalar against any shape
    (jaxprs carry scalar literals as rank-0 operands of elementwise eqns)."""
    if a.shape == b.shape:
        return a.shape, list(zip(a.los, a.his)), list(zip(b.los, b.his))
    if a.is_scalar():
        n = b.size
        return b.shape, [(a.los[0], a.his[0])] * n, list(zip(b.los, b.his))
    if b.is_scalar():
        n = a.size
        return a.shape, list(zip(a.los, a.his)), [(b.los[0], b.his[0])] * n
    raise IntervalError(f"shape mismatch {a.shape} vs {b.shape}")


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
    wrong end of that asymmetry."""
    if any(c.shape != which.shape for c in cases):
        raise IntervalError(
            f"select_n case shapes {[c.shape for c in cases]} != which {which.shape}"
        )
    n = which.size
    last = len(cases) - 1
    los, his = [], []
    for i in range(n):
        w_lo, w_hi = which.los[i], which.his[i]
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
    return IntervalArray(shape=which.shape, los=tuple(los), his=tuple(his))


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
