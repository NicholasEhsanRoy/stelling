# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The exact containment sweep behind the S10 entry in ``SOUNDNESS.md``.

**Why this is a shipped module and not a paragraph.** The S10 entry is the
evidence for an ``UNSOUND`` fix on a released surface, and its sweep table
was for one release an out-of-tree measurement: the counts appeared in the
log and `grep` for them found nothing in ``tests/`` or ``src/``. The
sibling figure three paragraphs below it had just been converted to an
exact ``assert`` against ``BOUNDARY_DIV_SWEEP_QUOTIENTS`` for precisely
that reason (audit 0.2.0 B5-4), which left the larger block next to it as
the biggest unverifiable number in the file — sitting beside the paragraph
explaining why such blocks are a problem. This module closes that gap the
way the sibling did: the counts SOUNDNESS.md quotes are module constants,
and :mod:`tests.test_ieee_zero_divisor_and_mul_exact` asserts the sweep
reproduces them exactly, so drift in either direction reddens the suite
instead of leaving prose to be believed (the B5 FOLLOW-UP audit's
sweep-table finding; see the note on finding IDs in SOUNDNESS.md).

**What it checks.** For every ordered endpoint pair drawn from an
adversarial pool it builds a box, pairs every box with every box, drives
the kernel, and asserts the returned box contains every value the
arithmetic can produce at points of the operand boxes — in EXACT rational
arithmetic, so no rounding can mask a miss.

**The yardstick differs by format, and that is not sloppiness.** For
binary64 the value the program has IS ``fl(x op y)``, and ``ieee_mul`` /
``ieee_div`` build their corner hull from exactly that native result with
no outward rounding — so those rows are checked against the native float,
and checking the REAL quotient there reported 24,400 misses that were not
misses on the first run of this file. The narrow rows are the opposite
case and are covered below.

**Why exact rationals suffice for the narrow formats.** The ``_fmt``
kernels compute their corner hull in binary64 and the propagator then
rounds the box OUTWARD onto the target's grid (``_ieee_round_box``). If
the outward-rounded box contains the exact REAL result ``p``, it also
contains the format's own ``RN_fmt(p)``: round-to-nearest lands on one of
``p``'s two neighbours on the format grid, and both lie inside a box whose
endpoints were rounded away from ``p``. So checking the exact real value
is SUFFICIENT, and it is stronger than checking a float emulation of the
format — it assumes nothing about how the format rounds. That is why there
is no IEEE emulator in this file: an emulator underneath a soundness log
is a second thing that can be wrong.

**Both zeros are kept apart on the input side.** ``-0.0 == +0.0``, so a
sweep that deduplicates its sample points with ``==`` sees one zero where
IEEE has two — which is exactly what the sweep this one replaced did not
do. Boxes here are built from POOL INDICES, never from value comparisons,
so ``[-0.0, +0.0]`` is a box and ``[+0.0, -0.0]`` is not.

**What it cannot do**, said plainly: it samples a finite subset of each
box (the endpoints and, when the box contains it, zero) rather than every
float, so it is a containment BATTERY, not a proof; it drives the kernels
and the ``_ieee_round_box`` composition directly, not the traced pipeline;
and it says nothing about transfers it does not name.

Runnable standalone to reproduce the table::

    PYTHONPATH=src python tests/ieee_containment_sweep.py
"""

from __future__ import annotations

import math
from fractions import Fraction

from stelling import interval as iv
from stelling.propagate import (
    _FLOAT_FORMATS,
    _ieee_format_min_normal,
    _ieee_round_box,
)

INF = math.inf
FMAX = 1.7976931348623157e308

# The adversarial pool, 28 values, in IEEE totalOrder (so -0.0 precedes
# +0.0 and box construction never has to compare two zeros). Every
# magnitude here is a boundary something in the pipeline case-splits on:
# the two zeros, the binary64 subnormal band, the subnormal and normal
# bands of each NARROWER format the propagator supports, an ulp either
# side of 1.0, the overflow edge, and both infinities.
_MAGNITUDES = (
    0.0,
    5e-324,                     # binary64 min subnormal
    2.2250738585072014e-308,    # binary64 min normal
    1e-45,                      # subnormal in float32, normal in binary64
    1.1754943508222875e-41,     # subnormal in float32, normal in binary64
    1.1754943508222875e-38,     # float32 min normal
    6e-08,                      # subnormal in float16
    6.103515625e-05,            # float16 min normal
    0.9999999999999999,         # nextafter(1.0, -inf)
    1.0,
    1.0000000000000002,         # nextafter(1.0, +inf)
    1e300,
    FMAX,
    INF,
)
# ASCENDING, and asserted so: boxes are built from POOL index pairs, so a
# pool out of order silently mints `lo > hi` boxes that the IntervalArray
# constructor then rejects as empty — 1,620 of them, counted as RAISED and
# read as a kernel defect, on the first run of this file.
assert all(
    _MAGNITUDES[i] < _MAGNITUDES[i + 1] for i in range(len(_MAGNITUDES) - 1)
), _MAGNITUDES

POOL = tuple(
    [-m for m in reversed(_MAGNITUDES)] + [0.0] + list(_MAGNITUDES[1:])
)
# -0.0 is the last of the negated half; python's unary minus on 0.0 gives
# -0.0, so POOL[13] is -0.0 and POOL[14] is +0.0 by construction.
assert len(POOL) == 2 * len(_MAGNITUDES), len(POOL)
assert math.copysign(1.0, POOL[len(_MAGNITUDES) - 1]) < 0
assert math.copysign(1.0, POOL[len(_MAGNITUDES)]) > 0

# Boxes are (i, j) index pairs with i <= j, so the two zeros stay distinct.
BOXES = tuple(
    (POOL[i], POOL[j])
    for i in range(len(POOL))
    for j in range(i, len(POOL))
)


def _s(lo: float, hi: float) -> iv.IntervalArray:
    return iv.IntervalArray(shape=(), los=(lo,), his=(hi,))


def _haze(lo: float, hi: float, min_normal: float) -> tuple[float, float]:
    """The operand band the kernel actually reasons over (DAZ)."""
    return iv._elt_haze_fmt(lo, hi, min_normal)


def _samples(lo: float, hi: float) -> list[float]:
    """The points of ``[lo, hi]`` this battery drives.

    The endpoints always, plus zero when the box straddles it — the
    interior zero is the point every NaN class and every division blowup
    is built on, and an endpoints-only sampler never sees it.
    """
    pts = [lo, hi]
    if lo < 0.0 < hi:
        pts.append(0.0)
    return pts


def _signed_inf(x: float, y: float) -> float:
    return math.copysign(1.0, x) * math.copysign(1.0, y) * INF


def _exact_mul(x: float, y: float, native: bool):
    """The value ``x * y`` has under the target semantics, or ``None`` = NaN.

    ``native=True`` (binary64) returns the FLOAT product, because that is
    what the program computes and what ``ieee_mul`` claims to contain: its
    corner hull is the native product with no outward rounding, so the
    real product is the wrong yardstick and would report misses that are
    not misses. ``native=False`` (a narrower format) returns the exact
    REAL product — see the module docstring for why that is the right,
    and sufficient, check once the box has been rounded outward onto the
    format's grid.
    """
    xinf, yinf = math.isinf(x), math.isinf(y)
    if (xinf and y == 0.0) or (yinf and x == 0.0):
        return None  # 0 * inf
    if xinf or yinf:
        return _signed_inf(x, y)
    return (x * y) if native else Fraction(x) * Fraction(y)


def _exact_div(x: float, y: float, native: bool):
    """The value ``x / y`` has, or ``None`` = NaN. See :func:`_exact_mul`."""
    xinf, yinf = math.isinf(x), math.isinf(y)
    if xinf and yinf:
        return None  # inf / inf
    if x == 0.0 and y == 0.0:
        return None  # 0 / 0
    if yinf:
        # finite / inf is a SIGNED ZERO under ieee; its value is 0
        return 0.0 if native else Fraction(0)
    if y == 0.0 or xinf:
        # x/0 for x != 0 is +-inf -- a VALUE under ieee, not NaN
        return _signed_inf(x, y)
    return (x / y) if native else Fraction(x) / Fraction(y)


def _contains(box: iv.IntervalArray, value) -> bool:
    lo, hi = box.los[0], box.his[0]
    if isinstance(value, float):
        if math.isinf(value):
            return (lo == -INF) if value < 0 else (hi == INF)
        return lo <= value <= hi
    # an exact rational: an infinite endpoint is unbounded on that side,
    # so only the finite endpoints constrain it
    v = Fraction(value)
    if lo == INF or hi == -INF:
        return False  # a degenerate [+-inf, +-inf] box holds no finite value
    if math.isfinite(lo) and v < Fraction(lo):
        return False
    if math.isfinite(hi) and v > Fraction(hi):
        return False
    return True


def prefix_ieee_div(a: iv.IntervalArray, b: iv.IntervalArray):
    """``ieee_div`` AS IT WAS BEFORE THE S10 FIX, for the positive control.

    The shipped kernel sends every zero-containing divisor to ⊤. The
    version this replaced case-split on WHERE the zero sat and kept a
    boundary-aware tightening for the one-sided shapes — sound over ℝ,
    where there is one zero, and wrong under IEEE, where `[0, hi]`
    contains BOTH `+0.0` and `-0.0` and `x / -0.0` is the opposite
    infinity from `x / +0.0`. Kept here, and only here, so the sweep can
    be shown to CATCH that defect rather than merely to pass in its
    absence: a battery that has never failed is not evidence.
    """
    def f(alo, ahi, blo, bhi):
        a0, b0 = alo <= 0.0 <= ahi, blo <= 0.0 <= bhi
        ainf = alo == -iv._INF or ahi == iv._INF
        binf = blo == -iv._INF or bhi == iv._INF
        made_nan = (a0 and b0) or (ainf and binf)
        if b0:
            if blo == 0.0 and bhi == 0.0:
                return -iv._INF, iv._INF, made_nan
            if blo < 0.0 < bhi:
                return -iv._INF, iv._INF, made_nan
            # the one-sided boundary arm the fix removed. It is NOT
            # wrapped in a try: this arm could RAISE out of a kernel whose
            # contract is to degrade (`ieee_div([-inf,-inf], [-inf, 0])`
            # computed `-inf/-inf = NaN`, which IntervalArray rejects), and
            # the sweep's RAISED column exists to count exactly that.
            r = iv.boundary_div(
                iv.IntervalArray(shape=(), los=(alo,), his=(ahi,)),
                iv.IntervalArray(shape=(), los=(blo,), his=(bhi,)),
            )
            return r.los[0], r.his[0], made_nan
        corners = (alo / blo, alo / bhi, ahi / blo, ahi / bhi)
        return iv._corner_hull(corners, made_nan)

    return iv._ieee_binary(a, b, f)


def sweep_ieee(op: str, fmt_name: str, kern=None):
    """Drive ``ieee_{op}`` for one format over the whole box-pair grid.

    Returns ``(samples, failures, nan_samples, raised, misses)`` where
    ``misses`` holds up to ten offending cases for a failure message.
    ``kern`` overrides the kernel under test (the positive control).
    """
    fmt = _FLOAT_FORMATS[fmt_name]
    min_normal = _ieee_format_min_normal(fmt)
    f64 = fmt == _FLOAT_FORMATS["float64"]
    if op == "mul":
        kern = kern or (iv.ieee_mul if f64 else iv.ieee_mul_fmt)
        value_of = _exact_mul
    else:
        kern = kern or (iv.ieee_div if f64 else iv.ieee_div_fmt)
        value_of = _exact_div

    # A box in format F has F-valued endpoints. The pool is written in
    # binary64, and -FMAX64 is not a float32 value at all, so a sweep that
    # sampled the raw pool would drive the kernel with points the program
    # can never hold and then call the resulting box wrong: 18,780 such
    # "misses" on the first run of this file, every one of them the
    # sampler's error and not the kernel's. Rounding the OPERAND boxes
    # outward onto the target grid first is what the propagator does to
    # every box in that format anyway, and it only ever widens.
    grid = [_ieee_round_box(_s(lo, hi), fmt) for lo, hi in BOXES]

    samples = failures = nan_samples = raised = 0
    misses: list[str] = []
    for ba in grid:
        for bb in grid:
            alo, ahi = ba.los[0], ba.his[0]
            blo, bhi = bb.los[0], bb.his[0]
            try:
                if f64:
                    box, made_nan = kern(ba, bb)
                else:
                    box, made_nan = kern(ba, bb, min_normal)
                    box = _ieee_round_box(box, fmt)
            except iv.IntervalError:
                # A kernel whose contract is to DEGRADE must not raise;
                # counted rather than propagated so one bad pair does not
                # hide the rest of the grid.
                raised += 1
                continue
            ha = _haze(alo, ahi, min_normal)
            hb = _haze(blo, bhi, min_normal)
            for x in _samples(*ha):
                for y in _samples(*hb):
                    v = value_of(x, y, f64)
                    samples += 1
                    if v is None:
                        nan_samples += 1
                        if not made_nan:
                            failures += 1
                            if len(misses) < 10:
                                misses.append(
                                    f"{op}/{fmt_name} NaN at {x}/{y} in "
                                    f"[{alo},{ahi}],[{blo},{bhi}] unflagged"
                                )
                        continue
                    if not _contains(box, v):
                        failures += 1
                        if len(misses) < 10:
                            misses.append(
                                f"{op}/{fmt_name} [{alo},{ahi}] x "
                                f"[{blo},{bhi}] -> [{box.los[0]},"
                                f"{box.his[0]}] misses {x} {op} {y}"
                            )
    return samples, failures, nan_samples, raised, misses


def sweep_real_mul():
    """Real-mode :func:`stelling.interval.mul` against exact products.

    No haze and no NaN class: real mode has one zero, no infinities among
    its VALUES, and the kernel's obligation is plain containment of the
    exact product at every point of the operand boxes. Boxes with an
    infinite endpoint are still driven — an unbounded declared range is
    representable — and an infinite endpoint contributes no finite sample,
    so the pair contributes only the products it can actually name.
    """
    samples = failures = raised = 0
    misses: list[str] = []
    for alo, ahi in BOXES:
        for blo, bhi in BOXES:
            try:
                box = iv.mul(_s(alo, ahi), _s(blo, bhi))
            except iv.IntervalError:
                raised += 1
                continue
            for x in _samples(alo, ahi):
                if math.isinf(x):
                    continue
                for y in _samples(blo, bhi):
                    if math.isinf(y):
                        continue
                    samples += 1
                    if not _contains(box, Fraction(x) * Fraction(y)):
                        failures += 1
                        if len(misses) < 10:
                            misses.append(
                                f"real mul [{alo},{ahi}] x [{blo},{bhi}] -> "
                                f"[{box.los[0]},{box.his[0]}] misses {x}*{y}"
                            )
    return samples, failures, 0, raised, misses


# THE TABLE SOUNDNESS.md QUOTES, as asserted constants.
#
# `(failures, samples, nan_samples, raised)` per row, from the run recorded
# in the S10 entry. Asserted exactly, never bounded, and for the reason the
# sibling constant `BOUNDARY_DIV_SWEEP_QUOTIENTS` is: a figure in a
# soundness log that no run in the tree reproduces is worth less than no
# figure (audit 0.2.0 B5-4, then the follow-up finding that this larger
# table needed the same treatment). Drift in either direction reddens the
# suite with "update both or neither" instead of leaving prose to be
# believed.
POST_FIX_ROWS = {
    ("real mul", "R"): (0, 851_929, 0, 0),
    ("ieee_mul", "float64"): (0, 962_361, 29_348, 0),
    ("ieee_div", "float64"): (0, 962_361, 67_373, 0),
    ("ieee_mul_fmt", "float32"): (0, 962_361, 104_632, 0),
    ("ieee_div_fmt", "float32"): (0, 962_361, 128_657, 0),
    ("ieee_mul_fmt", "float16"): (0, 962_361, 115_784, 0),
    ("ieee_div_fmt", "float16"): (0, 962_361, 151_505, 0),
    ("ieee_mul_fmt", "bfloat16"): (0, 962_361, 104_632, 0),
    ("ieee_div_fmt", "bfloat16"): (0, 962_361, 128_657, 0),
}

# The positive control: the same sweep against the PRE-FIX `ieee_div`.
# `RAISED` is 0 here and that is not a hole — the raise the S10 entry
# records (`ieee_div([-inf,-inf], [-inf, 0])` computing `-inf/-inf`) was
# closed inside `boundary_div` itself, by a different fix, so the pre-fix
# TRANSFER driven through today's kernel degrades instead of crashing.
PRE_FIX_IEEE_DIV_F64 = (15_048, 962_361, 67_373, 0)

POOL_SIZE = 28
BOX_COUNT = 406
BOX_PAIRS = 164_836

ROWS = (
    ("real mul", "R", sweep_real_mul),
    ("ieee_mul", "float64", lambda: sweep_ieee("mul", "float64")),
    ("ieee_div", "float64", lambda: sweep_ieee("div", "float64")),
    ("ieee_mul_fmt", "float32", lambda: sweep_ieee("mul", "float32")),
    ("ieee_div_fmt", "float32", lambda: sweep_ieee("div", "float32")),
    ("ieee_mul_fmt", "float16", lambda: sweep_ieee("mul", "float16")),
    ("ieee_div_fmt", "float16", lambda: sweep_ieee("div", "float16")),
    ("ieee_mul_fmt", "bfloat16", lambda: sweep_ieee("mul", "bfloat16")),
    ("ieee_div_fmt", "bfloat16", lambda: sweep_ieee("div", "bfloat16")),
)


def run_all() -> dict:
    """Every row, keyed as :data:`POST_FIX_ROWS` is and COLUMNED AS IT IS:
    ``{(kernel, format): (failures, samples, nan_samples, raised)}``, plus a
    ``(kernel, format, "misses")`` entry wherever a row produced examples.

    **THE COLUMN ORDER USED TO BE THE OTHER ONE** — audit 0.2.0 B8a, item 7.
    This function's contract read ``(samples, fails, nan, raised)`` and it
    built exactly that, one screen below a table in the same module written
    ``(failures, samples, nan_samples, raised)`` — so the two spellings of
    "the row" in a module whose whole purpose is stopping numbers from
    drifting were transposed against each other. It was also DEAD: nothing
    in the tree called it, and the one test of the table iterated
    :data:`ROWS` itself. A dead helper documenting the opposite convention
    from its own module's constants is a trap for whoever writes the next
    caller, so it is realigned AND made the single row-runner: the test and
    the ``__main__`` reproduction both go through it now, which is what
    stops the two orders from parting again.
    """
    out = {}
    for name, fmt, fn in ROWS:
        s, f, n, r, misses = fn()
        out[(name, fmt)] = (f, s, n, r)
        if misses:
            out[(name, fmt, "misses")] = misses
    return out


if __name__ == "__main__":  # pragma: no cover - reproduction entry point
    print(f"pool={len(POOL)} boxes={len(BOXES)} "
          f"box_pairs={len(BOXES) ** 2}")
    print(f"{'kernel':<14}{'format':<10}{'failures/samples':>26}"
          f"{'NaN':>12}{'RAISED':>9}")
    rows = run_all()
    for name, fmt, _fn in ROWS:
        f, s, n, r = rows[(name, fmt)]
        print(f"{name:<14}{fmt:<10}{f'{f:,} / {s:,}':>26}{n:>12,}{r:>9}")
        for m in rows.get((name, fmt, "misses"), ()):
            print(f"    {m}")
